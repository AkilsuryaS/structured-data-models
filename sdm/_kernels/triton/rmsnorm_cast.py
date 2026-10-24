# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _load_input(
    x_ptr,
    cos_ptr,
    sin_ptr,
    row,
    c,
    mask,
    channels: tl.constexpr,
    seq_len,
    num_heads,
    row_length: tl.constexpr = 0,
    row_stride=0,
    token_stride=0,
    channel_stride: tl.constexpr = 1,
) -> tl.tensor:
    if row_length == 0:
        offsets = row[:, None] * channels
    else:
        offsets = (
            (row // row_length) * row_stride
            + (row % row_length) * token_stride
        )[:, None]
    value = tl.load(
        x_ptr + offsets + c * channel_stride, mask=mask, other=0
    ).to(tl.float32)
    if cos_ptr is not None:
        first = c < channels // 2
        partner = tl.where(first, c + channels // 2, c - channels // 2)
        other = tl.load(
            x_ptr + offsets + partner * channel_stride, mask=mask, other=0
        ).to(tl.float32)
        table = (row // num_heads % seq_len)[:, None] * (channels // 2)
        table += c % (channels // 2)
        cos = tl.load(cos_ptr + table, mask=mask, other=0).to(tl.float32)
        sin = tl.load(sin_ptr + table, mask=mask, other=0).to(tl.float32)
        # Eager RoPE rounds both products and their sum in the input dtype.
        dtype = x_ptr.dtype.element_ty
        a = (value * cos).to(dtype).to(tl.float32)
        b = (other * sin).to(dtype).to(tl.float32)
        value = tl.where(first, a - b, a + b).to(dtype).to(tl.float32)
    return value


@triton.jit(
    do_not_specialize=[
        "rows",
        "seq_len",
        "num_heads",
        "row_stride",
        "token_stride",
    ]
)
def _rmsnorm_cast_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    rows,
    seq_len,
    num_heads,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
    weight2_ptr=None,
    out2_ptr=None,
    row_length: tl.constexpr = 0,
    row_stride=0,
    token_stride=0,
    channel_stride: tl.constexpr = 1,
) -> None:
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    offsets = row[:, None] * channels
    mask = row[:, None] < rows
    groups = tl.arange(0, channels // 4)[None, :]
    total = tl.full((block_rows, channels // 4), 0, tl.float32)
    for i in tl.static_range(4):
        v = _load_input(  # ty: ignore[invalid-argument-type]
            x_ptr,
            cos_ptr,
            sin_ptr,
            row,
            groups * 4 + i,
            mask,
            channels,
            seq_len,
            num_heads,
            row_length,
            row_stride,
            token_stride,
            channel_stride,
        )
        total = tl.fma(v, v, total)
    if channels == 32:
        # Preserve CUDA's shuffle order when Triton packs adjacent groups.
        for shift in tl.static_range(2, -1, -1):
            indices = tl.broadcast_to(groups ^ (1 << shift), (block_rows, 8))
            total += tl.gather(total, indices, axis=1)
        total = tl.where(groups == 0, total, 0)
    variance = tl.sum(total, 1)  # ty: ignore[invalid-argument-type]
    c = tl.arange(0, channels)[None, :]
    value = _load_input(  # ty: ignore[invalid-argument-type]
        x_ptr,
        cos_ptr,
        sin_ptr,
        row,
        c,
        mask,
        channels,
        seq_len,
        num_heads,
        row_length,
        row_stride,
        token_stride,
        channel_stride,
    )
    value *= tl.rsqrt(variance[:, None] / channels + eps)
    # Each affine is applied before its BF16 output store. Never quantize the
    # unweighted normalized value and reuse it as an intermediate.
    value1 = value
    if weight_ptr is not None:
        value1 = value1 * tl.load(weight_ptr + c).to(tl.float32)
    tl.store(out_ptr + offsets + c, value1, mask=mask)
    if out2_ptr is not None:
        value2 = value
        if weight2_ptr is not None:
            value2 = value2 * tl.load(weight2_ptr + c).to(tl.float32)
        tl.store(out2_ptr + offsets + c, value2, mask=mask)


def rmsnorm_cast(
    x: Tensor,
    weight: Tensor | None,
    eps: float,
    rope: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    x = x.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    cos, sin = (
        (None, None)
        if rope is None
        else (table.contiguous() for table in rope)
    )
    channels = x.size(-1)
    rows = x.numel() // channels
    block_rows = 8
    grid = ((rows + block_rows - 1) // block_rows,)
    out = torch.empty_like(x, dtype=torch.get_autocast_dtype("cuda"))
    with torch.cuda.device(x.device):
        cast(Any, _rmsnorm_cast_kernel)[grid](
            x,
            weight,
            cos,
            sin,
            out,
            rows,
            x.size(-3) if rope is not None else 1,
            x.size(-2) if rope is not None else 1,
            channels,
            eps,
            block_rows,
            weight2_ptr=None,
            out2_ptr=None,
            row_length=0,
            row_stride=0,
            token_stride=0,
            channel_stride=1,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def paired_rmsnorm_cast(
    x: Tensor,
    weight1: Tensor | None,
    weight2: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Inference-only paired affine RMSNorm for a promised shared input.

    Supports 2-D rows and 3-D row/token tensors directly, including transposed
    row/token storage. Single-output calls keep the original kernel defaults.
    """
    channels = x.size(-1)
    rows = x.numel() // channels
    output_dtype = torch.get_autocast_dtype("cuda")
    out1 = torch.empty(x.shape, device=x.device, dtype=output_dtype)
    out2 = torch.empty(x.shape, device=x.device, dtype=output_dtype)
    weight1 = None if weight1 is None else weight1.contiguous()
    weight2 = None if weight2 is None else weight2.contiguous()
    row_length = x.size(-2) if x.dim() == 3 else 1
    row_stride = x.stride(-3) if x.dim() == 3 else x.stride(-2)
    token_stride = x.stride(-2) if x.dim() == 3 else 0
    block_rows = 8
    with torch.cuda.device(x.device):
        cast(Any, _rmsnorm_cast_kernel)[
            ((rows + block_rows - 1) // block_rows,)
        ](
            x,
            weight1,
            None,
            None,
            out1,
            rows,
            1,
            1,
            channels,
            eps,
            block_rows,
            weight2_ptr=weight2,
            out2_ptr=out2,
            row_length=row_length,
            row_stride=row_stride,
            token_stride=token_stride,
            channel_stride=x.stride(-1),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out1, out2
