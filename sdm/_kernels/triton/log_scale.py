# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["numel"])
def _log_scale_kernel(
    query_ptr,
    length_ptr,
    head_ptr,
    gate_ptr,
    out_ptr,
    numel,
    heads: tl.constexpr,
    channels: tl.constexpr,
    block: tl.constexpr,
) -> None:
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    mask = index < numel
    query = tl.load(query_ptr + index, mask=mask, other=0)
    dtype = query.dtype
    length = tl.load(length_ptr).to(tl.float32)
    head = tl.load(head_ptr + index // channels % heads).to(tl.float32)
    # Eager execution rounds each product and gate operation separately.
    out = (query.to(tl.float32) * length).to(dtype).to(tl.float32)
    out = (out * head).to(dtype).to(tl.float32)
    if gate_ptr is not None:
        gate = tl.load(gate_ptr + index, mask=mask, other=0).to(tl.float32)
        gate = libdevice.tanh(gate).to(dtype).to(tl.float32)
        gate = (1.0 + gate).to(dtype).to(tl.float32)
        out = out * gate
    tl.store(out_ptr + index, out, mask=mask)


def log_scale(
    query: Tensor,
    log_key_len: Tensor,
    head_scale: Tensor,
    gate: Tensor | None,
) -> Tensor:
    out = torch.empty_like(query)
    grid = ((query.numel() + 1023) // 1024,)
    with torch.cuda.device(query.device):
        cast(Any, _log_scale_kernel)[grid](
            query,
            log_key_len,
            head_scale,
            gate,
            out,
            query.numel(),
            query.size(-2),
            query.size(-1),
            1024,
            enable_fp_fusion=False,
        )
    return out
