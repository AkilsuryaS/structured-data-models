# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental BF16 GEMM with a rounded, exact-erf GELU epilogue."""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice


@triton.jit
def _linear_bias_gelu(
    A,
    W,
    B,
    OUT,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FUSE_GELU: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = (pid % width) // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Identical tiled traversal and FP32 dot-add accumulation to the selected
    # Inductor MM template. W is the original contiguous [N,K] parameter.
    am = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    bn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ki in range(tl.cdiv(K, BLOCK_K)):
        kk = rk + ki * BLOCK_K
        a = tl.load(A + am[:, None] * K + kk[None, :])
        w = tl.load(W + bn[None, :] * K + kk[:, None])
        acc += tl.dot(a, w, allow_tf32=False, out_dtype=tl.float32)
    bias = tl.load(B + rn, rn < N, 0).to(tl.float32)
    rounded = (acc + bias[None, :]).to(tl.bfloat16)
    if FUSE_GELU:
        x = rounded.to(tl.float32)
        half = x * 0.5
        erf = libdevice.erf(x * 0.7071067811865476)
        result = half * (erf + 1.0)
    else:
        result = rounded
    tl.store(
        OUT + rm[:, None] * N + rn[None, :],
        result,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@torch.library.triton_op("sdm_round3::linear_bias_gelu_bf16", mutates_args={})
def linear_bias_gelu_bf16(
    a: Tensor, w: Tensor, b: Tensor, fuse_gelu: bool = True
) -> Tensor:
    """BF16, 2-D, contiguous operands; final two dimensions fixed by model."""
    torch._check(a.dim() == 2)
    torch._check(w.dim() == 2)
    torch._check(b.dim() == 1)
    k, n = a.shape[1], w.shape[0]
    torch._check((k == 1024 and n == 2048) or (k == 256 and n == 512))
    torch._check(w.shape[1] == k)
    torch._check(b.shape[0] == n)
    torch._check(a.shape[0] >= (131072 if k == 256 else 2048))
    torch._check(a.shape[0] <= (2097152 if k == 256 else 262144))
    assert a.dtype == w.dtype == b.dtype == torch.bfloat16
    assert a.is_contiguous()
    assert w.is_contiguous()
    assert b.is_contiguous()
    assert a.device == w.device == b.device
    assert a.device.type == "cuda"
    out = torch.empty((a.shape[0], n), device=a.device, dtype=torch.bfloat16)
    torch.library.wrap_triton(_linear_bias_gelu)[
        (triton.cdiv(a.shape[0], 128) * triton.cdiv(n, 128),)
    ](
        a,
        w,
        b,
        out,
        a.shape[0],
        N=n,
        K=k,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_M=8,
        FUSE_GELU=fuse_gelu,
        num_warps=4,
        num_stages=3,
    )
    return out
