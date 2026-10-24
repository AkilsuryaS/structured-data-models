# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in SM120 inference prototype with the original PyTorch fallback.

The fixed tile and shape families were benchmarked with Torch 2.14.
Unsupported devices, layouts, shapes and gradient-enabled calls use PyTorch.
"""

import os
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

_ENABLED = os.environ.get("SDM_EXPERIMENTAL_MLP_GELU_SM120") == "1"
_EMBEDDING_ENABLED = (
    os.environ.get("SDM_EXPERIMENTAL_MLP_GELU_EMBEDDING") == "1"
)
_SUPPORTED_DEVICE = (
    _ENABLED
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
)

_SUPPORTED_DEVICE_INDEX = (
    torch.cuda.current_device() if _SUPPORTED_DEVICE else None
)

if _SUPPORTED_DEVICE:
    from sdm._kernels.triton.linear_gelu import linear_bias_gelu


def linear_gelu(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Apply exact GELU after linear; keep embedding fusion opt-in.

    CUDA autocast selects FP16 or BF16. Without autocast, matching operands
    retain their FP16, BF16 or FP32 dtype, including the linear-output round
    before GELU. FP32 dot products avoid TF32 but measured slower than PyTorch
    at a screened shape; this entire prototype is experimental and opt-in.
    """
    if (
        _SUPPORTED_DEVICE
        and not torch.is_grad_enabled()
        and x.device.type == "cuda"
        and x.device.index == _SUPPORTED_DEVICE_INDEX
        and x.dim() >= 2
        and bias is not None
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and bias.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        k = x.shape[-1]
        rows = x.numel() // k if k in (256, 1024) else 0
        icl = (
            k == 1024
            and weight.shape == (2048, 1024)
            and 2048 <= rows <= 262144
        )
        embedding = (
            _EMBEDDING_ENABLED
            and k == 256
            and weight.shape == (512, 256)
            and 131072 <= rows <= 2097152
        )
        autocast = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda") if autocast else x.dtype
        matching_dtype = autocast or x.dtype == weight.dtype == bias.dtype
        if (icl or embedding) and matching_dtype:
            shape = x.shape[:-1]
            a = x.reshape(-1, k).to(dtype)
            w = weight.to(dtype)
            b = bias.to(dtype)
            return linear_bias_gelu(a, w, b).view(*shape, weight.shape[0])
    return F.gelu(F.linear(x, weight, bias), approximate="none")


class LinearGeluMLP(torch.nn.Sequential):
    """Keep the original Sequential checkpoint parameter names."""

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        x = input
        if not _SUPPORTED_DEVICE or torch.is_grad_enabled():
            return super().forward(x)
        if len(self) == 4:
            x = self[0](x)
            first, second = self[1], self[3]
        else:
            first, second = self[0], self[2]
        first = cast(torch.nn.Linear, first)
        return second(linear_gelu(x, first.weight, first.bias))
