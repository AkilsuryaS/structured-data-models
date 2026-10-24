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
    from sdm._kernels.triton.linear_gelu_bf16 import linear_bias_gelu_bf16


def linear_gelu(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Apply exact GELU after linear; keep embedding fusion opt-in."""
    if (
        _SUPPORTED_DEVICE
        and not torch.is_grad_enabled()
        and x.device.type == "cuda"
        and x.device.index == _SUPPORTED_DEVICE_INDEX
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and x.dim() >= 2
        and bias is not None
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias.is_contiguous()
        and x.dtype in (torch.float32, torch.bfloat16)
        and weight.dtype in (torch.float32, torch.bfloat16)
        and bias.dtype in (torch.float32, torch.bfloat16)
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
        if icl or embedding:
            shape = x.shape[:-1]
            a = x.reshape(-1, k).to(torch.bfloat16)
            w = weight.to(torch.bfloat16)
            b = bias.to(torch.bfloat16)
            return linear_bias_gelu_bf16(a, w, b).view(*shape, weight.shape[0])
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
