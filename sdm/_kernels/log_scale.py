# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import Tensor

_LogScale: TypeAlias = Callable[
    [Tensor, Tensor, Tensor, Tensor | None], Tensor
]

_triton_log_scale: _LogScale | None = None
try:
    from sdm._kernels.triton.log_scale import (
        log_scale as _triton_log_scale_impl,
    )
except ImportError:
    pass
else:
    _triton_log_scale = _triton_log_scale_impl


def log_scale(
    query: Tensor,
    log_key_len: Tensor,
    head_scale: Tensor,
    gate: Tensor | None = None,
) -> Tensor:
    """Scale query heads, preserving each eager operation's rounding."""
    inputs = (query, log_key_len, head_scale)
    if gate is not None:
        inputs += (gate,)
    if (
        _triton_log_scale is not None
        and query.is_cuda
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and query.numel() > 0
        and log_key_len.numel() == 1
        and log_key_len.dim() <= query.dim() - 2
        and all(
            x.device == query.device
            and x.dtype == query.dtype
            and x.is_contiguous()
            for x in inputs
        )
        and (gate is None or gate.shape == query.shape)
        and not (
            torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
        )
    ):
        return _triton_log_scale(query, log_key_len, head_scale, gate)

    head_scale = head_scale.reshape(*((1,) * (query.dim() - 2)), -1, 1)
    out = query * log_key_len[..., None, None] * head_scale
    if gate is not None:
        out = out * (1 + gate.tanh())
    return out
