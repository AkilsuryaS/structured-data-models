# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor


def _ndtri(x: Tensor) -> Tensor:
    # MPS does not implement 'torch.special.ndtri', so use 'erfinv' instead.
    if x.device.type == "mps":
        return x.mul(2.0).sub_(1.0).erfinv_().mul_(math.sqrt(2.0))
    return torch.special.ndtri(x)


def _isfinite(x: Tensor) -> Tensor:
    # Equal to 'x.isfinite()', which allocates 'x.abs()' on the way.
    return x.gt(-math.inf).logical_and_(x.lt(math.inf))


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    num_samples: int | Tensor,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = num_samples * eps * var + (num_samples * mean * eps) ** 2
    return var <= upper_bound


def _standardize_stats(
    numerical: Tensor, *, eps: float = 0.0
) -> tuple[Tensor, Tensor]:
    finite = _isfinite(numerical)
    count = finite.sum(dim=-2, keepdim=True)
    finite_or_nan = numerical.masked_fill(~finite, torch.nan)

    mean = finite_or_nan.nansum(-2, keepdim=True).div_(count)
    mean.masked_fill_(mean.isnan(), 0.0)

    var = finite_or_nan.sub_(mean).square_().nansum(-2, keepdim=True)
    var /= count
    var.masked_fill_(var.isnan(), 0.0)

    scale = var.sqrt()
    if eps == 0:
        mask = _constant_feature_mask(var, mean, num_samples=count)
        scale[mask] = 1.0
    else:
        scale += eps
    return mean, scale
