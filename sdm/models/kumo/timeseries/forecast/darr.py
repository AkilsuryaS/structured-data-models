# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor

# Kumo-TS af74798bbf7d020d3298f545fb9be383560a18ce:
# Kumo-Forecast/sdk/forecasting.py and Kumo-Forecast/backbone.py.


def embed(encoded: Tensor) -> Tensor:
    """Pool and normalize encoded patches for cosine retrieval."""
    pooled = encoded.float().mean(dim=-3).mean(dim=-2)
    pooled = pooled.nan_to_num(nan=0, posinf=0, neginf=0)
    return F.normalize(pooled, dim=-1, eps=1e-12)


def knn(
    query: Tensor,  # [..., D], unit-normalized
    keys: Tensor,  # [..., N, D], unit-normalized
    values: Tensor,  # [..., N, C, H]
    k: int,
    temperature: float | None,
) -> Tensor:  # [..., C, H]
    """Retrieve cosine neighbors and average their observed continuations."""
    with torch.amp.autocast(query.device.type, enabled=False):
        similarity = query.float().unsqueeze(-2) @ keys.float().transpose(
            -1, -2
        )
        similarity, index = similarity.squeeze(-2).topk(
            k=min(k, keys.size(-2)), dim=-1
        )
    if temperature is None or temperature < 0:
        weights = torch.ones_like(similarity) / similarity.size(-1)
    else:
        weights = (similarity / max(temperature, 1e-12)).softmax(dim=-1)
    neighbors = values.gather(
        dim=-3,
        index=index[..., None, None].expand(
            *index.size(), *values.size()[-2:]
        ),
    ).float()  # [..., K, C, H]
    valid = neighbors.isfinite()
    weights = weights[..., None, None] * valid
    total = (weights * neighbors.masked_fill(~valid, 0)).sum(dim=-3)
    return total / weights.sum(dim=-3).clamp_min(1e-12)
