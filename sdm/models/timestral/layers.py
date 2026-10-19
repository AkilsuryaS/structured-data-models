# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor, nn

from sdm.models.timestral.block import TimestralTransformerBlock
from sdm.models.timestral.config import LAYER_KIND_AXES
from sdm.nn import RMSNorm, RotaryEmbedding


class _Transformer(nn.Module):
    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        rope = RotaryEmbedding(
            channels=embed_dim // num_heads,
            layout="split_half",
            theta=rope_theta,
            partial_rotary_factor=rope_dim / (embed_dim // num_heads),
            requires_grad=False,
        )
        self.axes = tuple(
            is_time
            for kind in block_schedule
            for is_time in LAYER_KIND_AXES[kind]
        )
        self.layers = nn.ModuleList()
        for kind in block_schedule:
            axes = LAYER_KIND_AXES[kind]
            for index, is_time in enumerate(axes):
                self.layers.append(
                    TimestralTransformerBlock(
                        channels=embed_dim,
                        num_heads=num_heads,
                        hidden_channels=mlp_hidden_dim
                        if index == len(axes) - 1
                        else None,
                        rope=rope if is_time else None,
                    )
                )
        self.out_norm = RMSNorm(embed_dim, eps=norm_eps)

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor | None,
        variate_mask: Tensor | None,
        positions: Tensor,
    ) -> Tensor:
        for is_time, layer in zip(self.axes, self.layers):
            if not is_time:
                x = x.transpose(0, 1)
            x = layer(
                query=x,
                attn_mask=time_mask if is_time else variate_mask,
                query_pos=positions if is_time else None,
                is_causal=is_time and time_mask is None,
            )
            if not is_time:
                x = x.transpose(0, 1)
        return self.out_norm(x)
