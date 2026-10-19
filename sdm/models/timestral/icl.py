# Copyright (c) 2026 Mistral AI
# Licensed under Apache-2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor, nn

from sdm.models.timestral.block import TimestralTransformerBlock
from sdm.nn import RMSNorm, RotaryEmbedding


class ICLBlock(nn.Module):
    """Attention stack from `Timestral <https://github.com/mistralai/timestral>`_.

    Every layer applies row attention, followed by column attention every
    ``column_every`` layers, and one MLP. Row attention uses rotary positions.
    A ``None`` row mask selects causal attention; a ``None`` column mask
    allows unrestricted attention.

    Args:
        channels: Hidden channels.
        num_layers: Number of layers.
        num_heads: Number of attention heads.
        hidden_channels: MLP hidden channels.
        column_every: Apply column attention every this many layers.
        device: The device.
        dtype: The data type.
    """

    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        hidden_channels: int,
        column_every: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=10_000,
            partial_rotary_factor=0.5,
            requires_grad=False,
            **factory_kwargs,
        )
        self._row_attention: list[bool] = []
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            column = (i + 1) % column_every == 0
            self._row_attention.append(True)
            self.layers.append(
                TimestralTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    hidden_channels=None if column else hidden_channels,
                    rope=rope,
                    **factory_kwargs,
                )
            )
            if column:
                self._row_attention.append(False)
                self.layers.append(
                    TimestralTransformerBlock(
                        channels=channels,
                        num_heads=num_heads,
                        hidden_channels=hidden_channels,
                        **factory_kwargs,
                    )
                )
        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        row_mask: Tensor | None,
        col_mask: Tensor | None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """Apply attention across rows and columns.

        Args:
            x: Input with shape ``[..., C, R, D]``, with columns ``C``,
                row tokens ``R``, and hidden channels ``D``.
            row_mask: Allowed row attention, broadcastable to
                ``[..., C, H, R, R]``, where ``H`` is the number of heads,
                or ``None`` for causal attention.
            col_mask: Allowed column attention, broadcastable to
                ``[..., R, H, C, C]``, or ``None`` for unrestricted attention.
            pos: Row positions with shape ``[R]`` or ``[..., C, R]``.
                Defaults to contiguous positions.

        Returns:
            Tensor with the same shape as ``x``.
        """
        for i, (is_row, layer) in enumerate(
            zip(self._row_attention, self.layers)
        ):
            if not is_row:
                x = x.transpose(-3, -2)
            x = layer(
                query=x,
                attn_mask=row_mask if is_row else col_mask,
                query_pos=pos if is_row else None,
                is_causal=is_row and row_mask is None,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() or i == 0 else x,
            )
            if not is_row:
                x = x.transpose(-3, -2)
        return self.norm(x)
