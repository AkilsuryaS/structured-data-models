# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, Linear, ModuleList, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import LogScale, RMSNorm


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        num_key_value_heads_for_query: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.kv_heads = num_key_value_heads_for_query

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, bias=False, **factory_kwargs)

        self.layers = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                query_scaling=LogScale(
                    num_heads=num_heads,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.norm = RMSNorm(channels, **factory_kwargs)
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, out_channels, **factory_kwargs),
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, out_channels]
        R_train = y.size(-1)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))  # [..., R_train, D]

            x[..., :R_train, :] += y_emb.to(x.dtype)
            del y_emb

        for i, layer in enumerate(self.layers):
            cache_key = f"icl_block.layer{i}"
            last_layer = i == len(self.layers) - 1

            if cache is not None and cache.is_recording:
                query = x[..., R_train:, :] if last_layer else x
                result, key, value = _forward_context_layer(
                    layer=layer,
                    x=x,
                    num_context_rows=R_train,
                    num_cache_heads=self.kv_heads,
                    last_layer=last_layer,
                )
                if torch.is_grad_enabled():
                    x = result
                else:
                    query.copy_(result)
                    x = query
                cache[cache_key] = KVCacheEntry(key, value)
                continue

            if self.kv_heads is None or cache is not None:
                result = layer(
                    query=x[..., R_train:, :] if last_layer else x,
                    key_value=(
                        cast(KVCacheEntry, cache[cache_key])
                        if cache is not None and cache.is_replaying
                        else x[..., :R_train, :]
                    ),
                    return_key_value=cache is not None and cache.is_recording,
                    out=None
                    if torch.is_grad_enabled()
                    else x[..., R_train:, :]
                    if last_layer
                    else x,
                )

                x = result
                del result
                continue

            x_context, (key, value) = layer(
                query=x[..., :0, :] if last_layer else x[..., :R_train, :],
                key_value=x[..., :R_train, :],
                return_key_value=True,
                out=None
                if torch.is_grad_enabled()
                else x[..., :0, :]
                if last_layer
                else x[..., :R_train, :],
            )
            key_value = KVCacheEntry(
                key=key[..., : self.kv_heads, :].contiguous(),
                value=value[..., : self.kv_heads, :].contiguous(),
            )
            del key, value
            x_query = layer(
                query=x[..., R_train:, :],
                key_value=key_value,
                out=None if torch.is_grad_enabled() else x[..., R_train:, :],
            )
            if last_layer:
                x = x_query
            elif torch.is_grad_enabled():
                x = torch.cat([x_context, x_query], -2)

        return self.head(self.norm(x))


def _forward_context_layer(
    layer: torch.nn.Module,
    x: Tensor,
    num_context_rows: int,
    num_cache_heads: int | None,
    last_layer: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    query = x[..., num_context_rows:, :] if last_layer else x
    out = None
    if not torch.is_grad_enabled():
        # Compiled regions cannot return an alias of their input buffer.
        out = (
            torch.empty_like(query) if torch.compiler.is_compiling() else query
        )
    x, (key, value) = layer(
        query=query,
        key_value=x[..., :num_context_rows, :],
        return_key_value=True,
        out=out,
    )
    if num_cache_heads is not None:
        key = key[..., :num_cache_heads, :].contiguous()
        value = value[..., :num_cache_heads, :].contiguous()
    return x, key, value


if hasattr(torch.compiler, "nested_compile_region"):
    # Compact each layer's cache before proceeding to the next compiled layer.
    _forward_context_layer = torch.compiler.nested_compile_region(
        _forward_context_layer
    )
