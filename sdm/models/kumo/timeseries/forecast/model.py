# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecast.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.forecast.cross import CrossChannelAttention
from sdm.models.kumo.timeseries.forecast.darr import _embedding, _knn_forecast
from sdm.models.kumo.timeseries.forecast.icl import ICLBlock
from sdm.models.kumo.timeseries.forecast.recipe import default_recipe
from sdm.nn import RevIN, SinusoidalEmbedding
from sdm.tensor.table import TableSchema


class KumoForecast(ICLModel):
    r"""The time series forecasting foundation model from the
    `"Memory-Augmented Forecasting: Scalability and Generalization Across
    Temporal Domains"
    <https://ieeexplore.ieee.org/abstract/document/11402442>`__ paper.

    .. figure:: /images/kumo_forecast_light.png
        :width: 100%

    :class:`KumoForecast` is a time-series foundation model for multivariate
    forecasting, with optional historical covariates. Architecturally, it
    combines patch-based time-series embeddings with a
    `MOMENT <https://arxiv.org/abs/2402.03885>`__ transformer encoder and a
    pretrained forecasting head.
    Optional cross-channel mixing lets forecasts use dependencies between
    variates after temporal encoding. The forecasting head predicts the native
    up to 72-step horizon directly from a 512-step history.
    Set ``alpha < 1`` to blend direct forecasts with DARR retrieval from the
    full supplied history. Retrieval uses complete historical windows and
    their observed continuations, spaced 72 steps apart. Only the last 512
    observations are encoded for the current forecast. If history is too
    short for a complete retrieval window, the model forecasts directly.
    Horizons beyond 72 steps use autoregressive direct forecasts, up to 512.

    All context observations must be available at the forecast origin.
    Leading batch dimensions have independent retrieval memories. Non-finite
    continuation values are excluded from the retrieved average; an entirely
    missing continuation uses zero in the recipe's transformed target space.

    Within the :class:`~sdm.models.ICLModel` protocol, :class:`KumoForecast`
    treats rows as ordered time steps. In a default forward pass,
    ``x_context`` contains past-only historical covariates, ``y_context``
    contains one or more past target series, and ``x_query`` has no columns.
    The number of query rows determines the forecast horizon.

    Args:
        cross_channel: Whether to mix variates after encoding.
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
        alpha: Weight of the direct forecast in
            ``alpha * direct + (1 - alpha) * retrieved``. Defaults to ``1``
            to disable retrieval; upstream DARR defaults to ``0.01``.
        k: Maximum number of historical neighbors to retrieve.
        temperature: Softmax temperature for cosine similarities. ``None`` or
            a negative value uses uniform weights; zero uses ``1e-12``.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_multi_target: ClassVar[bool] = True
    supports_context_only_columns: ClassVar[bool] = True
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        cross_channel: bool = True,
        pretrained: bool = True,
        device: torch.device | str | None = None,
        *,
        alpha: float = 1.0,
        k: int = 64,
        temperature: float | None = 0.05,
    ) -> None:
        super().__init__(task=Task.regression)
        if not 0 <= alpha <= 1:
            raise ValueError("Expected 'alpha' between zero and one")
        self.alpha = alpha
        self.k = k
        self.temperature = temperature

        self.model = _KumoForecast(
            cross_channel=cross_channel,
            device="meta" if pretrained else device,
        )

        if pretrained:
            self._load_from_pretrained(cross_channel, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        use_cross_channel: bool,
        device: torch.device | str | None,
    ) -> KumoForecast:
        device = torch.get_default_device() if device is None else device

        path = download_checkpoint(
            repo_id="nvidia/Kumo-Forecast",
            filename="run8_best_model_cr.pt"
            if use_cross_channel
            else "moment_head_512_6hr.pt",
            revision="abff20a58834638b28227ff4ab934f26206e4b09",
        )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(remap_ckpt(ckpt), assign=True)

        return self

    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)
        if x_query.size(-1) > 0:
            warn_once(
                key="kumo-forecast-future-covariate",
                message=(
                    f"{self.__class__.__name__!r} does not support future "
                    f"covariates. Query columns will be ignored."
                ),
            )
            x_query = x_query[..., :0]

        return super().forward(x_context, y_context, x_query, *args, **kwargs)

    def fit(self, x: Tensor | TableTensor, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        if isinstance(x, TableTensor):
            kwargs["context_only_columns"] = [
                column for columns in x.columns.values() for column in columns
            ]
        else:
            kwargs["context_only_columns"] = range(x.size(-1))

        return super().fit(x, *args, **kwargs)

    def predict(
        self,
        x: Tensor | TableTensor,
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if x.size(-1) > 0:
            warn_once(
                key="kumo-forecast-future-covariate",
                message=(
                    f"{self.__class__.__name__!r} does not support future "
                    f"covariates. Query columns will be ignored."
                ),
            )
            x = x[..., :0]

        return super().predict(x, *args, **kwargs)

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,  # [..., R_context, D_only]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> TableTensor:

        if (
            y_context is not None
            and y_context.size(-2) < self.model.context_size
        ):
            raise ValueError(
                f"Expected at least {self.model.context_size} context rows "
                f"(got {y_context.size(-2)})"
            )

        if x_query is not None and not 1 <= x_query.size(-2) <= 512:
            raise ValueError(
                "Number of query rows must be between 1 and 512 steps "
                f"(got {x_query.size(-2)})"
            )

        length = 0 if x_query is None else x_query.size(-2)
        if cache is None or cache.is_recording:
            assert y_context is not None
            y_schema = y_context.schema
            x = y_context.numerical
            if x_context_only is not None and x_context_only.size(-1) > 0:
                if self.model.cross_channel is None and self.alpha == 1:
                    warn_once(
                        key="kumo-forecast-historical-covariate",
                        message=(
                            "Historical covariates have no effect when "
                            "'cross_channel' is disabled."
                        ),
                    )
                x = torch.cat([x, x_context_only.numerical], dim=-1)

            x = x.transpose(-1, -2)  # [..., C, R]
            history = x[..., -self.model.context_size :]
            encoded, mean, scale = self.model.encode(history)
            out = self.model.decode(encoded, mean, scale)
            keys = query = None
            memory_horizon = max(self.model.prediction_size, length)
            if self.alpha < 1 and x.size(-1) >= (
                self.model.context_size + memory_horizon
            ):
                keys = self.model.encode_memory(
                    x=x, horizon=memory_horizon, batch_size=batch_size
                )
                query = _embedding(encoded)

            if cache is not None:
                cache["history"] = x if keys is not None else history.clone()
                cache["out"] = out
                cache["keys"] = keys
                cache["query"] = query
        else:
            y_schema = cast(TableSchema, cache["y_schema"])
            x = cast(Tensor, cache["history"])
            out = cast(Tensor, cache["out"])
            keys = cast(Tensor | None, cache["keys"])
            query = cast(Tensor | None, cache["query"])

        memory_horizon = max(self.model.prediction_size, length)
        has_memory = keys is not None and x.size(-1) >= (
            self.model.context_size + memory_horizon
        )
        if length > self.model.prediction_size and (
            self.alpha != 0 or not has_memory
        ):
            history = x[..., -self.model.context_size :]
            direct = out
            rollout = [direct]
            for _ in range(
                self.model.prediction_size, length, self.model.prediction_size
            ):
                history = torch.cat([history, direct], dim=-1)
                history = history[..., -self.model.context_size :]
                direct = self.model(history)
                rollout.append(direct)
            out = torch.cat(rollout, dim=-1)
        out = out[..., :length]

        if length > 0 and has_memory:
            assert keys is not None
            assert query is not None
            windows = x.unfold(
                -1,
                self.model.context_size + memory_horizon,
                self.model.prediction_size,
            ).transpose(-3, -2)  # [..., N, C, L + H]
            retrieved = _knn_forecast(
                query=query,
                keys=keys[..., : windows.size(-3), :],
                values=windows[
                    ...,
                    self.model.context_size : self.model.context_size + length,
                ],
                k=self.k,
                temperature=self.temperature,
            )
            out = (
                retrieved
                if self.alpha == 0
                else self.alpha * out + (1 - self.alpha) * retrieved
            )

        return TableTensor(
            columns={Stype.numerical: y_schema.columns[Stype.numerical]},
            numerical=out[
                ..., : len(y_schema.columns[Stype.numerical]), :
            ].transpose(-1, -2),
        )


class _KumoForecast(torch.nn.Module):
    def __init__(
        self,
        cross_channel: bool,
        context_size: int = 512,
        prediction_size: int = 72,
        patch_size: int = 8,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        cross_channel_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.context_size = context_size
        self.prediction_size = prediction_size
        self.patch_size = patch_size

        self.revin = RevIN(eps=1e-5, **factory_kwargs)
        self.lin = torch.nn.Linear(
            in_features=patch_size,
            out_features=channels,
            bias=False,
            **factory_kwargs,
        )
        torch.nn.init.orthogonal_(self.lin.weight, gain=1.41)
        self.pe = SinusoidalEmbedding()

        self.icl = ICLBlock(
            channels=channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_buckets=32,
            max_distance=128,
            **factory_kwargs,
        )
        self.cross_channel: CrossChannelAttention | None = None
        if cross_channel:
            self.cross_channel = CrossChannelAttention(
                channels=channels,
                num_heads=cross_channel_heads,
                **factory_kwargs,
            )

        self.head = torch.nn.Linear(
            in_features=channels * (context_size // patch_size),
            out_features=prediction_size,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., C, R]
    ) -> Tensor:  # [..., C, prediction_size]
        """Forecast from the shared patch encoder."""
        encoded, mean, scale = self.encode(x)
        return self.decode(encoded, mean, scale)

    def decode(self, encoded: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """Restore forecast units after applying the forecasting head."""
        return self.revin.inverse(self.head(encoded.flatten(-2)), mean, scale)

    def encode_memory(
        self,
        x: Tensor,  # [..., C, R]
        horizon: int,
        batch_size: int,
    ) -> Tensor:  # [..., N, D]
        """Encode historical windows with fully observed continuations."""
        windows = x.unfold(
            -1, self.context_size + horizon, self.prediction_size
        ).transpose(-3, -2)  # [..., N, C, L + H]
        keys = x.new_empty(
            (*x.size()[:-2], windows.size(-3), self.lin.out_features),
            dtype=torch.float32,
        )
        for start in range(0, windows.size(-3), batch_size):
            history = windows[
                ..., start : start + batch_size, :, : self.context_size
            ]
            encoded, _, _ = self.encode(history)
            keys[..., start : start + batch_size, :] = _embedding(encoded)
        return keys

    def encode(
        self,
        x: Tensor,  # [..., C, R]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return encoded patches and the instance-normalization statistics."""
        mean, scale = self.revin.compute_stats(x)
        x = self.revin(x, mean, scale)
        x = x.nan_to_num(nan=0, posinf=0, neginf=0)
        x = x.unflatten(-1, (-1, self.patch_size))  # [..., C, N, P]
        x = self.pe(self.lin(x))  # [..., C, N, D]

        x = self.icl(x)  # [..., C, N, D]
        if self.cross_channel is not None:
            x = self.cross_channel(x)  # [..., C, N, D]

        return x, mean, scale
