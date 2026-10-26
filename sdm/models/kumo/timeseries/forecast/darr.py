# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.models.kumo.timeseries.forecast.model import KumoForecast
from sdm.models.kumo.timeseries.forecast.recipe import standardizer


class DARR(torch.nn.Module):
    """Retrieval-augmented forecasting with a :class:`KumoForecast` encoder.

    Fit a memory of historical windows and their observed continuations, then
    retrieve the nearest windows using cosine similarity of mean-pooled
    encoder embeddings. The output is ``alpha * direct + (1 - alpha) *
    retrieved``, in the input's units, for every numerical column.

    Reference and query series must have the same variables, units and sampling
    interval. Rows must be ordered. Only supply reference observations that
    would have been available at the query's forecast origin: this API does not
    inspect timestamps or remove overlapping evaluation data automatically.
    Batched reference series contribute to one shared memory, without windows
    crossing series boundaries. Query batch dimensions remain independent.

    Uses the checkpoint's frozen normalization, not a user-fitted recipe.
    Non-finite continuation values are excluded independently at each channel
    and horizon step; when all selected values are missing, the retrieval
    prediction is the checkpoint standardizer's mean. Missing history follows
    :class:`KumoForecast` normalization. Equal-similarity neighbor ordering is
    unspecified. This is an inference-only wrapper; call ``eval()`` before use.

    The wrapped model's weights are shared. Refit after changing weights.
    Memory moves with ``to()`` but is not included in ``state_dict()``; refit
    after loading weights. No additional checkpoint or dependencies are needed.

    Ported from the Apache-2.0 Kumo-TS implementation at
    ``af74798bbf7d020d3298f545fb9be383560a18ce`` (``Kumo-Forecast/sdk``).

    Args:
        model: The pretrained forecaster to reuse.
        k: Maximum number of historical neighbors to retrieve.
        temperature: Softmax temperature for cosine similarities. ``None`` or
            a negative value uses uniform weights; zero uses ``1e-12``.
        alpha: Weight of the direct forecast, between zero and one. The SDK
            default is ``0.01``. One gives direct-only predictions.
    """

    _keys: Tensor | None
    _values: Tensor | None

    def __init__(
        self,
        model: KumoForecast,
        *,
        k: int = 64,
        temperature: float | None = 0.05,
        alpha: float = 0.01,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("Expected at least one neighbor")
        if not 0 <= alpha <= 1:
            raise ValueError("Expected 'alpha' between zero and one")

        self.model = model.model
        self.standardizer = standardizer().to(self.model.lin.weight.device)
        self.k = k
        self.temperature = temperature
        self.alpha = alpha
        self.register_buffer("_keys", None, persistent=False)
        self.register_buffer("_values", None, persistent=False)
        self._columns: tuple[str, ...] | None = None
        self.eval()

    @torch.no_grad()
    def fit(
        self,
        reference: Tensor | TableTensor,  # [..., R, C]
        *,
        horizon: int = 72,
        stride: int = 72,
        batch_size: int = 8,
    ) -> None:
        """Replace retrieval memory using only complete historical windows.

        Args:
            reference: Ordered numerical observations of shape ``[..., R, C]``.
                Leading dimensions identify independent reference series.
                Every continuation must be known before predicting a query.
            horizon: Number of future steps to retrieve, between 1 and 512.
                Beyond 72 steps, direct forecasts are autoregressive.
                Windows require ``512 + max(72, horizon)`` reference rows.
            stride: Step between reference windows, defaulting to the native
                forecast horizon, as in the SDK.
            batch_size: Maximum number of reference windows encoded together.
        """
        self._check_eval()
        if not 1 <= horizon <= 512:
            raise ValueError("Expected 'horizon' between 1 and 512")
        table = self._as_table(reference)
        length = self.model.context_size + max(
            self.model.prediction_size, horizon
        )
        if table.size(-2) < length:
            raise ValueError(f"Expected at least {length} reference rows")

        x = self.standardizer(table).numerical
        x = x.reshape(-1, x.size(-2), x.size(-1))
        num_windows = (x.size(-2) - length) // stride + 1
        keys, values = [], []
        offsets = torch.arange(length, device=x.device)
        # Gather bounded batches without materializing all overlapping windows.
        for start in range(0, x.size(0) * num_windows, batch_size):
            index = torch.arange(
                start,
                min(start + batch_size, x.size(0) * num_windows),
                device=x.device,
            )
            rows = (index % num_windows).unsqueeze(-1) * stride + offsets
            window = x[(index // num_windows).unsqueeze(-1), rows]
            history = window[:, : self.model.context_size].transpose(-1, -2)
            encoded, _, _ = self.model.encode(history)
            keys.append(_embedding(encoded))
            values.append(
                window[
                    :,
                    self.model.context_size : self.model.context_size
                    + horizon,
                ].float()
            )

        self._keys = torch.cat(keys)
        self._values = torch.cat(values)
        self._columns = table.numerical_columns

    @torch.no_grad()
    def forward(
        self,
        context: Tensor | TableTensor,  # [..., R, C]
        *,
        batch_size: int = 8,
    ) -> TableTensor:  # [..., horizon, C]
        """Forecast every column using a previously fitted reference memory.

        Args:
            context: Ordered observed history of shape ``[..., R, C]``. Uses
                the last 512 rows. Named columns are aligned to the fitted
                reference; plain tensors use positional order. Returns columns
                in reference order, with all leading query dimensions retained.
            batch_size: Maximum number of query windows processed together.

        Returns:
            Numerical forecasts in original units, with the fitted horizon.
            The output retains the query's device and floating-point dtype.
            Select target columns from the output when other columns are
            historical covariates; covariates are also forecast for rollouts.
        """
        self._check_eval()
        if self._keys is None or self._values is None:
            raise RuntimeError("Call 'fit()' with reference history first")
        table = self._as_table(context, columns=self._columns)
        if table.size(-2) < self.model.context_size:
            raise ValueError(
                f"Expected at least {self.model.context_size} context rows"
            )

        x = self.standardizer(table[..., -self.model.context_size :, :])
        history = x.numerical.transpose(-1, -2)
        history = history.reshape(-1, history.size(-2), history.size(-1))
        horizon = self._values.size(-2)
        predictions = []
        for window in history.split(batch_size):
            encoded, mean, scale = self.model.encode(window)
            if self.alpha != 1:
                retrieved = _knn_forecast(
                    query=_embedding(encoded),
                    keys=self._keys,
                    values=self._values,
                    k=self.k,
                    temperature=self.temperature,
                )
            if self.alpha == 0:
                predictions.append(retrieved)
                continue

            direct = self.model.revin.inverse(
                x=self.model.head(encoded.flatten(-2)),
                mean=mean,
                scale=scale,
            )
            rollout = [direct]
            for _ in range(
                self.model.prediction_size, horizon, self.model.prediction_size
            ):
                window = torch.cat([window, direct], dim=-1)
                window = window[..., -self.model.context_size :]
                direct = self.model(window)
                rollout.append(direct)
            direct = torch.cat(rollout, dim=-1)[..., :horizon].transpose(
                -1, -2
            )
            predictions.append(
                direct
                if self.alpha == 1
                else self.alpha * direct + (1 - self.alpha) * retrieved
            )

        out = torch.cat(predictions).reshape(*table.size()[:-2], horizon, -1)
        out = TableTensor.from_tensor(out, columns=self._columns)
        out = self.standardizer.inverse_transform(out)
        return cast(TableTensor, out.to(dtype=context.dtype))

    def clear(self) -> None:
        """Discard reference memory without changing model weights."""
        self._keys = self._values = None
        self._columns = None

    def _check_eval(self) -> None:
        """Require deterministic inference embeddings for retrieval."""
        if self.training or self.model.training:
            raise RuntimeError("DARR is inference-only; call 'eval()' first")

    def _as_table(
        self,
        x: Tensor | TableTensor,
        columns: tuple[str, ...] | None = None,
    ) -> TableTensor:
        """Normalize inputs and align reference and query variables."""
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x, columns=columns)
        if x.size(-1) == 0 or x.active_stypes != {Stype.numerical}:
            raise ValueError("Expected one or more numerical columns only")
        if columns is not None:
            if set(x.numerical_columns) != set(columns):
                raise ValueError("Context columns must match the reference")
            x = x.select_columns(columns)
        return cast(TableTensor, x.to(dtype=self.model.lin.weight.dtype))


def _embedding(encoded: Tensor) -> Tensor:
    """Match the SDK's channel-then-patch mean pooling and cosine keys."""
    pooled = encoded.mean(dim=-3).mean(dim=-2).float()
    pooled = pooled.nan_to_num(nan=0, posinf=0, neginf=0)
    return F.normalize(pooled, dim=-1, eps=1e-12)


def _knn_forecast(
    query: Tensor,  # [B, D], unit-normalized
    keys: Tensor,  # [N, D], unit-normalized
    values: Tensor,  # [N, H, C]
    k: int,
    temperature: float | None,
) -> Tensor:  # [B, H, C]
    """Retrieve cosine neighbors and average their observed continuations."""
    with torch.amp.autocast(query.device.type, enabled=False):
        similarity, index = (query.float() @ keys.float().t()).topk(
            k=min(k, keys.size(0)), dim=-1
        )
    if temperature is None or temperature < 0:
        weights = torch.ones_like(similarity) / similarity.size(-1)
    else:
        weights = (similarity / max(temperature, 1e-12)).softmax(dim=-1)
    neighbors = values[index].float()  # [B, K, H, C]
    valid = neighbors.isfinite()
    weights = weights[..., None, None] * valid
    total = (weights * neighbors.masked_fill(~valid, 0)).sum(dim=1)
    return total / weights.sum(dim=1).clamp_min(1e-12)
