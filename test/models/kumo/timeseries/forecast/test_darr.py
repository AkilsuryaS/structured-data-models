# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import pytest
import torch

from sdm import TableTensor
from sdm.models import KumoForecast
from sdm.models.kumo.timeseries.forecast import DARR
from sdm.models.kumo.timeseries.forecast import model as forecasting
from sdm.models.kumo.timeseries.forecast.darr import _embedding, _knn_forecast
from sdm.testing import withCUDA


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> KumoForecast:
    """Build the same forecast architecture at a small test scale."""
    monkeypatch.setattr(
        forecasting,
        "_KumoForecast",
        partial(
            forecasting._KumoForecast,
            context_size=16,
            prediction_size=8,
            patch_size=4,
            channels=16,
            hidden_channels=24,
            num_layers=2,
            num_heads=4,
            cross_channel_heads=4,
        ),
    )
    return KumoForecast(pretrained=False)


@withCUDA
@pytest.mark.parametrize("temperature", [None, -1.0, 0.0, 0.05, 2.0])
@pytest.mark.parametrize("k", [1, 3, 20])
def test_retrieval_matches_numpy(
    temperature: float | None,
    k: int,
    device: torch.device,
) -> None:
    """Match the upstream cosine/top-k/weighted-continuation equations."""
    # Distinct similarities avoid depending on unspecified top-k tie order.
    keys = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    query = np.array([[0.6, 0.8], [-0.8, 0.6]], dtype=np.float32)
    values = np.arange(32, dtype=np.float32).reshape(4, 4, 2)
    values[0, 0, 0] = np.nan
    values[1, 0, 0] = np.inf
    values[:, 3, 1] = np.nan
    similarity = query @ keys.T
    index = np.argsort(-similarity, axis=-1)[:, : min(k, len(keys))]
    scores = np.take_along_axis(similarity, index, axis=-1)
    if temperature is None or temperature < 0:
        weights = np.ones_like(scores) / scores.shape[-1]
    else:
        scores = scores / max(temperature, 1e-12)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
    neighbors = values[index]
    valid = np.isfinite(neighbors)
    weights = weights[..., None, None] * valid
    expected = (weights * np.where(valid, neighbors, 0)).sum(axis=1)
    expected /= np.maximum(weights.sum(axis=1), 1e-12)

    out = _knn_forecast(
        query=torch.from_numpy(query).to(device),
        keys=torch.from_numpy(keys).to(device),
        values=torch.from_numpy(values).to(device),
        k=k,
        temperature=temperature,
    )
    torch.testing.assert_close(out, torch.from_numpy(expected).to(device))
    assert out.device == device


def test_embedding_pooling() -> None:
    """Pool channels and then patches as the SDK does, including zero keys."""
    encoded = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    pooled = encoded.mean(dim=1).sum(dim=1) / 3
    expected = pooled / pooled.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(_embedding(encoded), expected)
    torch.testing.assert_close(
        _embedding(torch.zeros_like(encoded)), torch.zeros(2, 4)
    )


@withCUDA
def test_retrieval_uses_query_history(
    model: KumoForecast,
    device: torch.device,
) -> None:
    """Retrieve the matching continuation from a shared two-series memory."""
    model = model.to(device)
    time = torch.arange(16, device=device).float()
    history = torch.stack([time.square(), time.flip(0).square()])
    future = torch.stack([time[:8] + 100, time[:8] - 100])
    reference = torch.cat([history, future], dim=-1).unsqueeze(-1)
    darr = DARR(model, k=1, alpha=0)
    darr.fit(reference, horizon=8, batch_size=1)
    out = darr(reference[:, :16], batch_size=2).numerical
    torch.testing.assert_close(out, reference[:, 16:])


@withCUDA
def test_retrieval_autocast(device: torch.device) -> None:
    """Keep similarity and weight accumulation in float32 under autocast."""
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    values = torch.tensor([[[1.0]], [[3.0]]], device=device)
    query = torch.tensor([[0.50001, 0.5]], device=device)
    expected = _knn_forecast(query, keys, values, k=2, temperature=1e-5)
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = _knn_forecast(query, keys, values, k=2, temperature=1e-5)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out.dtype == torch.float32


@withCUDA
@pytest.mark.parametrize("cross_channel", [False, True])
@pytest.mark.parametrize("horizon", [3, 8, 19])
def test_blending_and_rollout(
    model: KumoForecast,
    cross_channel: bool,
    horizon: int,
    device: torch.device,
) -> None:
    """Match both blending endpoints and SDK-style native-block rollouts."""
    model = model.to(device)
    if not cross_channel:
        model.model.cross_channel = None
    reference = torch.randn(35, 2, device=device)
    context = torch.randn(2, 3, 20, 2, device=device)
    darr = DARR(model, temperature=None, alpha=0)
    darr.fit(reference, horizon=horizon, stride=4, batch_size=2)
    retrieved = darr(context, batch_size=2).numerical
    starts = range(0, 35 - 16 - max(8, horizon) + 1, 4)
    expected = torch.stack(
        [reference[s + 16 : s + 16 + horizon] for s in starts]
    ).mean(0)
    torch.testing.assert_close(retrieved, expected.expand(2, 3, -1, -1))

    # Independent use of the existing public forecaster for each native block.
    current = context.reshape(-1, 20, 2)
    rollout = []
    for _ in range(0, horizon, 8):
        prediction = model(
            x_context=current[..., :0],
            y_context=current,
            x_query=current.new_empty(current.size(0), 8, 0),
            num_estimators=1,
        ).numerical
        rollout.append(prediction)
        current = torch.cat([current, prediction], dim=-2)[..., -16:, :]
    expected_direct = torch.cat(rollout, dim=-2)[..., :horizon, :].reshape(
        2, 3, horizon, 2
    )
    darr.alpha = 1
    direct = darr(context).numerical
    torch.testing.assert_close(direct, expected_direct, rtol=2e-5, atol=2e-5)
    darr.alpha = 0.25
    mixed = darr(context).numerical
    torch.testing.assert_close(mixed, 0.25 * direct + 0.75 * retrieved)
    assert not mixed.requires_grad
    assert mixed.device == device


@withCUDA
def test_reference_boundaries(
    model: KumoForecast, device: torch.device
) -> None:
    """Never cross independent series or borrow incomplete future windows."""
    model = model.to(device)
    reference = (
        torch.stack(
            [
                torch.arange(28, device=device),
                torch.arange(28, device=device) + 100,
            ]
        )
        .float()
        .unsqueeze(-1)
    )
    darr = DARR(model, temperature=None, alpha=0)
    darr.fit(reference, horizon=3, stride=8, batch_size=1)
    out = darr(reference[:, :16]).numerical
    # Even a short horizon requires a complete native 8-step continuation.
    expected = reference[:, 16:19].mean(dim=0).expand(2, -1, -1)
    torch.testing.assert_close(out, expected)
    # A query's extra leading history must not augment or alter fitted memory.
    longer = torch.cat([reference * 1000, reference[:, :16]], dim=-2)
    torch.testing.assert_close(darr(longer).numerical, out)
    darr.fit(reference, horizon=3, stride=8, batch_size=8)
    torch.testing.assert_close(darr(longer).numerical, out)


def test_column_alignment_and_refit(model: KumoForecast) -> None:
    """Align named channels, preserve units, and replace memory on refit."""
    reference = TableTensor.from_tensor(torch.randn(24, 2), columns=["a", "b"])
    darr = DARR(model, k=1, alpha=0)
    darr.fit(reference, horizon=8)
    context = reference[:16].select_columns(["b", "a"])
    out = darr(context)
    assert out.numerical_columns == ("a", "b")
    torch.testing.assert_close(out.numerical, reference[16:].numerical)
    torch.testing.assert_close(
        darr(reference[:16].numerical).numerical, out.numerical
    )
    with pytest.raises(ValueError, match="columns must match"):
        darr(TableTensor.from_tensor(torch.randn(16, 2), columns=["a", "c"]))
    reference = reference.replace_blocks(numerical=reference.numerical + 50)
    darr.fit(reference, horizon=8)
    torch.testing.assert_close(
        darr(context).numerical, reference[16:].numerical
    )
    darr.clear()
    with pytest.raises(RuntimeError, match="fit"):
        darr(context)


def test_missing_continuations(model: KumoForecast) -> None:
    """Renormalize finite neighbors; all-missing values use the frozen mean."""
    reference = torch.randn(2, 24, 2)
    reference[0, 16:, 0] = torch.nan
    reference[:, 16:, 1] = torch.inf
    darr = DARR(model, temperature=None, alpha=0)
    darr.fit(reference, horizon=8)
    out = darr(reference[1, :16]).numerical
    torch.testing.assert_close(out[:, 0], reference[1, 16:, 0])
    torch.testing.assert_close(out[:, 1], torch.full((8,), -5.671202659606934))


@withCUDA
def test_memory_conversion(model: KumoForecast, device: torch.device) -> None:
    """Move memory with weights; exclude observations from state dicts."""
    reference = torch.randn(24, 1)
    darr = DARR(model, alpha=0)
    before = set(darr.state_dict())
    darr.fit(reference, horizon=8)
    assert set(darr.state_dict()) == before
    darr = darr.to(device=device, dtype=torch.float64)
    out = darr(reference[:16].to(device)).numerical
    assert out.dtype == reference.dtype
    torch.testing.assert_close(
        out,
        reference[16:].to(device=device, dtype=out.dtype),
        rtol=2e-6,
        atol=2e-6,
    )
    assert out.device == device


def test_input_contract(model: KumoForecast) -> None:
    """Reject absent memory, insufficient history, and incompatible data."""
    darr = DARR(model)
    with pytest.raises(RuntimeError, match="fit"):
        darr(torch.zeros(16, 1))
    with pytest.raises(ValueError, match="24 reference rows"):
        darr.fit(torch.zeros(23, 1), horizon=3)
    with pytest.raises(ValueError, match="horizon"):
        darr.fit(torch.zeros(24, 1), horizon=0)
    with pytest.raises(ValueError, match="numerical columns"):
        darr.fit(TableTensor.from_tensor(torch.ones(24, 1, dtype=torch.long)))
    darr.fit(torch.zeros(24, 1), horizon=8)
    with pytest.raises(ValueError, match="16 context rows"):
        darr(torch.zeros(15, 1))
    darr.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        darr(torch.zeros(16, 1))
