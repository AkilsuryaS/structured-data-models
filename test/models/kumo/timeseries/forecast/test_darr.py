# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from functools import partial

import numpy as np
import pytest
import torch

import sdm.processing as sp
from sdm import TableTensor
from sdm.models import KumoForecast
from sdm.models.kumo.timeseries.forecast import model as forecasting
from sdm.models.kumo.timeseries.forecast.darr import knn
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

    out = knn(
        query=torch.from_numpy(query).to(device),
        keys=torch.from_numpy(keys).to(device).expand(2, -1, -1),
        values=torch.from_numpy(values)
        .to(device)
        .transpose(-1, -2)
        .expand(2, -1, -1, -1),
        k=k,
        temperature=temperature,
    )
    torch.testing.assert_close(
        out.transpose(-1, -2), torch.from_numpy(expected).to(device)
    )
    assert out.device == device


@withCUDA
@pytest.mark.parametrize("cross_channel", [False, True])
@pytest.mark.parametrize("horizon", [3, 8, 19])
def test_blending_and_rollout(
    model: KumoForecast,
    cross_channel: bool,
    horizon: int,
    device: torch.device,
) -> None:
    """Retrieve from full history with distinct targets and covariates."""
    model = model.to(device)
    if not cross_channel:
        model.model.cross_channel = None
    reference = torch.randn(2, 3, 48, 3, device=device)
    features = TableTensor.from_tensor(reference[..., 2:], columns=["past"])
    targets = TableTensor.from_tensor(reference[..., :2], columns=["a", "b"])
    query = TableTensor.from_tensor(reference.new_empty(2, 3, horizon, 0))

    retrieved = model(
        features,
        targets,
        query,
        alpha=0,
        temperature=None,
        num_estimators=1,
        batch_size=2,
    )
    starts = range(0, 48 - 16 - max(8, horizon) + 1, 8)
    expected = torch.stack(
        [reference[..., s + 16 : s + 16 + horizon, :2] for s in starts]
    ).mean(0)
    torch.testing.assert_close(retrieved.numerical, expected)
    assert retrieved.numerical_columns == ("a", "b")

    # Independently assemble native forecasts through the public model API.
    current = reference[..., -16:, :]
    blocks = []
    for _ in range(0, horizon, 8):
        prediction = model(
            current[..., :0],
            current,
            current.new_empty(2, 3, 8, 0),
            num_estimators=1,
        ).numerical
        blocks.append(prediction)
        current = torch.cat([current, prediction], dim=-2)[..., -16:, :]
    expected_direct = torch.cat(blocks, dim=-2)[..., :horizon, :2]
    # Keep covariates in their existing context-only role for direct forecasts.
    with (
        nullcontext()
        if cross_channel
        else pytest.warns(RuntimeWarning, match="Historical covariates")
    ):
        direct = model(features, targets, query, num_estimators=1).numerical
    torch.testing.assert_close(direct, expected_direct, rtol=2e-5, atol=2e-5)
    mixed = model(
        features,
        targets,
        query,
        alpha=0.25,
        temperature=None,
        num_estimators=1,
    ).numerical
    torch.testing.assert_close(mixed, 0.25 * direct + 0.75 * expected)
    assert not mixed.requires_grad
    assert mixed.device == device


@withCUDA
def test_retrieval_uses_query_history(
    model: KumoForecast, device: torch.device
) -> None:
    """Retrieve matching windows independently in each batch."""
    model = model.to(device)
    history = torch.randn(2, 16, 1, device=device)
    future = torch.randn(2, 8, 1, device=device) * 100
    reference = torch.cat([history, future, history], dim=-2)
    out = model(
        reference[..., :0],
        reference,
        reference.new_empty(2, 8, 0),
        alpha=0,
        k=1,
        num_estimators=1,
    )
    torch.testing.assert_close(out.numerical, future, rtol=2e-5, atol=2e-5)


@withCUDA
@pytest.mark.parametrize("num_estimators", [None, 3])
@pytest.mark.parametrize("estimator_batch_size", [1, 2, None])
def test_estimator_batching(
    model: KumoForecast,
    device: torch.device,
    num_estimators: int | None,
    estimator_batch_size: int | None,
) -> None:
    """Match separate histories and repeated estimators across batch sizes."""
    model = model.to(device)
    reference = torch.randn(3, 48, 2, device=device)
    query = reference.new_empty(8, 0)
    predictions = [
        model(
            history[:, :0],
            history,
            query,
            alpha=0.2,
            k=2,
            temperature=None,
        ).numerical
        for history in reference
    ]
    if num_estimators is None:
        expected = torch.stack(predictions).mean(0)
        query = reference.new_empty(3, 8, 0)
    else:
        reference = reference[0]
        expected = predictions[0]
    actual = model(
        reference[..., :0],
        reference,
        query,
        alpha=0.2,
        k=2,
        temperature=None,
        num_estimators=num_estimators,
        estimator_batch_size=estimator_batch_size,
    )
    torch.testing.assert_close(actual.numerical, expected)


@withCUDA
@pytest.mark.parametrize("horizon", [8, 19])
def test_direct_default_and_short_history(
    model: KumoForecast, device: torch.device, horizon: int
) -> None:
    """Default forecasts ignore old history and need no retrieval memory."""
    model = model.to(device)
    history = torch.randn(16, 2, device=device)
    reference = torch.cat([history + 100, history], dim=-2)
    query = history.new_empty(horizon, 0)
    direct = model(history[:, :0], history, query).numerical
    torch.testing.assert_close(
        model(reference[:, :0], reference, query).numerical, direct
    )
    for alpha in [0, 0.25]:
        torch.testing.assert_close(
            model(history[:, :0], history, query, alpha=alpha).numerical,
            direct,
        )


def test_recipe_and_missing_continuations(model: KumoForecast) -> None:
    """Use the target recipe for both retrieved and direct predictions."""
    reference = torch.randn(24, 2)
    recipe = sp.Recipe(
        target=sp.Standardize().with_state(7.0, 3.0),
        output=sp.AverageEstimators(),
    )
    query = reference.new_empty(8, 0)
    out = model(reference[:, :0], reference, query, alpha=0, recipe=recipe)
    torch.testing.assert_close(out.numerical, reference[16:])
    reference[16:, 1] = torch.nan
    out = model(reference[:, :0], reference, query, alpha=0, recipe=recipe)
    torch.testing.assert_close(out.numerical[:, 0], reference[16:, 0])
    torch.testing.assert_close(out.numerical[:, 1], torch.full((8,), 7.0))


def test_call_options(model: KumoForecast) -> None:
    """Keep retrieval settings independent across forward calls."""
    reference = torch.randn(48, 2)
    features = reference[:, :0]
    query = reference.new_empty(8, 0)
    direct = model(features, reference, query).numerical
    model(features, reference, query, alpha=0, k=1, temperature=None)
    torch.testing.assert_close(
        model(features, reference, query).numerical, direct
    )


def test_one_shot_api(model: KumoForecast) -> None:
    """Reject the fitted lifecycle in favor of complete forecast calls."""
    history = torch.randn(24, 1)
    with pytest.raises(NotImplementedError, match="forward"):
        model.fit(history[:, :0], history)
    with pytest.raises(NotImplementedError, match="forward"):
        model.predict(torch.empty(8, 0))


@withCUDA
def test_retrieval_autocast(device: torch.device) -> None:
    """Keep similarity and weight accumulation in float32 under autocast."""
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    values = torch.tensor([[[1.0]], [[3.0]]], device=device)
    query = torch.tensor([0.50001, 0.5], device=device)
    expected = knn(query, keys, values, k=2, temperature=1e-5)
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = knn(query, keys, values, k=2, temperature=1e-5)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out.dtype == torch.float32
