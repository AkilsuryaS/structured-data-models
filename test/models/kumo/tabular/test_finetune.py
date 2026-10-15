# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, cast

import pytest
import torch

import sdm.models.kumo.tabular.finetune as ft
import sdm.processing as sp
from sdm import Recipe, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel, KumoTabular
from sdm.models.kumo.tabular.finetune import (
    _capacity,
    _context_order,
    _context_rows,
    _split,
)
from sdm.models.kumo.tabular.model import MODEL_KWARGS
from sdm.processing.execution import RecipeExecution


def test_split_and_context_sampling() -> None:
    labels = torch.arange(100).remainder(2)
    generator = torch.Generator().manual_seed(7)
    train, validation = _split(labels, generator)
    order = _context_order(train, labels, generator)
    assert len(validation) == 10
    assert len(train) == len(order) == 90
    assert not torch.isin(validation, train).any()
    assert not torch.isin(validation, order).any()
    assert order.unique().numel() == len(order)
    assert labels[order[:2]].unique().numel() == 2
    generator.manual_seed(7)
    repeated_train, repeated_validation = _split(labels, generator)
    assert train.equal(repeated_train)
    assert validation.equal(repeated_validation)
    assert order.equal(_context_order(repeated_train, labels, generator))


@pytest.mark.parametrize("counts", [(2, 18), (2, 98), (98, 2)])
def test_split_imbalanced(counts: tuple[int, int]) -> None:
    labels = torch.cat([torch.zeros(counts[0]), torch.ones(counts[1])])
    train, validation = _split(labels, None)
    assert labels[train].unique().numel() == 2
    assert labels[validation].unique().numel() == 2


def test_split_rejects_insufficient_classes() -> None:
    with pytest.raises(ValueError, match="two classes"):
        _split(torch.tensor([0, 0, 1]), None)
    with pytest.raises(ValueError, match="10% validation"):
        _split(torch.tensor([0, 0, 1, 1]), None)


@pytest.mark.parametrize(
    ("capacity", "estimate"), [(2, 100), (73, 10), (99, 80)]
)
def test_capacity(capacity: int, estimate: int) -> None:
    size, measurements = _capacity(
        100, estimate, lambda n: (n <= capacity, n * 24)
    )
    assert size == capacity
    assert all(m["context_size"] >= 2 for m in measurements)
    assert any(m["peak_bytes"] == capacity * 24 for m in measurements)


def test_capacity_cap_and_minimum_failure() -> None:
    size, _ = _capacity(262_144, 100, lambda n: (True, n))
    assert size == 262_144
    with pytest.raises(RuntimeError, match="both classes"):
        _capacity(100, 40, lambda n: (False, 0))


@pytest.fixture
def cuda_model(monkeypatch: pytest.MonkeyPatch) -> KumoTabular:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 16,
            "num_embedding_layers": 1,
            "num_embedding_heads": 2,
            "num_inducing_points": 4,
            "group_size": 1,
            "num_frequencies": 4,
            "num_readout_tokens": 1,
            "icl_channels": 16,
            "num_icl_layers": 2,
            "num_icl_heads": 2,
        },
    )
    model = KumoTabular(
        task="classification", size="small", pretrained=False, device="cuda"
    )
    for p in model.parameters():
        if not p.any():
            torch.nn.init.normal_(p, std=0.1)
    return model


def _recipe() -> Recipe:
    return Recipe(
        features=sp.Standardize(),
        target=sp.AlignCategories(sort_by="value"),
        output=[sp.AverageEstimators(), sp.Softmax()],
    )


@pytest.mark.parametrize(
    ("value", "rows", "expected"),
    [
        (0, 24, 0),
        (False, 24, 0),
        (True, 24, 21),
        (True, 36_408, 32_767),
        (True, 36_409, 32_768),
        (True, 100_000, 32_768),
        (0.0, 24, 0),
        (2, 24, 2),
        (10.0, 24, 10),
        (0.5, 25, 12),
        (0.125, 100, 12),
        (300_000, 400_000, 300_000),
    ],
)
def test_context_rows(value: int | float, rows: int, expected: int) -> None:
    assert _context_rows(value, rows) == expected


@pytest.mark.parametrize(
    "value",
    [-1, -0.5, 1, 1.0, 1.5, float("nan"), float("inf"), 0.01],
)
def test_invalid_context_rows(value: int | float) -> None:
    with pytest.raises(ValueError, match=r"finetune|at least two rows"):
        _context_rows(value, 24)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, 21), (2, 2), (10, 10), (0.5, 12), (0.875, 21)],
)
def test_requested_context_always_tunes(
    cuda_model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
    value: int | float,
    expected: int,
) -> None:
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    selected = {}

    def ordinary(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Positive finetune must not attempt ordinary inference.")

    def observer(event: str, data: dict[str, Any]) -> None:
        if event in ("selection", "context"):
            selected.update(data)

    monkeypatch.setattr(ICLModel, "fit", ordinary)
    cuda_model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=value,
        query_batch_size=8,
        max_epochs=1,
        _finetune_observer=observer,
    )
    result = cuda_model.finetune_result_
    assert result is not None
    assert result["status"] == "tuned"
    assert result["context_size"] == expected
    assert result["selection"] == "stratified"
    indices = selected["context_indices"]
    assert len(indices) == indices.unique().numel() == expected
    assert y[indices].unique().numel() == 2
    assert not torch.isin(indices, selected["validation_indices"]).any()
    assert cuda_model.predict(x[:3]).numerical.isfinite().all()


@pytest.mark.parametrize("value", [22, 0.95])
def test_requested_context_exceeds_training_pool(
    cuda_model: KumoTabular, value: int | float
) -> None:
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    with pytest.raises(ValueError, match="after the 10% validation holdout"):
        cuda_model.fit(x, y, finetune=value)
    assert cuda_model.finetune_result_ is None


def test_requested_context_has_no_automatic_row_cap(
    cuda_model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = []

    def train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        sizes.append(len(args[3]))
        return {}

    monkeypatch.setattr(ft, "_train", train)
    x = torch.zeros(300_000, 1)
    y = torch.arange(300_000).remainder(2).reshape(-1, 1)
    cuda_model.fit(x, y, finetune=270_000)
    assert sizes == [270_000, 270_000]
    assert cuda_model.finetune_result_ is not None
    assert cuda_model.finetune_result_["context_size"] == 270_000


@pytest.mark.parametrize("shuffle", [False, True])
def test_tuned_prediction_and_lifecycle(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch, shuffle: bool
) -> None:
    model = cuda_model
    x = torch.randn(24, 3)
    y = (x[:, :1] > 0).long() * 10
    weights = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }
    flags = [p.requires_grad for p in model.parameters()]

    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full context does not fit")

    original_fit = ICLModel.fit
    model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=21,
        query_batch_size=8,
        max_epochs=3,
        min_epochs=2,
        patience=1,
        min_improvement=100,
        learning_rate=0,
        min_learning_rate=0,
        shuffle_columns=shuffle,
        generator=torch.Generator().manual_seed(4),
    )
    result = model.finetune_result_
    assert result is not None
    assert result["status"] == "tuned"
    assert result["completed_epochs"] == 2
    assert result["completed_steps"] == 6
    assert result["best_validation_loss"] == result["initial_validation_loss"]
    assert result["overlap_rate"] == 1.0
    assert result["memory"]["peak_bytes"] <= result["memory"]["budget_bytes"]
    assert [p.requires_grad for p in model.parameters()] == flags
    for name, p in model.named_parameters():
        assert p.equal(weights[name])
        assert p.grad is None
    output = model.predict(x[:4])
    torch.testing.assert_close(
        output.numerical, model.predict(x[:4]).numerical
    )
    assert output.columns[Stype.numerical] == ("0", "10")
    assert output.numerical.isfinite().all()
    model.clear()
    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(x[:4])
    monkeypatch.setattr(ICLModel, "fit", original_fit)
    model.fit(x.cuda(), y.cuda(), recipe=_recipe())
    assert model.finetune_result_ is None
    assert model.predict(x[:4].cuda()).numerical.isfinite().all()


def test_updates_and_direct_prediction(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = cuda_model
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    changed = []
    original_step = torch.optim.Adam.step

    def step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        parameter = optimizer.param_groups[0]["params"][0]
        before = parameter.detach().clone()
        assert parameter.dtype == torch.float32
        assert parameter.grad.isfinite().all()
        result = original_step(optimizer, *args, **kwargs)
        changed.append(not parameter.equal(before))
        # A real Adam probe has allocated both moments.
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert (
            optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape
        )
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step)

    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    net = model.models["classification"]
    recorded = []

    def capture(module: Any, args: Any, kwargs: Any) -> None:
        cache = kwargs.get("cache")
        if cache is not None and cache.is_recording:
            recorded.append((args[0].detach().clone(), args[1].clone()))

    hook = net.icl_block.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model.fit(
            x,
            y,
            recipe=_recipe(),
            finetune=21,
            query_batch_size=8,
            max_epochs=2,
            min_improvement=0,
            generator=torch.Generator().manual_seed(3),
        )
    finally:
        hook.remove()
    assert all(changed)
    embedding, labels = recorded[-1]
    assert model._cache is not None
    cache = cast(Cache, model._cache[0])
    execution = cast(RecipeExecution, model._cache["recipe_execution"])
    query = execution.transform(x[:5], None)[0].x.numerical.cuda()
    dtype = cast(torch.dtype, cache["finetune_dtype"])
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        query_embedding = net.encode(
            query,
            labels[:0],
            cast(torch.Tensor, cache["categorical_mask"]),
            cache=cache,
        )
        direct = net.icl_block(torch.cat([embedding, query_embedding]), labels)
        expected = direct[..., :2].float().softmax(-1)
    actual = model.predict(x[:5]).numerical
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("stage", ["probe", "training"])
def test_requested_context_oom_does_not_shrink_and_restores_flags(
    cuda_model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    model = cuda_model
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    original_train = ft._train
    attempted = []
    flags = [p.requires_grad for p in model.parameters()]

    def train(*args: Any, **kwargs: Any) -> Any:
        attempted.append(len(args[3]))
        if stage == "probe" or not kwargs["probing"]:
            raise torch.cuda.OutOfMemoryError("Requested context does not fit")
        return original_train(*args, **kwargs)

    monkeypatch.setattr(ft, "_train", train)
    with pytest.raises(RuntimeError, match=r"21 rows.*smaller finetune"):
        model.fit(
            x,
            y,
            recipe=_recipe(),
            finetune=21,
            query_batch_size=8,
            max_epochs=1,
        )
    assert attempted == ([21] if stage == "probe" else [21, 21])
    assert [p.requires_grad for p in model.parameters()] == flags
    assert model.finetune_result_ is None
    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(x[:1])


@pytest.mark.parametrize("case", ["multiclass", "regression"])
def test_unsupported_tuning_requests(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(3 if case == "multiclass" else 2)
    y = y.reshape(-1, 1).float() if case == "regression" else y.reshape(-1, 1)
    with pytest.raises(NotImplementedError):
        cuda_model.fit(
            x,
            y,
            finetune=21,
            n_estimators=1,
        )


def test_fp16_scaling_and_nonfinite_failure(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    cuda_model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=21,
        query_batch_size=8,
        max_epochs=1,
    )
    assert cuda_model._cache is not None
    assert cast(Cache, cuda_model._cache[0])["finetune_dtype"] == torch.float16
    assert cuda_model.predict(x[:4]).numerical.isfinite().all()
    flags = [p.requires_grad for p in cuda_model.parameters()]
    with torch.no_grad():
        cuda_model.models["classification"].icl_block.head[-1].bias.fill_(
            float("nan")
        )
    with pytest.raises(RuntimeError, match="Nonfinite"):
        cuda_model.fit(
            x,
            y,
            recipe=_recipe(),
            finetune=21,
            query_batch_size=8,
            max_epochs=1,
        )
    assert [p.requires_grad for p in cuda_model.parameters()] == flags
    assert cuda_model._cache is None


def test_observer_gate_preserves_weights_and_flags(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = cuda_model
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    flags = [p.requires_grad for p in model.parameters()]
    before = [p.detach().clone() for p in model.parameters()]
    events = []

    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    def observer(event: str, data: dict[str, Any]) -> None:
        events.append(event)
        if event == "preflight":
            assert data["projected_seconds"] > 0
            assert data["timings"]["step_seconds"] > 0
            raise RuntimeError("Benchmark runtime gate")

    with pytest.raises(RuntimeError, match="runtime gate"):
        model.fit(
            x,
            y,
            recipe=_recipe(),
            finetune=21,
            query_batch_size=8,
            max_epochs=2,
            _finetune_observer=observer,
        )
    assert "probe" in events
    assert "context" not in events
    assert [p.requires_grad for p in model.parameters()] == flags
    for parameter, initial in zip(model.parameters(), before, strict=True):
        assert parameter.equal(initial)
    assert model._cache is None


def test_observer_exports_restored_context(
    cuda_model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = cuda_model
    x = TableTensor.from_tensor(torch.randn(24, 3))
    # Arrow categorical codes use int32, while cross-entropy needs int64.
    y = torch.arange(24, dtype=torch.int32).remainder(2).reshape(-1, 1)
    epochs = []

    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "epoch":
            epochs.append(data)
        if event == "restored":
            snapshot = {
                name: value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value.cpu()
                if isinstance(value, Cache)
                else value
                for name, value in data.items()
            }
            torch.save(snapshot, tmp_path / "optimized-context.pt")

    model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=21,
        query_batch_size=8,
        max_epochs=2,
        _finetune_observer=observer,
    )
    assert len(epochs) == 2
    assert model.finetune_result_ is not None
    assert model.finetune_result_["best_epoch"] in (0, 1, 2)
    state = torch.load(tmp_path / "optimized-context.pt", weights_only=False)
    execution = state["execution"]
    net = model.models["classification"]
    labels = state["labels"].cuda()
    mask = state["categorical_mask"].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        query = execution.transform(x[:5], None)[0].x.numerical.cuda()
        query_embedding = net.encode(
            query,
            labels[:0],
            mask,
            cache=state["encoder_cache"].to("cuda"),
        )
        context = (
            (state["base"].float() + state["adjustment"]).cuda().bfloat16()
        )
        output = net.icl_block(torch.cat([context, query_embedding]), labels)
        expected = output[..., :2].float().softmax(-1)
        actual = model.predict(x[:5]).numerical
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_zero_adjustment_matches_ordinary_context(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = cuda_model
    x = TableTensor.from_tensor(torch.randn(24, 3))
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    selected: dict[str, Any] = {}
    original_fit = ICLModel.fit

    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "context":
            selected.update(data)

    model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=21,
        query_batch_size=8,
        max_epochs=1,
        learning_rate=0,
        min_learning_rate=0,
        _finetune_observer=observer,
    )
    expected = model.predict(x[:5]).numerical
    monkeypatch.setattr(ICLModel, "fit", original_fit)
    indices = selected["context_indices"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.fit(
            x[indices],
            y[indices],
            recipe=_recipe(),
            finetune=False,
            _context_device=torch.device("cuda"),
        )
        actual = model.predict(x[:5]).numerical
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_approximate_capacity() -> None:
    selected, measurements = _capacity(
        2000, 500, lambda size: (size <= 949, size), tolerance=10
    )
    assert 940 <= selected <= 949
    assert (
        min(m["context_size"] for m in measurements if not m["fits"])
        - selected
        <= 10
    )


def test_smoke_run_update_budget(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full context does not fit")

    x = torch.randn(40, 3)
    y = torch.arange(40).remainder(2).reshape(-1, 1)
    cuda_model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=21,
        query_batch_size=4,
        _finetune_max_steps=2,
    )
    result = cuda_model.finetune_result_
    assert result is not None
    assert result["completed_steps"] == result["planned_steps"] == 2
    assert result["history"][-1]["partial_epoch"]
    assert result["history"][-1]["learning_rate"] == pytest.approx(1e-5)
    assert torch.isfinite(cuda_model.predict(x[:3]).numerical).all()


def test_capacity_counts_backward_with_adam_state(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_step = torch.optim.Adam.step

    def step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        parameter = optimizer.param_groups[0]["params"][0]
        if optimizer.state and parameter.size(0) > 10:
            raise torch.cuda.OutOfMemoryError("Backward with Adam state")
        return original_step(optimizer, *args, **kwargs)

    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference does not fit")

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    x = torch.randn(40, 3)
    y = torch.arange(40).remainder(2).reshape(-1, 1)
    with pytest.raises(RuntimeError, match=r"11 rows.*memory budget"):
        cuda_model.fit(
            x,
            y,
            recipe=_recipe(),
            finetune=11,
            query_batch_size=4,
            max_epochs=1,
        )
    cuda_model.fit(
        x,
        y,
        recipe=_recipe(),
        finetune=10,
        query_batch_size=4,
        max_epochs=1,
    )
    result = cuda_model.finetune_result_
    assert result is not None
    assert result["context_size"] == 10
    assert result["completed_steps"] == 9


@pytest.mark.parametrize("name", ["n_estimators", "num_estimators"])
def test_tuned_ensemble_predictions_and_lifecycle(
    cuda_model: KumoTabular, name: str
) -> None:
    model = cuda_model
    x = torch.randn(40, 3)
    y = torch.arange(40).remainder(2).reshape(-1, 1) * 10 + 5
    weights = [p.detach().clone() for p in model.parameters()]
    flags = [p.requires_grad for p in model.parameters()]
    selections: list[dict[str, Any]] = []
    contexts: list[torch.Tensor] = []
    caches: list[Cache] = []

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "selection":
            selections.append(data)
        if event == "context":
            contexts.append(data["context_indices"].clone())
        if event == "restored":
            assert model._cache is not None
            caches.append(model._cache)

    kwargs: dict[str, Any] = dict(
        recipe=_recipe(),
        finetune=8,
        query_batch_size=4,
        _finetune_max_steps=2,
        _finetune_observer=observer,
        **{name: 3},
    )
    assert (
        model.fit(x, y, generator=torch.Generator().manual_seed(4), **kwargs)
        is None
    )
    result = model.finetune_result_
    assert result is not None
    assert result["num_estimators"] == 3
    assert result["total_context_rows"] == 24
    assert result["completed_steps"] == 6
    assert len(result["members"]) == 3
    assert all(r["completed_steps"] == 2 for r in result["members"])
    assert [s["estimator"] for s in selections] == [0, 1, 2]
    assert all(
        selections[0]["validation_indices"].equal(s["validation_indices"])
        for s in selections
    )
    assert not contexts[0].equal(contexts[1])
    for indices in contexts:
        assert len(indices) == indices.unique().numel() == 8
        assert not torch.isin(
            indices, selections[0]["validation_indices"]
        ).any()
    ensemble_cache = model._cache
    assert ensemble_cache is not None
    # Compare the public ensemble with independently replayed member caches.
    try:
        predictions = []
        for cache in caches:
            model._cache = cache
            predictions.append(model.predict(x[:5]).numerical)
    finally:
        model._cache = ensemble_cache
    expected = torch.stack(predictions).mean(0)
    for query in (x[:5], x[:5].cuda(), x[:5]):
        actual = model.predict(query)
        assert actual.columns[Stype.numerical] == ("5", "15")
        torch.testing.assert_close(actual.numerical, expected)
        torch.testing.assert_close(
            actual.numerical.sum(-1), torch.ones(5, device=actual.device)
        )
    assert [p.requires_grad for p in model.parameters()] == flags
    for p, before in zip(model.parameters(), weights, strict=True):
        assert p.equal(before)
    assert not model.training
    original_contexts = list(contexts)
    model.fit(x, y, generator=torch.Generator().manual_seed(4), **kwargs)
    for original, repeated in zip(
        original_contexts, contexts[3:], strict=True
    ):
        assert original.equal(repeated)
    torch.testing.assert_close(model.predict(x[:5]).numerical, expected)
    model.clear()
    assert model._cache is None
    assert model.finetune_result_ is None
    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(x[:1])
    model.fit(
        x,
        y,
        recipe=_recipe(),
        n_estimators=1,
        _context_device=torch.device("cuda"),
    )
    assert model.finetune_result_ is None
    assert model.predict(x[:5]).numerical.isfinite().all()


def test_estimator_defaults_and_alias(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = []

    def fit(*args: Any, **kwargs: Any) -> None:
        counts.append(kwargs["num_estimators"])

    monkeypatch.setattr(ICLModel, "fit", fit)
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    cuda_model.fit(x, y)
    cuda_model.fit(x, y, finetune=False)
    cuda_model.fit(x, y, n_estimators=3)
    cuda_model.fit(x, y, num_estimators=2)
    assert counts == [8, 8, 3, 2]
    cuda_model.fit(x, y, finetune=8, recipe=_recipe(), _finetune_max_steps=1)
    assert cuda_model.finetune_result_ is not None
    assert cuda_model.finetune_result_["num_estimators"] == 1


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_estimator_count(cuda_model: KumoTabular, count: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        cuda_model.fit(
            torch.zeros(24, 3), torch.zeros(24, 1), n_estimators=count
        )


def test_estimator_alias_conflict(cuda_model: KumoTabular) -> None:
    with pytest.raises(ValueError, match="only one"):
        cuda_model.fit(
            torch.zeros(24, 3),
            torch.zeros(24, 1),
            n_estimators=2,
            num_estimators=2,
        )


def test_later_estimator_failure_discards_ensemble(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ft._train
    active = 0
    baselines = []
    flags = [p.requires_grad for p in cuda_model.parameters()]

    def observer(event: str, data: dict[str, Any]) -> None:
        nonlocal active
        active = data["estimator"]

    def train(*args: Any, **kwargs: Any) -> Any:
        if kwargs["probing"]:
            baselines.append(torch.cuda.memory_allocated())
        if active == 1:
            raise torch.cuda.OutOfMemoryError("Second context exceeds budget")
        return original(*args, **kwargs)

    monkeypatch.setattr(ft, "_train", train)
    with pytest.raises(RuntimeError, match="memory budget"):
        cuda_model.fit(
            torch.randn(40, 3),
            torch.arange(40).remainder(2).reshape(-1, 1),
            n_estimators=2,
            finetune=8,
            recipe=_recipe(),
            _finetune_max_steps=1,
            _finetune_observer=observer,
        )
    assert len(baselines) == 2
    assert baselines[1] > baselines[0]
    assert cuda_model._cache is None
    assert cuda_model.finetune_result_ is None
    assert [p.requires_grad for p in cuda_model.parameters()] == flags


@pytest.mark.parametrize(
    ("counts", "size", "expected_first"),
    [
        ((80, 20), 2, 1),
        ((80, 20), 17, 14),
        ((80, 20), 100, 80),
        ((2, 98), 2, 1),
        ((2, 98), 17, 1),
        ((2, 98), 100, 2),
        ((98, 2), 2, 1),
        ((98, 2), 17, 16),
        ((98, 2), 100, 98),
    ],
)
def test_stratified_context_proportions(
    counts: tuple[int, int], size: int, expected_first: int
) -> None:
    labels = torch.cat([torch.zeros(counts[0]), torch.ones(counts[1])])
    train = torch.arange(len(labels))
    order = _context_order(
        train, labels, torch.Generator().manual_seed(17), context_rows=size
    )
    chosen = order[:size]
    assert int((labels[chosen] == 0).sum()) == expected_first
    assert labels[chosen].unique().numel() == 2
    assert len(order) == order.unique().numel() == len(train)
    repeated = _context_order(
        train, labels, torch.Generator().manual_seed(17), context_rows=size
    )
    assert order.equal(repeated)


def test_automatic_context_cap(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = []

    def train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        sizes.append(len(args[3]))
        return {}

    monkeypatch.setattr(ft, "_train", train)
    cuda_model.fit(
        torch.zeros(40_000, 1),
        torch.arange(40_000).remainder(2).reshape(-1, 1),
        finetune=True,
    )
    assert sizes == [32_768, 32_768]
    assert cuda_model.finetune_result_ is not None
    assert cuda_model.finetune_result_["context_size"] == 32_768
    assert cuda_model.finetune_result_["num_estimators"] == 1


def test_automatic_stratified_ensemble(cuda_model: KumoTabular) -> None:
    x = torch.randn(100, 3)
    y = torch.cat([torch.zeros(80), torch.ones(20)]).long().reshape(-1, 1)
    contexts = []
    validation = []

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "selection":
            validation.append(data["validation_indices"])
        if event == "context":
            contexts.append(data["context_indices"])

    cuda_model.fit(
        x,
        y,
        finetune=True,
        n_estimators=2,
        recipe=_recipe(),
        _finetune_max_steps=1,
        _finetune_observer=observer,
    )
    assert len(contexts) == 2
    assert validation[0].equal(validation[1])
    for indices in contexts:
        assert len(indices) == 90
        assert int((y[indices] == 0).sum()) == 72
        assert int((y[indices] == 1).sum()) == 18
        assert not torch.isin(indices, validation[0]).any()
    report = cuda_model.finetune_result_
    assert report is not None
    assert report["selection"] == "stratified"
    assert report["context_size"] == 90
    assert report["total_context_rows"] == 180
    prediction = cuda_model.predict(x[:5]).numerical
    assert prediction.isfinite().all()
    torch.testing.assert_close(
        prediction.sum(-1), torch.ones(5, device=prediction.device)
    )
