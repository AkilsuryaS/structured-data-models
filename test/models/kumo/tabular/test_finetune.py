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
    _validation_rows,
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
    with pytest.raises(ValueError, match="validation holdout"):
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
        task="classification",
        size="small",
        pretrained=False,
        device="cuda",
        n_estimators=1,
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
        (True, 24, 22),
        (True, 36_408, 32_768),
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
    [(True, 22), (2, 2), (10, 10), (0.5, 12), (0.875, 21)],
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


@pytest.mark.parametrize("value", [23, 0.99])
def test_requested_context_exceeds_training_pool(
    cuda_model: KumoTabular, value: int | float
) -> None:
    x = torch.randn(24, 3)
    y = torch.arange(24).remainder(2).reshape(-1, 1)
    with pytest.raises(ValueError, match="after the validation holdout"):
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
        finetune=22,
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


def test_missing_regression_head(cuda_model: KumoTabular) -> None:
    with pytest.raises(ValueError, match="head is not initialized"):
        cuda_model.fit(torch.randn(24, 3), torch.randn(24, 1), finetune=8)


def test_fp16_scaling_and_nonfinite_failure(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    def oom(*args: Any, **kwargs: Any) -> None:
        raise torch.cuda.OutOfMemoryError("Full inference failed")

    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda **kwargs: False
    )
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
        patience=7,
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


@pytest.mark.parametrize(
    ("rows", "losses", "best_step", "reason", "patience", "alpha"),
    [
        (80, [1.0] * 4, 0, "ema", 3, 0.98),
        (80, [1.0, 0.9, 2.0, 2.0], 1, "ema", 3, 0.98),
        # A temporary increase does not stop a still-improving EMA.
        (80, [1.0, 0.5, 0.5, 1.5, 0.5, 1.5], 1, "ema", 3, 0.98),
        # Tiny improvements remain improvements, without a delta threshold.
        (
            80,
            [1.0 - step * 0.00005 for step in range(10)],
            9,
            "epoch_limit",
            3,
            0.98,
        ),
        (
            80,
            [1.0 - step * 0.01 for step in range(10)],
            9,
            "epoch_limit",
            None,
            0.98,
        ),
        (
            81,
            [1.0 - step * 0.01 for step in range(11)],
            10,
            "epoch_limit",
            None,
            0.98,
        ),
        (160, [1.0] * 11, 0, "ema", 10, 0.98),
        (240, [1.0] * 21, 0, "ema", None, 0.98),
        (400, [1.0] + [0.9] * 20 + [1.5] * 3, 1, "ema", None, 0.98),
        (80, [1.0, 0.9, 1.1], 1, "ema", 1, 0.0),
        (80, [1.0, 0.5, 1.0, 1.0, 1.0], 1, "ema", 3, 0.5),
    ],
)
def test_batch_validation_stopping_and_best_context(
    cuda_model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    losses: list[float],
    best_step: int,
    reason: str,
    patience: int | None,
    alpha: float,
) -> None:
    # Control only the holdout scores; gradients and Adam updates stay real.
    train = ft._train
    cross_entropy = ft.cross_entropy
    adam_step = torch.optim.Adam.step
    probing = True
    validation_losses = iter(losses)
    updates = []
    validations = []
    restored: dict[str, Any] = {}
    preflight: dict[str, Any] = {}

    def run(*args: Any, **kwargs: Any) -> Any:
        nonlocal probing
        probing = kwargs["probing"]
        return train(*args, **kwargs)

    def objective(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not probing and not torch.is_grad_enabled():
            return output.new_tensor(next(validation_losses))
        return cross_entropy(output, target)

    def step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        result = adam_step(optimizer, *args, **kwargs)
        if not probing:
            parameter = optimizer.param_groups[0]["params"][0]
            updates.append(parameter.detach().clone())
        return result

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "validation":
            validations.append(data)
        elif event == "restored":
            restored.update(data)
        elif event == "preflight":
            preflight.update(data)

    monkeypatch.setattr(ft, "_train", run)
    monkeypatch.setattr(ft, "cross_entropy", objective)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    options: dict[str, Any] = (
        {} if patience is None else {"patience": patience}
    )
    cuda_model.fit(
        torch.randn(rows, 3),
        torch.arange(rows).remainder(2).reshape(-1, 1),
        recipe=_recipe(),
        finetune=16,
        query_batch_size=8,
        ema_alpha=alpha,
        _finetune_observer=observer,
        **options,
    )
    report = cuda_model.finetune_result_
    assert report is not None
    budget = (rows - 8 + 7) // 8
    assert report["planned_steps"] == report["steps_per_epoch"] == budget
    assert preflight["planned_steps"] == budget
    assert preflight["planned_validation_checks"] == budget + 1
    timings = preflight["timings"]
    assert preflight["projected_seconds"] == pytest.approx(
        timings["prepare_seconds"]
        + timings["step_seconds"] * budget
        + timings["validation_seconds"] * (budget + 1)
        + timings["cache_seconds"]
    )
    assert report["completed_steps"] == len(updates) == len(losses) - 1
    assert report["validation_checks"] == len(losses)
    assert len(validations) == len(report["history"]) == len(updates)
    assert [v["steps"] for v in validations] == list(range(1, len(losses)))
    assert [v["validation_loss"] for v in validations] == pytest.approx(
        losses[1:]
    )
    ema = losses[0]
    expected_emas = [ema]
    for loss in losses[1:]:
        ema = alpha * ema + (1 - alpha) * loss
        expected_emas.append(ema)
    assert [v["validation_loss_ema"] for v in validations] == pytest.approx(
        expected_emas[1:]
    )
    lookback = patience if patience is not None else 20
    assert report["ema_alpha"] == alpha
    assert report["ema_lookback"] == lookback
    assert report["final_validation_loss_ema"] == pytest.approx(ema)
    for index, entry in enumerate(validations, 1):
        reference = entry["reference_validation_loss_ema"]
        if index < lookback:
            assert reference is None
        else:
            assert reference == pytest.approx(expected_emas[index - lookback])
            if index < len(updates) or reason == "epoch_limit":
                assert entry["validation_loss_ema"] < reference
    if reason == "ema":
        assert ema >= expected_emas[len(updates) - lookback]
    assert report["stop_reason"] == reason
    assert report["best_step"] == restored["best_step"] == best_step
    assert report["best_validation_loss"] == pytest.approx(min(losses))
    assert report["completed_epochs"] == len(updates) // budget
    expected = (
        updates[best_step - 1] if best_step else torch.zeros_like(updates[0])
    )
    torch.testing.assert_close(
        restored["adjustment"], expected, rtol=0, atol=0
    )
    if reason == "epoch_limit":
        assert report["history"][-1]["learning_rate"] == pytest.approx(1e-5)
        assert not report["history"][-1]["partial_epoch"]
    else:
        assert report["history"][-1]["partial_epoch"]
    assert cuda_model.predict(torch.randn(4, 3)).numerical.isfinite().all()


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
        patience=10,
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
    cuda_model.n_estimators = 8
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
    cuda_model.n_estimators = 1
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
        assert len(indices) == 95
        assert int((y[indices] == 0).sum()) == 76
        assert int((y[indices] == 1).sum()) == 19
        assert not torch.isin(indices, validation[0]).any()
    report = cuda_model.finetune_result_
    assert report is not None
    assert report["selection"] == "stratified"
    assert report["context_size"] == 95
    assert report["total_context_rows"] == 190
    prediction = cuda_model.predict(x[:5]).numerical
    assert prediction.isfinite().all()
    torch.testing.assert_close(
        prediction.sum(-1), torch.ones(5, device=prediction.device)
    )


@pytest.mark.parametrize("num_classes", [3, 10, 12])
def test_multiclass_selection(num_classes: int) -> None:
    counts = torch.arange(2, num_classes + 2) * 10
    labels = torch.repeat_interleave(torch.arange(num_classes), counts)
    generator = torch.Generator().manual_seed(17)
    train, validation = _split(labels, generator)
    assert len(validation) == (len(labels) + 9) // 10
    assert labels[train].unique().numel() == num_classes
    assert labels[validation].unique().numel() == num_classes
    size = num_classes * 3
    context = _context_order(train, labels, generator, context_rows=size)[
        :size
    ]
    assert len(context.unique()) == size
    assert labels[context].unique().numel() == num_classes
    assert not torch.isin(context, validation).any()
    generator.manual_seed(17)
    repeated_train, repeated_validation = _split(labels, generator)
    assert train.equal(repeated_train)
    assert validation.equal(repeated_validation)
    assert context.equal(
        _context_order(train, labels, generator, context_rows=size)[:size]
    )
    with pytest.raises(ValueError, match="every class"):
        _context_order(train, labels, generator, context_rows=num_classes - 1)


@pytest.mark.parametrize(
    "kind", ["native", "native10", "ecoc", "regression", "constant"]
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_all_task_prediction_and_serialization(
    cuda_model: KumoTabular,
    kind: str,
    shuffle: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regression = kind in ("regression", "constant")
    model = (
        KumoTabular(
            task="regression",
            size="small",
            pretrained=False,
            device="cuda",
            n_estimators=1,
        )
        if regression
        else cuda_model
    )
    if regression:
        for parameter in model.parameters():
            if not parameter.any():
                torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(160, 4)
    if regression:
        y = (
            torch.full((160, 1), 123.0)
            if kind == "constant"
            else (100 + 30 * x[:, :1])
        )
        recipe = Recipe(
            features=sp.Standardize(),
            target=[sp.Standardize(), sp.FlipSign(probability=1.0)],
            output=[sp.SortQuantiles(), sp.AverageEstimators()],
        )
    else:
        count = 12 if kind == "ecoc" else 10 if kind == "native10" else 3
        y = torch.arange(160).remainder(count).reshape(-1, 1) * 10 + 5
        recipe = _recipe()
    before = [p.detach().clone() for p in model.parameters()]
    flags = [p.requires_grad for p in model.parameters()]
    events: dict[str, Any] = {}
    updated = []
    adam_step = torch.optim.Adam.step

    def step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        parameter = optimizer.param_groups[0]["params"][0]
        assert parameter.shape == (24, 16)
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
        old = parameter.detach().clone()
        out = adam_step(optimizer, *args, **kwargs)
        updated.append(not old.equal(parameter))
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", step)

    def observer(event: str, data: dict[str, Any]) -> None:
        if event in ("selection", "context", "restored"):
            events.update(data)

    model.fit(
        x,
        y,
        recipe=recipe,
        finetune=24,
        query_batch_size=16,
        shuffle_columns=shuffle,
        _finetune_max_steps=3,
        _finetune_observer=observer,
        generator=torch.Generator().manual_seed(17),
    )
    assert any(updated)
    report = model.finetune_result_
    assert report is not None
    assert report["completed_steps"] == 3
    assert report["loss_name"] == (
        "pinball" if regression else "cross_entropy"
    )
    assert report["best_validation_loss"] <= report["initial_validation_loss"]
    assert (
        report["ecoc_tasks"] > 1
        if kind == "ecoc"
        else report["ecoc_tasks"] == 1
    )
    if kind == "ecoc":
        assert events["base"].dim() == 3
        assert not events["base"][0].equal(events["base"][1])
        assert model._cache is not None
        assert events["codebook"].equal(
            cast(Cache, model._cache[0])["ecoc_codebook"]
        )
    train, validation = events["train_indices"], events["validation_indices"]
    assert not torch.isin(events["context_indices"], validation).any()
    assert not torch.isin(train, validation).any()
    for parameter, old, flag in zip(
        model.parameters(), before, flags, strict=True
    ):
        assert parameter.equal(old)
        assert parameter.requires_grad == flag
        assert parameter.grad is None
    assert not model.training
    actual = model.predict(x[:7])
    torch.testing.assert_close(
        actual.numerical, model.predict(x[:7].cuda()).numerical
    )
    net = model.models["regression" if regression else "classification"]
    execution = events["execution"]
    features = execution.transform(x[:7], None)[0].x.numerical.cuda()
    cy = events["icl_labels"]
    if events["codebook"] is not None:
        features = features.expand(len(events["codebook"]), *features.shape)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        query = net.encode(
            features,
            cy[..., :0],
            events["categorical_mask"],
            cache=events["encoder_cache"],
        )
        context = (events["base"].float() + events["adjustment"]).to(dtype)
        direct = net.icl_block(torch.cat([context, query], dim=-2), cy).float()
    if events["codebook"] is not None:
        direct = model.ecoc._decode(direct, events["codebook"])
    elif not regression:
        direct = direct[..., : report["num_classes"]]
    raw = TableTensor(
        columns={Stype.numerical: actual.numerical_columns}, numerical=direct
    )
    if regression:
        raw = execution.inverse_transform_target([raw])[0]
    expected = execution.transform_output([raw])
    torch.testing.assert_close(
        actual.numerical, expected.numerical, rtol=0.02, atol=0.02
    )
    if regression:
        assert actual.shape == (7, 999)
        assert (actual.numerical.diff(dim=-1) >= 0).all()
    else:
        assert actual.shape == (7, report["num_classes"])
        torch.testing.assert_close(
            actual.numerical.sum(-1), torch.ones(7, device="cuda")
        )
    path = tmp_path / "fitted.pt"
    torch.save(model, path)
    restored = torch.load(path, weights_only=False)
    torch.testing.assert_close(
        actual.numerical, restored.predict(x[:7]).numerical
    )
    model.clear()
    assert model.finetune_result_ is None
    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(x[:7])


@pytest.mark.parametrize("kind", ["ecoc", "regression"])
def test_task_ensemble_target_inversion(
    cuda_model: KumoTabular, kind: str
) -> None:
    regression = kind == "regression"
    model = (
        KumoTabular(
            task="regression",
            size="small",
            pretrained=False,
            device="cuda",
            n_estimators=1,
        )
        if regression
        else cuda_model
    )
    x = torch.randn(160, 3)
    y = (
        (500 + 100 * x[:, :1])
        if regression
        else torch.arange(160).remainder(12).reshape(-1, 1)
    )
    members = []

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "restored":
            members.append(model._cache)

    model.fit(
        x,
        y,
        finetune=24,
        n_estimators=3,
        query_batch_size=16,
        _finetune_max_steps=2,
        _finetune_observer=observer,
        generator=torch.Generator().manual_seed(17),
    )
    ensemble = model._cache
    try:
        predictions = []
        for cache in members:
            model._cache = cache
            predictions.append(model.predict(x[:8]))
    finally:
        model._cache = ensemble
    output = model.predict(x[:8])
    actual = output.numerical
    expected = torch.stack(
        [
            prediction.numerical[
                :,
                [
                    prediction.numerical_columns.index(column)
                    for column in output.numerical_columns
                ],
            ]
            for prediction in predictions
        ]
    ).mean(0)
    torch.testing.assert_close(actual, expected)
    assert actual.isfinite().all()
    assert model.finetune_result_ is not None
    assert model.finetune_result_["completed_steps"] == 6


@pytest.mark.parametrize(
    "invalid", ["rare", "small_holdout", "missing", "nonfinite", "multiple"]
)
def test_task_target_errors(cuda_model: KumoTabular, invalid: str) -> None:
    x = torch.randn(40, 3)
    y = torch.arange(40).remainder(3).reshape(-1, 1)
    match = ""
    if invalid == "rare":
        y[-1] = 99
        match = "two rows each"
    elif invalid == "small_holdout":
        y = torch.arange(40).remainder(5).reshape(-1, 1)
        match = "validation holdout"
    elif invalid == "missing":
        y = TableTensor.from_tensor(y)
        y.categorical.code[-1] = -1
        match = "nonmissing"
    elif invalid == "nonfinite":
        cuda_model = KumoTabular(
            task="regression",
            size="small",
            pretrained=False,
            device="cuda",
            n_estimators=1,
        )
        y = torch.randn(40, 1)
        y[-1] = torch.inf
        match = "finite regression"
    else:
        y = torch.cat([y, y], dim=-1)
        match = "one target"
    with pytest.raises(ValueError, match=match):
        cuda_model.fit(x, y, finetune=12, _finetune_max_steps=1)
    assert cuda_model._cache is None


def test_column_permutations_match_context_and_queries(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    net = cuda_model.models["classification"]
    encode = net.encode
    context = None
    matched = []

    def capture(
        x: torch.Tensor, y: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        nonlocal context
        if y.numel():
            context = x.sort(dim=-2).values
        elif context is not None and len(x) == len(context):
            torch.testing.assert_close(x.sort(dim=-2).values, context)
            matched.append(True)
        return encode(x, y, *args, **kwargs)

    monkeypatch.setattr(net, "encode", capture)
    cuda_model.fit(
        torch.randn(24, 4),
        torch.arange(24).remainder(2).reshape(-1, 1),
        recipe=_recipe(),
        finetune=True,
        shuffle_columns=True,
        query_batch_size=4096,
        max_epochs=2,
        _finetune_max_steps=2,
    )
    assert len(matched) == 4  # Two preflight steps and two real updates.


@pytest.mark.parametrize(
    ("rows", "context", "queries", "expected"),
    [
        (100000, 32768, 4096, 4096),
        (36864, 32768, 4096, 1844),
        (36865, 32768, 4096, 4096),
        (1000, 950, 64, 50),
        (1000, 128, 64, 64),
        (41, 39, 4096, 3),
        (40, 38, 4096, 2),
    ],
)
def test_validation_sizing_rule(
    rows: int, context: int, queries: int, expected: int
) -> None:
    assert _validation_rows(rows, context, queries) == expected


@pytest.mark.parametrize(
    ("rows", "queries"),
    [(100, 1), (5000, 128), (34000, 64), (34000, 4096), (36864, 4096)],
)
def test_automatic_context_and_holdout_are_consistent(
    rows: int, queries: int
) -> None:
    context = _context_rows(True, rows, queries)
    holdout = _validation_rows(rows, context, queries)
    assert context <= min(32768, rows - holdout)
    if context < 32768:
        larger = context + 1
        assert larger > rows - _validation_rows(rows, larger, queries)


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize(
    ("rows", "context", "queries", "expected"),
    [(80, 24, 16, 16), (40, 30, 16, 2), (40, 24, 16, 2)],
)
def test_fit_validation_partition(
    cuda_model: KumoTabular,
    regression: bool,
    rows: int,
    context: int,
    queries: int,
    expected: int,
) -> None:
    model = (
        KumoTabular(
            task="regression",
            size="small",
            pretrained=False,
            device="cuda",
            n_estimators=2,
        )
        if regression
        else cuda_model
    )
    selection = []
    x = torch.randn(rows, 3)
    y = (
        torch.randn(rows, 1)
        if regression
        else torch.arange(rows).remainder(2).reshape(-1, 1)
    )

    def observer(event: str, data: dict[str, Any]) -> None:
        if event == "selection":
            selection.append(data)

    model.fit(
        x,
        y,
        finetune=context,
        query_batch_size=queries,
        n_estimators=2,
        _finetune_max_steps=1,
        _finetune_observer=observer,
        generator=torch.Generator().manual_seed(17),
    )
    report = model.finetune_result_
    assert report is not None
    assert report["validation_rows"] == expected
    assert report["query_batch_size"] == queries
    assert len(selection) == 2
    assert selection[0]["validation_indices"].equal(
        selection[1]["validation_indices"]
    )
    for member in selection:
        validation = member["validation_indices"]
        assert len(validation) == expected
        assert not torch.isin(validation, member["train_indices"]).any()
        assert not torch.isin(
            validation, member["context_order"][:context]
        ).any()
    assert model.predict(x[:3]).numerical.isfinite().all()


@pytest.mark.parametrize("count", [1, 3, 8])
def test_constructor_estimators_used_for_tuning(
    cuda_model: KumoTabular, count: int
) -> None:
    # Use the constructor itself so the eight-member default is exercised.
    options = {} if count == 8 else {"n_estimators": count}
    model = KumoTabular(
        task="classification",
        size="small",
        pretrained=False,
        device="cuda",
        **options,
    )
    assert model.n_estimators == count
    x = torch.randn(80, 3)
    y = torch.arange(80).remainder(2).reshape(-1, 1)
    model.fit(x, y, finetune=8, query_batch_size=16, _finetune_max_steps=1)
    result = model.finetune_result_
    assert result is not None
    assert result["num_estimators"] == count
    assert result["completed_steps"] == count
    assert model.predict(x[:4]).numerical.isfinite().all()
    model.fit(
        x,
        y,
        finetune=8,
        n_estimators=1,
        query_batch_size=16,
        _finetune_max_steps=1,
    )
    assert model.n_estimators == count
    assert model.finetune_result_ is not None
    assert model.finetune_result_["num_estimators"] == 1


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_constructor_estimators(count: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        KumoTabular(n_estimators=count)


@pytest.mark.parametrize("bf16", [False, True])
def test_native_mixed_precision_and_scaling(
    cuda_model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
    bf16: bool,
) -> None:
    if bf16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native BF16 hardware required")
    dtype = torch.bfloat16 if bf16 else torch.float16
    checks = []
    capabilities = []
    scalers = []
    scaler = torch.amp.GradScaler
    loss = ft.cross_entropy

    def supported(*, including_emulation: bool = True) -> bool:
        capabilities.append(including_emulation)
        return bf16

    def make_scaler(*args: Any, **kwargs: Any) -> Any:
        scalers.append(kwargs["enabled"])
        return scaler(*args, **kwargs)

    def check_loss(
        output: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> Any:
        assert output.dtype == torch.float32
        return loss(output, target, **kwargs)

    def capture(
        module: torch.nn.Module, args: Any, output: torch.Tensor
    ) -> None:
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") == dtype
        assert output.dtype == dtype
        checks.append(True)

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    monkeypatch.setattr(torch.amp, "GradScaler", make_scaler)
    monkeypatch.setattr(ft, "cross_entropy", check_loss)
    net = cuda_model.models["classification"]
    hook = net.icl_block.register_forward_hook(capture)
    try:
        x = torch.randn(80, 3)
        y = torch.arange(80).remainder(2).reshape(-1, 1)
        cuda_model.fit(
            x, y, finetune=8, query_batch_size=16, _finetune_max_steps=1
        )
        assert cuda_model.predict(x[:4]).numerical.isfinite().all()
    finally:
        hook.remove()
    assert checks
    assert capabilities[0] is False
    assert scalers == [not bf16, not bf16]
    assert cuda_model.finetune_result_ is not None
    assert cuda_model.finetune_result_["precision"] == str(dtype).removeprefix(
        "torch."
    )


def test_constructor_estimator_zero_shot_default(
    cuda_model: KumoTabular, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = KumoTabular(
        task="classification",
        size="small",
        pretrained=False,
        device="cuda",
        n_estimators=3,
    )
    counts = []

    def fit(*args: Any, **kwargs: Any) -> None:
        counts.append(kwargs["num_estimators"])

    monkeypatch.setattr(ICLModel, "fit", fit)
    x = torch.randn(40, 3)
    y = torch.arange(40).remainder(2).reshape(-1, 1)
    model.fit(x, y)
    model.fit(x, y, n_estimators=2)
    model.fit(x, y, num_estimators=1)
    model.fit(x, y, finetune=False)
    assert counts == [3, 2, 1, 3]
    assert model.n_estimators == 3
