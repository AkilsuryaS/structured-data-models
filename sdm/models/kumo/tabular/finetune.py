# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen-model context optimization, separate from inference cache replay."""

from __future__ import annotations

import copy
import gc
import math
import time
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from sdm import EnsembleTable, Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models.base import _categorical_mask
from sdm.processing.execution import MemberQuery, RecipeExecution

if TYPE_CHECKING:
    from sdm.models.kumo.tabular.model import KumoTabular, _KumoTabular


def _context_rows(finetune: bool | int | float, rows: int) -> int:
    """Resolve automatic sizing, a count, or a fraction of supplied rows."""
    if not isinstance(finetune, Real):
        raise ValueError(
            "finetune must be a boolean, a row count, or a fraction in (0, 1)."
        )
    if finetune is True:
        count = min(32_768, rows - math.ceil(rows * 0.1))
    elif isinstance(finetune, Integral):
        count = int(finetune)
    else:
        if not math.isfinite(finetune) or finetune < 0:
            raise ValueError("finetune must be finite and nonnegative.")
        if finetune == 0:
            return 0
        if finetune < 1:
            count = math.floor(finetune * rows)
        elif finetune == int(finetune):
            count = int(finetune)
        else:
            raise ValueError("finetune values >= 1 must be whole row counts.")
    if count == 0 and finetune == 0:
        return 0
    if count < 2:
        raise ValueError("The tuning context must contain at least two rows.")
    return count


def _timestamp(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter()


def _randperm(n: int, generator: torch.Generator | None) -> Tensor:
    device = "cpu" if generator is None else generator.device
    return torch.randperm(n, device=device, generator=generator).cpu()


def _split(
    labels: Tensor, generator: torch.Generator | None
) -> tuple[Tensor, Tensor]:
    """Stratified holdout with exactly ceil(10% of rows), two classes each."""
    classes, counts = labels.unique(return_counts=True)
    if classes.numel() != 2 or counts.min() < 2:
        raise ValueError(
            "Context tuning needs two classes with at least two rows each."
        )
    n_validation = math.ceil(labels.numel() * 0.1)
    if n_validation < 2:
        raise ValueError(
            "The 10% validation holdout must contain both classes."
        )
    first = round(n_validation * int(counts[0]) / labels.numel())
    first = max(1, min(first, int(counts[0]) - 1, n_validation - 1))
    second = n_validation - first
    if second >= counts[1]:
        second = int(counts[1]) - 1
        first = n_validation - second
    train, validation = [], []
    for label, n in zip(classes, (first, second), strict=True):
        indices = (labels == label).nonzero().flatten()
        indices = indices[_randperm(len(indices), generator)]
        validation.append(indices[:n])
        train.append(indices[n:])
    return torch.cat(train), torch.cat(validation)


def _context_order(
    train: Tensor,
    labels: Tensor,
    generator: torch.Generator | None,
    *,
    context_rows: int = 2,
) -> Tensor:
    """Sample a stratified context prefix, retaining both classes."""
    order = train[_randperm(len(train), generator)]
    ordered_labels = labels[order]
    classes, counts = ordered_labels.unique(return_counts=True)
    first_count = round(context_rows * int(counts[0]) / len(train))
    first_count = max(
        1,
        context_rows - int(counts[1]),
        min(first_count, int(counts[0]), context_rows - 1),
    )
    first = order[ordered_labels == classes[0]]
    second = order[ordered_labels == classes[1]]
    second_count = context_rows - first_count
    chosen = torch.cat([first[:first_count], second[:second_count]])
    chosen = chosen[_randperm(context_rows, generator)]
    return torch.cat([chosen, first[first_count:], second[second_count:]])


def _capacity(
    limit: int, estimate: int, probe: Any, *, tolerance: int = 1
) -> tuple[int, list[dict[str, Any]]]:
    """Bracket with an estimate, then probe to the requested resolution."""
    low, high = 1, limit + 1
    candidate = min(limit, max(2, estimate))
    measurements = []
    while low + (1 if low == 1 else tolerance) < high:
        fits, peak = probe(candidate)
        measurements.append(
            {"context_size": candidate, "fits": fits, "peak_bytes": peak}
        )
        if fits:
            low = candidate
            if candidate == limit:
                break
            candidate = (
                min(limit, candidate * 2)
                if high == limit + 1
                else (low + high) // 2
            )
        else:
            high = candidate
            candidate = (
                max(2, candidate // 2) if low == 1 else (low + high) // 2
            )
    if low < 2:
        raise RuntimeError(
            "CUDA memory cannot fit a context containing both classes."
        )
    return low, measurements


class _TunedRecipeExecution(RecipeExecution):
    """Replay independently fitted recipes and average member probabilities."""

    def __init__(self, executions: Sequence[RecipeExecution]) -> None:
        super().__init__(executions[0].recipe)
        self.executions = tuple(executions)

    @property
    def num_members(self) -> int:
        return len(self.executions)

    def transform(
        self,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[MemberQuery, ...]:
        # Each member's preprocessing was fitted on its sampled CPU context.
        x = x.cpu()
        return tuple(
            execution.transform(x, related_tables)[0]
            for execution in self.executions
        )

    def transform_output(self, outputs: Sequence[TableTensor]) -> TableTensor:
        predictions = [
            execution.transform_output([output])
            for execution, output in zip(self.executions, outputs, strict=True)
        ]
        stacked = cast(
            TableTensor, torch.stack(cast(list[Tensor], predictions), dim=0)
        )
        return TableTensor(
            columns={Stype.numerical: stacked.columns[Stype.numerical]},
            numerical=stacked.numerical.mean(dim=0),
        )


def fit_context(
    model: KumoTabular,
    x: Tensor | TableTensor,
    y: Tensor | TableTensor,
    *,
    fit_kwargs: dict[str, Any],
    context_rows: int,
    query_batch_size: int,
    learning_rate: float,
    min_learning_rate: float,
    max_epochs: int,
    patience: int,
    min_epochs: int,
    min_improvement: float,
    shuffle_columns: bool,
    memory_fraction: float,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    max_steps: int | None = None,
) -> None:
    """Optimize exactly the requested number of sampled binary context rows."""
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1].")
    if (
        query_batch_size < 1
        or max_epochs < 1
        or patience < 1
        or min_epochs < 0
        or (max_steps is not None and max_steps < 1)
    ):
        raise ValueError(
            "Batch size, epochs and patience must define a positive budget."
        )
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(
            "Positive finetune requires a model on a local CUDA GPU."
        )
    model.clear()
    model.eval()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    baseline = torch.cuda.memory_allocated(device)
    allocation_limit = int(
        total * torch.cuda.get_per_process_memory_fraction(device)
    )
    budget = baseline + int(
        min(free, allocation_limit - baseline) * memory_fraction
    )
    generator = fit_kwargs.get("generator")
    if generator is not None and generator.device.type != "cpu":
        seed = int(
            torch.empty(
                (), dtype=torch.int64, device=generator.device
            ).random_(generator=generator)
        )
        generator = torch.Generator().manual_seed(seed)
    fit_kwargs = {**fit_kwargs, "generator": generator}
    if x.dim() != 2 or y.dim() not in (1, 2):
        raise ValueError(
            "Context fine-tuning requires an unbatched training table."
        )
    num_estimators = fit_kwargs.get("num_estimators") or 1
    if (
        fit_kwargs.get("callbacks")
        or fit_kwargs.get("related_tables")
        or fit_kwargs.get("context_only_columns")
    ):
        raise NotImplementedError(
            "Context tuning does not support callbacks, related tables "
            "or context-only tables."
        )
    x = cast(
        TableTensor,
        (
            x.cpu()
            if isinstance(x, TableTensor)
            else TableTensor.from_tensor(x.cpu())
        ),
    )
    y = cast(
        TableTensor,
        (
            y.cpu()
            if isinstance(y, TableTensor)
            else TableTensor.from_tensor(y.cpu().reshape(-1, 1))
        ),
    )
    if y.categorical.size(-1) != 1 or y.numerical.size(-1) != 0:
        raise NotImplementedError(
            "Context tuning currently supports binary classification only."
        )
    labels = y.categorical.code.squeeze(-1)
    if labels.unique().numel() != 2 or len(y.categorical.categories[0]) != 2:
        raise NotImplementedError(
            "Context tuning currently supports binary classification only."
        )
    train, validation = _split(labels, generator)
    if context_rows > len(train):
        raise ValueError(
            f"Requested tuning context has {context_rows} rows, but only "
            f"{len(train)} remain after the 10% validation holdout."
        )
    with torch.cuda.device(device):
        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    reports: list[dict[str, Any]] = []
    caches: list[Cache] = []
    try:

        def fit_member(estimator: int) -> dict[str, Any]:
            measurements: list[dict[str, Any]] = []
            probe_timings: dict[int, dict[str, float]] = {}
            member_observer = observer
            if observer is not None and num_estimators > 1:

                def member_observer(event: str, data: dict[str, Any]) -> None:
                    assert observer is not None
                    observer(event, {**data, "estimator": estimator})

            order = _context_order(
                train, labels, generator, context_rows=context_rows
            )
            if member_observer is not None:
                member_observer(
                    "selection",
                    {
                        "train_indices": train,
                        "validation_indices": validation,
                        "context_order": order,
                    },
                )
            # Probes must not consume the real training RNG sequence.
            rng_state = None if generator is None else generator.get_state()
            cpu_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(device)

            def run(size: int, probing: bool) -> dict[str, Any]:
                return _train(
                    model,
                    x,
                    y,
                    order[:size],
                    train,
                    validation,
                    recipe=fit_kwargs.get("recipe"),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                    budget=budget,
                    query_batch_size=query_batch_size,
                    learning_rate=learning_rate,
                    min_learning_rate=min_learning_rate,
                    max_epochs=max_epochs,
                    patience=patience,
                    min_epochs=min_epochs,
                    min_improvement=min_improvement,
                    shuffle_columns=shuffle_columns,
                    probing=probing,
                    observer=member_observer,
                    max_steps=max_steps,
                )

            def probe(size: int) -> tuple[bool, int]:
                torch.cuda.reset_peak_memory_stats(device)
                started = (
                    _timestamp(device) if member_observer is not None else 0.0
                )
                try:
                    trial = run(size, True)
                    if member_observer is not None:
                        probe_timings[size] = trial["probe_timings"]
                    fits = True
                except torch.cuda.OutOfMemoryError:
                    fits = False
                finally:
                    model.clear()
                    if generator is not None:
                        generator.set_state(cast(Tensor, rng_state))
                    torch.random.set_rng_state(cpu_state)
                    torch.cuda.set_rng_state(cuda_state, device)
                peak = torch.cuda.max_memory_allocated(device)
                if member_observer is not None:
                    reserved = torch.cuda.max_memory_reserved(device)
                    member_observer(
                        "probe",
                        {
                            "context_size": size,
                            "fits": fits,
                            "peak_allocated_bytes": peak,
                            "peak_reserved_bytes": reserved,
                            "seconds": _timestamp(device) - started,
                            "timings": probe_timings.get(size),
                        },
                    )
                gc.collect()
                torch.cuda.empty_cache()
                return fits, peak

            fits, peak = probe(context_rows)
            measurements.append(
                {
                    "context_size": context_rows,
                    "fits": fits,
                    "peak_bytes": peak,
                }
            )
            if not fits:
                raise RuntimeError(
                    f"Requested tuning context of {context_rows} rows exceeds "
                    "the CUDA memory budget; choose a smaller finetune value."
                )
            torch.cuda.reset_peak_memory_stats(device)
            if member_observer is not None:
                timings = probe_timings[context_rows]
                steps_per_epoch = math.ceil(len(train) / query_batch_size)
                planned_updates = max_epochs * steps_per_epoch
                if max_steps is not None:
                    planned_updates = min(planned_updates, max_steps)
                planned_epochs = math.ceil(planned_updates / steps_per_epoch)
                projected = (
                    timings["step_seconds"] * planned_updates
                    + timings["validation_seconds"] * (planned_epochs + 1)
                    + timings["prepare_seconds"]
                    + timings["cache_seconds"]
                )
                member_observer(
                    "preflight",
                    {
                        "context_size": context_rows,
                        "max_epochs": max_epochs,
                        "steps_per_epoch": steps_per_epoch,
                        "planned_steps": planned_updates,
                        "planned_epochs": planned_epochs,
                        "projected_seconds": projected,
                        "timings": timings,
                    },
                )
            try:
                result = run(context_rows, False)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"Requested tuning context of {context_rows} rows ran out "
                    "of CUDA memory during training; "
                    "choose a smaller finetune value."
                ) from error
            result.update(
                status="tuned",
                reason="tuning context requested",
                selection="stratified",
                context_size=context_rows,
            )
            result["memory"] = {
                "budget_bytes": budget,
                "total_bytes": total,
                "peak_bytes": torch.cuda.max_memory_allocated(device),
                "probes": measurements,
            }
            result["num_estimators"] = 1
            return result

        for estimator in range(num_estimators):
            reports.append(fit_member(estimator))
            if model._cache is not None:
                caches.append(model._cache)
        if num_estimators == 1:
            model.finetune_result_ = reports[0]
        else:
            executions = [
                cast(RecipeExecution, cache["recipe_execution"])
                for cache in caches
            ]
            model._cache = Cache(
                {
                    "recipe_execution": _TunedRecipeExecution(executions),
                    "kwargs": {},
                    "num_batches": num_estimators,
                    **{i: cache[0] for i, cache in enumerate(caches)},
                }
            ).freeze()
            model.finetune_result_ = {
                "status": "tuned",
                "reason": "tuning contexts requested",
                "selection": "stratified",
                "num_estimators": num_estimators,
                "context_size": context_rows,
                "total_context_rows": context_rows * num_estimators,
                "completed_steps": sum(r["completed_steps"] for r in reports),
                "members": reports,
                "memory": {
                    "budget_bytes": budget,
                    "total_bytes": total,
                    "peak_bytes": max(
                        r["memory"]["peak_bytes"] for r in reports
                    ),
                },
            }
    except BaseException:
        model.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise
    finally:
        for p, flag in zip(model.parameters(), flags, strict=True):
            p.requires_grad_(flag)
        model.eval()


def _train(
    model: KumoTabular,
    x: TableTensor,
    y: TableTensor,
    context_indices: Tensor,
    train_indices: Tensor,
    validation_indices: Tensor,
    *,
    recipe: Recipe | None,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
    budget: int,
    query_batch_size: int,
    learning_rate: float,
    min_learning_rate: float,
    max_epochs: int,
    patience: int,
    min_epochs: int,
    min_improvement: float,
    shuffle_columns: bool,
    probing: bool,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    started = _timestamp(device) if observer is not None else 0.0
    if observer is not None and not probing:
        observer(
            "context",
            {
                "context_indices": context_indices,
                "generator_state": None
                if generator is None
                else generator.get_state(),
                "cpu_rng_state": torch.random.get_rng_state(),
            },
        )
    net = cast("_KumoTabular", model.models[Task.classification])
    execution = RecipeExecution(
        model.default_recipe() if recipe is None else copy.deepcopy(recipe)
    )
    with torch.no_grad():
        context = execution.fit_transform(
            x=x[context_indices],
            y=y[context_indices],
            related_tables=None,
            num_members=1,
            generator=generator,
        )[0]
        model._prepare_context(context, ())
        cx = context.x.numerical.to(device)
        cy = context.y.categorical.code.squeeze(-1).to(
            device=device, dtype=torch.long
        )
        mask = _categorical_mask([context]).to(device)

    def encode_context(
        permutation: Tensor | None = None,
    ) -> tuple[Tensor, Cache]:
        cache = Cache()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            base = net.encode(
                cx if permutation is None else cx[:, permutation],
                cy,
                mask if permutation is None else mask[permutation],
                cache=cache,
            )
        return base, cache.freeze()

    base, encoder = encode_context()
    adjustment = torch.nn.Parameter(
        torch.zeros_like(base, dtype=torch.float32)
    )
    best = adjustment.detach().clone()
    optimizer = torch.optim.Adam([adjustment], lr=learning_rate)
    batch_size = min(query_batch_size, len(train_indices))
    steps_per_epoch = math.ceil(len(train_indices) / query_batch_size)
    planned_steps = max_epochs * steps_per_epoch
    if max_steps is not None:
        planned_steps = min(planned_steps, max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=planned_steps,
        eta_min=min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    def check_memory() -> None:
        if torch.cuda.max_memory_allocated(device) > budget:
            raise torch.cuda.OutOfMemoryError(
                "Context exceeds configured CUDA memory budget."
            )

    def query(
        indices: Tensor, cache: Cache, permutation: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            qx = execution.transform(x[indices], None)[0].x.numerical.to(
                device
            )
            # Use the fitted mapping, including category permutations.
            qy = (
                execution.recipe.target.transform_ensemble(
                    EnsembleTable.from_table(y[indices], num_members=1)
                )[0]
                .categorical.code.squeeze(-1)
                .to(device=device, dtype=torch.long)
            )
            with torch.autocast("cuda", dtype=dtype):
                embedding = net.encode(
                    qx if permutation is None else qx[:, permutation],
                    cy[:0],
                    mask if permutation is None else mask[permutation],
                    cache=cache,
                )
        return embedding, qy

    def logits(embedding: Tensor, context_base: Tensor) -> Tensor:
        with torch.autocast("cuda", dtype=dtype):
            rows = torch.cat(
                [(context_base.float() + adjustment).to(dtype), embedding]
            )
            return net.icl_block(rows, cy)[..., :2].float()

    def prediction_cache() -> Cache:
        cache = Cache(dict(encoder))
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            net.icl_block(
                (base.float() + adjustment).to(dtype), cy, cache=cache
            )
        return cache

    def validate() -> float:
        cache = prediction_cache().freeze()
        loss_sum = 0.0
        with torch.no_grad():
            for indices in validation_indices.split(query_batch_size):
                embedding, target = query(indices, encoder)
                with torch.autocast("cuda", dtype=dtype):
                    output = net.icl_block(embedding, cy[:0], cache=cache)[
                        ..., :2
                    ]
                loss = cross_entropy(output.float(), target, reduction="sum")
                if not loss.isfinite():
                    raise RuntimeError("Nonfinite context validation loss.")
                loss_sum += float(loss)
                check_memory()
        return loss_sum / len(validation_indices)

    prepared = _timestamp(device) if observer is not None else 0.0
    initial_loss = best_loss = validate()
    validated = _timestamp(device) if observer is not None else 0.0
    step_seconds = 0.0
    history: list[dict[str, Any]] = []
    best_epoch = 0
    stale = steps = overlap = sampled = epochs = 0
    context_members = torch.zeros(len(x), dtype=torch.bool)
    context_members[context_indices] = True
    for epoch in range(1 if probing else max_epochs):
        epoch_started = _timestamp(device) if observer is not None else 0.0
        epoch_steps = min(steps_per_epoch, planned_steps - steps)
        # A second backward pass includes persistent Adam-state allocations.
        trial_steps = min(2, planned_steps) if probing else epoch_steps
        for _ in range(trial_steps):
            step_started = (
                _timestamp(device) if observer is not None and probing else 0.0
            )
            indices = train_indices[
                _randperm(len(train_indices), generator)[:batch_size]
            ]
            permutation = None
            step_base, step_cache = base, encoder
            if shuffle_columns:
                permutation = _randperm(cx.size(-1), generator).to(device)
                step_base, step_cache = encode_context(permutation)
            embedding, target = query(indices, step_cache, permutation)
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(logits(embedding, step_base), target)
            if not loss.isfinite():
                raise RuntimeError("Nonfinite context training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [adjustment], 1.0, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            overlap += int(context_members[indices].sum())
            sampled += len(indices)
            check_memory()
            if observer is not None:
                if probing:
                    step_seconds = _timestamp(device) - step_started
                elif steps % (2 if max_steps is not None else 16) == 0:
                    observer(
                        "progress",
                        {
                            "steps": steps,
                            "epoch": epoch + 1,
                            "training_batch_loss": float(loss.detach()),
                        },
                    )
            del embedding, loss, step_base, step_cache
        epochs = epoch + 1
        validation_loss = validate()
        improvement = best_loss - validation_loss
        stale = 0 if improvement > min_improvement else stale + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best.copy_(adjustment.detach())
            best_epoch = epochs
        entry = {
            "epoch": epochs,
            "steps": steps,
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "partial_epoch": not probing and epoch_steps < steps_per_epoch,
        }
        history.append(entry)
        if observer is not None and not probing:
            observer(
                "epoch",
                {**entry, "seconds": _timestamp(device) - epoch_started},
            )
        if steps >= planned_steps or (
            epochs >= min_epochs and stale >= patience
        ):
            break
    with torch.no_grad():
        adjustment.copy_(best)
    cache_started = _timestamp(device) if observer is not None else 0.0
    cache = prediction_cache()
    cache_seconds = (
        _timestamp(device) - cache_started if observer is not None else 0.0
    )
    cache.update(
        x_schemas=(context.x.schema,),
        y_schema=context.y.schema,
        related_tables_schema=None,
        classes=context.y.categorical.categories[0],
        class_values=[None],
        categorical_mask=mask,
        finetune_dtype=dtype,
    )
    check_memory()
    model._cache = Cache(
        {
            "recipe_execution": execution,
            "kwargs": {},
            "num_batches": 1,
            0: cache,
        }
    ).freeze()
    if observer is not None and not probing:
        observer(
            "restored",
            {
                "execution": execution,
                "context_indices": context_indices,
                "base": base,
                "adjustment": adjustment.detach(),
                "labels": cy,
                "categorical_mask": mask,
                "encoder_cache": encoder,
                "best_epoch": best_epoch,
            },
        )
    return {
        "probe_timings": {
            "prepare_seconds": prepared - started,
            "validation_seconds": validated - prepared,
            "step_seconds": step_seconds,
            "cache_seconds": cache_seconds,
        },
        "history": history,
        "best_epoch": best_epoch,
        "completed_steps": steps,
        "planned_steps": planned_steps,
        "steps_per_epoch": steps_per_epoch,
        "completed_epochs": epochs,
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best_loss,
        "overlap_rate": overlap / sampled,
    }
