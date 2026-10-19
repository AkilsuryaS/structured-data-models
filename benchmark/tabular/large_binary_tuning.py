# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-only update schedule for frozen-model context optimization.

The neural encoding and cache construction mirror the production trainer.
Holdout identities and update-based validation belong to the benchmark.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from sdm import EnsembleTable, Recipe, TableTensor, Task
from sdm.cache import Cache
from sdm.models.base import _categorical_mask
from sdm.models.kumo.tabular.finetune import _randperm, _timestamp
from sdm.processing.execution import RecipeExecution

if TYPE_CHECKING:
    from sdm.models.kumo.tabular.model import KumoTabular, _KumoTabular


def train_context(
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
    validation_interval: int = 100,
    validation_batch_size: int = 1024,
    warmup: int = 50,
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
    warmup = min(warmup, planned_steps - 1)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        fraction = (step - warmup) / max(1, planned_steps - warmup)
        floor = min_learning_rate / learning_rate
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
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
            for indices in validation_indices.split(validation_batch_size):
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
    sampling = np.random.default_rng(
        generator.initial_seed() if generator else None
    )
    train_array = train_indices.numpy()
    for epoch in range(
        1 if probing else math.ceil(planned_steps / validation_interval)
    ):
        epoch_started = _timestamp(device) if observer is not None else 0.0
        epoch_steps = min(validation_interval, planned_steps - steps)
        # A second backward pass includes persistent Adam-state allocations.
        trial_steps = min(2, planned_steps) if probing else epoch_steps
        for _ in range(trial_steps):
            step_started = (
                _timestamp(device) if observer is not None and probing else 0.0
            )
            indices = torch.from_numpy(
                train_array[
                    sampling.choice(
                        len(train_array), batch_size, replace=False
                    )
                ]
            )
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
                "x_schema": context.x.schema,
                "y_schema": context.y.schema,
                "classes": context.y.categorical.categories[0],
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
