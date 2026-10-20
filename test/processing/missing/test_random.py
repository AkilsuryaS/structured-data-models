# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, EnsembleTable, TableTensor
from sdm.testing import withCUDA


def test_masks_cells_at_the_configured_rate() -> None:
    rate = 0.05
    table = TableTensor(
        numerical=torch.randn(4, 4000, 3),
        categorical=CategoricalTensor(
            code=torch.randint(3, (4, 4000, 2)),
            categories=(torch.arange(3),) * 2,
        ),
    )

    output = sp.MaskRandom(rate=(rate, rate)).transform(table)

    numerical_missing = output.numerical.isnan()
    categorical_missing = output.categorical.code == -1
    missing = torch.cat((numerical_missing, categorical_missing), dim=-1)
    actual_rate = missing.float().mean(dim=(-2, -1))
    torch.testing.assert_close(
        actual_rate,
        torch.full_like(actual_rate, rate),
        rtol=0.0,
        atol=0.01,
    )
    assert output.numerical[~numerical_missing].equal(
        table.numerical[~numerical_missing]
    )
    assert output.categorical.code[~categorical_missing].equal(
        table.categorical.code[~categorical_missing]
    )
    assert all(
        actual.equal(expected)
        for actual, expected in zip(
            output.categorical.categories,
            table.categorical.categories,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "processor",
    [
        sp.MaskRandom(rate=(0.1, 0.3)),
        sp.Sequential(sp.MaskRandom(rate=(0.1, 0.3)), sp.Identity()),
        sp.Choice(sp.MaskRandom(rate=(0.1, 0.3)), method="round_robin"),
        sp.StypeDispatch(numerical=sp.MaskRandom(rate=(0.1, 0.3))),
        sp.TaskDispatch(regression=sp.MaskRandom(rate=(0.1, 0.3))),
        sp.TableDispatch(task=sp.MaskRandom(rate=(0.1, 0.3))),
        sp.EnsembleProcessorAdapter(sp.MaskRandom(rate=(0.1, 0.3))),
        sp.Sequential(
            sp.StypeDispatch(
                numerical=sp.Choice(
                    sp.MaskRandom(rate=(0.1, 0.3)), method="round_robin"
                )
            ),
            sp.Identity(),
        ),
    ],
    ids=lambda processor: type(processor).__name__,
)
@pytest.mark.parametrize(
    "method",
    [
        "transform",
        "forward",
        "fit_transform",
        "transform_ensemble",
        "fit_transform_ensemble",
    ],
)
@withCUDA
def test_generator_controls_each_transform(
    processor: sp.Processor,
    method: str,
    device: torch.device,
) -> None:
    processor = deepcopy(processor).to(device)
    if isinstance(processor, sp.TaskDispatch):
        processor._task = "regression"
    if isinstance(processor, sp.TableDispatch):
        processor._route = "task"
    table = TableTensor.from_tensor(torch.ones(1000, 4, device=device))
    ensemble = method.endswith("_ensemble")
    if ensemble:
        ensemble_processor = sp.EnsembleProcessor.as_processor(processor)
        value = EnsembleTable.from_table(table, num_members=2)
        ensemble_processor.fit_ensemble(value)
        transform = getattr(ensemble_processor, method)
    else:
        value = table
        processor.fit(table)
        transform = getattr(processor, method)

    generator = torch.Generator(device=device).manual_seed(0)
    first = transform(value, generator=generator)
    second = transform(value, generator=generator)
    repeated = transform(
        value, generator=torch.Generator(device=device).manual_seed(0)
    )

    if ensemble:
        first, second, repeated = first[0], second[0], repeated[0]
    assert first.allclose(repeated, rtol=0, atol=0, equal_nan=True)
    assert not first.numerical.isnan().equal(second.numerical.isnan())
