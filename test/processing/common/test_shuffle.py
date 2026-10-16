# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.models._batching import _stack_ensemble
from sdm.processing import (
    Cast,
    EnsembleProcessor,
    SelectColumns,
    ShuffleColumns,
)
from sdm.testing import withCUDA


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )


@pytest.mark.parametrize("method", ["random", "latin"])
def test_shuffle_columns_scalar_fit_transform_and_inverse(
    method: Literal["random", "latin"],
) -> None:
    table = _table()
    processor = ShuffleColumns(method=method)

    processor.fit(table)
    transformed = processor.transform(table)
    restored = processor.inverse_transform(transformed)

    assert restored.equal(table)


@pytest.mark.parametrize("method", ["random", "latin"])
def test_shuffle_columns_is_reproducible_with_generator(
    method: Literal["random", "latin"],
) -> None:
    table = _table()

    first = ShuffleColumns(method=method)
    first_output = first.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleColumns(method=method)
    second_output = second.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert first_output.equal(second_output)


def test_random_ensemble_matches_independent_shuffles() -> None:
    context = _table()
    query = context.replace_blocks(numerical=context.numerical + 10)
    ensemble = ShuffleColumns(method="random")
    ensemble_generator = torch.Generator().manual_seed(7)

    context_output = ensemble.fit_transform_ensemble(
        EnsembleTable.from_table(context, num_members=8),
        generator=ensemble_generator,
    )
    query_output = ensemble.transform_ensemble(
        EnsembleTable.from_table(query, num_members=8)
    )
    restored = ensemble.inverse_transform_ensemble(context_output)

    reference_generator = torch.Generator().manual_seed(7)
    references = [ShuffleColumns(method="random") for _ in range(8)]
    for member_id, processor in enumerate(references):
        expected_context = processor.fit_transform(
            context,
            generator=reference_generator,
        )
        expected_query = processor.transform(query)
        assert context_output[member_id].equal(expected_context)
        assert query_output[member_id].equal(expected_query)
        assert restored[member_id].equal(context)


@pytest.mark.parametrize("method", ["random", "latin"])
def test_shuffled_members_share_batched_storage(
    method: Literal["random", "latin"],
) -> None:
    values = torch.arange(60.0).reshape(5, 4, 3).requires_grad_()
    source = EnsembleTable(
        groups=(TableTensor(numerical=values),),
        locations=tuple((0, i) for i in range(5)),
    )
    shuffle = ShuffleColumns(method=method)
    out = shuffle.fit_transform_ensemble(
        source, generator=torch.Generator().manual_seed(42)
    )
    storage = out._positional_storage
    assert storage is not None
    assert storage.num_groups == 1
    for batch in (slice(0, 4), slice(1, 5)):
        combined = _stack_ensemble(out, batch=batch)
        expected = torch.stack([out[i].numerical for i in range(5)[batch]])
        torch.testing.assert_close(combined.numerical, expected)
        assert combined.columns == out[range(5)[batch][0]].columns
        assert combined.numerical.untyped_storage().data_ptr() == (
            storage._groups[0].numerical.untyped_storage().data_ptr()
        )
    selected = out[[4, 1, 4, 0]]
    torch.testing.assert_close(
        _stack_ensemble(selected, batch=slice(None)).numerical,
        torch.stack([out[i].numerical for i in (4, 1, 4, 0)]),
    )
    _stack_ensemble(out, batch=slice(None)).numerical.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    with torch.no_grad():
        selected[0].numerical.add_(7)
    torch.testing.assert_close(
        _stack_ensemble(selected, batch=slice(None)).numerical,
        torch.stack([selected[i].numerical for i in range(len(selected))]),
    )

    cast = EnsembleProcessor.as_processor(
        Cast(torch.float64)
    ).transform_ensemble(out)
    selected = (
        SelectColumns(2)
        .transform_ensemble(cast)
        .select_stypes(Stype.numerical)
    )
    assert selected._positional_storage is not None
    for i in range(5):
        expected = out[i].numerical[:, :2].double()
        torch.testing.assert_close(selected[i].numerical, expected)
        assert (
            selected[i].columns[Stype.numerical]
            == (out[i].columns[Stype.numerical][:2])
        )
    with torch.no_grad():
        selected[0].numerical.add_(5)
    torch.testing.assert_close(
        _stack_ensemble(selected, batch=slice(None)).numerical,
        torch.stack([selected[i].numerical for i in range(len(selected))]),
    )
    moved = selected.to("cpu")
    assert moved._positional_storage is not None
    for i in range(5):
        torch.testing.assert_close(moved[i].numerical, selected[i].numerical)

    replaced = selected.replace_groups(
        [
            group.replace_blocks(numerical=group.numerical + 10)
            for group in selected._iter_groups()
        ]
    )
    assert replaced._positional_storage is None
    torch.testing.assert_close(
        _stack_ensemble(replaced, batch=slice(None)).numerical,
        _stack_ensemble(selected, batch=slice(None)).numerical + 10,
    )


@withCUDA
def test_latin_ensemble_couples_member_permutations(
    device: torch.device,
) -> None:
    ensemble = EnsembleTable.from_tables(
        tables=(
            TableTensor.from_tensor(
                torch.arange(8, dtype=torch.float32, device=device).view(2, 4)
            ),
            TableTensor.from_tensor(
                torch.arange(4, dtype=torch.float32, device=device).view(2, 2)
            ),
        ),
        member_table_ids=(0, 0, 0, 0, 1, 1, 1, 1),
    )
    output = ShuffleColumns(method="latin").fit_transform_ensemble(ensemble)
    permutations = tuple(
        output[member_id].columns[Stype.numerical]
        for member_id in range(len(output))
    )
    # Each member must contain every source column once
    # and reorder its values accordingly.
    for member_id, permutation in enumerate(permutations):
        source = ensemble[member_id]
        result = output[member_id]
        source_columns = source.columns[Stype.numerical]
        indices = torch.tensor(
            [source_columns.index(column) for column in permutation],
            dtype=torch.long,
            device=source.numerical.device,
        )
        assert sorted(permutation) == sorted(source_columns)
        assert torch.equal(
            result.numerical,
            source.numerical.index_select(-1, indices),
        )

    # Per position, each of 4 columns must occur once
    # and each of 2 columns twice.
    for member_ids in (range(4), range(4, 8)):
        source_columns = ensemble[member_ids.start].columns[Stype.numerical]
        expected_columns = sorted(
            source_columns * (len(member_ids) // len(source_columns))
        )
        for position in zip(
            *(permutations[member_id] for member_id in member_ids),
            strict=True,
        ):
            assert sorted(position) == expected_columns


@pytest.mark.parametrize(
    "method_name",
    ["transform_ensemble", "inverse_transform_ensemble"],
)
def test_shuffle_columns_checks_num_members(method_name: str) -> None:
    processor = ShuffleColumns(method="latin")
    processor.fit_ensemble(
        EnsembleTable.from_table(_table(), num_members=8),
        generator=torch.Generator().manual_seed(9),
    )

    with pytest.raises(
        RuntimeError,
        match="was fitted with 8 ensemble members, but got 7",
    ):
        getattr(processor, method_name)(
            EnsembleTable.from_table(_table(), num_members=7)
        )
