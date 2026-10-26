# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import AddCategoryCounts
from sdm.processing.categorical._categorical import (
    _check_categorical_codes,
    _check_categories,
)
from sdm.testing import withCUDA


def _table(
    codes: Tensor, numbers: Tensor, data: Tensor, offset: Tensor
) -> TableTensor:
    return TableTensor(
        categorical=CategoricalTensor(
            code=codes,
            categories=(
                numbers,
                StringTensor(data=data, offset=offset, valid=None, size=(2,)),
            ),
        ),
    )


@withCUDA
@pytest.mark.parametrize("processor", [False, True])
def test_category_checks_compile_and_reject_changed_inputs(
    compilation: None,
    device: torch.device,
    processor: bool,
) -> None:
    codes = torch.tensor(
        [[[0, 0], [1, 1], [0, -1]], [[1, 1], [1, 0], [-2, -1]]],
        device=device,
    )
    numbers = torch.tensor([10, 20], device=device)
    strings = StringTensor.from_list(["a", "b"], device=device)
    table = _table(codes, numbers, strings._data, strings._offset)
    fitted = AddCategoryCounts().fit(table)
    categories = BufferList(table.categorical.categories)

    def transform(
        codes: Tensor, numbers: Tensor, data: Tensor, offset: Tensor
    ) -> Tensor:
        table = _table(codes, numbers, data, offset)
        if processor:
            return fitted.transform(table).numerical
        _check_categories(table, categories)
        _check_categorical_codes(table)
        return codes + 1

    compiled = torch.compile(transform, backend="inductor", fullgraph=True)
    expected = fitted.transform(table).numerical if processor else codes + 1
    actual = compiled(codes, numbers, strings._data, strings._offset)
    torch.testing.assert_close(actual, expected)

    # Change data after capture, preserving shapes to exercise runtime checks.
    for changed_numbers, changed_strings, column in (
        (numbers.flip(0), strings, "cat_0"),
        (numbers, StringTensor.from_list(["b", "a"], device=device), "cat_1"),
    ):
        with pytest.raises(
            ValueError,
            match=(
                f"vocabulary for categorical column '{column}'"
                ".*values and order"
            ),
        ):
            compiled(
                codes,
                changed_numbers,
                changed_strings._data,
                changed_strings._offset,
            )

    invalid_codes = codes.clone()
    invalid_codes[1, 2, 1] = 2
    with pytest.raises(
        ValueError,
        match=(
            "Categorical column 'cat_1' contains a code "
            "outside its category vocabulary"
        ),
    ):
        compiled(invalid_codes, numbers, strings._data, strings._offset)

    # Valid input remains usable after a rejected call.
    torch.testing.assert_close(
        compiled(codes, numbers, strings._data, strings._offset), expected
    )


def test_category_checks_preserve_column_count_and_disable_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[3], [-1]]),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    categories: BufferList[Tensor] = BufferList()
    with pytest.raises(
        ValueError, match=r"Expected 0 fitted categorical columns \(got 1\)"
    ):
        _check_categories(table, categories)
    with pytest.raises(ValueError, match="outside its category vocabulary"):
        _check_categorical_codes(table)

    monkeypatch.setenv("SDM_ENABLE_CATEGORY_CHECKS", "0")
    _check_categories(table, categories)
    _check_categorical_codes(table)


def test_compiled_category_equality_preserves_string_views(
    compilation: None,
) -> None:
    strings = StringTensor.from_list(["a", "b", "c", "d"])
    categories: BufferList[Tensor] = BufferList(
        (StringTensor.from_list(["b", "d"]),)
    )

    def check(data: Tensor, offset: Tensor) -> Tensor:
        category = StringTensor(
            data=data,
            offset=offset,
            valid=None,
            size=(2,),
            stride=(2,),
            storage_offset=1,
        )
        table = TableTensor(
            categorical=CategoricalTensor(
                code=torch.tensor([[0], [1]]),
                categories=(category,),
            ),
        )
        _check_categories(table, categories)
        return data + 1

    compiled = torch.compile(check, backend="inductor", fullgraph=True)
    torch.testing.assert_close(
        compiled(strings._data, strings._offset), strings._data + 1
    )
    with pytest.raises(ValueError, match="match the fitted values and order"):
        compiled(strings._data + 1, strings._offset)
