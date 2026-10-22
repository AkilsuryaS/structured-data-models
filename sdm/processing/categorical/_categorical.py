# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import cast

import torch
from torch import Tensor

from sdm import NullableTensor, StringTensor, TableTensor, VarLenTensor
from sdm.nn._buffer import BufferList
from sdm.tensor.categorical import _iter_categories
from sdm.tensor.var_len import _contiguous

_CATEGORY_TYPES = (Tensor, NullableTensor, VarLenTensor, StringTensor)


def _check_categorical_codes(table: TableTensor) -> None:
    """Raise if a categorical code exceeds its column's category vocabulary.

    Negative codes encode missing values and are allowed.

    Args:
        table: The table whose categorical codes are validated.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table.categorical_columns
    categorical = table.categorical
    for index, category in enumerate(_iter_categories(categorical)):
        codes = categorical.code[..., index]
        if (codes >= category.numel()).any():
            raise ValueError(
                f"Categorical column {columns[index]!r} contains a code "
                "outside its category vocabulary."
            )


def _check_categories(
    table: TableTensor,
    categories: BufferList[Tensor],
) -> None:
    """Raise if a category vocabulary differs from the fitted one.

    Args:
        table: The table whose category vocabularies are validated.
        categories: The fitted category vocabulary of every column.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table.categorical_columns
    categorical = table.categorical
    if categorical.code.size(-1) != len(categories):
        raise ValueError(
            f"Expected {len(categories)} fitted categorical "
            f"columns (got {len(columns)})."
        )
    for index, (actual, expected) in enumerate(
        zip(_iter_categories(categorical), categories)
    ):
        expected = expected.to(device=actual.device)
        message = (
            "Expected the category vocabulary for categorical column "
            f"{columns[index]!r} to match the fitted values and order. "
            "Use 'AlignCategories' before this processor for "
            "independently tensorized inputs."
        )
        if (
            torch.compiler.is_compiling()
            and type(actual) in _CATEGORY_TYPES
            and type(expected) in _CATEGORY_TYPES
        ):
            actual_tensors, actual_layout = _pack_category(actual)
            expected_tensors, expected_layout = _pack_category(expected)
            # The message overload keeps this runtime check in the graph.
            torch._assert_async(
                _check_category_values(
                    actual_tensors,
                    actual_layout,
                    expected_tensors,
                    expected_layout,
                    message,
                ),
                message,
            )
        elif not actual.equal(expected):
            raise ValueError(message)


def _pack_category(category: Tensor) -> tuple[list[Tensor | None], list[int]]:
    """Pack vocabulary data for the operator without tensor wrappers."""
    kind = _CATEGORY_TYPES.index(type(category))
    if type(category) is Tensor:
        return [category], [kind]
    if type(category) is NullableTensor:
        category = cast(NullableTensor, category)
        return [category._data, category._valid], [kind]
    category = cast(VarLenTensor, category)
    return [category._data, category._offset, category._valid], [
        kind,
        category.numel(),
        category.stride(0),
        category._logical_storage_offset,
    ]


def _unpack_category(
    tensors: list[Tensor | None], layout: list[int]
) -> Tensor:
    cls = _CATEGORY_TYPES[layout[0]]
    if cls is Tensor:
        return cast(Tensor, tensors[0])
    if cls is NullableTensor:
        return NullableTensor(
            cast(Tensor, tensors[0]), cast(Tensor, tensors[1])
        )
    return cls(
        data=cast(Tensor, tensors[0]),
        offset=cast(Tensor, tensors[1]),
        valid=tensors[2],
        size=(layout[1],),
        stride=(layout[2],),
        storage_offset=layout[3],
    )


@torch.library.custom_op("sdm::check_category_values", mutates_args=())
def _check_category_values(
    actual: list[Tensor | None],
    actual_layout: list[int],
    expected: list[Tensor | None],
    expected_layout: list[int],
    message: str,
) -> Tensor:
    actual_category = _unpack_category(actual, actual_layout)
    expected_category = _unpack_category(expected, expected_layout)
    if isinstance(actual_category, VarLenTensor):
        actual_category = _contiguous(actual_category)
    if isinstance(expected_category, VarLenTensor):
        expected_category = _contiguous(expected_category)
    if not actual_category.equal(expected_category):
        raise ValueError(message)
    return torch.ones((), dtype=torch.bool, device="cpu")


@_check_category_values.register_fake
def _check_category_values_fake(
    actual: list[Tensor | None],
    actual_layout: list[int],
    expected: list[Tensor | None],
    expected_layout: list[int],
    message: str,
) -> Tensor:
    return torch.empty((), dtype=torch.bool, device="cpu")
