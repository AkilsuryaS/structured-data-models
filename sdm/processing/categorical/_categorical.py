# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch import Tensor
from torch._higher_order_ops.effects import (
    _EffectType,
    _register_effectful_op,
)

from sdm import StringTensor, Stype, TableTensor, VarLenTensor
from sdm.nn._buffer import BufferList
from sdm.tensor.var_len import _compact


@torch.library.custom_op("sdm::_check_category_condition", mutates_args=())
def _check_condition(condition: Tensor, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _category_parts(
    category: Tensor,
) -> tuple[Tensor, Tensor | None, list[int]]:
    if isinstance(category, VarLenTensor):
        view = [
            int(isinstance(category, StringTensor)),
            int(category.numel()),
            int(category.stride(0)),
            category._storage_offset,
        ]
        return category._data, category._offset, view
    return category, None, []


def _category_values(
    data: Tensor,
    offset: Tensor | None,
    view: list[int],
) -> tuple[Tensor, Tensor | None]:
    if offset is None:
        return data, None
    _, size, stride, storage_offset = view
    if stride == 1:
        offset = offset[storage_offset : storage_offset + size + 1]
        return data[int(offset[0]) : int(offset[-1])], offset - offset[0]
    start = torch.as_strided(
        offset,
        size=(size,),
        stride=(stride,),
        storage_offset=storage_offset,
    )
    end = torch.as_strided(
        offset,
        size=(size,),
        stride=(stride,),
        storage_offset=storage_offset + 1,
    )
    offset, index = _compact(start, end)
    return data[index], offset


@torch.library.custom_op("sdm::_check_category_vocabulary", mutates_args=())
def _check_vocabulary(
    actual_data: Tensor,
    actual_offset: Tensor | None,
    actual_view: list[int],
    expected_data: Tensor,
    expected_offset: Tensor | None,
    expected_view: list[int],
    message: str,
) -> None:
    if actual_view[:2] != expected_view[:2]:
        raise ValueError(message)
    actual, actual_offset = _category_values(
        data=actual_data,
        offset=actual_offset,
        view=actual_view,
    )
    expected, expected_offset = _category_values(
        data=expected_data,
        offset=expected_offset,
        view=expected_view,
    )
    if actual_offset is not None:
        assert expected_offset is not None
        if not actual_offset.equal(expected_offset):
            raise ValueError(message)
    if not actual.equal(expected):
        raise ValueError(message)


def _fake_check(*args: object, **kwargs: object) -> None:
    return None


_check_condition.register_fake(_fake_check)
_check_vocabulary.register_fake(_fake_check)
_register_effectful_op(
    torch.ops.sdm._check_category_condition.default,
    _EffectType.ORDERED,
)
_register_effectful_op(
    torch.ops.sdm._check_category_vocabulary.default,
    _EffectType.ORDERED,
)
# Older PyTorch versions otherwise remove validation with unused tokens.
torch.fx.node.has_side_effect(torch.ops.prims._sink_tokens.default)


def _check_categorical_codes(table: TableTensor) -> None:
    """Raise if a categorical code exceeds its column's category vocabulary.

    Negative codes encode missing values and are allowed.

    Args:
        table: The table whose categorical codes are validated.
    """
    if os.getenv("SDM_ENABLE_CATEGORY_CHECKS", "1") == "0":
        return
    columns = table._columns[Stype.categorical]
    categorical = table.categorical
    for index in range(categorical.size(-1)):
        category = getattr(categorical, f"_category_{index}")
        codes = categorical.code[..., index]
        valid = (codes < category.numel()).all()
        message = (
            f"Categorical column {columns[index]!r} contains a code "
            "outside its category vocabulary."
        )
        _check_condition(valid, message)


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
    columns = table._columns[Stype.categorical]
    if table.categorical.size(-1) != len(categories):
        raise ValueError(
            f"Expected {len(categories)} fitted categorical "
            f"columns (got {len(columns)})."
        )
    for index in range(len(categories)):
        actual = getattr(table.categorical, f"_category_{index}")
        expected = categories[index]
        expected = expected.to(device=actual.device)
        message = (
            "Expected the category vocabulary for categorical column "
            f"{columns[index]!r} to match the fitted values and order. "
            "Use 'AlignCategories' before this processor for "
            "independently tensorized inputs."
        )
        if torch.compiler.is_compiling():
            actual_data, actual_offset, actual_view = _category_parts(actual)
            expected_data, expected_offset, expected_view = _category_parts(
                expected,
            )
            _check_vocabulary(
                actual_data=actual_data,
                actual_offset=actual_offset,
                actual_view=actual_view,
                expected_data=expected_data,
                expected_offset=expected_offset,
                expected_view=expected_view,
                message=message,
            )
        elif not actual.equal(expected):
            raise ValueError(message)
