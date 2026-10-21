# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from sdm import ColumnarTensor, StringTensor, TableTensor
from sdm.relational.join import join_index
from sdm.tensor.var_len import _cat, _contiguous


def _flatten_strings(
    categories: tuple[Tensor, ...],
) -> tuple[list[Tensor | None], list[int]]:
    tensors: list[Tensor | None] = []
    layout: list[int] = []
    for category in categories:
        category = cast(StringTensor, category)
        tensors.extend((category._data, category._offset, category._valid))
        layout.extend(
            (
                category.numel(),
                category.stride(0),
                category._logical_storage_offset,
            )
        )
    return tensors, layout


def _unflatten_strings(
    tensors: list[Tensor | None], layout: list[int]
) -> tuple[StringTensor, ...]:
    return tuple(
        StringTensor(
            data=cast(Tensor, tensors[3 * i]),
            offset=cast(Tensor, tensors[3 * i + 1]),
            valid=tensors[3 * i + 2],
            size=(layout[3 * i],),
            stride=(layout[3 * i + 1],),
            storage_offset=layout[3 * i + 2],
        )
        for i in range(len(layout) // 3)
    )


def _concatenate_strings(categories: tuple[StringTensor, ...]) -> StringTensor:
    # Avoid re-entering subclass dispatch from the opaque operator kernel.
    contiguous = tuple(
        category if category.is_contiguous() else _contiguous(category)
        for category in categories
    )
    return cast(StringTensor, _cat(contiguous))


@torch.library.custom_op("sdm::string_category_lookup", mutates_args=())
def _string_category_lookup(
    inputs: list[Tensor | None],
    input_layout: list[int],
    fitted: list[Tensor | None],
    fitted_layout: list[int],
    codes: Tensor,
) -> Tensor:
    """Keep the backend join's dynamic row pairs inside a fixed-size lookup."""
    input_categories = _unflatten_strings(inputs, input_layout)
    fitted_categories = _unflatten_strings(fitted, fitted_layout)
    # Pair IDs for each input category: (total_input_categories,).
    left_pair = torch.cat(
        [
            codes.new_full((categories.numel(),), pair_index)
            for pair_index, categories in enumerate(input_categories)
        ]
    )
    # Pair IDs for each fitted category: (total_fitted_categories,).
    right_pair = torch.cat(
        [
            codes.new_full((categories.numel(),), pair_index)
            for pair_index, categories in enumerate(fitted_categories)
        ]
    )
    left_index, right_index = join_index(
        left_table=TableTensor(
            columns={"id": ("pair", "value")},
            id=ColumnarTensor(
                (left_pair, _concatenate_strings(input_categories)),
            ),
        ),
        right_table=TableTensor(
            columns={"id": ("pair", "value")},
            id=ColumnarTensor(
                (right_pair, _concatenate_strings(fitted_categories)),
            ),
        ),
        left_keys=["pair", "value"],
        right_keys=["pair", "value"],
        dtype=codes.dtype,
    )
    fitted_codes = torch.cat(
        [
            torch.arange(
                categories.numel(),
                dtype=codes.dtype,
                device=codes.device,
            )
            for categories in fitted_categories
        ]
    )
    # Fitted code for each input category: (total_input_categories,).
    lookup = codes.new_full(
        (sum(categories.numel() for categories in input_categories),),
        -1,
    )
    lookup[left_index] = fitted_codes[right_index]
    return lookup


@_string_category_lookup.register_fake
def _string_category_lookup_fake(
    inputs: list[Tensor | None],
    input_layout: list[int],
    fitted: list[Tensor | None],
    fitted_layout: list[int],
    codes: Tensor,
) -> Tensor:
    return codes.new_empty((sum(input_layout[::3]),))
