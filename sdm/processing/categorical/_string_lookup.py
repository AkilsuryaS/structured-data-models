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
    """Pack vocabulary tensors and their (length, stride, offset) metadata."""
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


def _vocabulary_table(
    tensors: list[Tensor | None], layout: list[int], codes: Tensor
) -> TableTensor:
    categories = tuple(
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
    # Custom operations bypass tensor-subclass dispatch; call the kernels.
    contiguous = tuple(
        category if category.is_contiguous() else _contiguous(category)
        for category in categories
    )
    # Pair IDs keep join matches within the corresponding vocabulary.
    pairs = torch.cat(
        [
            codes.new_full((category.numel(),), pair_index)
            for pair_index, category in enumerate(categories)
        ]
    )
    return TableTensor(
        columns={"id": ("pair", "value")},
        id=ColumnarTensor((pairs, _cat(contiguous))),
    )


@torch.library.custom_op("sdm::string_category_lookup", mutates_args=())
def _string_category_lookup(
    inputs: list[Tensor | None],
    input_layout: list[int],
    fitted: list[Tensor | None],
    fitted_layout: list[int],
    codes: Tensor,
) -> Tensor:
    """Keep the backend join's dynamic row pairs inside a fixed-size lookup."""
    left_index, right_index = join_index(
        left_table=_vocabulary_table(inputs, input_layout, codes),
        right_table=_vocabulary_table(fitted, fitted_layout, codes),
        left_keys=["pair", "value"],
        right_keys=["pair", "value"],
        dtype=codes.dtype,
    )
    fitted_codes = torch.cat(
        [
            torch.arange(size, dtype=codes.dtype, device=codes.device)
            for size in fitted_layout[::3]
        ]
    )
    lookup = codes.new_full((sum(input_layout[::3]),), -1)
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
