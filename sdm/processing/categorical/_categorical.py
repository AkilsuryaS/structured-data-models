# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.nn._buffer import BufferList
from sdm.tensor.categorical import _iter_categories


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
    codes = categorical.code
    if codes.numel() == 0:
        return
    sizes = codes.new_tensor(
        [category.numel() for category in _iter_categories(categorical)],
        dtype=torch.int64,
    )
    maxima = codes.amax(dim=tuple(range(codes.dim() - 1)))
    invalid = maxima >= sizes
    # Synchronize once; identify the first invalid column only on failure.
    if invalid.any():
        index = int(invalid.long().argmax())
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
        if not actual.equal(expected):
            raise ValueError(
                "Expected the category vocabulary for categorical column "
                f"{columns[index]!r} to match the fitted values and order. "
                "Use 'AlignCategories' before this processor for "
                "independently tensorized inputs."
            )
