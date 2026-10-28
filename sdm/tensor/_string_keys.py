# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact fixed-width keys for tensor-native UTF-8 operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sdm.tensor.string import StringTensor


def pack(
    values: StringTensor, width: int | None = None
) -> tuple[Tensor, Tensor]:
    """Convert strings into integer keys that compare like the strings.

    Each string becomes a row of ``int64`` words, so ordinary tensor
    operations can compare strings: two strings are equal exactly when their
    rows are equal, and comparing rows word by word gives UTF-8 byte order.

    Bytes are stored as ``byte + 1`` and padding as zero, so a prefix sorts
    before its extensions even when the next byte is NUL. These 257 values
    need nine bits, so each word holds seven bytes in its 63 non-negative
    bits, with the first byte highest. Null strings get all-zero keys; the
    returned mask tells them apart from empty strings.

    Args:
        values: One-dimensional strings with a known length bound.
        width: Bytes to encode per string. Defaults to the length bound and
            must not be smaller, so strings are never truncated.

    Returns:
        Keys of shape ``[N, W]`` with ``W = max(1, ceil(width / 7))``
        words, and a validity mask of shape ``[N]``.
    """
    assert values.ndim == 1
    bound = values._max_length
    if bound is None:
        raise ValueError("String keys require a known byte bound")
    if width is None:
        width = bound
    if width < bound:
        raise ValueError(
            "String key width is smaller than the stored byte bound"
        )

    # Each string's start and length in the byte buffer.
    rows = torch.arange(values.numel(), device=values.device)
    rows = rows * values.stride(0) + values._logical_storage_offset
    start = values._offset[rows].long()
    length = values._offset[rows + 1].long() - start
    valid = (
        torch.ones_like(rows, dtype=torch.bool)
        if values._valid is None
        else values._valid[rows]
    )
    # Read each string into a [rows, padded_width] grid of byte + 1 symbols,
    # with zero where a string has no byte; absent slots read a dummy byte.
    padded_width = max(7, ((width + 6) // 7) * 7)
    positions = torch.arange(padded_width, device=values.device)
    present = (positions < length[:, None]) & valid[:, None]
    data = torch.cat((values._data, values._data.new_zeros(1)))
    index = torch.where(
        present, start[:, None] + positions, values._data.numel()
    )
    symbols = torch.where(present, data[index].long() + 1, 0)
    # Nine bits per symbol, first byte highest: 7 * 9 = 63 bits per word.
    shifts = torch.arange(6, -1, -1, device=values.device) * 9
    # [rows, words, seven symbols] -> [rows, words].
    words = (
        symbols.reshape(values.numel(), padded_width // 7, 7) << shifts
    ).sum(-1)
    return words, valid
