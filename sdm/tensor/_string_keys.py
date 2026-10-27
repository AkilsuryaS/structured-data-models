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
    """Pack one-dimensional strings into ordered keys and a validity mask.

    Seven byte-plus-one symbols occupy each nonnegative 63-bit word. Zero
    padding sorts prefixes before extensions, including embedded NUL bytes.
    Null keys are zero; their separate validity mask distinguishes them from
    empty strings. The width is a byte bound, never a truncation length: it
    must cover the conservative per-element bound established at ingestion.
    Unknown ingestion bounds remain None in storage metadata; packing uses
    the total byte count as a safe fallback, never a string-length cutoff.
    """
    assert values.ndim == 1
    bound = values._max_length
    if bound is None:
        bound = values._data.numel()
    if width is None:
        width = bound
    if width < bound:
        raise ValueError(
            "String key width is smaller than the stored byte bound"
        )

    rows = torch.arange(values.numel(), device=values.device)
    rows = rows * values.stride(0) + values._logical_storage_offset
    start = values._offset[rows].long()
    length = values._offset[rows + 1].long() - start
    valid = (
        torch.ones_like(rows, dtype=torch.bool)
        if values._valid is None
        else values._valid[rows]
    )
    padded_width = max(7, ((width + 6) // 7) * 7)
    positions = torch.arange(padded_width, device=values.device)
    present = (positions < length[:, None]) & valid[:, None]
    data = torch.cat((values._data, values._data.new_zeros(1)))
    index = torch.where(
        present, start[:, None] + positions, values._data.numel()
    )
    symbols = torch.where(present, data[index].long() + 1, 0)
    shifts = torch.arange(6, -1, -1, device=values.device) * 9
    # [rows, words, seven symbols] -> [rows, words].
    words = (
        symbols.reshape(values.numel(), padded_width // 7, 7) << shifts
    ).sum(-1)
    return words, valid
