# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from sdm import EnsembleTable, VarLenTensor
from sdm.nn._buffer import BufferList
from sdm.processing.categorical._routing import narrow, narrow_length
from sdm.tensor.categorical import _iter_categories
from sdm.tensor.var_len import _select_unique


def _lengths(values: Sequence[Tensor]) -> Tensor:
    return torch.stack(
        [
            torch.scalar_tensor(
                value.numel(), dtype=torch.long, device=value.device
            )
            for value in values
        ]
    )


class RaggedColumn:
    def __init__(self, values: Sequence[Tensor], aliases: Tensor):
        self.kind = type(values[0])
        arange = torch.arange(len(values), device=aliases.device)
        same = aliases[:, None] == aliases[None, :]
        representative = torch.where(same, arange, len(values)).amin(-1)
        self.lengths = _lengths(values)[representative]
        if isinstance(values[0], VarLenTensor):
            strings = cast(Sequence[VarLenTensor], values)
            self.data = torch.cat([value._data for value in strings])
            self.offset = torch.cat([value._offset for value in strings])
            self.valid = (
                torch.cat([cast(Tensor, value._valid) for value in strings])
                if strings[0]._valid is not None
                else None
            )
            data_lengths = _lengths([value._data for value in strings])
            offset_lengths = _lengths([value._offset for value in strings])
            self.data_lengths = data_lengths[representative]
            self.data_starts = (data_lengths.cumsum(0) - data_lengths)[
                representative
            ]
            self.offset_starts = (offset_lengths.cumsum(0) - offset_lengths)[
                representative
            ]
        else:
            self.data = torch.cat(tuple(values))
        lengths = _lengths(values)
        self.starts = (lengths.cumsum(0) - lengths)[representative]

    def member(self, index: int) -> Tensor:
        length = self.lengths[index]
        if issubclass(self.kind, VarLenTensor):
            n = cast(int, length.item())
            torch._check(n >= 0)
            data = narrow(
                self.data, self.data_starts[index], self.data_lengths[index]
            )
            torch._check(data.numel() <= torch.iinfo(self.offset.dtype).max)
            return self.kind(
                data=data,
                offset=narrow_length(
                    self.offset, self.offset_starts[index], n + 1
                ),
                valid=narrow_length(self.valid, self.starts[index], n)
                if self.valid is not None
                else None,
                size=(n,),
            )
        return narrow(self.data, self.starts[index], length)


class PermutedCategories:
    def __init__(
        self,
        source: EnsembleTable,
        permutations: BufferList[BufferList[Tensor]],
        groups: Tensor,
    ) -> None:
        values: dict[tuple[object, ...], list[Tensor]] = {}
        members: dict[tuple[object, ...], list[int]] = {}
        self.columns: list[list[tuple[int, int]]] = [
            [] for _ in range(len(source))
        ]
        keys = []
        for i in range(len(source)):
            for column, (category, permutation) in enumerate(
                zip(
                    _iter_categories(source[i].categorical),
                    permutations[i],
                    strict=True,
                )
            ):
                order = permutation.argsort()
                selected = (
                    _select_unique(category, order)
                    if isinstance(category, VarLenTensor)
                    else category[order]
                )
                key = (
                    column,
                    type(category),
                    category.dtype,
                    category.device,
                    category._offset.dtype
                    if isinstance(category, VarLenTensor)
                    else None,
                    category._valid is None
                    if isinstance(category, VarLenTensor)
                    else True,
                )
                if key not in values:
                    keys.append(key)
                    values[key] = []
                    members[key] = []
                bucket = keys.index(key)
                self.columns[i].append((bucket, len(values[key])))
                values[key].append(selected)
                members[key].append(i)
        self.records = [
            RaggedColumn(values[key], groups[members[key]]) for key in keys
        ]
        width = max(len(columns) for columns in self.columns)
        self.aliases = groups[:, None].expand(-1, width)

    def member(self, index: int) -> tuple[Tensor, ...]:
        return tuple(
            self.records[bucket].member(row)
            for bucket, row in self.columns[index]
        )
