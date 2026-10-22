# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

from sdm import StringTensor, VarLenTensor
from sdm.tensor.string import _sort_indices
from sdm.tensor.var_len import _permute_banks

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class ValueColumn:
    """Fresh sorted storage with one shared full-retention slot."""

    def __init__(self, categories: Tensor, codes: Tensor, min_frequency: int):
        self.original = categories
        self.known_empty = statically_known_true(categories.numel() == 0)
        b, _ = codes.shape
        k = categories.numel()
        if self.known_empty:
            self.rows = codes.new_zeros(b, dtype=torch.long)
            self.counts = self.rows
            self.aligned = torch.full_like(codes, -1)
            return
        # An unknown zero input would require choosing an input alias.
        # Keep that unsupported case explicit; retained counts may be zero.
        torch._check(k > 0)
        if type(categories) is StringTensor:
            permutation = _sort_indices(categories)
            ordered = categories
        elif categories.dtype in _UNSIGNED_DTYPES and categories.is_cuda:
            key = categories.to(torch.int64)
            if categories.dtype == torch.uint64:
                key = key.bitwise_xor(torch.iinfo(torch.int64).min)
            permutation = key.argsort()
            ordered = categories[permutation]
        else:
            ordered, permutation = categories.sort()
        mask = codes >= 0
        indices = codes.clamp_min(0).long()
        frequencies = codes.new_zeros((b, k))
        frequencies.scatter_add_(1, indices, mask.to(codes.dtype))
        order = permutation.expand(b, -1)
        observed = frequencies.gather(1, order) >= min_frequency
        logical = torch.arange(k, device=codes.device).expand(b, -1)
        selected_first = torch.where(observed, logical, k).argsort(
            dim=1, stable=True
        )
        self.rows = torch.where(
            observed.all(dim=1), 0, torch.arange(b, device=codes.device) + 1
        )
        self.counts = observed.sum(dim=1)
        self.kind = type(ordered)
        if isinstance(ordered, VarLenTensor):
            orders = torch.cat(
                (
                    torch.arange(k, device=codes.device).view(1, -1),
                    selected_first,
                )
            )
            # Compose selection with the sort permutation before packing, so
            # payload bounds still come from the original backing storage.
            self.bank = _permute_banks(
                cast(VarLenTensor, categories), permutation[orders]
            )
        else:
            self.bank = torch.cat(
                (ordered.view(1, -1), ordered[selected_first])
            )
        rank = observed.cumsum(dim=1, dtype=codes.dtype) - 1
        lookup = torch.full_like(frequencies, -1)
        lookup.scatter_(1, order, torch.where(observed, rank, -1))
        self.aligned = torch.where(mask, lookup.gather(1, indices), -1)

    def member(self, index: int) -> Tensor:
        if self.known_empty:
            return self.original
        row = cast(int, self.rows[index].item())
        count = cast(int, self.counts[index].item())
        torch._check(row >= 0)
        torch._check(row <= self.rows.size(0))
        torch._check(count >= 0)
        torch._check(count <= self.original.numel())
        if isinstance(self.bank, tuple):
            data, offset, valid = self.bank
            offsets = offset.narrow(0, row, 1).squeeze(0)
            byte_count = cast(
                int,
                offsets.gather(0, self.counts[index : index + 1])[0].item(),
            )
            torch._check(byte_count >= 0)
            torch._check(byte_count <= data.size(1))
            torch._check(byte_count <= torch.iinfo(offset.dtype).max)
            return cast(type[VarLenTensor], self.kind)(
                data=torch.as_strided(
                    data, (byte_count,), (1,), row * data.size(1)
                ),
                offset=torch.as_strided(
                    offset, (count + 1,), (1,), row * offset.size(1)
                ),
                valid=torch.as_strided(
                    valid, (count,), (1,), row * valid.size(1)
                )
                if valid is not None
                else None,
                size=(count,),
            )
        return self.bank.narrow(0, row, 1).squeeze(0).narrow(0, 0, count)


class ValueCategories:
    def __init__(
        self,
        groups: Sequence[Sequence[ValueColumn]],
        members: Sequence[tuple[int, int]],
    ) -> None:
        self.groups = tuple(tuple(columns) for columns in groups)
        self.members = tuple(members)
        offsets = []
        offset = 0
        for columns in groups:
            offsets.append(offset)
            offset += columns[0].rows.size(0) + 1
        self.aliases = torch.stack(
            [
                torch.stack(
                    [
                        column.rows[position] + offsets[group]
                        for column in groups[group]
                    ]
                )
                for group, position in members
            ]
        )

    def member(self, index: int) -> tuple[Tensor, ...]:
        group, position = self.members[index]
        return self.physical_member(group, position)

    def physical_member(self, group: int, position: int) -> tuple[Tensor, ...]:
        return tuple(column.member(position) for column in self.groups[group])
