# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Sequence
from typing import Protocol, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, EnsembleTable, TableTensor


class CategoryStorage(Protocol):
    aliases: Tensor

    def member(self, index: int) -> tuple[Tensor, ...]: ...


def source_locations(source: EnsembleTable) -> Tensor:
    locations = source._route_locations
    if locations is None:
        locations = torch.tensor(
            source._locations, dtype=torch.long, device=source[0].device
        )
    return locations


def grouped_locations(source: Tensor, same_group: Tensor) -> Tensor:
    """Preserve source-group order and first referenced position order."""
    n = source.size(0)
    arange = torch.arange(n, device=source.device)
    first = torch.where(same_group, arange, n).amin(-1)
    is_first = first == arange
    keys = source[:, 0] * (n + 1) + first
    groups = ((keys[:, None] > keys[None, :]) & is_first[None, :]).sum(-1)
    same_position = same_group & (source[:, 1, None] == source[None, :, 1])
    first_position = torch.where(same_position, arange, n).amin(-1)
    positions = (
        same_group
        & (first_position == arange)[None, :]
        & (arange[None, :] <= first_position[:, None])
    ).sum(-1) - 1
    return torch.stack((groups, positions), dim=-1)


def narrow(tensor: Tensor, start: Tensor, count: Tensor) -> Tensor:
    return narrow_length(tensor, start, cast(int, count.item()))


def narrow_length(tensor: Tensor, start: Tensor, length: int) -> Tensor:
    offset = cast(int, start.item())
    torch._check(offset >= 0)
    torch._check(length >= 0)
    torch._check(offset + length <= tensor.size(0))
    # These banks are freshly concatenated, contiguous 1D tensors. Explicit
    # view sizes avoid slice-clamping expressions for unbacked lengths.
    return torch.as_strided(tensor, (length,), (1,), offset)


class CategoricalRoutes:
    """Store code banks and logical routes without retaining member views."""

    def __init__(
        self,
        source: EnsembleTable,
        codes: Sequence[Tensor],
        locations: Tensor,
        categories: CategoryStorage,
        static_locations: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        self.source = source
        self.locations = locations
        self.categories = categories
        self.static_locations = static_locations
        self.materialized: EnsembleTable | None = None
        buckets: dict[tuple[object, ...], list[int]] = {}
        for i, code in enumerate(codes):
            table = source[i]
            key = (table.column_groups, code.shape, code.dtype, code.device)
            buckets.setdefault(key, []).append(i)
        self.banks = []
        self.bucket_ids = [-1] * len(source)
        self.rows = locations.new_empty(len(source))
        for bucket, indices in enumerate(buckets.values()):
            for i in indices:
                self.bucket_ids[i] = bucket
            loc = locations[indices]
            arange = torch.arange(len(indices), device=loc.device)
            same = (loc[:, None, :] == loc[None, :, :]).all(-1)
            first = torch.where(same, arange, len(indices)).amin(-1)
            is_first = first == arange
            keys = loc[:, 0] * (len(source) + 1) + loc[:, 1]
            order = torch.where(is_first, keys, keys.amax() + 1).argsort(
                stable=True
            )
            self.banks.append(torch.stack([codes[i] for i in indices])[order])
            self.rows[indices] = (
                (keys[:, None] > keys[None, :]) & is_first[None, :]
            ).sum(-1)

    def __len__(self) -> int:
        return len(self.source)

    def schema_tables(self) -> Iterator[TableTensor]:
        # Categorical alignment and permutation preserve column layouts.
        return self.source._iter_schema_tables()

    def categorical_code(self, index: int) -> Tensor:
        bank = self.banks[self.bucket_ids[index]]
        row = cast(int, self.rows[index].item())
        torch._check(row >= 0)
        torch._check(row < bank.size(0))
        return bank.narrow(0, row, 1).squeeze(0)

    def member(self, index: int) -> TableTensor:
        return self.source[index].replace_blocks(
            categorical=CategoricalTensor(
                self.categorical_code(index),
                self.categories.member(index),
            )
        )

    def materialize(self) -> EnsembleTable:
        if self.materialized is not None:
            return self.materialized
        if self.static_locations is not None:
            locations = self.static_locations
            groups = tuple(
                self.member(
                    next(i for i, (g, _) in enumerate(locations) if g == group)
                ).unsqueeze(0)
                for group in range(max(g for g, _ in locations) + 1)
            )
            output = EnsembleTable(
                groups=cast(tuple[TableTensor, ...], groups),
                locations=locations,
            )
            # Keep views fresh while capturing permissive resumed frames.
            if not torch.compiler.is_compiling():
                self.materialized = output
            return output
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "Logical group iteration needs a tensor-aware consumer"
            )
        locations = tuple(tuple(row) for row in self.locations.tolist())
        source_locations_ = self.source._locations
        aliases = self.categories.aliases.tolist()
        category_cache: dict[tuple[int, int], Tensor] = {}
        groups = []
        for group_id in range(max(row[0] for row in locations) + 1):
            members = [
                i for i, (g, _) in enumerate(locations) if g == group_id
            ]
            first = members[0]
            source_group = source_locations_[first][0]
            by_position = {
                locations[i][1]: source_locations_[i][1] for i in members
            }
            positions = tuple(by_position[i] for i in range(len(by_position)))
            source = self.source._groups[source_group]
            start = positions[0]
            step = positions[1] - start if len(positions) > 1 else 1
            stop = start + step * len(positions)
            if step > 0 and positions == tuple(range(start, stop, step)):
                group = source[start:stop:step]
            else:
                group = cast(
                    TableTensor,
                    torch.stack([source[pos] for pos in positions]),
                )
            categories = []
            for column, category in enumerate(self.categories.member(first)):
                key = (column, aliases[first][column])
                categories.append(category_cache.setdefault(key, category))
            bank = self.banks[self.bucket_ids[first]]
            row = int(self.rows[first])
            groups.append(
                group.replace_blocks(
                    categorical=CategoricalTensor(
                        bank.narrow(0, row, len(positions)), categories
                    )
                )
            )
        self.materialized = EnsembleTable(groups=groups, locations=locations)
        return self.materialized
