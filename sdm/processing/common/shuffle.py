# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype, StypeLike, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class ShuffleColumns(EnsembleProcessor, EnsembleInvertibleMixin):
    """Permute numerical feature columns and their names.

    Convert non-numerical feature stypes before this step, for example with
    :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"random"`` draws independent
            permutations, and ``"latin"`` draws coupled Latin
            permutations.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        method: Literal["random", "latin"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[Tensor] = BufferList()
        self._host_permutations: tuple[tuple[int, ...], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, ...], ...]:
        r""":meta private:"""  # noqa: D415
        return self._host_permutations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._host_permutations = cast(tuple[tuple[int, ...], ...], state)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = next(ensemble_table._iter_groups()).device
        widths = [
            ensemble_table[member_id].numerical.size(-1)
            for member_id in range(len(ensemble_table))
        ]
        permutations = []
        if self.method == "latin":
            max_features = max(widths)
            base_rank = torch.randperm(
                max_features,
                generator=generator,
                device=device,
            )
            row_rank = torch.randperm(
                max_features,
                generator=generator,
                device=device,
            )
            for member_id, n_features in enumerate(widths):
                if n_features <= 1:
                    permutation = torch.arange(n_features, device=device)
                else:
                    base = base_rank
                    rows = row_rank
                    if n_features < max_features:
                        base = base[base < n_features]
                        rows = rows[rows < n_features]
                    pattern = member_id % n_features
                    permutation = base[(pattern - rows) % n_features]
                permutations.append(permutation)
        else:
            assert self.method == "random"
            for n_features in widths:
                if n_features <= 1:
                    permutation = torch.arange(n_features, device=device)
                else:
                    permutation = torch.randperm(
                        n_features,
                        generator=generator,
                        device=device,
                    )
                permutations.append(permutation)

        self._permutations = BufferList(permutations)
        # TODO: Add to EnsembleTable directly.
        member_ids_by_group: list[list[int]] = [
            [] for _ in range(ensemble_table.num_groups)
        ]
        for member_id, (group_id, _) in enumerate(ensemble_table._locations):
            member_ids_by_group[group_id].append(member_id)
        host_permutations = [()] * len(permutations)
        for member_ids in member_ids_by_group:
            host_rows = torch.stack(
                [permutations[member_id] for member_id in member_ids]
            ).tolist()
            for member_id, host_row in zip(member_ids, host_rows, strict=True):
                host_permutations[member_id] = tuple(host_row)
        self._host_permutations = tuple(host_permutations)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutations) != len(ensemble_table):
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._permutations)} ensemble members, but got "
                f"{len(ensemble_table)}."
            )
        if len(ensemble_table) > 1 and all(
            group.active_stypes <= {Stype.numerical}
            for group in ensemble_table._iter_groups()
        ):
            return self._transform_numerical_groups(ensemble_table)
        tables: list[TableTensor] = []
        for member_id in range(len(ensemble_table)):
            table = ensemble_table[member_id]
            host_permutation = self._host_permutations[member_id]
            shuffled = table.__class__(
                columns={
                    Stype.numerical: tuple(
                        table.columns[Stype.numerical][i]
                        for i in host_permutation
                    )
                },
                numerical=table.numerical.index_select(
                    -1, self._permutations[member_id]
                ),
            )
            tables.append(
                cast(
                    TableTensor,
                    torch.cat(
                        (table.drop_stypes(Stype.numerical), shuffled),
                        dim=-1,
                    ),
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _transform_numerical_groups(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable:
        source = ensemble_table._positional_storage
        if source is None:
            source = ensemble_table
        groups = []
        locations = [(-1, -1)] * len(source)
        columns = []
        member_ids_by_group: list[list[int]] = [
            [] for _ in range(source.num_groups)
        ]
        for member_id, (group_id, _) in enumerate(source._locations):
            member_ids_by_group[group_id].append(member_id)
        for member_id in range(len(source)):
            names = ensemble_table[member_id].columns[Stype.numerical]
            columns.append(
                {
                    Stype.numerical: tuple(
                        names[i] for i in self._host_permutations[member_id]
                    )
                }
            )
        for group_id, member_ids in enumerate(member_ids_by_group):
            group = source._groups[group_id]
            positions = tuple(source._locations[i][1] for i in member_ids)
            permutations = torch.stack(
                [self._permutations[i] for i in member_ids]
            )
            groups.append(
                group.__class__(
                    columns=cast(
                        Mapping[StypeLike, Sequence[str]],
                        columns[member_ids[0]],
                    ),
                    numerical=_gather_columns(
                        group.numerical, positions, permutations
                    ),
                )
            )
            for position, member_id in enumerate(member_ids):
                locations[member_id] = (group_id, position)
        storage = EnsembleTable(groups=groups, locations=locations)
        return EnsembleTable._from_positional_storage(storage, columns)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutations) != len(ensemble_table):
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._permutations)} ensemble members, but got "
                f"{len(ensemble_table)}."
            )
        tables: list[TableTensor] = []
        for member_id in range(len(ensemble_table)):
            table = ensemble_table[member_id]
            host_permutation = self._host_permutations[member_id]
            inverse_host_permutation = [0] * len(host_permutation)
            for destination, source in enumerate(host_permutation):
                inverse_host_permutation[source] = destination
            shuffled = table.__class__(
                columns={
                    Stype.numerical: tuple(
                        table.columns[Stype.numerical][i]
                        for i in inverse_host_permutation
                    )
                },
                numerical=table.numerical.index_select(
                    -1, self._permutations[member_id].argsort()
                ),
            )
            tables.append(
                cast(
                    TableTensor,
                    torch.cat(
                        (table.drop_stypes(Stype.numerical), shuffled),
                        dim=-1,
                    ),
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )


def _gather_columns(
    values: Tensor, positions: tuple[int, ...], permutations: Tensor
) -> Tensor:
    """Shuffle members without materializing cyclically repeated inputs."""
    member_shape = values.shape[1:]
    repeats = len(positions) // values.size(0)
    if positions == tuple(range(values.size(0))) * repeats:
        # Keep repetition as a broadcast dimension until after the gather.
        values = values.unsqueeze(0).expand(repeats, *values.shape)
        permutations = permutations.view(
            repeats, values.size(1), permutations.size(-1)
        )
    else:
        index = torch.tensor(positions, device=values.device)
        values = values.index_select(0, index)
    index = permutations.view(
        *permutations.shape[:-1],
        *(1,) * (len(member_shape) - 1),
        permutations.size(-1),
    ).expand_as(values)
    return values.gather(-1, index).reshape(len(positions), *member_shape)
