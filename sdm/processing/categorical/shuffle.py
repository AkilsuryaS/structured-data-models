# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    EnsembleTable,
    Stype,
    TableTensor,
    VarLenTensor,
)
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor
from sdm.tensor.categorical import _iter_categories
from sdm.tensor.var_len import _permute_banks


class ShuffleCategories(EnsembleProcessor):
    """Permute the integer codes of categorical columns.

    One permutation per categorical column is drawn when the processor is
    fitted. Codes and their corresponding category vectors are permuted
    together so decoded values remain unchanged. Negative codes represent
    missing values and are preserved unchanged. Only categorical columns
    are supported; use :class:`~sdm.processing.StypeDispatch` to apply this
    processor to the categorical block of a mixed feature table.

    Args:
        method: Permutation strategy. ``"random"`` remaps codes
            with an independent permutation. ``"shift"`` uses a cyclic
            offset, cycling through every offset in random order.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = True
    _representatives: Tensor

    def __init__(
        self,
        method: Literal["shift", "random"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[BufferList[Tensor]] = BufferList()
        self.register_buffer(
            "_representatives", torch.empty(0, dtype=torch.long)
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        representatives_key = prefix + "_representatives"
        legacy_key = prefix + "_extra_state"
        if representatives_key not in state_dict and legacy_key in state_dict:
            permutation_ids = state_dict.pop(legacy_key)
            permutation_prefix = prefix + "_permutations."
            previous = {
                key.removeprefix(permutation_prefix): state_dict.pop(key)
                for key in tuple(state_dict)
                if key.startswith(permutation_prefix)
            }
            for member_id, permutation_id in enumerate(permutation_ids):
                group_prefix = f"{permutation_id}."
                for key, value in previous.items():
                    if key.startswith(group_prefix):
                        state_dict[
                            f"{permutation_prefix}{member_id}."
                            + key.removeprefix(group_prefix)
                        ] = value
            state_dict[permutation_prefix + "_extra_state"] = (True,) * len(
                permutation_ids
            )
            device = next(
                (
                    value.device
                    for value in previous.values()
                    if isinstance(value, Tensor)
                ),
                torch.device("cpu"),
            )
            state_dict[representatives_key] = torch.tensor(
                [permutation_ids.index(index) for index in permutation_ids],
                dtype=torch.long,
                device=device,
            )
        if (
            representatives_key in state_dict
            and self._representatives.shape
            != state_dict[representatives_key].shape
        ):
            self.register_buffer(
                "_representatives", state_dict[representatives_key].clone()
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _draw_permutations(
        self,
        table: TableTensor,
        *,
        history: dict[str, list[tuple[int | Tensor, int | Tensor, Tensor]]],
        generator: torch.Generator | None = None,
    ) -> list[Tensor]:
        device = table.categorical.device
        generator_kwargs = (
            {"generator": generator} if generator is not None else {}
        )
        permutations: list[Tensor] = []
        for column_index, category in enumerate(
            _iter_categories(table.categorical)
        ):
            n_classes = category.numel()
            count = (
                torch.scalar_tensor(n_classes, dtype=torch.long, device=device)
                if torch.compiler.is_compiling()
                else n_classes
            )
            if self.method == "random":
                permutation = torch.cond(
                    count > 1,
                    lambda n: torch.randperm(
                        n, device=device, **generator_kwargs
                    ),
                    lambda n: torch.arange(n, device=device),
                    (n_classes,),
                )
            else:
                assert self.method == "shift"
                column_name = table.categorical_columns[column_index]
                # Learned sizes stay in tensor history, not Python keys.
                prior = history.setdefault(column_name, [])
                rank = count * 0
                for previous_count, _, _ in prior:
                    rank = rank + (previous_count == count)
                denominator = (
                    count.clamp_min(1)
                    if isinstance(count, Tensor)
                    else max(count, 1)
                )
                position = rank % denominator
                draw = torch.cond(
                    (count > 1) & (position == 0),
                    lambda n: torch.randperm(
                        n, device=device, **generator_kwargs
                    ),
                    lambda n: torch.arange(n, device=device),
                    (n_classes,),
                )
                prior.append((count, rank, draw))
                # Reuse the draw that began this column's current offset cycle.
                cycle_start = rank - position
                offset = draw.new_zeros(())
                for previous_count, previous_rank, previous_draw in prior:
                    if isinstance(position, Tensor):
                        selected = torch.where(
                            torch.arange(previous_draw.numel(), device=device)
                            == position,
                            previous_draw,
                            0,
                        ).sum()
                        offset = offset + torch.where(
                            cast(
                                Tensor,
                                (previous_count == count)
                                & (previous_rank == cycle_start),
                            ),
                            selected,
                            0,
                        )
                    elif (
                        previous_count == count
                        and previous_rank == cycle_start
                    ):
                        offset = previous_draw[position : position + 1]
                permutation = (
                    torch.arange(n_classes, device=device) - offset
                ) % denominator
            permutations.append(permutation)
        return permutations

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        per_member = []
        history: dict[
            str, list[tuple[int | Tensor, int | Tensor, Tensor]]
        ] = {}
        for member_id in range(len(ensemble_table)):
            per_member.append(
                self._draw_permutations(
                    ensemble_table[member_id],
                    history=history,
                    generator=generator,
                )
            )

        # Keep module structure independent of the sampled permutation values.
        self._permutations = BufferList(
            BufferList(permutations) for permutations in per_member
        )
        device = ensemble_table[0].device
        representatives = torch.arange(len(ensemble_table), device=device)
        for member_id, permutations in enumerate(per_member):
            for previous in reversed(range(member_id)):
                other = per_member[previous]
                if len(permutations) != len(other):
                    continue
                equal = torch.ones((), dtype=torch.bool, device=device)
                for permutation, prior in zip(
                    permutations, other, strict=True
                ):
                    length = torch.sym_min(permutation.numel(), prior.numel())
                    equal = (
                        equal
                        & (
                            torch.scalar_tensor(
                                permutation.numel(),
                                dtype=torch.long,
                                device=device,
                            )
                            == torch.scalar_tensor(
                                prior.numel(), dtype=torch.long, device=device
                            )
                        )
                        & (
                            permutation.narrow(0, 0, length)
                            == prior.narrow(0, 0, length)
                        ).all()
                    )
                representatives[member_id] = torch.where(
                    equal,
                    representatives[previous],
                    representatives[member_id],
                )
        self._set_buffer("_representatives", representatives)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutations) != len(ensemble_table):
            raise RuntimeError(
                "ShuffleCategories must be fitted with the same number of "
                "ensemble members before transform."
            )
        candidates = [
            self._permute(
                ensemble_table[member_id],
                self._permutations[member_id],
                permute_categories=False,
            )
            for member_id in range(len(ensemble_table))
        ]
        return EnsembleTable._from_routes(
            _PermutationRoutes(
                ensemble_table,
                candidates,
                self._representatives,
                self._permutations,
            )
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        permutations: BufferList[Tensor],
        *,
        permute_categories: bool = True,
    ) -> TableTensor:
        code = table.categorical.code.clone()
        valid_mask = table.categorical.isfinite()
        categories: list[Tensor] = []
        for index, category in zip(
            range(len(permutations)),
            _iter_categories(table.categorical),
            strict=True,
        ):
            permutation = permutations[index]
            codes = code[..., index]
            valid = valid_mask[..., index]
            nonempty = permutation.numel() > 0
            code[..., index] = torch.cond(
                nonempty,
                lambda values, mask, order: torch.where(
                    mask,
                    order[values.clamp_min(0).long()].to(values.dtype),
                    values,
                ),
                lambda values, mask, order: values.clone(),
                (codes, valid, permutation),
            )
            categories.append(
                category[permutation.argsort()]
                if permute_categories
                else category
            )

        categorical = CategoricalTensor(
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )


class _PermutationRoutes:
    """Keep banks, not member views, across compiled graph boundaries."""

    def __init__(
        self,
        source: EnsembleTable,
        candidates: Sequence[TableTensor],
        representatives: Tensor,
        permutations: BufferList[BufferList[Tensor]],
    ) -> None:
        self.source = source
        self.locations = representatives.new_empty((len(source), 2))
        self.code_rows = representatives.new_empty(len(source))
        self.category_rows = representatives.new_empty(len(source))
        self.codes: dict[int, Tensor] = {}
        self.categories: dict[
            int, list[Tensor | tuple[Tensor, Tensor, Tensor | None]]
        ] = {}
        self.materialized: EnsembleTable | None = None
        offset = representatives.new_zeros(())
        for group_id, group in enumerate(source._groups):
            indices = [
                i
                for i, (g, _) in enumerate(source._locations)
                if g == group_id
            ]
            if not indices:
                continue
            arange = torch.arange(len(indices), device=representatives.device)
            ids = representatives[indices]
            positions = ids.new_tensor(
                [source._locations[i][1] for i in indices]
            )
            same_group = ids[:, None] == ids[None, :]
            first_group = torch.where(same_group, arange, len(indices)).amin(
                -1
            )
            is_first_group = first_group == arange
            group_ids = is_first_group.long().cumsum(0) - 1
            same_position = same_group & (
                positions[:, None] == positions[None, :]
            )
            first_position = torch.where(
                same_position, arange, len(indices)
            ).amin(-1)
            is_first_position = first_position == arange
            position_ids = (
                same_group
                & is_first_position[None, :]
                & (arange[None, :] <= first_position[:, None])
            ).sum(-1) - 1
            self.locations[indices, 0] = group_ids[first_group] + offset
            self.locations[indices, 1] = position_ids
            offset = offset + is_first_group.sum()
            # Distinct logical positions occupy a sorted prefix. The unused
            # duplicate rows are private capacity, not extra logical members.
            keys = (
                self.locations[indices, 0] * (len(indices) + 1) + position_ids
            )
            order = torch.where(
                is_first_position, keys, keys.amax() + 1
            ).argsort(stable=True)
            self.codes[group_id] = torch.stack(
                [candidates[i].categorical.code for i in indices]
            ).index_select(0, order)
            self.code_rows[indices] = (
                (keys[:, None] > keys[None, :]) & is_first_position[None, :]
            ).sum(-1)
            self.category_rows[indices] = first_group
            banks = []
            for column, category in enumerate(
                _iter_categories(group.categorical)
            ):
                orders = torch.stack(
                    [permutations[i][column].argsort() for i in indices]
                )
                banks.append(
                    _permute_banks(category, orders)
                    if isinstance(category, VarLenTensor)
                    else category[orders]
                )
            self.categories[group_id] = banks
        arange = torch.arange(len(source), device=representatives.device)
        all_unique = (representatives == arange).all()
        self.locations = torch.where(
            all_unique,
            torch.stack((arange, torch.zeros_like(arange)), dim=-1),
            self.locations,
        )

    def __len__(self) -> int:
        return len(self.source)

    def _categories(self, group: int, row: int) -> tuple[Tensor, ...]:
        categories = []
        originals = tuple(
            _iter_categories(self.source._groups[group].categorical)
        )
        for original, bank in zip(
            originals, self.categories[group], strict=True
        ):
            if isinstance(bank, tuple):
                data, offsets, valid = bank
                categories.append(
                    cast(VarLenTensor, original).__class__(
                        data=data.narrow(0, row, 1).squeeze(0),
                        offset=offsets.narrow(0, row, 1).squeeze(0),
                        valid=valid.narrow(0, row, 1).squeeze(0)
                        if valid is not None
                        else None,
                        size=original.shape,
                        storage_offset=0,
                    )
                )
            else:
                categories.append(bank.narrow(0, row, 1).squeeze(0))
        return tuple(categories)

    def member(self, index: int) -> TableTensor:
        group, position = self.source._locations[index]
        code_row = cast(int, self.code_rows[index].item())
        category_row = cast(int, self.category_rows[index].item())
        torch._check(code_row >= 0)
        torch._check(code_row < self.codes[group].size(0))
        torch._check(category_row >= 0)
        torch._check(category_row < self.codes[group].size(0))
        # Recreate views here: retaining them in an ephemeral object lets
        # resumed graphs retain an earlier invocation's alias pattern.
        return self.source._groups[group][position].replace_blocks(
            categorical=CategoricalTensor(
                code=self.codes[group].narrow(0, code_row, 1).squeeze(0),
                categories=self._categories(group, category_row),
            ),
        )

    def materialize(self) -> EnsembleTable:
        if self.materialized is not None:
            return self.materialized
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "Logical group iteration needs a tensor-aware consumer"
            )
        locations = tuple(tuple(row) for row in self.locations.tolist())
        groups = []
        for group_id in range(max(row[0] for row in locations) + 1):
            members = [
                i for i, (g, _) in enumerate(locations) if g == group_id
            ]
            source_group = self.source._locations[members[0]][0]
            by_position = {
                locations[i][1]: self.source._locations[i][1] for i in members
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
            code_row = int(self.code_rows[members[0]])
            category_row = int(self.category_rows[members[0]])
            groups.append(
                group.replace_blocks(
                    categorical=CategoricalTensor(
                        code=self.codes[source_group].narrow(
                            0, code_row, len(positions)
                        ),
                        categories=self._categories(
                            source_group, category_row
                        ),
                    )
                )
            )
        self.materialized = EnsembleTable(groups=groups, locations=locations)
        return self.materialized
