# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor
from sdm.tensor.categorical import _iter_categories


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

    def __init__(
        self,
        method: Literal["shift", "random"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[BufferList[Tensor]] = BufferList()
        self._permutation_ids: tuple[int, ...] = ()

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._permutation_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._permutation_ids = cast(tuple[int, ...], state)

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

        permutations_by_id = []
        permutation_ids = []
        permutation_id_by_key: dict[tuple[tuple[int, ...], ...], int] = {}

        history: dict[
            str, list[tuple[int | Tensor, int | Tensor, Tensor]]
        ] = {}
        for member_id in range(len(ensemble_table)):
            permutations = self._draw_permutations(
                ensemble_table[member_id],
                history=history,
                generator=generator,
            )
            key = tuple(
                tuple(permutation.tolist()) for permutation in permutations
            )
            permutation_id = permutation_id_by_key.get(key)
            if permutation_id is None:
                permutation_id = len(permutations_by_id)
                permutation_id_by_key[key] = permutation_id
                permutations_by_id.append(permutations)
            permutation_ids.append(permutation_id)

        self._permutations = BufferList(
            BufferList(permutations) for permutations in permutations_by_id
        )
        self._permutation_ids = tuple(permutation_ids)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutation_ids) != len(ensemble_table):
            raise RuntimeError(
                "ShuffleCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        member_ids_by_permutation: dict[int, list[int]] = {}
        for member_id, permutation_id in enumerate(self._permutation_ids):
            member_ids_by_permutation.setdefault(permutation_id, []).append(
                member_id
            )

        if len(member_ids_by_permutation) == len(ensemble_table):
            member_tables: list[TableTensor] = []
            for member_id, permutation_id in enumerate(self._permutation_ids):
                permutations = self._permutations[permutation_id]
                member_tables.append(
                    self._permute(
                        ensemble_table[member_id],
                        permutations,
                    )
                )
            return EnsembleTable.from_tables(
                tables=member_tables,
                member_table_ids=range(len(ensemble_table)),
            )

        outputs: dict[int, EnsembleTable] = {}
        for permutation_id, member_ids in member_ids_by_permutation.items():
            selected = ensemble_table[member_ids]
            permutations = self._permutations[permutation_id]
            outputs[permutation_id] = selected.replace_groups(
                [
                    self._permute(group, permutations)
                    for group in selected._iter_groups()
                ]
            )

        output_tables = []
        member_ids = []
        next_member_id_by_permutation: dict[int, int] = {}
        for permutation_id in self._permutation_ids:
            output_tables.append(outputs[permutation_id])
            member_id = next_member_id_by_permutation.get(permutation_id, 0)
            member_ids.append(member_id)
            next_member_id_by_permutation[permutation_id] = member_id + 1

        return ensemble_table.gather_members(
            tables=output_tables,
            member_ids=member_ids,
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        permutations: BufferList[Tensor],
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
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
