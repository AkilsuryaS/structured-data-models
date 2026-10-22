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
)
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor
from sdm.processing.categorical._routing import (
    CategoricalRoutes,
    grouped_locations,
    source_locations,
)
from sdm.processing.categorical._storage import PermutedCategories
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
                    lambda n=n_classes: torch.randperm(
                        n, device=device, **generator_kwargs
                    ),
                    lambda n=n_classes: torch.arange(n, device=device),
                    (),
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
                    lambda n=n_classes: torch.randperm(
                        n, device=device, **generator_kwargs
                    ),
                    lambda n=n_classes: torch.arange(n, device=device),
                    (),
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
            self._permute_codes(
                ensemble_table._categorical_code(member_id),
                self._permutations[member_id],
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
    def _permute_codes(
        code: Tensor, permutations: BufferList[Tensor]
    ) -> Tensor:
        code = code.clone()
        for index, permutation in enumerate(permutations):
            codes = code[..., index]
            code[..., index] = torch.cond(
                permutation.numel() > 0,
                lambda values, order: torch.where(
                    values >= 0,
                    order[values.clamp_min(0).long()].to(values.dtype),
                    values,
                ),
                lambda values, order: values.clone(),
                (codes, permutation),
            )
        return code

    @staticmethod
    def _permute(
        table: TableTensor,
        permutations: BufferList[Tensor],
    ) -> TableTensor:
        return table.replace_blocks(
            categorical=CategoricalTensor(
                ShuffleCategories._permute_codes(
                    table.categorical.code, permutations
                ),
                [
                    category[permutation.argsort()]
                    for category, permutation in zip(
                        _iter_categories(table.categorical),
                        permutations,
                        strict=True,
                    )
                ],
            )
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )


class _PermutationRoutes(CategoricalRoutes):
    def __init__(
        self,
        source: EnsembleTable,
        candidates: Sequence[Tensor],
        representatives: Tensor,
        permutations: BufferList[BufferList[Tensor]],
    ) -> None:
        locations = source_locations(source)
        same = (locations[:, 0, None] == locations[None, :, 0]) & (
            representatives[:, None] == representatives[None, :]
        )
        output_locations = grouped_locations(locations, same)
        arange = torch.arange(len(source), device=representatives.device)
        output_locations = torch.where(
            (representatives == arange).all(),
            torch.stack((arange, torch.zeros_like(arange)), dim=-1),
            output_locations,
        )
        super().__init__(
            source,
            candidates,
            output_locations,
            PermutedCategories(source, permutations, output_locations[:, 0]),
            ((0, 0),) if len(source) == 1 else None,
        )
