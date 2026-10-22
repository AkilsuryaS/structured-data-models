# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleProcessor


class DropConstantColumns(EnsembleProcessor):
    """Remove non-informative numerical columns learned during fit.

    Columns are retained when they have more than ``threshold`` distinct
    values. When the number of samples is less than or equal to ``threshold``,
    all columns are preserved. NaN is counted once as a distinct value.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.
    Fitting expects data with shape ``[N, C]``, where ``N`` is the number of
    rows and ``C`` is the number of numerical columns. The learned selection
    can transform later tables with shape ``[..., C]``.

    Args:
        threshold: Columns with at most this many unique values are removed.
            Must be positive.
    """

    _shared_keep: torch.Tensor
    _shared_index: torch.Tensor | None

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    def __init__(
        self,
        *,
        threshold: int = 1,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.threshold = threshold
        # TODO: Consider recording the fitted column names if transforms should
        # verify that the numerical schema and order match fit.
        self._kept_indices: tuple[tuple[int, ...], ...] = ()
        self.register_buffer("_shared_keep", torch.empty(0, dtype=torch.bool))
        self.register_buffer("_shared_index", None, persistent=False)
        self._shared_members = 0
        self._keep_all = False

    def get_extra_state(self) -> object:
        r""":meta private:"""  # noqa: D415
        return (self._kept_indices, self._shared_members, self._keep_all)

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        if (
            isinstance(state, tuple)
            and len(state) == 3
            and isinstance(state[1], int)
        ):
            self._kept_indices, self._shared_members, self._keep_all = cast(
                tuple[tuple[tuple[int, ...], ...], int, bool], state
            )
        else:
            self._kept_indices = cast(tuple[tuple[int, ...], ...], state)
            self._shared_members = 0
            self._keep_all = False

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
        # Older checkpoints store only the Python selection in extra state.
        key = f"{prefix}_shared_keep"
        if key not in state_dict:
            state_dict[key] = self._shared_keep.new_empty(0)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.register_buffer("_shared_index", None, persistent=False)

    def _get_shared_index(self) -> torch.Tensor:
        index = getattr(self, "_shared_index", None)
        if index is None:
            index = self._shared_keep.nonzero().flatten()
            self.register_buffer("_shared_index", index, persistent=False)
        return index

    def _keep_mask(self, data: torch.Tensor) -> torch.Tensor:
        # [N, C] or [..., N, C] -> [C] or [..., C].
        # Preserve the schema when too few rows can exceed the threshold.
        if data.size(-2) <= self.threshold:
            return data.new_ones(
                (*data.shape[:-2], data.size(-1)),
                dtype=torch.bool,
            )
        if self.threshold == 1:
            # Any mismatch with the first row proves a second unique value.
            different = (data != data[..., :1, :]).any(dim=-2)
            return different & ~data.isnan().all(dim=-2)

        # A sorted column with k unique values has k - 1 transitions.
        values = data.sort(dim=-2).values
        left, right = values[..., :-1, :], values[..., 1:, :]
        changed = (right != left) & ~(right.isnan() & left.isnan())
        return changed.sum(dim=-2) >= self.threshold

    @staticmethod
    def _select_columns(
        table: TableTensor,
        kept_indices: tuple[int, ...],
    ) -> TableTensor:
        numerical_columns = table.numerical_columns
        if len(kept_indices) == len(numerical_columns):
            return table
        kept_numerical_columns = tuple(
            numerical_columns[index] for index in kept_indices
        )
        columns = [
            column
            for stype, stype_columns in table.column_groups
            for column in (
                kept_numerical_columns
                if stype == Stype.numerical
                else stype_columns
            )
        ]
        return table.select_columns(columns)

    def _cache_shared_indices(self) -> None:
        if not self._kept_indices:
            indices = tuple(
                index
                for index, kept in enumerate(self._shared_keep.tolist())
                if kept
            )
            self._kept_indices = (indices,) * self._shared_members

    def _select_shared_columns(self, table: TableTensor) -> TableTensor:
        # A fitted Python selection with static names keeps the native views.
        _, _, pools, indices = table._column_context
        if self._kept_indices and not (pools or indices):
            return self._select_columns(table, self._kept_indices[0])
        if not torch.compiler.is_compiling():
            self._cache_shared_indices()
            return self._select_columns(table, self._kept_indices[0])
        # Dynamic compaction preserves values and names, but does not preserve
        # eager storage aliases when the selected columns form a slice.
        return table._select_numerical_indices(self._get_shared_index())

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._set_buffer("_shared_keep", self._keep_mask(table.numerical))
        self._shared_members = 1
        self._keep_all = table.numerical.size(-2) <= self.threshold
        self._kept_indices = ()
        self.register_buffer("_shared_index", None, persistent=False)
        if torch.compiler.is_compiling() and not self._keep_all:
            self._get_shared_index()
        if not torch.compiler.is_compiling():
            self._cache_shared_indices()

    def _transform(self, table: TableTensor) -> TableTensor:
        if self._shared_members:
            if self._keep_all and (
                table.numerical.size(-1) == self._shared_keep.size(0)
            ):
                return table
            return self._select_shared_columns(table)
        return self._select_columns(table, self._kept_indices[0])

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._fit(table, generator=generator)
        return self._transform(table)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        groups = tuple(ensemble_table._iter_groups())
        if sum(group.size(0) for group in groups) == 1:
            group = groups[0]
            self._set_buffer(
                "_shared_keep", self._keep_mask(group.numerical)[0]
            )
            self._shared_members = len(ensemble_table)
            self._keep_all = group.numerical.size(-2) <= self.threshold
            self._kept_indices = ()
            self.register_buffer("_shared_index", None, persistent=False)
            if torch.compiler.is_compiling() and not self._keep_all:
                self._get_shared_index()
            if not torch.compiler.is_compiling():
                self._cache_shared_indices()
            return

        self._shared_members = 0
        self._keep_all = False
        self.register_buffer("_shared_index", None, persistent=False)
        masks = ensemble_table.replace_groups(
            [
                TableTensor.from_tensor(
                    tensor=self._keep_mask(group.numerical)
                    .to(dtype=group.numerical.dtype)
                    .unsqueeze(-2),
                    columns=group.numerical_columns,
                )
                for group in groups
            ]
        )
        masks_by_size_and_device: dict[
            tuple[int, torch.device],
            list[tuple[int, torch.Tensor]],
        ] = {}
        for member_id in range(len(masks)):
            table = masks[member_id]
            key = (table.numerical.size(-1), table.device)
            masks_by_size_and_device.setdefault(key, []).append(
                (member_id, table.numerical[0].bool())
            )

        kept_indices_by_member: list[tuple[int, ...]]
        kept_indices_by_member = [()] * len(masks)
        for member_masks in masks_by_size_and_device.values():
            keep_by_member = torch.stack(
                [keep for _, keep in member_masks]
            ).tolist()
            for (member_id, _), keep in zip(
                member_masks,
                keep_by_member,
                strict=True,
            ):
                kept_indices_by_member[member_id] = tuple(
                    index for index, kept in enumerate(keep) if kept
                )
        self._kept_indices = tuple(kept_indices_by_member)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._shared_members:
            if self._shared_members != len(ensemble_table):
                raise RuntimeError(
                    "DropConstantColumns must be fitted with the same "
                    "number of ensemble members before transform."
                )
            if self._keep_all and all(
                group.numerical.size(-1) == self._shared_keep.size(0)
                for group in ensemble_table._iter_groups()
            ):
                return ensemble_table
            return ensemble_table.replace_groups(
                [
                    self._select_shared_columns(group)
                    for group in ensemble_table._iter_groups()
                ]
            )

        if len(self._kept_indices) != len(ensemble_table):
            raise RuntimeError(
                "DropConstantColumns must be fitted with the same number of "
                "ensemble members before transform."
            )

        member_ids_by_kept_indices: dict[tuple[int, ...], list[int]] = {}
        for member_id, kept_indices in enumerate(self._kept_indices):
            member_ids_by_kept_indices.setdefault(kept_indices, []).append(
                member_id
            )

        if len(member_ids_by_kept_indices) == 1:
            kept_indices = next(iter(member_ids_by_kept_indices))
            return ensemble_table.replace_groups(
                [
                    self._select_columns(group, kept_indices)
                    for group in ensemble_table._iter_groups()
                ]
            )

        if sum(
            group.size(0) for group in ensemble_table._iter_groups()
        ) == len(ensemble_table):
            tables = [
                self._select_columns(
                    ensemble_table[member_id],
                    kept_indices,
                )
                for member_id, kept_indices in enumerate(self._kept_indices)
            ]
            return ensemble_table.replace_tables(
                tables=tables,
                member_table_ids=range(len(ensemble_table)),
            )

        outputs: dict[tuple[int, ...], EnsembleTable] = {}
        for kept_indices, member_ids in member_ids_by_kept_indices.items():
            selected = ensemble_table[member_ids]
            outputs[kept_indices] = selected.replace_groups(
                [
                    self._select_columns(group, kept_indices)
                    for group in selected._iter_groups()
                ]
            )

        tables = []
        member_ids = []
        next_member_id_by_kept_indices: dict[tuple[int, ...], int] = {}
        for kept_indices in self._kept_indices:
            tables.append(outputs[kept_indices])
            member_id = next_member_id_by_kept_indices.get(kept_indices, 0)
            member_ids.append(member_id)
            next_member_id_by_kept_indices[kept_indices] = member_id + 1

        return ensemble_table.gather_members(
            tables=tables,
            member_ids=member_ids,
        )
