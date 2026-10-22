# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from itertools import repeat
from typing import Any, cast

import torch
from torch.nn import ModuleList

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    InvertibleMixin,
    Processor,
)


class EnsembleProcessorAdapter(EnsembleProcessor, EnsembleInvertibleMixin):
    """Adapt an ordinary processor to ensemble-aware processing.

    The adapter turns a :class:`~sdm.processing.base.Processor` into an
    :class:`~sdm.processing.ensemble.EnsembleProcessor`. It copies and fits the
    processor separately for each group of compatible tables in an
    :class:`~sdm.EnsembleTable`.

    The wrapped processor must preserve row and leading dimensions as required
    by the :class:`~sdm.processing.base.Processor` contract. A processor that
    changes the ensemble structure must implement
    :class:`~sdm.processing.ensemble.EnsembleProcessor` directly. Inverse
    transformation requires the wrapped processor to implement
    :class:`~sdm.processing.base.InvertibleMixin`.

    Args:
        processor: Processor to fit separately for each ensemble table group.
    """

    _conditional_count: torch.Tensor
    _conditional_locations: torch.Tensor

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit

        self._processors: ModuleList[Processor] = ModuleList()
        self._fitted_locations: tuple[tuple[int, int], ...] = ()
        self._conditional_state = False

    def _materialize_conditional_state(self) -> None:
        # Restore legacy module aliases and locations at Python API boundaries.
        if not self._conditional_state:
            return
        count = int(self._conditional_count)
        self._processors = ModuleList(list(self._processors)[:count])
        self._fitted_locations = cast(
            tuple[tuple[int, int], ...],
            tuple(
                tuple(location)
                for location in self._conditional_locations.tolist()
            ),
        )

    def _save_to_state_dict(
        self, destination: dict[str, Any], prefix: str, keep_vars: bool
    ) -> None:
        self._materialize_conditional_state()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def get_extra_state(
        self,
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        r""":meta private:"""  # noqa: D415
        self._materialize_conditional_state()
        return len(self._processors), self._fitted_locations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        num_processors, self._fitted_locations = cast(
            tuple[int, tuple[tuple[int, int], ...]], state
        )
        self._processors = ModuleList(
            [
                self.processor if index == 0 else copy.deepcopy(self.processor)
                for index in range(num_processors)
            ]
        )
        self._conditional_state = False

    def fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        r""":meta private:"""  # noqa: D415
        groups = tuple(ensemble_table._iter_groups())
        if not (
            torch.compiler.is_compiling()
            and len(groups) == 1
            and self.processor._supports_conditional_fit(groups[0])
            and self.requires_fit
        ):
            self._materialize_conditional_state()
            return super().fit_transform_ensemble(
                ensemble_table, generator=generator
            )

        device = groups[0].device
        if not self._conditional_state:
            self.register_buffer(
                "_conditional_count",
                torch.tensor(len(self._processors), device=device),
                persistent=False,
            )
            self.register_buffer(
                "_conditional_locations",
                torch.tensor(
                    self._fitted_locations, dtype=torch.int64, device=device
                ).reshape(-1, 2),
                persistent=False,
            )
            self._conditional_state = True
        if not self._processors:
            # The processor already exists; only its fitted membership varies.
            self._processors.append(self.processor)

        self._defer_fitted()
        out, active = self.processor._fit_transform_conditional(groups[0])
        locations = torch.tensor(
            ensemble_table._locations, dtype=torch.int64, device=device
        ).reshape(-1, 2)
        selected_locations = torch.cond(
            active,
            lambda current, previous: current.clone(),
            lambda current, previous: previous.clone(),
            (locations, self._conditional_locations),
        )
        self._set_buffer(
            "_conditional_count",
            torch.where(active, 1, self._conditional_count),
        )
        self._set_buffer("_conditional_locations", selected_locations)
        self._set_buffer("_fitted_state", self._fitted_state | active)
        return ensemble_table.replace_groups([out])

    def transform(self, table: TableTensor) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if (
            torch.compiler.is_compiling()
            and self._conditional_state
            and self.processor._supports_conditional_transform(table)
        ):
            return self.transform_ensemble(
                EnsembleTable.from_table(table, num_members=1)
            )[0]
        return super().transform(table)

    def transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable:
        r""":meta private:"""  # noqa: D415
        groups = tuple(ensemble_table._iter_groups())
        if not (
            torch.compiler.is_compiling()
            and self._conditional_state
            and len(groups) == 1
            and self.processor._supports_conditional_transform(groups[0])
        ):
            return super().transform_ensemble(ensemble_table)

        saved = self._conditional_locations
        current = saved.new_tensor(ensemble_table._locations).reshape(-1, 2)
        padded = torch.cat((saved, saved.new_full((1, 2), -1)))
        index = torch.arange(current.size(0), device=saved.device).clamp_max(
            saved.size(0)
        )
        aligned = padded.index_select(0, index)
        count_matches = torch.scalar_tensor(
            saved.size(0), device=saved.device, dtype=torch.int64
        ) == current.size(0)
        batch_matches = (aligned[:, 1] == current[:, 1]).all()
        shared_batch = (aligned[:, 1] == aligned[0, 1]).all()
        valid = (
            count_matches
            & (aligned[:, 0] == 0).all()
            & (shared_batch | batch_matches)
            & self._fitted_state
        )
        out = self.processor._transform_conditional(groups[0], valid)
        return ensemble_table.replace_groups([out])

    @property
    def handles_stypes(self) -> tuple[Stype, ...]:
        r""":meta private:"""  # noqa: D415
        return self.processor.handles_stypes

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._materialize_conditional_state()
        self._conditional_state = False
        self._fitted_locations = ensemble_table._locations
        self._processors = ModuleList()
        for group_id, group in enumerate(ensemble_table._iter_groups()):
            if group_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            processor.fit(group, generator=generator)
            self._processors.append(processor)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        self._materialize_conditional_state()
        self._conditional_state = False
        self._fitted_locations = ensemble_table._locations
        outputs = []
        self._processors = ModuleList()
        for group_id, group in enumerate(ensemble_table._iter_groups()):
            if group_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            outputs.append(processor.fit_transform(group, generator=generator))
            self._processors.append(processor)
        return ensemble_table.replace_groups(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            processors = self._aligned_processors(ensemble_table)
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)
        outputs = [
            processor.transform(group)
            for group, processor in zip(
                ensemble_table._iter_groups(), processors, strict=True
            )
        ]
        return ensemble_table.replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            processors = self._aligned_processors(ensemble_table)
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)

        outputs = []
        for group, processor in zip(
            ensemble_table._iter_groups(), processors, strict=True
        ):
            if not isinstance(processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            outputs.append(processor.inverse_transform(group))
        return ensemble_table.replace_groups(outputs)

    def _aligned_processors(
        self,
        ensemble_table: EnsembleTable,
    ) -> tuple[Processor, ...]:
        self._materialize_conditional_state()
        if ensemble_table._locations == self._fitted_locations:
            return tuple(self._processors)

        # One fitted processor per transform ensemble group. Multiple
        # transform groups can share the same fitted processor.
        processors: list[Processor | None] = [None] * ensemble_table.num_groups
        # Current and fitted batch of every member sharing a fitted processor.
        batches_by_processor: dict[int, list[tuple[int, int, int]]] = {}
        for member_id, (group_id, batch_id) in enumerate(
            ensemble_table._locations
        ):
            fitted_group_id, fitted_batch_id = self._fitted_locations[
                member_id
            ]
            processor = self._processors[fitted_group_id]
            if processors[group_id] is None:
                processors[group_id] = processor
            elif processors[group_id] is not processor:
                raise RuntimeError(
                    "Cannot apply fitted processor state to an ensemble "
                    "group containing members from different fitted groups"
                )
            batches_by_processor.setdefault(fitted_group_id, []).append(
                (group_id, batch_id, fitted_batch_id)
            )

        # A processor fitted on a stacked tensor table stores per-batch state.
        # It can only operate on stacked tensor tables with the same layout,
        # but not on splits or reordered batches.
        for batches in batches_by_processor.values():
            if len({fitted for _, _, fitted in batches}) > 1 and (
                len({group for group, _, _ in batches}) > 1
                or any(current != fitted for _, current, fitted in batches)
            ):
                raise RuntimeError(
                    "Cannot apply position-dependent fitted processor state "
                    "after its ensemble group was split or reordered"
                )

        return tuple(cast(Processor, processor) for processor in processors)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)
