# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _isfinite


class LinearDetrend(Processor, InvertibleMixin):
    """Remove a per-column linear trend over step coordinates.

    Non-finite values are ignored when fitting and preserved during the
    transform.

    Args:
        id_column: The ID column holding the step coordinate of each row.
        threshold: If set, detrend a column only when its residual standard
            deviation is less than ``threshold`` times its original standard
            deviation.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        id_column: str,
        *,
        threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.id_column = id_column
        self.threshold = threshold
        self.register_buffer("origin", torch.empty(0, dtype=torch.long))
        self.register_buffer("slope", torch.empty(0))
        self.register_buffer("intercept", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        x = table.numerical.to(
            dtype=torch.promote_types(table.numerical.dtype, torch.float32)
        )
        time = table[self.id_column].id[..., 0]
        # Shift steps to end at zero to keep large coordinates precise
        self.origin = time[..., -1:].clone()
        time = (time - self.origin).to(x.dtype).unsqueeze(-1)  # [..., R, 1]

        # Per-column statistics over finite values
        valid = _isfinite(x)
        count = valid.sum(dim=-2, keepdim=True, dtype=x.dtype).clamp(min=1)
        time_mean = torch.where(valid, time, 0.0).sum(-2, keepdim=True) / count
        time_sq = torch.where(valid, time.square(), 0.0).sum(-2, keepdim=True)
        time_var = time_sq / count - time_mean.square()
        x = x.masked_fill(~valid, 0.0)
        mean = x.sum(dim=-2, keepdim=True) / count
        x.sub_(mean).masked_fill_(~valid, 0.0)
        cov = (time.transpose(-2, -1) @ x) / count

        # Least-squares fit of `x = intercept + slope * time`
        slope = torch.where(time_var > 0, cov / time_var, 0.0)
        if self.threshold is not None:
            # The residual variance is `var - slope * cov`
            var = x.square_().sum(dim=-2, keepdim=True) / count
            keep = var - slope * cov < self.threshold**2 * var
            slope = torch.where(keep, slope, 0.0)
            mean = torch.where(keep, mean, 0.0)

        self.slope = slope
        self.intercept = mean - slope * time_mean

    def _transform(self, table: TableTensor) -> TableTensor:
        time = table[self.id_column].id[..., 0] - self.origin
        time = time.to(self.slope.dtype).unsqueeze(-1)
        numerical = table.numerical - self.intercept
        numerical.addcmul_(self.slope, time, value=-1)
        return table.replace_blocks(
            numerical=numerical.to(table.numerical.dtype)
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        time = table[self.id_column].id[..., 0] - self.origin
        time = time.to(self.slope.dtype).unsqueeze(-1)
        numerical = table.numerical + self.intercept
        numerical.addcmul_(self.slope, time)
        return table.replace_blocks(
            numerical=numerical.to(table.numerical.dtype)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"id_column={self.id_column!r}, threshold={self.threshold})"
        )
