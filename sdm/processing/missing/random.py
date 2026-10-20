# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import Processor


class MaskRandom(Processor):
    """Mask numerical and categorical cells completely at random.

    Each transform draws a missing-cell probability per table and masks each
    cell independently with that probability.

    Args:
        rate: Lower and upper bounds for uniform sampling of each table's
            missing-cell probability.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = False

    def __init__(self, *, rate: tuple[float, float] = (0.01, 0.08)) -> None:
        super().__init__()
        self.rate = rate

    def _transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        missing_rate = torch.empty(
            size=table.size()[:-2],
            device=table.device,
        ).uniform_(*self.rate, generator=generator)
        num_numerical = table.numerical.size(-1)
        num_columns = num_numerical + table.categorical.size(-1)
        u = torch.rand(
            (*table.size()[:-1], num_columns),
            device=table.device,
            generator=generator,
        )
        missing = u < missing_rate[..., None, None]
        return table.replace_blocks(
            numerical=table.numerical.masked_fill(
                missing[..., :num_numerical], torch.nan
            ),
            categorical=CategoricalTensor(
                code=table.categorical.code.masked_fill(
                    missing[..., num_numerical:], -1
                ),
                categories=table.categorical.categories,
            ),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}(rate={self.rate})"
