# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import Processor


class MaskRandom(Processor):
    """Mask numerical and categorical cells completely at random.

    Each table draws a missing-cell probability when the processor is fitted.
    Transform masks each cell independently with this probability. Inputs
    of the same shape use the same missing-cell selections. Apply the
    processor to all rows in one call.

    Args:
        rate: Lower and upper bounds for uniform sampling of each table's
            missing-cell probability.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = True

    _missing_rate: Tensor

    def __init__(self, *, rate: tuple[float, float] = (0.01, 0.08)) -> None:
        super().__init__()
        self.rate = rate
        self._seed = 0
        self.register_buffer("_missing_rate", torch.empty(0))

    def get_extra_state(self) -> int:
        r""":meta private:"""  # noqa: D415
        return self._seed

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._seed = cast(int, state)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = None if generator is None else generator.device
        self._seed = int(
            torch.randint(2**62, (), device=device, generator=generator)
        )
        self._set_buffer(
            "_missing_rate",
            torch.empty(
                size=table.size()[:-2],
                device=table.device,
                dtype=table.numerical.dtype,
            ).uniform_(*self.rate, generator=generator),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        num_numerical = table.numerical.size(-1)
        num_columns = num_numerical + table.categorical.size(-1)
        u = torch.rand(
            (*table.size()[:-1], num_columns),
            device="cpu",
            generator=torch.Generator().manual_seed(self._seed),
        )
        missing = u.to(table.device) < self._missing_rate[..., None, None]
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
