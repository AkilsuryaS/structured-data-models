# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm._memory import _split_tensor


def test_bounded_symbolic_split_compiles(compilation: None) -> None:
    def transform(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[:, mask]
        torch._check(selected.size(-1) <= values.size(-1))
        return torch.cat(
            [chunk.square() for chunk in _split_tensor(selected, 8, dim=-1)],
            dim=-1,
        )

    values = torch.arange(24.0).reshape(3, 8)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(transform, backend="inductor", fullgraph=True)
        for width in (0, 1, 3, 8):
            mask = torch.arange(8) < width
            torch.testing.assert_close(
                compiled(values, mask), values[:, mask].square()
            )


@pytest.mark.parametrize("shape", [(3, 8), (0, 8), (3, 0)])
@pytest.mark.parametrize("size", [1, 3, 8])
def test_split_preserves_chunked_operations(
    shape: tuple[int, int], size: int
) -> None:
    values = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(
        shape
    )
    actual = torch.cat(
        [chunk.cumsum(dim=-1) for chunk in _split_tensor(values, size, -1)],
        dim=-1,
    )
    expected = torch.cat(
        [chunk.cumsum(dim=-1) for chunk in values.split(size, dim=-1)],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
