# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
import torch

from sdm import ColumnarTensor, NullableTensor, TableTensor
from sdm.relational._integer_join import _join_integer
from sdm.relational.join import join_index
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_join_index(dtype: torch.dtype, device: torch.device) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(8, device=device),)),
    )

    left_index, right_index = join_index(
        left_table=left_table,
        right_table=right_table,
        left_keys=["id"],
        right_keys=["id"],
        dtype=dtype,
    )

    assert left_index.dtype == dtype
    assert right_index.dtype == dtype
    assert left_index.device == device
    assert right_index.device == device
    assert left_index.sort()[0].equal(torch.arange(8, device=device))
    assert right_index.equal(left_index)


def test_join_index_cast() -> None:
    left_table = TableTensor(
        columns={"id": ("friend",)},
        id=ColumnarTensor((torch.tensor([1.0, float("nan"), 3.0]),)),
    )
    right_table = TableTensor(
        columns={"id": ("user_id",)},
        id=ColumnarTensor((torch.tensor([1, 2, 3]),)),
    )

    left_index, right_index = join_index(
        left_table=left_table,
        right_table=right_table,
        left_keys=["friend"],
        right_keys=["user_id"],
    )

    assert left_index.equal(torch.tensor([0, 2]))
    assert right_index.equal(torch.tensor([0, 2]))


def test_join_index_compile_rejects_unsafe_narrowing(
    compilation: None,
) -> None:
    left = TableTensor(
        columns={"id": ("key",)},
        id=ColumnarTensor((torch.tensor([1], dtype=torch.int32),)),
    )
    right = TableTensor(
        columns={"id": ("key",)},
        id=ColumnarTensor((torch.tensor([1, 2**40], dtype=torch.int64),)),
    )

    def join(
        left: TableTensor, right: TableTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return join_index(left, right, ["key"], ["key"])

    with torch.inference_mode():
        with pytest.raises(pa.ArrowInvalid) as eager_error:
            join(left, right)
        with pytest.raises(pa.ArrowInvalid) as compiled_error:
            torch.compile(
                join, backend="inductor", fullgraph=False, dynamic=True
            )(left, right)

    assert str(compiled_error.value) == str(eager_error.value)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_invalid_dtype(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(8, device=device),)),
    )

    with pytest.raises(TypeError, match="requires an integer input type"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=dtype,
        )


@withCUDA
def test_overflow(device: torch.device) -> None:
    left_table = right_table = TableTensor(
        columns={"id": ("id",)},
        id=ColumnarTensor((torch.arange(200, device=device),)),
    )

    with pytest.raises(ValueError, match="row indices up to 199"):
        join_index(
            left_table=left_table,
            right_table=right_table,
            left_keys=["id"],
            right_keys=["id"],
            dtype=torch.int8,
        )


@withCUDA
def test_join_index_compile_nullable(
    device: torch.device,
    compilation: None,
) -> None:
    left = TableTensor(
        columns={"id": ("b", "a")},
        id=ColumnarTensor(
            (
                torch.tensor([[10, 10], [20, 30]], device=device),
                NullableTensor(
                    torch.tensor([[1, 1], [2, 3]], device=device),
                    torch.tensor([[True, False], [True, True]], device=device),
                ),
            )
        ),
    )
    right = TableTensor(
        columns={"id": ("x", "y")},
        id=ColumnarTensor(
            (
                torch.tensor([[1, 2, 1]], dtype=torch.int32, device=device),
                torch.tensor([[10, 20, 10]], dtype=torch.int32, device=device),
            )
        ),
    )

    def join(
        left: TableTensor, right: TableTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return join_index(
            left, right, ["a", "b"], ["x", "y"], dtype=torch.int32
        )

    with (
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
    ):
        actual = torch.compile(join, fullgraph=True, dynamic=True)(left, right)
    assert sorted(zip(actual[0].tolist(), actual[1].tolist())) == [
        (0, 0),
        (0, 2),
        (2, 1),
    ]
    assert all(index.dtype == torch.int32 for index in actual)
    assert all(index.device == device for index in actual)


@withCUDA
def test_join_index_compile_hash_collision(
    device: torch.device,
    compilation: None,
) -> None:
    # Distinct original keys with the same strong hash exercise exact fallback.
    a = (-(1 << 63), (1 << 63) - 1)
    b = ((1 << 63) - 1, -2044408044210139470)
    c, d, e = (1, 2), (3, 4), (5, 6)

    def table(rows: list[tuple[int, int]]) -> TableTensor:
        keys = torch.tensor(rows, dtype=torch.int64, device=device).reshape(
            -1, 2
        )
        return TableTensor(
            columns={"id": ("a", "b")},
            id=ColumnarTensor(keys.unbind(-1)),
        )

    def join(
        left: TableTensor, right: TableTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return join_index(left, right, ["a", "b"], ["a", "b"])

    with (
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
    ):
        compiled = torch.compile(join, fullgraph=True, dynamic=True)
        for left, right in [
            ([a, c, d], [c, d, e]),
            ([a, c, d], [a, b, e]),
            ([a, a, a], [a, a, a]),
            ([a, a, a], [b, b, b]),
            ([], [a]),
            ([a], [a]),
        ]:
            actual = compiled(table(left), table(right))
            expected = [
                (i, j)
                for i, left_key in enumerate(left)
                for j, right_key in enumerate(right)
                if left_key == right_key
            ]
            assert (
                sorted(zip(actual[0].tolist(), actual[1].tolist())) == expected
            )


@withCUDA
def test_integer_join_compile_filtered_chain(
    device: torch.device,
    compilation: None,
) -> None:
    keys = torch.ones((3, 1), dtype=torch.int64, device=device)
    valid = torch.ones(3, dtype=torch.bool, device=device)

    def join_filtered(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        filtered = keys[keep]
        filtered_valid = torch.ones_like(filtered[:, 0], dtype=torch.bool)
        index, _ = _join_integer(filtered, keys, filtered_valid, valid)
        gathered = filtered[index]
        gathered_valid = torch.ones_like(gathered[:, 0], dtype=torch.bool)
        return _join_integer(gathered, keys, gathered_valid, valid)

    with (
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True),
    ):
        compiled = torch.compile(join_filtered, fullgraph=True, dynamic=True)
        for keep in ([False, False, False], [True, False, False], [True] * 3):
            actual = compiled(torch.tensor(keep, device=device))
            expected = [(i, j) for i in range(3 * sum(keep)) for j in range(3)]
            assert (
                sorted(zip(actual[0].tolist(), actual[1].tolist())) == expected
            )


@withCUDA
@pytest.mark.parametrize(
    ("left_values", "right_values"),
    [([1.0, float("nan"), 3.0], [1, 3]), (["a", None, "c"], ["a", "c"])],
)
def test_join_index_compile_external(
    device: torch.device,
    compilation: None,
    left_values: list[float | str | None],
    right_values: list[int | str],
) -> None:
    left = TableTensor.from_columns(
        {"key": left_values}, stypes={"key": "id"}, device=device
    )
    right = TableTensor.from_columns(
        {"key": right_values}, stypes={"key": "id"}, device=device
    )

    def join(
        left: TableTensor, right: TableTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return join_index(left, right, ["key"], ["key"])

    with torch.inference_mode():
        actual = torch.compile(join, fullgraph=False, dynamic=True)(
            left, right
        )
    assert sorted(zip(actual[0].tolist(), actual[1].tolist())) == [
        (0, 0),
        (2, 1),
    ]
