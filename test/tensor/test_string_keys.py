# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from typing import cast

import pyarrow as pa
import pytest
import torch

from sdm import StringTensor
from sdm.tensor._string_keys import pack
from sdm.testing import withCUDA

pytestmark = pytest.mark.usefixtures("compilation")


def _reference_keys(values: list[str | None], width: int) -> torch.Tensor:
    words = max(1, (width + 6) // 7)
    result = []
    for value in values:
        symbols = (
            [] if value is None else [byte + 1 for byte in value.encode()]
        )
        symbols += [0] * (words * 7 - len(symbols))
        result.append(
            [
                sum(symbols[column * 7 + i] << (9 * (6 - i)) for i in range(7))
                for column in range(words)
            ]
        )
    return torch.tensor(result, dtype=torch.int64).reshape(len(values), words)


@withCUDA
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("length", [0, 7, 8, 14, 15])
def test_pack_keys(device: torch.device, compiled: bool, length: int) -> None:
    texts = (
        ["", None]
        if length == 0
        else ["", "\0", "a", "a\0", "é", "🙂", "a" * length, None]
    )
    values = StringTensor.from_list(texts, device=device)
    fn = torch.compile(pack, fullgraph=True) if compiled else pack
    keys, valid = fn(values)
    width = cast(int, values._max_length)
    assert torch.equal(keys.cpu(), _reference_keys(texts, width))
    assert valid.tolist() == [text is not None for text in texts]
    non_null = [i for i, text in enumerate(texts) if text is not None]
    assert sorted(non_null, key=lambda i: tuple(keys[i].tolist())) == sorted(
        non_null, key=lambda i: cast(str, texts[i]).encode()
    )


@withCUDA
def test_pack_strided_views(device: torch.device) -> None:
    values = StringTensor.from_list(
        ["ignored", "é", "ignored", None, "ignored", "\0"], device=device
    )
    view = cast(StringTensor, values[1::2])
    expected = _reference_keys(["é", None, "\0"], 7)
    for fn in (pack, torch.compile(pack, fullgraph=True)):
        keys, valid = fn(view)
        assert torch.equal(keys.cpu(), expected)
        assert valid.tolist() == [True, False, True]


@withCUDA
def test_pack_empty_and_null_payload(device: torch.device) -> None:
    for texts in ([], ["", None, ""]):
        values = StringTensor.from_list(texts, device=device)
        keys, valid = torch.compile(pack, fullgraph=True)(values)
        assert torch.equal(keys.cpu(), _reference_keys(texts, 0))
        assert valid.tolist() == [text is not None for text in texts]
    # A null slot may hold arbitrary payload bytes.
    array = pa.Array.from_buffers(
        pa.string(),
        1,
        [
            pa.py_buffer(bytes([0])),
            pa.py_buffer(torch.tensor([0, 7], dtype=torch.int32).numpy()),
            pa.py_buffer(bytearray(b"ignored")),
        ],
    )
    keys, valid = pack(StringTensor.from_arrow(array, device=device))
    assert not keys.any()
    assert not valid.any()


def test_pack_rejects_narrow_width() -> None:
    values = StringTensor.from_list(["abcdefgX", "abcdefgY"])
    with pytest.raises(ValueError, match="byte bound"):
        pack(values, 7)


def test_pack_unknown_storage_bound() -> None:
    values = StringTensor(
        data=torch.tensor(list(b"abcdef"), dtype=torch.uint8),
        offset=torch.tensor([0, 3, 6]),
        valid=None,
        size=(2,),
    )
    assert values._max_length is None
    for width in (None, 6):
        with pytest.raises(ValueError, match="known byte bound"):
            pack(values, width)


@pytest.mark.parametrize(
    ("length", "bound"),
    [(0, 0), (1, 1), (3, 3), (4, 7), (127, 127), (128, 255)],
)
def test_bound_rounds_up(length: int, bound: int) -> None:
    values = StringTensor.from_list(["a" * length, ""])
    assert values._max_length == bound


@withCUDA
@pytest.mark.parametrize("bound", [None, 0, 15])
def test_bound_provenance(device: torch.device, bound: int | None) -> None:
    texts = ["", None] if bound == 0 else ["abcdefgh", "é"]
    # A non-null Arrow slice retains the longest string's backing storage.
    array = pa.array(texts)
    if bound == 15:
        array = pa.array(["x" * 12, *texts]).slice(1)
    values = StringTensor.from_arrow(array, device=device)
    if bound is None:
        values = StringTensor(
            data=values._data,
            offset=values._offset,
            valid=values._valid,
            size=values.size(),
        )
    compiled = torch.compile(lambda value: value.detach(), fullgraph=True)
    for copied, expected in (
        (values[:], texts),
        (values[::2], texts[::2]),
        (values[torch.tensor([1, 0], device=device)], texts[::-1]),
        (values.clone(), texts),
        (values.detach(), texts),
        (values.to(device, copy=True), texts),
        (compiled(values), texts),
        (pickle.loads(pickle.dumps(values)), texts),
    ):
        copied = cast(StringTensor, copied)
        assert copied._max_length == bound
        assert copied.tolist() == expected
    compiled_pack = torch.compile(pack, fullgraph=True)
    for suffix in ("", "x" * 16):
        extra = StringTensor.from_list([suffix], device=device)
        joined = cast(StringTensor, torch.cat((values, extra)))
        new_bound = None if bound is None else max(bound, 31 if suffix else 0)
        assert joined._max_length == new_bound
        if new_bound is None:
            continue
        keys, valid = compiled_pack(joined)
        expected = [*texts, suffix]
        assert torch.equal(keys.cpu(), _reference_keys(expected, new_bound))
        assert valid.tolist() == [text is not None for text in expected]
