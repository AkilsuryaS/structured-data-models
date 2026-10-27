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
@pytest.mark.parametrize("width", [0, 7, 8, 14, 15])
def test_pack_keys(device: torch.device, compiled: bool, width: int) -> None:
    texts = (
        ["", None]
        if width == 0
        else ["", "\0", "a", "a\0", "é", "🙂", "a" * width, None]
    )
    values = StringTensor.from_list(texts, device=device)
    fn = torch.compile(pack, fullgraph=True) if compiled else pack
    keys, valid = fn(values, width)
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
    values = StringTensor(
        data=torch.tensor(list(b"ignored"), dtype=torch.uint8, device=device),
        offset=torch.tensor([0, 7], device=device),
        valid=torch.tensor([False], device=device),
        size=(1,),
    )
    keys, valid = pack(values)
    assert not keys.any()
    assert not valid.any()


def test_pack_rejects_narrow_width() -> None:
    values = StringTensor.from_list(["abcdefgX", "abcdefgY"])
    with pytest.raises(ValueError, match="byte bound"):
        pack(values, 7)


@withCUDA
def test_byte_bound_after_copy_and_selection(device: torch.device) -> None:
    values = StringTensor.from_arrow(
        pa.array(["xxxxxxxxxxxx", "é", "", "a"]).slice(1)
    )
    # The Arrow slice still exposes preceding storage, so its bound covers it.
    for selected in (
        values[::2],
        values[torch.tensor([2, 0])],
        values.clone(),
        values.detach(),
        values.to(device),
    ):
        selected = cast(StringTensor, selected)
        keys, valid = pack(selected)
        assert torch.equal(keys.cpu(), _reference_keys(selected.tolist(), 12))
        assert valid.tolist() == [
            text is not None for text in selected.tolist()
        ]


@withCUDA
def test_pack_concatenation(device: torch.device) -> None:
    left = StringTensor.from_list(["a", "é"], device=device)
    right = StringTensor.from_list([None, "12345678"], device=device)
    values = cast(StringTensor, torch.cat((left, right)))
    keys, valid = torch.compile(pack, fullgraph=True)(values)
    assert torch.equal(
        keys.cpu(), _reference_keys(["a", "é", None, "12345678"], 8)
    )
    assert valid.tolist() == [True, True, False, True]


def test_pack_unknown_storage_bound() -> None:
    values = StringTensor(
        data=torch.tensor(list(b"abcdef"), dtype=torch.uint8),
        offset=torch.tensor([0, 3, 6]),
        valid=None,
        size=(2,),
    )
    assert values._max_length is None
    assert torch.equal(pack(values)[0], _reference_keys(["abc", "def"], 6))
    with pytest.raises(ValueError, match="byte bound"):
        pack(values, 3)


@withCUDA
@pytest.mark.parametrize("bound", [None, 0, 8])
def test_bound_provenance(device: torch.device, bound: int | None) -> None:
    texts = ["", None] if bound == 0 else ["abcdefgh", "é"]
    values = StringTensor.from_list(texts, device=device)
    if bound is None:
        values = StringTensor(
            data=values._data,
            offset=values._offset,
            valid=values._valid,
            size=values.size(),
        )
    compiled = torch.compile(lambda value: value.detach(), fullgraph=True)
    for copied in (
        values[:],
        values.clone(),
        values.detach(),
        values.to(device),
        compiled(values),
        pickle.loads(pickle.dumps(values)),
    ):
        copied = cast(StringTensor, copied)
        assert copied._max_length == bound
        assert copied.tolist() == texts
    concatenated = cast(
        StringTensor,
        torch.cat((values, StringTensor.from_list([""], device=device))),
    )
    assert concatenated._max_length == bound
    assert concatenated.tolist() == [*texts, ""]
