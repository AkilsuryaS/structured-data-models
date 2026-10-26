# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NullableTensor,
    StringTensor,
    TableTensor,
    VarLenTensor,
)
from sdm.testing import onlyCUDA


@onlyCUDA
def test_record_stream() -> None:
    device = torch.device("cuda")
    tensors = (
        CategoricalTensor(
            code=torch.tensor([[0]], device=device),
            categories=(torch.tensor([1], device=device),),
        ),
        ColumnarTensor((torch.tensor([1], device=device),)),
        NullableTensor.from_list([1, None], device=device),
        StringTensor.from_list(["hi", None], device=device),
        TableTensor.from_tensor(torch.tensor([[1.0]], device=device)),
        VarLenTensor.from_list([[1], None], device=device),
    )
    stream = torch.cuda.Stream()

    for tensor in tensors:
        assert tensor.record_stream(stream) is None


def _stream_tensor(kind: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    data = torch.arange(4096, device="cuda")
    valid = data.remainder(3).ne(0)
    if kind == "categorical":
        code = data.remainder(4096).view(-1, 1)
        return CategoricalTensor(code, (data,)), (code, data)
    if kind == "columnar":
        return ColumnarTensor((data, NullableTensor(data, valid))), (
            data,
            valid,
        )
    if kind == "nullable":
        return NullableTensor(data, valid), (data, valid)
    if kind in ("var_len", "string"):
        offset = torch.arange(4097, device="cuda")
        cls = StringTensor if kind == "string" else VarLenTensor
        if kind == "string":
            data = data.remainder(26).add(97).to(torch.uint8)
        return cls(data, offset, valid, size=(4096,)), (data, offset, valid)
    assert kind == "table"
    return TableTensor(
        numerical=NullableTensor(data.view(-1, 1), valid.view(-1, 1))
    ), (data, valid)


@onlyCUDA
@pytest.mark.usefixtures("compilation")
@pytest.mark.parametrize(
    "kind",
    ["categorical", "columnar", "nullable", "var_len", "string", "table"],
)
def test_record_stream_fullgraph_lifetime(kind: str) -> None:
    tensor, sources = _stream_tensor(kind)
    expected = tuple(source.cpu() for source in sources)
    outputs = tuple(
        torch.empty_like(value, pin_memory=True) for value in expected
    )
    specs = tuple((source.shape, source.dtype) for source in sources)
    stream = torch.cuda.Stream()

    def record(inp: torch.Tensor, stream: torch.cuda.Stream) -> None:
        inp.record_stream(stream)

    compiled_record = torch.compile(record, fullgraph=True)
    compiled_record(tensor, stream)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Keep copies pending while the allocation stream reuses storage.
        torch.cuda._sleep(50_000_000)
        for output, source in zip(outputs, sources):
            output.copy_(source, non_blocking=True)
    del tensor, sources, source
    replacements = [
        torch.full(shape, 0, dtype=dtype, device="cuda")
        for shape, dtype in specs
    ]
    stream.synchronize()
    for output, value in zip(outputs, expected):
        torch.testing.assert_close(output, value)
    del replacements
