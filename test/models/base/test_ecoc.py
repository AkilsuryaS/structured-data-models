# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import cast

import pytest
import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models import ECOC
from sdm.testing import withCUDA


class MyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        temperature: float = 1.0,
        cache: Cache | None = None,
    ) -> Tensor:
        probs = (x[..., y.size(-1) :, :] / temperature).softmax(dim=-1)
        if cache is not None:
            y = cast(Tensor, cache.setdefault("y", y))
        out = probs.new_zeros((*probs.shape[:-1], self.num_classes))
        return out.scatter_add(
            dim=-1,
            index=y.unsqueeze(-2).expand_as(probs),
            src=probs,
        ).log()


@withCUDA
@pytest.mark.parametrize(
    ("max_classes", "num_classes", "batch_shape"),
    [(10, 3, ()), (10, 11, ()), (10, 201, (2, 3)), (2, 12, ())],
)
def test_ecoc(
    device: torch.device,
    max_classes: int,
    num_classes: int,
    batch_shape: tuple[int, ...],
) -> None:
    model = MyModel(max_classes)
    ecoc = ECOC(max_classes=max_classes)
    forward = partial(
        ecoc, model=model, num_classes=num_classes, temperature=0.7
    )
    # Scramble the labels and leave one class absent from each context.
    num_context = num_classes - 1
    x = torch.randn(
        *batch_shape,
        num_context + 3,
        num_context,
        device=device,
        dtype=torch.float64,
    )
    y = torch.rand(*batch_shape, num_classes, device=device).argsort(dim=-1)[
        ..., :num_context
    ]

    scores = forward(x=x, y=y)
    probs = (x[..., -3:, :] / 0.7).softmax(dim=-1)
    expected = x.new_zeros((*batch_shape, 3, num_classes)).scatter(
        dim=-1,
        index=y.unsqueeze(-2).expand_as(probs),
        src=probs,
    )

    torch.testing.assert_close(scores.exp(), expected)

    cache = Cache()
    recorded = forward(x=x[..., :num_context, :], y=y, cache=cache)
    assert recorded.shape == (*batch_shape, 0, num_classes)
    cache.freeze()
    for query, expected in zip(
        x[..., num_context:, :].split(2, dim=-2),
        scores.split(2, dim=-2),
        strict=True,
    ):
        actual = forward(x=query, y=y[..., :0], cache=cache)
        torch.testing.assert_close(actual, expected)

    if num_classes > ecoc.max_classes:
        with pytest.raises(ValueError, match=r"num_classes.*cached"):
            forward(
                x=x[..., -1:, :],
                y=y[..., :0],
                num_classes=num_classes + 1,
                cache=cache,
            )


@withCUDA
def test_ecoc_members(device: torch.device) -> None:
    model = MyModel(10)
    ecoc = ECOC(max_classes=10)
    num_members, num_classes, num_context = 3, 12, 11
    x = torch.randn(
        num_members,
        num_context + 2,
        num_context,
        device=device,
        dtype=torch.float64,
    )
    y = torch.rand(num_members, num_classes, device=device).argsort(dim=-1)[
        ..., :num_context
    ]
    forward = partial(
        ecoc, model=model, num_classes=num_classes, num_members=num_members
    )
    scores = forward(
        x=x, y=y, generator=torch.Generator(device).manual_seed(0)
    )

    cache = Cache()
    recorded = forward(
        x=x[..., :num_context, :],
        y=y,
        cache=cache,
        generator=torch.Generator(device).manual_seed(0),
    )
    assert recorded.shape == (num_members, 0, num_classes)
    cache.freeze()
    replayed = forward(x=x[..., num_context:, :], y=y[..., :0], cache=cache)
    torch.testing.assert_close(replayed, scores)


class NeuralModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 10)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.linear(x[..., -2:, :]).tanh()


@withCUDA
@pytest.mark.parametrize("num_members", [1, 2])
@pytest.mark.skipif(
    tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 14),
    reason="Unbacked output strides in torch.cond require PyTorch 2.14",
)
def test_ecoc_dynamic_classes(
    device: torch.device, compilation: None, num_members: int
) -> None:
    model = NeuralModel().to(device).eval()
    ecoc = ECOC(max_classes=10)
    batch_shape = () if num_members == 1 else (num_members,)
    x = torch.randn(*batch_shape, 14, 4, device=device)

    def forward(x: Tensor, y: Tensor) -> Tensor:
        return ecoc(
            model=model,
            x=x,
            y=y,
            num_classes=y.unique().numel(),
            num_members=num_members,
        )

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(forward, backend="inductor", fullgraph=True)
        # Twelve labels let the learned count cross the head's capacity.
        for num_classes in (3, 10, 11, 12, 3):
            y = (torch.arange(12, device=device) % num_classes).expand(
                *batch_shape, 12
            )
            # Sampling must consume the same random stream as eager execution.
            torch.manual_seed(42)
            with torch.no_grad():
                expected = forward(x, y)
            expected_next = torch.rand(5, device=device)
            torch.manual_seed(42)
            with torch.no_grad():
                actual = compiled(x, y)
            actual_next = torch.rand(5, device=device)
            assert actual.shape == (*batch_shape, 2, num_classes)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                actual_next, expected_next, rtol=0, atol=0
            )


@withCUDA
@pytest.mark.parametrize(
    ("num_classes", "use_generator"), [(3, False), (3, True), (11, False)]
)
def test_ecoc_compile_static_classes(
    device: torch.device,
    compilation: None,
    num_classes: int,
    use_generator: bool,
) -> None:
    model = NeuralModel().to(device).eval()
    ecoc = ECOC(max_classes=10)
    generator = torch.Generator(device) if use_generator else None
    x = torch.randn(14, 4, device=device)
    y = torch.arange(12, device=device) % num_classes

    def forward(x: Tensor, y: Tensor) -> Tensor:
        return ecoc(model, x, y, num_classes=num_classes, generator=generator)

    torch.manual_seed(42)
    with torch.no_grad():
        expected = forward(x, y)
    expected_next = torch.rand(5, device=device)
    torch.manual_seed(42)
    with torch.no_grad():
        actual = torch.compile(forward, backend="inductor", fullgraph=True)(
            x, y
        )
    actual_next = torch.rand(5, device=device)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)
