# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from typing import Self, cast

import pytest
import torch
from torch import Tensor

from sdm.cache import Cache, KVCacheEntry
from sdm.models.base.model import _Prefetcher
from sdm.tensor.mixin import DeviceMixin


@pytest.mark.parametrize("fullgraph", [False, True])
def test_prefetch_resident_replay_cache_compiles(fullgraph: bool) -> None:
    tensor = torch.ones(2)
    cache = Cache(
        value=tensor,
        nested={"entry": KVCacheEntry(tensor, tensor)},
    ).freeze()

    def predict(x: Tensor) -> Tensor:
        with _Prefetcher([cache, cache], x.device, None) as caches:
            for item in caches:
                x = x + cast(Tensor, item["value"])
        return x

    compiled = torch.compile(predict, backend="eager", fullgraph=fullgraph)
    torch.testing.assert_close(compiled(tensor), tensor + 2)
    torch.testing.assert_close(compiled(tensor + 1), tensor + 3)
    assert cache.is_replaying
    torch.testing.assert_close(cast(Tensor, cache["value"]), torch.ones(2))


def test_prefetch_recording_cache_remains_independent() -> None:
    cache = Cache(value=torch.ones(2))
    with _Prefetcher([cache], torch.device("cpu"), None) as caches:
        for item in caches:
            item["new"] = 1
    assert "new" not in cache


class _CustomTransfer(DeviceMixin):
    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor

    def _tensors(self) -> Iterator[Tensor]:
        yield self.tensor

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(fn(self.tensor) + 1)


def test_prefetch_preserves_custom_device_conversion() -> None:
    original = _CustomTransfer(torch.ones(2))
    cache = Cache(value=original).freeze()
    with _Prefetcher([cache], torch.device("cpu"), None) as caches:
        for item in caches:
            converted = cast(_CustomTransfer, item["value"])
            torch.testing.assert_close(converted.tensor, torch.full((2,), 2.0))
    torch.testing.assert_close(original.tensor, torch.ones(2))


def test_prefetch_normalizes_container_subclasses() -> None:
    class Metadata(tuple):
        pass

    cache = Cache(value=Metadata((torch.ones(2),))).freeze()
    with _Prefetcher([cache], torch.device("cpu"), None) as caches:
        for item in caches:
            assert type(item["value"]) is tuple


def test_prefetch_converts_nonresident_replay_cache() -> None:
    cache = Cache(value=torch.ones(2)).freeze()
    with _Prefetcher([cache], torch.device("meta"), None) as caches:
        for item in caches:
            assert item.is_replaying
            assert cast(Tensor, item["value"]).device.type == "meta"
    assert cast(Tensor, cache["value"]).device.type == "cpu"
