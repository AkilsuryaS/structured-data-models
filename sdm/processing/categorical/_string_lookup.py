# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match string categories with compact keys and exact byte comparison."""

from collections.abc import Callable

import torch
from torch import Tensor

from sdm import StringTensor


def _select(
    predicate: Tensor,
    when_true: Callable[[], Tensor],
    when_false: Callable[[], Tensor],
) -> Tensor:
    if torch.compiler.is_compiling():
        return torch.cond(predicate, when_true, when_false, ())
    # Structured ops invoke Dynamo in eager. Use the same scalar decisions
    # directly here; compiled execution retains structured control flow.
    return when_true() if bool(predicate) else when_false()


def _different(
    left_data: Tensor,
    left_start: Tensor,
    right_data: Tensor,
    right_start: Tensor,
    length: Tensor,
) -> Tensor:
    capacity = left_data.numel()
    if capacity == 0 or right_data.numel() == 0:
        return length != 0
    first = torch.cat(
        tensors=(length.new_zeros(1), length.cumsum(0, dtype=length.dtype))
    )
    left_start = torch.cat((left_start, left_start.new_zeros(1)))
    right_start = torch.cat((right_start, right_start.new_zeros(1)))
    different = torch.zeros(
        size=(length.numel() + 1,), dtype=length.dtype, device=left_data.device
    )

    chunk = (capacity + 7) // 8
    base = torch.arange(chunk, dtype=length.dtype, device=left_data.device)
    count = length.numel()

    def body(part: Tensor | int, carry: Tensor) -> tuple[Tensor | int, Tensor]:
        byte = base + part * chunk
        owner = (
            torch.searchsorted(
                sorted_sequence=first,
                input=byte,
                right=True,
                out_int32=length.dtype == torch.int32,
            )
            - 1
        )
        local = byte - first[owner]
        real = owner < count
        left_index = (left_start[owner] + local).where(real, 0)
        right_index = (right_start[owner] + local).where(real, 0)
        differs = (
            (left_data[left_index] != right_data[right_index]) & real
        ).to(length.dtype)
        return part + 1, carry.index_add(0, owner, differs)

    if torch.compiler.is_compiling():
        counter = torch.zeros((), dtype=length.dtype, device="cpu")
        _, different = torch.while_loop(
            cond_fn=lambda part, carry: part < 8,
            body_fn=body,
            carried_inputs=(counter, different),
        )
    else:
        for part in range(8):
            _, different = body(part, different)
    return different[:-1] != 0


def _rows(
    value: StringTensor, index_dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    size = 1 if value.numel() and value.stride(0) == 0 else value.numel()
    rows = torch.arange(size, dtype=index_dtype, device=value.device)
    rows = rows * value.stride(0) + value._logical_storage_offset
    start = value._offset[rows].to(index_dtype)
    return start, value._offset[rows + 1].to(index_dtype) - start


def _pack(data: Tensor, start: Tensor, length: Tensor, words: int) -> Tensor:
    keys = torch.zeros(
        size=(length.numel(), words), dtype=torch.int64, device=data.device
    )
    if data.numel() == 0:
        return torch.cat((keys, length[:, None]), dim=1)
    position = 8 * torch.arange(words, dtype=start.dtype, device=data.device)
    index = start[:, None] + position
    shift = (length - 8 * words).clamp_min(0)
    index[:, (words + 1) // 2 :].add_(shift[:, None])
    for byte in range(8):
        if byte == 4 and words % 2:
            index[:, words // 2].add_(shift)
        present = position + byte < length[:, None]
        index.clamp_max_(data.numel() - 1)
        symbol = data[index].masked_fill_(~present, 0)
        keys.add_(symbol, alpha=1 << (8 * byte))
        index.add_(1)
    return torch.cat((keys, length[:, None]), dim=1)


def _lookup(
    left: StringTensor, right: StringTensor, dtype: torch.dtype = torch.int32
) -> Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return torch.full(left.shape, -1, dtype=dtype, device=left.device)
    left_size = 1 if left.stride(0) == 0 else left.numel()
    right_size = 1 if right.stride(0) == 0 else right.numel()
    left_data, right_data = left._data, right._data
    capacity = left_data.numel() + right_data.numel()
    words = max(
        1,
        (capacity + 8 * (left_size + right_size) - 1)
        // (8 * (left_size + right_size)),
    )
    if left._max_length is not None and right._max_length is not None:
        words = min(
            words, max(1, (max(left._max_length, right._max_length) + 7) // 8)
        )
    extent = (
        capacity + 8 * words + left._offset.numel() + right._offset.numel()
    )
    index_dtype = (
        torch.int32 if extent < torch.iinfo(torch.int32).max else torch.int64
    )
    left_start, left_length = _rows(left, index_dtype)
    right_start, right_length = _rows(right, index_dtype)
    complete = left._max_length is not None and left._max_length <= words * 8
    if (
        torch.compiler.is_compiling()
        and not complete
        and left_data is not right_data
    ):
        right_data = right_data.clone()

    def match_at(width: int, full: bool) -> Tensor:
        # Flat buffers keep a fixed layout across higher-order control flow.
        left_keys = _pack(left_data, left_start, left_length, width).flatten()
        right_keys = _pack(
            data=right_data,
            start=right_start,
            length=right_length,
            words=width,
        ).flatten()
        columns = width + 1
        weights = torch.arange(
            start=1,
            end=2 * columns,
            step=2,
            dtype=torch.int64,
            device=left_data.device,
        )
        # Position-sensitive sums select candidates; full keys decide equality.
        left_hash = (left_keys.view(left_size, columns) * weights).sum(-1)
        sorted_hash, order = (
            (right_keys.view(right_size, columns) * weights)
            .sum(-1)
            .sort(stable=True)
        )
        del weights

        def resolve(check_bytes: bool) -> Tensor:
            position = (
                torch.searchsorted(sorted_hash, left_hash, right=True) - 1
            )
            active = (position >= 0) & (
                sorted_hash[position.clamp_min(0)] == left_hash
            )
            output = torch.full(
                size=left_length.shape,
                fill_value=-1,
                dtype=dtype,
                device=left_data.device,
            )

            def cond(
                position: Tensor, output: Tensor, active: Tensor
            ) -> Tensor:
                return active.any()

            def body(
                position: Tensor, output: Tensor, active: Tensor
            ) -> tuple[Tensor, Tensor, Tensor]:
                candidate = order[position.clamp_min(0)]
                matched = active & (
                    left_keys.view(left_size, columns)
                    == right_keys.view(right_size, columns)[candidate]
                ).all(-1)
                if check_bytes:
                    count = left_length.where(
                        matched & (left_length > 8 * width), 0
                    )
                    different = _different(
                        left_data=left_data,
                        left_start=left_start,
                        right_data=right_data,
                        right_start=right_start[candidate],
                        length=count,
                    )
                    matched = matched & ~different
                next_output = torch.where(matched, candidate.to(dtype), output)
                next_position = position - (active & ~matched).to(
                    position.dtype
                )
                next_active = active & ~matched & (next_position >= 0)
                next_active = next_active & (
                    sorted_hash[next_position.clamp_min(0)] == left_hash
                )
                return next_position, next_output, next_active

            if not check_bytes:
                position, output, active = body(position, output, active)
            if torch.compiler.is_compiling():
                _, result, _ = torch.while_loop(
                    cond_fn=cond,
                    body_fn=body,
                    carried_inputs=(position, output, active),
                )
            else:
                while bool(cond(position, output, active)):
                    position, output, active = body(position, output, active)
                result = output
            return result

        if full:
            return resolve(False)
        return _select(
            predicate=(left_length <= width * 8).all(),
            when_true=lambda: resolve(False),
            when_false=lambda: resolve(True),
        )

    if complete:
        output = match_at(words, True)
    else:
        maximum = left_length.amax()
        power = 1
        while power * 2 <= words:
            power *= 2

        def shrink(width: int) -> Tensor:
            if width == 1:
                return match_at(1, True)
            return _select(
                predicate=maximum <= 8 * (width // 2),
                when_true=lambda: shrink(width // 2),
                when_false=lambda: match_at(width, True),
            )

        output = _select(
            predicate=maximum <= power * 8,
            when_true=lambda: shrink(power),
            when_false=lambda: match_at(words, False),
        )

    if right.stride(0) == 0:
        output = torch.where(output >= 0, right.numel() - 1, output)
    return output.expand(left.numel()) if left.stride(0) == 0 else output


def _native_string_lookups(
    left: tuple[StringTensor, ...],
    right: tuple[StringTensor, ...],
    dtype: torch.dtype = torch.int32,
) -> tuple[Tensor, ...]:
    return tuple(
        _lookup(query, fitted, dtype)
        for query, fitted in zip(left, right, strict=True)
    )
