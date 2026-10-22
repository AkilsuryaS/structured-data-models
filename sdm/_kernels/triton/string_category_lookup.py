# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _lookup_kernel(
    input_data,
    input_offsets,
    input_valid,
    fitted_data,
    fitted_offsets,
    fitted_valid,
    output,
    input_stride: tl.constexpr,
    input_offset: tl.constexpr,
    fitted_stride: tl.constexpr,
    fitted_offset: tl.constexpr,
    fitted_count: tl.constexpr,
    output_offset: tl.constexpr,
    has_input_valid: tl.constexpr,
    has_fitted_valid: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    left = input_offset + row * input_stride
    start = tl.load(input_offsets + left)
    end = tl.load(input_offsets + left + 1)
    lane = tl.arange(0, block)
    right = fitted_offset + lane * fitted_stride
    active = lane < fitted_count
    right_start = tl.load(fitted_offsets + right, active, other=0)
    right_end = tl.load(fitted_offsets + right + 1, active, other=0)
    equal = active & (end - start == right_end - right_start)
    if has_input_valid:
        equal &= tl.load(input_valid + left)
    if has_fitted_valid:
        equal &= tl.load(fitted_valid + right, active, other=False)
    position = 0
    # Compare exact bytes; no hash collisions or encoding assumptions.
    while (position < end - start) & (tl.sum(equal.to(tl.int32), 0) > 0):
        value = tl.load(input_data + start + position)
        other = tl.load(fitted_data + right_start + position, equal, other=0)
        equal &= value == other
        position += 1
    index = tl.max(tl.where(equal, lane, -1), 0)
    # Duplicate right matches have backend-dependent indexed-write ordering.
    # Preserve that behavior by asking the original join to handle this call.
    ambiguous = tl.sum(equal.to(tl.int32), 0) > 1
    tl.store(output + output_offset + row, tl.where(ambiguous, -2, index))


def string_category_lookup(
    inputs: list[Tensor | None],
    input_layout: list[int],
    fitted: list[Tensor | None],
    fitted_layout: list[int],
    codes: Tensor,
) -> Tensor | None:
    """Bounded exact comparison for small GPU vocabularies, else defer."""
    if not codes.is_cuda:
        return None
    # Bound both category-pair and byte work using host-visible metadata only.
    # Eligibility creates no pairwise intermediate and reads no device scalars.
    work = 0
    for pair in range(len(input_layout) // 3):
        count = fitted_layout[3 * pair]
        data = cast(Tensor, inputs[3 * pair])
        if count > 256:
            return None
        right_data = cast(Tensor, fitted[3 * pair])
        byte_bound = min(data.numel(), right_data.numel())
        # Also bound the longest possible serial byte loop in each CTA.
        if byte_bound > 4096:
            return None
        work += count * input_layout[3 * pair] * (1 + byte_bound)
    if work > 1 << 22:
        return None
    output = codes.new_empty((sum(input_layout[::3]),))
    output_offset = 0
    with torch.cuda.device(codes.device):
        for pair in range(len(input_layout) // 3):
            n, stride, offset = input_layout[3 * pair : 3 * pair + 3]
            count, fitted_stride, fitted_offset = fitted_layout[
                3 * pair : 3 * pair + 3
            ]
            left = inputs[3 * pair : 3 * pair + 3]
            right = fitted[3 * pair : 3 * pair + 3]
            if n:
                cast(Any, _lookup_kernel)[(n,)](
                    *left,
                    *right,
                    output,
                    stride,
                    offset,
                    fitted_stride,
                    fitted_offset,
                    count,
                    output_offset,
                    left[2] is not None,
                    right[2] is not None,
                    max(32, triton.next_power_of_2(count)),
                    num_warps=1,
                )
            output_offset += n
    # One scalar synchronization preserves the original duplicate semantics.
    if output.numel() and output.min().item() == -2:
        return None
    return output
