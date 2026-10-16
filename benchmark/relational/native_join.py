# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact int64 inner join using hashes only to generate candidates.

No uniqueness claim is made for the hash; every candidate is checked against
all original key columns and validity masks. Output order is unspecified.
"""

import argparse
import random

import torch
from torch import Tensor


def _row_valid(keys: Tensor, valid: Tensor | None) -> Tensor:
    if valid is None:
        return torch.ones(keys.shape[0], dtype=torch.bool, device=keys.device)
    return valid.all(dim=1) if valid.ndim == 2 else valid


def _logical_rshift(x: Tensor, bits: int) -> Tensor:
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def _hash_rows(keys: Tensor) -> Tensor:
    # Wraparound affects the hash only, never the exact key comparison.
    h = torch.zeros(keys.shape[0], dtype=torch.int64, device=keys.device)
    for col in range(keys.shape[1]):
        x = keys[:, col]
        x = (x ^ _logical_rshift(x, 30)) * -4658895280553007687
        x = (x ^ _logical_rshift(x, 27)) * -7723592293110705685
        x = x ^ _logical_rshift(x, 31)
        h = (h ^ x) * -7046029254386353131
        h = h ^ _logical_rshift(h, 29)
    return h


def native_join(
    left: Tensor,
    right: Tensor,
    left_valid: Tensor | None = None,
    right_valid: Tensor | None = None,
    *,
    force_constant_hash: bool = False,
) -> tuple[Tensor, Tensor]:
    """Join [N,K]/[M,K] int64 tensors; masks are [N,K]/[M,K] or row masks.

    Requires int64 keys, matching positive K, and one device.
    Returns the original left/right row indices, including duplicate matches.
    This prototype is independent of SDM's current join backend.
    Worst-case collisions can generate N*M candidates, all verified exactly.
    """
    if left.shape[0] == 0 or right.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.int64, device=left.device)
        return empty, empty.clone()
    lv = _row_valid(left, left_valid)
    rv = _row_valid(right, right_valid)
    if force_constant_hash:
        lh = torch.zeros(left.shape[0], dtype=torch.int64, device=left.device)
        rh = torch.zeros(
            right.shape[0], dtype=torch.int64, device=right.device
        )
    else:
        lh = _hash_rows(left)
        rh = _hash_rows(right)
    # Separate invalid-side tags reduce null/null candidate blowups. The tags
    # may coincide with genuine hashes; final validity checks remain mandatory.
    lh = torch.where(lv, lh, -1)
    rh = torch.where(rv, rh, -2)
    # Keep right length shape-backed: do not compact valid rows before sorting.
    sorted_hash, order = torch.sort(rh, stable=True)
    lo = torch.searchsorted(sorted_hash, lh, side="left")
    hi = torch.searchsorted(sorted_hash, lh, side="right")
    counts = hi - lo
    lpos = torch.repeat_interleave(counts)
    starts = torch.cumsum(counts, 0) - counts
    rsorted = (
        lo[lpos]
        + torch.arange(lpos.numel(), device=left.device)
        - starts[lpos]
    )
    rpos = order[rsorted]
    exact = lv[lpos] & rv[rpos]
    for col in range(left.shape[1]):
        exact = exact & left[lpos, col].eq(right[rpos, col])
    return lpos[exact], rpos[exact]


def _selftest() -> None:
    rng = random.Random(1337)
    extrema = [
        -(1 << 63),
        (1 << 63) - 1,
        -1,
        0,
        1,
        -(1 << 53),
        1 << 53,
        (1 << 53) + 1,
    ]
    checked = 0
    for k in (1, 2, 4):
        for n, m in [(0, 0), (0, 5), (6, 0), (1, 1), (9, 12), (11, 10)]:
            left = torch.tensor(
                [[rng.choice(extrema) for _ in range(k)] for _ in range(n)],
                dtype=torch.int64,
            ).reshape(n, k)
            right = torch.tensor(
                [[rng.choice(extrema) for _ in range(k)] for _ in range(m)],
                dtype=torch.int64,
            ).reshape(m, k)
            if n and m:
                right[0] = left[0]
                if m > 1:
                    right[1] = left[0]
            for mask_mode in ("none", "column", "row", "invalid"):
                if mask_mode == "none":
                    lv = rv = None
                elif mask_mode == "row":
                    lv = torch.tensor(
                        [rng.random() > 0.2 for _ in range(n)],
                        dtype=torch.bool,
                    )
                    rv = torch.tensor(
                        [rng.random() > 0.2 for _ in range(m)],
                        dtype=torch.bool,
                    )
                else:
                    lv = torch.tensor(
                        [
                            [
                                mask_mode != "invalid" and rng.random() > 0.2
                                for _ in range(k)
                            ]
                            for _ in range(n)
                        ],
                        dtype=torch.bool,
                    ).reshape(n, k)
                    rv = torch.tensor(
                        [
                            [
                                mask_mode != "invalid" and rng.random() > 0.2
                                for _ in range(k)
                            ]
                            for _ in range(m)
                        ],
                        dtype=torch.bool,
                    ).reshape(m, k)
                lvalid, rvalid = _row_valid(left, lv), _row_valid(right, rv)
                expected = sorted(
                    (i, j)
                    for i in range(n)
                    for j in range(m)
                    if lvalid[i]
                    and rvalid[j]
                    and torch.equal(left[i], right[j])
                )
                for collide in (False, True):
                    li, ri = native_join(
                        left, right, lv, rv, force_constant_hash=collide
                    )
                    got = sorted(zip(li.tolist(), ri.tolist()))
                    assert got == expected, (
                        k,
                        n,
                        m,
                        mask_mode,
                        collide,
                        expected,
                        got,
                    )
                    checked += 1
    # Explicit nonmatching composite rows with equal forced hashes must reject.
    left = torch.tensor([[1, 2], [1, 2], [2, 1]], dtype=torch.int64)
    right = torch.tensor([[1, 3], [2, 1], [1, 2], [1, 2]], dtype=torch.int64)
    li, ri = native_join(left, right, force_constant_hash=True)
    assert sorted(zip(li.tolist(), ri.tolist())) == [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 1),
    ]
    print(
        torch.__version__,
        "PASS",
        checked + 1,
        "eager exact-oracle cases including forced hash collisions",
        flush=True,
    )


def _compile_check() -> None:
    # Variable-length matches require dynamic-output capture on PyTorch 2.7.
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    for fullgraph in (False, True):
        for constant in (False, True):
            torch._dynamo.reset()
            fn = torch.compile(
                native_join,
                backend="inductor",
                fullgraph=fullgraph,
                dynamic=True,
            )
            count = 0
            for n in (9, 0, 1):
                for m in (9, 0, 1):
                    left = torch.arange(n * 2, dtype=torch.long).reshape(n, 2)
                    right = torch.arange(m * 2, dtype=torch.long).reshape(m, 2)
                    if n and m:
                        left[0] = torch.tensor([-(1 << 63), (1 << 63) - 1])
                        right[0] = left[0]
                        if m > 1:
                            right[1] = left[0]
                    lv = torch.ones((n, 2), dtype=torch.bool)
                    rv = torch.ones((m, 2), dtype=torch.bool)
                    if n > 2:
                        lv[2, 0] = False
                    if m > 3:
                        rv[3, 1] = False
                    lvalid, rvalid = (
                        _row_valid(left, lv),
                        _row_valid(right, rv),
                    )
                    expected = sorted(
                        (i, j)
                        for i in range(n)
                        for j in range(m)
                        if lvalid[i]
                        and rvalid[j]
                        and torch.equal(left[i], right[j])
                    )
                    li, ri = fn(
                        left, right, lv, rv, force_constant_hash=constant
                    )
                    assert sorted(zip(li.tolist(), ri.tolist())) == expected, (
                        constant,
                        n,
                        m,
                    )
                    count += 1
            print(
                "PASS fullgraph=",
                fullgraph,
                "dynamic=True constant_hash=",
                constant,
                "cases=",
                count,
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        _selftest()
        if args.compile:
            _compile_check()
