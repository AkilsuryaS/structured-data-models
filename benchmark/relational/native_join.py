# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact int64 inner join using hashes only to generate candidates.

No uniqueness claim is made for the hash; every candidate is checked against
all original key columns and validity masks. Output order is unspecified.
"""

import argparse
import random
from collections.abc import Callable

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
    """Join int64 keys by generating and verifying hash-match candidates.

    Accepts row or column validity masks and preserves duplicate matches.
    Collision candidates can require quadratic temporary memory; use
    native_join_bounded when that allocation must be avoided.
    """
    n, m = left.shape[0], right.shape[0]
    if n == 0 or m == 0:
        return left.new_empty((0,)), right.new_empty((0,))
    lv = _row_valid(left, left_valid)
    rv = _row_valid(right, right_valid)
    if force_constant_hash:
        lh = left.new_zeros((n,))
        rh = right.new_zeros((m,))
    else:
        lh = _hash_rows(left)
        rh = _hash_rows(right)
    lh = torch.where(lv, lh, -1)
    rh = torch.where(rv, rh, -2)
    sorted_hash, order = torch.sort(rh, stable=True)
    lo = torch.searchsorted(sorted_hash, lh, side="left")
    # Scan right-group ends only when the right side is no larger than left.
    # Otherwise retain two searches instead of adding a larger right-side scan.
    if m <= n:
        last = torch.cat(
            (
                sorted_hash[1:] != sorted_hash[:-1],
                torch.ones((1,), dtype=torch.bool, device=left.device),
            )
        )
        end_flags = torch.where(
            last, torch.arange(1, m + 1, device=left.device), m
        )
        runend = end_flags.flip(0).cummin(0).values.flip(0)
        safe_lo = lo.clamp(max=m - 1)
        match = (lo < m) & (sorted_hash[safe_lo] == lh) & lv
        counts = torch.where(match, runend[safe_lo] - lo, 0)
    else:
        hi = torch.searchsorted(sorted_hash, lh, side="right")
        counts = (hi - lo) * lv.to(torch.int64)
    lpos = torch.repeat_interleave(counts)
    starts = counts.cumsum(0) - counts
    rsorted = (
        lo[lpos]
        + torch.arange(lpos.numel(), device=left.device)
        - starts[lpos]
    )
    rpos = order[rsorted]
    exact = rv[rpos]
    for col in range(left.shape[1]):
        exact = exact & left[lpos, col].eq(right[rpos, col])
    # One data-dependent output extent, shared by both index tensors.
    keep = exact.nonzero(as_tuple=True)[0]
    return lpos[keep], rpos[keep]


def native_join_bounded(
    left: Tensor,
    right: Tensor,
    left_valid: Tensor | None = None,
    right_valid: Tensor | None = None,
    *,
    force_constant_hash: bool = False,
) -> tuple[Tensor, Tensor]:
    """Join [N,K]/[M,K] int64 keys, accepting row or column validity masks.

    Requires matching positive K and one device. Duplicate matches retain their
    multiplicities. Temporary storage is O((N+M)*K + true_output).
    Hash equality
    never establishes a match without exact comparison of original key tuples.
    """
    nl, nr = left.shape[0], right.shape[0]
    if nl == 0 or nr == 0:
        empty = torch.empty(0, dtype=torch.int64, device=left.device)
        return empty, empty.clone()

    lv = _row_valid(left, left_valid)
    rv = _row_valid(right, right_valid)
    if force_constant_hash:
        left_hash = torch.zeros(nl, dtype=torch.int64, device=left.device)
        right_hash = torch.zeros(nr, dtype=torch.int64, device=right.device)
    else:
        left_hash = _hash_rows(left)
        right_hash = _hash_rows(right)

    sorted_hash, right_perm = torch.sort(right_hash, stable=True)
    sorted_right = right[right_perm]
    # A bucket is pure when all its right key tuples are identical. Left-side
    # collisions need no fallback: the fast branch compares an exact tuple.
    impure = (
        (sorted_hash[1:] == sorted_hash[:-1])
        & (sorted_right[1:] != sorted_right[:-1]).any(dim=1)
    ).any()

    def exact_metadata(
        left: Tensor,
        right: Tensor,
        left_hash: Tensor,
        sorted_hash: Tensor,
        right_perm: Tensor,
        lv: Tensor,
        rv: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        nl, nr = left.shape[0], right.shape[0]
        n = nl + nr
        keys = torch.cat((left, right), dim=0)
        order = torch.arange(n, device=left.device)
        for col in range(keys.shape[1] - 1, -1, -1):
            order = order[torch.argsort(keys[order, col], stable=True)]
        ordered_keys = keys[order]
        boundary = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=left.device),
                (ordered_keys[1:] != ordered_keys[:-1]).any(dim=1),
            )
        )
        group = boundary.to(torch.int64).cumsum(0) - 1
        take_right = (order >= nl) & torch.cat((lv, rv))[order]
        valid = take_right.to(torch.int64)
        before = valid.cumsum(0) - valid
        group_counts = torch.zeros(
            n, dtype=torch.int64, device=left.device
        ).scatter_add(0, group, valid)
        group_starts = group_counts.cumsum(0) - group_counts
        # Non-right rows write only to an unused dump slot.
        destination = torch.where(take_right, before, nr)
        right_order = torch.zeros(
            nr + 1, dtype=torch.int64, device=left.device
        ).scatter(0, destination, order - nl)[:nr]
        inverse = torch.empty_like(order).scatter(
            0, order, torch.arange(n, device=left.device)
        )
        left_group = group[inverse[:nl]]
        counts = group_counts[left_group] * lv.to(torch.int64)
        starts = group_starts[left_group]
        return counts, starts, right_order

    def pure_metadata(
        left: Tensor,
        right: Tensor,
        left_hash: Tensor,
        sorted_hash: Tensor,
        right_perm: Tensor,
        lv: Tensor,
        rv: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        nr = right.shape[0]
        boundary = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=right.device),
                sorted_hash[1:] != sorted_hash[:-1],
            )
        )
        group = boundary.to(torch.int64).cumsum(0) - 1
        valid = rv[right_perm].to(torch.int64)
        before = valid.cumsum(0) - valid
        group_counts = torch.zeros(
            nr, dtype=torch.int64, device=right.device
        ).scatter_add(0, group, valid)
        destination = torch.where(rv[right_perm], before, nr)
        right_order = torch.zeros(
            nr + 1, dtype=torch.int64, device=right.device
        ).scatter(0, destination, right_perm)[:nr]

        lower = torch.searchsorted(sorted_hash, left_hash, side="left")
        safe = lower.clamp(max=nr - 1)
        representative = right[right_perm[safe]]
        exact = (
            lv
            & (lower < nr)
            & (sorted_hash[safe] == left_hash)
            & (left == representative).all(dim=1)
        )
        counts = torch.where(exact, group_counts[group[safe]], 0)
        # Starts are unused for left rows with zero matches.
        starts = before[safe]
        return counts, starts, right_order

    counts, starts, right_order = torch.cond(
        impure,
        exact_metadata,
        pure_metadata,
        (left, right, left_hash, sorted_hash, right_perm, lv, rv),
    )
    # Both branches return fixed [N], [N], [M] metadata. This sole expansion is
    # proportional to true output cardinality, including under constant hashes.
    left_index = torch.repeat_interleave(counts)
    output_starts = counts.cumsum(0) - counts
    offset = starts - output_starts
    right_positions = offset[left_index] + torch.arange(
        left_index.numel(), device=left.device
    )
    return left_index, right_order[right_positions]


def _collision_check(join: Callable[..., tuple[Tensor, Tensor]]) -> None:
    # Different keys with the same real hash, exercising the production path.
    a = (-(1 << 63), (1 << 63) - 1)
    b = ((1 << 63) - 1, -2044408044210139470)
    hashes = _hash_rows(torch.tensor([a, b], dtype=torch.int64))
    assert hashes[0] == hashes[1]
    c, d, e = (1, 2), (3, 4), (5, 6)
    cases = [
        ([a, c, d], [c, d, e]),
        ([a, c, d], [a, b, e]),
        ([a, a, a], [a, a, a]),
        ([a, a, a], [b, b, b]),
        ([a, c, d], [a, d, e]),
        ([a, c, d], [a, b, e]),
    ]
    for left_rows, right_rows in cases:
        left = torch.tensor(left_rows, dtype=torch.int64)
        right = torch.tensor(right_rows, dtype=torch.int64)
        li, ri = join(left, right)
        expected = [
            (i, j)
            for i in range(3)
            for j in range(3)
            if left_rows[i] == right_rows[j]
        ]
        assert sorted(zip(li.tolist(), ri.tolist())) == expected
    print("PASS", len(cases), "real hash-collision cases", flush=True)


def _selftest(join: Callable[..., tuple[Tensor, Tensor]]) -> None:
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
                    li, ri = join(
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
    li, ri = join(left, right, force_constant_hash=True)
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


def _compile_check(join: Callable[..., tuple[Tensor, Tensor]]) -> None:
    # Variable-length matches require dynamic-output capture on PyTorch 2.7.
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    # Empty, singleton, and nonempty shapes need distinct compiled variants.
    # Raise the budget only for this deliberately varied validation matrix.
    torch._dynamo.config.cache_size_limit = 64
    for fullgraph in (False, True):
        for constant in (False, True):
            torch._dynamo.reset()
            fn = torch.compile(
                join,
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

        torch._dynamo.reset()
        _collision_check(
            torch.compile(
                join, backend="inductor", fullgraph=fullgraph, dynamic=True
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bounded", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    join = native_join_bounded if args.bounded else native_join
    with torch.inference_mode():
        _selftest(join)
        _collision_check(join)
        if args.compile:
            _compile_check(join)
