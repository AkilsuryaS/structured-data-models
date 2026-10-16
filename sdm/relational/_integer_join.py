# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact integer join with pure hash buckets or lexicographic groups.

Only true matches are expanded. torch.cond selects fixed-shape metadata,
with linear workspace for fixed key width even under adversarial collisions.
Output order is unspecified.
"""

import torch
from torch import Tensor


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


def _join_integer(
    left: Tensor,
    right: Tensor,
    left_valid: Tensor,
    right_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Join int64 keys with row validity masks and bounded workspace.

    Requires matching positive K and one device. Duplicate matches retain their
    multiplicities. Temporary storage is O((N+M)*K + true_output).
    Hash equality never establishes a match without comparing original keys.
    """
    # An invalid row keeps each physical side nonempty, even when a preceding
    # filter produces a data-dependent zero row count. Original row IDs stay
    # unchanged; invalid rows have zero match counts and are never returned.
    left = torch.cat((left, left.new_zeros((1, left.shape[1]))))
    right = torch.cat((right, right.new_zeros((1, right.shape[1]))))
    lv = torch.cat((left_valid, left_valid.new_zeros(1)))
    rv = torch.cat((right_valid, right_valid.new_zeros(1)))
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
