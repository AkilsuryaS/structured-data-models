# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=sp.Detrend(threshold=0.5),
        target=sp.Detrend(threshold=0.5),
        output=sp.Sequential(
            sp.SortQuantiles(num_quantiles=9),
            sp.AverageEstimators(),
        ),
    )
