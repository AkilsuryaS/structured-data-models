# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from sdm.models.kumo.tabular.memory import KumoMemoryEstimator

pytest.importorskip("scipy")

from benchmark.tabular.memory import equation, synthetic


def calibration() -> dict[str, Any]:
    return {
        "settings": {"query_rows": 1024},
        "models": {
            "small": {
                "model_bytes": 100,
                "checkpoint_sha256": "abc",
                "equations": {
                    name: {"coefficients": [7, 5, 3, 2]}
                    for name in [
                        "encoder",
                        "icl",
                        "predict",
                        "encoder_reserved",
                        "icl_reserved",
                        "predict_reserved",
                    ]
                },
                "safety_fraction": 0.1,
                "range": {
                    "min_rows": 8,
                    "max_rows": 16,
                    "min_columns": 4,
                    "max_columns": 8,
                },
            }
        },
    }


def test_memory_cardinality_bounds_and_checkpoint_identity() -> None:
    estimator = KumoMemoryEstimator(calibration())
    lower = estimator.estimate("small", 12, 2, 2, high_cardinality_columns=0)
    upper = estimator.estimate("small", 12, 2, 2)
    assert lower["predict_peak_bytes"] < upper["predict_peak_bytes"]
    assert upper["processed_columns"] == 6
    assert lower["processed_columns"] == 4
    assert estimator.estimate("abc", 12, 2, 2) == upper
    assert (
        estimator.estimate("small", 24, 2, 2)["inside_calibration_range"]
        is False
    )
    assert upper["conservative_peak_bytes"] >= upper["peak_allocated_bytes"]
    assert (
        estimator.estimate("small", 12, 100, 300)["processed_columns"] == 500
    )
    with pytest.raises(ValueError, match="no memory calibration"):
        estimator.estimate("unknown", 12, 2, 2)


def test_memory_equation_recovers_unseen_shape() -> None:
    records = [
        {
            "rows": r,
            "processed_columns": c,
            "model_bytes": 100,
            "peaks": {"predict": 100 + 7 + 5 * r + 3 * c + 2 * r * c},
        }
        for r in [8, 16, 32]
        for c in [4, 8, 12]
    ]
    bundle = calibration()
    bundle["models"]["small"]["equations"]["predict"] = equation(
        records, "predict"
    )
    estimate = KumoMemoryEstimator(bundle).estimate("small", 12, 6, 0)
    assert estimate["predict_peak_bytes"] == pytest.approx(
        100 + 7 + 5 * 12 + 3 * 6 + 2 * 12 * 6, abs=1
    )


def test_synthetic_data_preserves_types_and_missingness() -> None:
    x, y = synthetic(1024, 3, 4, 64, 17)
    assert x.shape == (2048, 7)
    assert y.shape == (1024, 1)
    assert torch.isnan(x.numerical).any()
    assert (x.categorical.code < 0).any()
    assert all(len(c) == 64 for c in x.categorical.categories)
    assert y.categorical.categories[0].tolist() == [0, 1]


def test_reserved_estimate_respects_allocation() -> None:
    bundle = calibration()
    for name, fitted in bundle["models"]["small"]["equations"].items():
        if name.endswith("_reserved"):
            fitted["coefficients"] = [0, 0, 0, 0]
    result = KumoMemoryEstimator(bundle).estimate("small", 12, 2, 2)
    assert result["peak_reserved_bytes"] >= result["peak_allocated_bytes"]
    assert (
        result["predict_peak_reserved_bytes"] >= result["predict_peak_bytes"]
    )
    assert result["fit_peak_reserved_bytes"] >= result["fit_peak_bytes"]
