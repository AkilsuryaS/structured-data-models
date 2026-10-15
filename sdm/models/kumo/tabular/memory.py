# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empirical memory estimates for a calibrated Kumo inference configuration."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class KumoMemoryEstimator:
    """Estimate fit and cached inference peaks from a synthetic calibration.

    Args:
        calibration: JSON-compatible measurements and fitted coefficients
            produced by ``benchmark.tabular.memory``. Precision, allocator
            budget, query size, task and estimator count belong to the
            calibration; this class does not assume they are interchangeable.
    """

    def __init__(self, calibration: Mapping[str, Any]) -> None:
        self.calibration = calibration

    @classmethod
    def from_file(cls, path: str | Path) -> "KumoMemoryEstimator":
        """Load a calibration without loading model weights or using CUDA."""
        return cls(json.loads(Path(path).read_text()))

    def estimate(
        self,
        checkpoint: str,
        context_rows: int,
        numeric_columns: int,
        categorical_columns: int,
        *,
        high_cardinality_columns: int | None = None,
    ) -> dict[str, Any]:
        """Estimate allocated and reserved GPU bytes for binary inference.

        ``checkpoint`` is a calibrated model name or its checkpoint SHA256.
        Unknown categorical cardinalities assume all categorical features add
        count columns. The result also includes the low-cardinality estimate.
        The default recipe keeps at most 500 processed columns; constant
        features can reduce the actual count. Synthetic calibration uses
        nonconstant columns and reports accuracy on separate test cases.
        """
        model = self.calibration["models"].get(checkpoint)
        if model is None:
            model = next(
                (
                    item
                    for item in self.calibration["models"].values()
                    if item["checkpoint_sha256"] == checkpoint
                ),
                None,
            )
        if model is None:
            raise ValueError("Checkpoint has no memory calibration.")
        maximum = categorical_columns
        high = (
            maximum
            if high_cardinality_columns is None
            else high_cardinality_columns
        )
        if not 0 <= high <= maximum:
            raise ValueError(
                "High-cardinality count must be within categorical columns."
            )
        columns = min(numeric_columns + categorical_columns + high, 500)
        low_columns = min(numeric_columns + categorical_columns, 500)

        def predict(processed: int) -> dict[str, int]:
            features = [1, context_rows, processed, context_rows * processed]
            stages = {}
            for name, equation in model["equations"].items():
                terms = features + [
                    min(context_rows * processed, knot)
                    for knot in equation.get("rc_knots", [])
                ]
                if "row_cap" in equation:
                    terms.append(
                        min(
                            context_rows
                            * (processed + equation["readout_tokens"]),
                            equation["row_cap"],
                        )
                    )
                value = model["model_bytes"] + sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        equation["coefficients"], terms, strict=True
                    )
                )
                stages[name] = max(int(value), model["model_bytes"])
            for phase in ("encoder", "icl", "predict"):
                stages[phase + "_reserved"] = max(
                    stages[phase + "_reserved"], stages[phase]
                )
            return stages

        stages = predict(columns)
        low = predict(low_columns)
        allocated = max(stages[name] for name in ("encoder", "icl", "predict"))
        reserved = max(
            stages[name]
            for name in (
                "encoder_reserved",
                "icl_reserved",
                "predict_reserved",
            )
        )
        safety = model["safety_fraction"]
        safe = int(max(allocated, reserved) * (1 + safety))
        return {
            "fit_peak_bytes": max(stages["encoder"], stages["icl"]),
            "predict_peak_bytes": stages["predict"],
            "fit_peak_reserved_bytes": max(
                stages["encoder_reserved"], stages["icl_reserved"]
            ),
            "predict_peak_reserved_bytes": stages["predict_reserved"],
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "conservative_peak_bytes": safe,
            "low_cardinality_peak_bytes": max(low.values()),
            "processed_columns": columns,
            "assumed_all_categories_high_cardinality": high_cardinality_columns
            is None,
            "inside_calibration_range": (
                model["range"]["min_rows"]
                <= context_rows
                <= model["range"]["max_rows"]
                and model["range"]["min_columns"]
                <= columns
                <= model["range"]["max_columns"]
            ),
            "stages": stages,
            "settings": self.calibration["settings"],
        }
