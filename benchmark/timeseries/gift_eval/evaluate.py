# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM forecasting model results on GIFT-Eval."""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from benchmark.timeseries.gift_eval import GIFT_EVAL_URL
from benchmark.timeseries.model import MODEL_CONFIGS

BASELINE = "Seasonal_Naive"
REFERENCES = ("TimesFM-3", "TimesFM-2.5", "Chronos-2", BASELINE)
METRICS = {
    "eval_metrics/MASE[0.5]": "MASE",
    "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
}


def main() -> None:
    benchmark_dir = Path(__file__).parent.parent
    result_root = benchmark_dir / "gift_eval_out"
    output_root = benchmark_dir / "evals" / "gift_eval"

    results = []
    for model_config in MODEL_CONFIGS.values():
        path = result_root / model_config.name / "all_results.csv"
        if path.exists():
            print(f"Loading local results: {path}", flush=True)
            results.append(pd.read_csv(path))

    if not results:
        raise FileNotFoundError(
            f"No GIFT-Eval results found under {result_root}"
        )

    # Compare with published results on configurations run by all models:
    configs = set.intersection(*(set(result["dataset"]) for result in results))
    for name in REFERENCES:
        path = f"{GIFT_EVAL_URL}/results/{name}/all_results.csv"
        print(f"Loading published results: {name}", flush=True)
        with urlopen(path, timeout=30) as response:
            results.append(pd.read_csv(response))

    df = pd.concat(results)
    df = df[df["dataset"].isin(configs)].set_index(["model", "dataset"])
    df = df[list(METRICS)].rename(columns=METRICS)

    # Normalize by the seasonal naive baseline and aggregate by geometric mean:
    log_scores = np.log(df.div(df.loc[BASELINE], level="dataset"))
    terms = log_scores.index.get_level_values("dataset").str.split("/").str[-1]

    output_root.mkdir(parents=True, exist_ok=True)
    for term, scores in [("overall", log_scores), *log_scores.groupby(terms)]:
        leaderboard = np.exp(scores.groupby("model").mean())
        leaderboard = leaderboard.sort_values("CRPS")
        leaderboard.to_csv(output_root / f"leaderboard_{term}.csv")
        print(f"\nGIFT-Eval leaderboard ({term}):")
        print(leaderboard.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
