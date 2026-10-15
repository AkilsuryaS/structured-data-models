# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce binary context compression under a CUDA allocation limit.

Run from the repository root with:
    python -m benchmark.tabular.context_finetune --memory-gib 0.5

The generated features stay on CPU. An external test partition is excluded
from both context tuning and its internal validation holdout. The requested
context size is explicit; the allocator limit emulates a smaller GPU.
"""

import argparse
import json
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from sdm import Stype
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.finetune import _context_order, _split


def main() -> None:
    """Compare the same sampled context before and after optimization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-gib", type=float, default=0.5)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=500)
    parser.add_argument("--context-rows", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.rows + 256, args.columns, generator=generator)
    score = x[:, 0] + x[:, 1] * 0.5 + x[:, 2] * x[:, 3]
    y = (score > 0).long().unsqueeze(-1)
    x_train, x_test = x[: args.rows], x[args.rows :]
    y_train, y_test = y[: args.rows], y[args.rows :]
    model = KumoTabular(task="classification", size="small", device="cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 2**30 / total)
    torch.cuda.synchronize()
    start = time.perf_counter()
    sampling = torch.Generator().manual_seed(args.seed)
    model.fit(
        x_train,
        y_train,
        finetune=args.context_rows,
        query_batch_size=args.batch_size,
        max_epochs=args.epochs,
        generator=sampling,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = model.finetune_result_
    assert result is not None

    def probabilities(features: torch.Tensor) -> torch.Tensor:
        output = model.predict(features)
        indices = [
            output.columns[Stype.numerical].index(str(i)) for i in (0, 1)
        ]
        return output.numerical[:, indices].float()

    def test_loss() -> float:
        losses = []
        for indices in torch.arange(len(x_test)).split(args.batch_size):
            prediction = probabilities(x_test[indices])
            losses.append(
                -prediction.gather(1, y_test[indices].cuda())
                .clamp_min(1e-30)
                .log()
            )
        return float(torch.cat(losses).mean())

    tuned_loss = test_loss()
    # Reproduce the split, context identities and preprocessing RNG state.
    sampling.manual_seed(args.seed)
    train, _ = _split(y_train.squeeze(-1), sampling)
    order = _context_order(train, y_train.squeeze(-1), sampling)
    indices = order[: result["context_size"]]
    model.clear()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        model.fit(
            x_train[indices],
            y_train[indices],
            generator=sampling,
            _context_device=torch.device("cuda"),
        )
        baseline_probabilities = torch.cat(
            [probabilities(batch) for batch in x_test.split(args.batch_size)]
        )
    baseline_loss = float(
        -baseline_probabilities.gather(1, y_test.cuda())
        .clamp_min(1e-30)
        .log()
        .mean()
    )
    report: dict[str, Any] = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
            ).strip()
        ),
        "code_sha256": {
            path: sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "sdm/models/kumo/tabular/model.py",
                "sdm/models/kumo/tabular/finetune.py",
                "benchmark/tabular/context_finetune.py",
            )
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": "KumoTabular small",
        "checkpoint": "nvidia/Kumo-Tabular v1.0.1 small/classifier.pt",
        "dataset": "synthetic Gaussian binary: x0 + 0.5*x1 + x2*x3 > 0",
        "train_rows": args.rows,
        "test_rows": 256,
        "columns": args.columns,
        "validation": "fixed stratified 10% of training rows",
        "preprocessing": "default_recipe fitted on sampled context",
        "seed": args.seed,
        "precision": f"{dtype} neural execution, FP32 adjustment/loss",
        "hardware": torch.cuda.get_device_name(0),
        "allocation_limit_gib": args.memory_gib,
        "runtime_seconds": elapsed,
        "sampled_context_test_loss": baseline_loss,
        "tuned_context_test_loss": tuned_loss,
        "finetune_result": result,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
