# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grid-search Kumo fit/cache/prediction memory with 1,024 query rows."""

import argparse
import gc
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.optimize import nnls

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.tabular.memory import KumoMemoryEstimator
from sdm.models.kumo.tabular.model import MODEL_KWARGS
from sdm.tensor.table import TableSchema

QUERY_ROWS = 1024
SEED = 17
LIMIT_BYTES = 40_000_000_000
STAGES = ("encoder", "icl", "predict")


def digest(path: Path) -> str:
    """Hash checkpoints and source without reading them all into memory."""
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish a complete measurement atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def synthetic(
    rows: int, numeric: int, categorical: int, cardinality: int, seed: int
) -> tuple[TableTensor, TableTensor]:
    """Nonconstant mixed features with explicit 2% missing values on CPU."""
    generator = torch.Generator().manual_seed(seed)
    count = rows + QUERY_ROWS
    numerical = torch.randn(count, numeric, generator=generator)
    numerical.masked_fill_(
        torch.rand(count, numeric, generator=generator) < 0.02, float("nan")
    )
    codes = torch.randint(
        cardinality, (count, categorical), generator=generator
    )
    codes.masked_fill_(
        torch.rand(count, categorical, generator=generator) < 0.02, -1
    )
    x = TableTensor(
        columns={
            Stype.numerical: [f"n{i}" for i in range(numeric)],
            Stype.categorical: [f"c{i}" for i in range(categorical)],
        },
        numerical=numerical,
        categorical=CategoricalTensor(
            code=codes,
            categories=tuple(
                torch.arange(cardinality) for _ in range(categorical)
            ),
        ),
    )
    y = TableTensor.from_tensor(
        torch.arange(rows).remainder(2).reshape(-1, 1), columns=["target"]
    )
    return x, y


def stamp() -> float:
    """Measure elapsed CUDA work rather than enqueue time."""
    torch.cuda.synchronize()
    return time.perf_counter()


def measure(run: Path, case: dict[str, Any]) -> None:
    """Measure fresh-process fit and cold/warm prediction peaks."""
    torch.set_num_threads(8)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(LIMIT_BYTES / total)
    x, y = synthetic(
        case["rows"],
        case["numeric"],
        case["categorical"],
        case["cardinality"],
        SEED,
    )
    model = KumoTabular(
        task="classification", size=case["size"], device="cuda"
    )
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    fingerprints = {
        name: digest(Path(name))
        for name in (
            "benchmark/tabular/memory.py",
            "sdm/models/kumo/tabular/memory.py",
            "sdm/models/kumo/tabular/model.py",
        )
    }
    archive = run / "code"
    archive.mkdir(exist_ok=True)
    for name, fingerprint in fingerprints.items():
        (archive / (fingerprint + ".py")).write_bytes(Path(name).read_bytes())
    report: dict[str, Any] = {
        **case,
        "status": "running",
        "model_bytes": baseline,
        "code_sha256": fingerprints,
    }
    peak: dict[str, int] = {}
    net = model.models["classification"]
    stage = "fit"

    def begin(name: str) -> None:
        nonlocal stage
        stage = name
        torch.cuda.reset_peak_memory_stats()

    def end(name: str) -> None:
        torch.cuda.synchronize()
        peak[name] = torch.cuda.max_memory_allocated()
        peak[name + "_reserved"] = torch.cuda.max_memory_reserved()

    def encoder_pre(module: Any, inputs: Any) -> None:
        begin("encoder")

    def encoder_post(module: Any, inputs: Any, output: Any) -> None:
        end("encoder")

    def icl_pre(module: Any, inputs: Any) -> None:
        begin("icl")

    def icl_post(module: Any, inputs: Any, output: Any) -> None:
        end("icl")

    hooks = [
        net.row_embedding.register_forward_pre_hook(encoder_pre),
        net.row_embedding.register_forward_hook(encoder_post),
        net.icl_block.register_forward_pre_hook(icl_pre),
        net.icl_block.register_forward_hook(icl_post),
    ]
    started = stamp()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.fit(
                x[: case["rows"]],
                y,
                finetune=False,
                num_estimators=1,
                estimator_batch_size=1,
                generator=torch.Generator().manual_seed(SEED),
                _context_device=torch.device("cuda"),
            )
        report["fit_seconds"] = stamp() - started
        for hook in hooks:
            hook.remove()
        assert model._cache is not None
        cache = cast(Cache, model._cache[0])
        report["cache_bytes"] = cache.size()
        report["processed_columns"] = len(
            cast(tuple[TableSchema, ...], cache["x_schemas"])[0].columns[
                Stype.numerical
            ]
        )
        report["persistent_after_fit_bytes"] = torch.cuda.memory_allocated()
        stage = "predict"
        torch.cuda.reset_peak_memory_stats()
        started = stamp()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first = model.predict(x[case["rows"] :]).numerical
            repeat = model.predict(x[case["rows"] :]).numerical
        torch.testing.assert_close(first, repeat, atol=1e-6, rtol=1e-6)
        assert torch.isfinite(first).all()
        report["prediction_seconds_two_calls"] = stamp() - started
        end("predict")
        report.update(status="complete", peaks=peak)
    except torch.cuda.OutOfMemoryError as error:
        report.update(
            status="oom",
            failed_stage=stage,
            reason=str(error),
            peaks=peak,
            failed_peak_bytes=torch.cuda.max_memory_allocated(),
        )
    finally:
        for hook in hooks:
            hook.remove()
        write_json(run / (case["id"] + ".json"), report)
    print(
        json.dumps(
            {"id": case["id"], "status": report["status"], "peaks": peak}
        ),
        flush=True,
    )


def grid(sizes: list[str]) -> list[dict[str, Any]]:
    """Factorial calibration and separate held-out shapes."""
    configs = [
        (n, c)
        for width in (8, 32, 80, 160)
        for n, c in ((width, 0), (width // 2, width // 2), (0, width))
    ]
    cases = []
    for size, rows, (numeric, categorical) in itertools.product(
        sizes, [1024, 8192, 32768, 131072], configs
    ):
        for cardinality in [16, 64] if categorical else [16]:
            cases.append(
                {
                    "size": size,
                    "rows": rows,
                    "numeric": numeric,
                    "categorical": categorical,
                    "cardinality": cardinality,
                    "split": "calibration",
                }
            )
    for size, rows, (numeric, categorical, cardinality) in itertools.product(
        sizes,
        [6144, 24576, 98304],
        [(12, 12, 16), (37, 21, 64), (0, 48, 64), (96, 0, 16)],
    ):
        cases.append(
            {
                "size": size,
                "rows": rows,
                "numeric": numeric,
                "categorical": categorical,
                "cardinality": cardinality,
                "split": "validation",
            }
        )
    for size, (rows, numeric, categorical, cardinality) in itertools.product(
        sizes,
        [
            (262144, 40, 40, 64),
            (524288, 80, 0, 16),
            (1048576, 80, 0, 16),
            (32768, 300, 100, 64),
            (8192, 0, 300, 64),
        ],
    ):
        cases.append(
            {
                "size": size,
                "rows": rows,
                "numeric": numeric,
                "categorical": categorical,
                "cardinality": cardinality,
                "split": "validation",
            }
        )
    for i, case in enumerate(cases):
        case["id"] = (
            f"{i:04d}-{case['size']}-{case['rows']}-{case['numeric']}-{case['categorical']}-{case['cardinality']}"
        )
    return cases


def equation(
    records: list[dict[str, Any]],
    stage: str,
    *,
    row_cap: float = 2e9 / (15 * 128 * 2),
    readout_tokens: int = 4,
) -> dict[str, Any]:
    """Nonnegative coefficients preserve monotonicity of resource estimates."""
    matrix = np.array(
        [
            [
                1,
                r["rows"],
                r["processed_columns"],
                r["rows"] * r["processed_columns"],
            ]
            for r in records
        ],
        dtype=np.float64,
    )
    knots = [2**20, 2**22, 2**24]
    matrix = np.column_stack(
        [
            matrix,
            *[np.minimum(matrix[:, 3], knot) for knot in knots],
            np.minimum(
                matrix[:, 1] * (matrix[:, 2] + readout_tokens), row_cap
            ),
        ]
    )
    target = np.array(
        [r["peaks"][stage] - r["model_bytes"] for r in records],
        dtype=np.float64,
    )
    scale = np.maximum(matrix.max(0), 1)
    measured = target + np.array([r["model_bytes"] for r in records])
    coefficients, _ = nnls(
        matrix / scale / measured[:, None], target / measured
    )
    coefficients /= scale
    return {
        "coefficients": coefficients.tolist(),
        "basis": [
            "1",
            "R",
            "C",
            "R*C",
            *[f"min(R*C,{knot})" for knot in knots],
            f"min(R*(C+{readout_tokens}),{row_cap})",
        ],
        "rc_knots": knots,
        "row_cap": row_cap,
        "readout_tokens": readout_tokens,
    }


def calibrate(run: Path) -> dict[str, Any]:
    """Fit on successful calibration cases; report held-out errors and OOMs."""
    manifest = json.loads((run / "manifest.json").read_text())
    records = [
        json.loads((run / (case["id"] + ".json")).read_text())
        for case in manifest["cases"]
        if (run / (case["id"] + ".json")).exists()
    ]
    models: dict[str, Any] = {}
    for size in manifest["sizes"]:
        training = [
            r
            for r in records
            if r["size"] == size
            and r["status"] == "complete"
            and r["split"] == "calibration"
        ]
        if len(training) < 4:
            continue
        stages = [
            stage for name in STAGES for stage in (name, name + "_reserved")
        ]
        architecture = manifest["checkpoints"][size]["architecture"]
        chunk_bytes = manifest["settings"]["allocator_limit_bytes"] * (
            manifest["settings"].get("chunk_memory_fraction", 0.05)
        )
        models[size] = {
            "model_bytes": training[0]["model_bytes"],
            "checkpoint_sha256": manifest["checkpoints"][size]["sha256"],
            "equations": {
                stage: equation(
                    training,
                    stage,
                    row_cap=chunk_bytes
                    / (15 * 2 * architecture["cell_channels"]),
                    readout_tokens=architecture["num_readout_tokens"],
                )
                for stage in stages
            },
            "range": {
                "min_rows": min(r["rows"] for r in training),
                "max_rows": max(r["rows"] for r in training),
                "min_columns": min(r["processed_columns"] for r in training),
                "max_columns": max(r["processed_columns"] for r in training),
            },
            "safety_fraction": 0.1,
        }
    bundle = {
        "settings": manifest["settings"],
        "models": models,
        "provenance": {k: v for k, v in manifest.items() if k != "cases"},
    }
    estimator = KumoMemoryEstimator(bundle)
    errors = []
    for r in records:
        if r["status"] != "complete" or r["size"] not in models:
            continue
        estimate = estimator.estimate(
            r["size"],
            r["rows"],
            r["numeric"],
            r["categorical"],
            high_cardinality_columns=r["categorical"]
            if r["cardinality"] >= 50
            else 0,
        )
        measured = max(r["peaks"][stage] for stage in STAGES)
        predicted = estimate["peak_allocated_bytes"]
        errors.append(
            {
                "id": r["id"],
                "size": r["size"],
                "split": r["split"],
                "measured_bytes": measured,
                "predicted_bytes": predicted,
                "relative_error": (predicted - measured) / measured,
                "required_margin": max(
                    0,
                    max(r["peaks"].values())
                    / max(
                        estimate["peak_allocated_bytes"],
                        estimate["peak_reserved_bytes"],
                    )
                    - 1,
                ),
                "predict_relative_error": (
                    estimate["predict_peak_bytes"] - r["peaks"]["predict"]
                )
                / r["peaks"]["predict"],
            }
        )
    for size, model in models.items():
        calibration_errors = [
            e
            for e in errors
            if e["size"] == size and e["split"] == "calibration"
        ]
        model["safety_fraction"] = max(
            0.1, max(e["required_margin"] for e in calibration_errors) + 0.05
        )
    bundle["calibration_code_sha256"] = digest(Path(__file__))
    (run / "code" / (bundle["calibration_code_sha256"] + ".py")).write_bytes(
        Path(__file__).read_bytes()
    )
    bundle["estimator_code_sha256"] = digest(
        Path("sdm/models/kumo/tabular/memory.py")
    )
    (run / "code" / (bundle["estimator_code_sha256"] + ".py")).write_bytes(
        Path("sdm/models/kumo/tabular/memory.py").read_bytes()
    )
    bundle["completed_cases"] = len(records)
    bundle["requested_cases"] = len(manifest["cases"])
    bundle["grid_complete"] = len(records) == len(manifest["cases"])
    bundle["errors"] = errors
    bundle["oom_cases"] = [r for r in records if r["status"] == "oom"]
    write_json(run / "calibration.json", bundle)
    lines = [
        "# Kumo synthetic inference-memory calibration",
        "",
        (
            "Allocated and reserved peaks measured independently. "
            "R = context rows; C = processed columns. Query rows = 1,024. "
            "Equations give additional bytes above loaded model memory. "
            "OOM peaks are censored and excluded from regression."
        ),
        "",
        "For each phase: M = W + a + bR + cC + dRC "
        "+ k1*min(RC,1048576) + k2*min(RC,4194304) "
        "+ k3*min(RC,16777216) + t*min(R*(C+K),T). "
        "Coefficients and W are bytes. K is readout tokens; "
        "T = chunk_budget/(15*2*cell_channels). "
        "Allocated peak is max(encoder,icl,predict).",
        "",
    ]
    for size, model in models.items():
        lines += [
            f"## {size}",
            "",
            f"Model memory: {model['model_bytes'] / 1e9:.6f} GB. "
            f"K={model['equations']['encoder']['readout_tokens']}, "
            f"T={model['equations']['encoder']['row_cap']:.3f}.",
            "",
            "| Stage | a | bR | cC | dRC | k1 | k2 | k3 | token cap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stage, e in model["equations"].items():
            lines.append(
                "| "
                + stage
                + " | "
                + " | ".join(f"{c:.6g}" for c in e["coefficients"])
                + " |"
            )
        held = [
            e
            for e in errors
            if e["size"] == size and e["split"] == "validation"
        ]
        if held:
            absolute = np.abs([e["relative_error"] for e in held])
            lines += [
                "",
                (
                    f"Held-out cases: {len(held)}. "
                    f"Median absolute peak error: {np.median(absolute):.2%}; "
                    f"maximum: {max(absolute):.2%}. "
                    "Safety fraction fitted on calibration cases: "
                    f"{model['safety_fraction']:.2%}."
                ),
                "",
            ]
    lines += [
        (
            f"Completed cases: {len(errors)}; "
            f"OOM cases: {len(bundle['oom_cases'])}."
        ),
        "",
        (
            "Binary inference with one estimator, the default recipe, "
            "CPU source tables and GPU neural execution. Unknown categories "
            "use C=min(numeric+2*categorical,500); low-cardinality columns "
            "do not add count features. Hardware, checkpoint and code hashes, "
            "allocator budget and precision are recorded in manifest.json. "
            "Extrapolation and different settings require fresh measurement."
        ),
    ]
    (run / "equations.md").write_text("\n".join(lines) + "\n")
    return bundle


def verify(run: Path) -> dict[str, Any]:
    """Audit measurements, identities and held-out estimates."""
    manifest = json.loads((run / "manifest.json").read_text())
    bundle = json.loads((run / "calibration.json").read_text())
    assert bundle["grid_complete"]
    for item in manifest["checkpoints"].values():
        assert digest(Path(item["path"])) == item["sha256"]
    for key in ("calibration_code_sha256", "estimator_code_sha256"):
        fingerprint = bundle[key]
        assert digest(run / "code" / (fingerprint + ".py")) == fingerprint
    estimator = KumoMemoryEstimator(bundle)
    errors: dict[str, list[float]] = {size: [] for size in manifest["sizes"]}
    conservative_misses = []
    oom_checks = []
    successful = 0
    for case in manifest["cases"]:
        report = json.loads((run / (case["id"] + ".json")).read_text())
        for name, value in case.items():
            assert report[name] == value
        fingerprints = report.get("code_sha256", manifest["code_sha256"])
        for fingerprint in fingerprints.values():
            assert digest(run / "code" / (fingerprint + ".py")) == fingerprint
        estimate = estimator.estimate(
            case["size"],
            case["rows"],
            case["numeric"],
            case["categorical"],
            high_cardinality_columns=case["categorical"]
            if case["cardinality"] >= 50
            else 0,
        )
        if report["status"] == "oom":
            predicted = estimate["peak_allocated_bytes"] > LIMIT_BYTES
            oom_checks.append(
                {
                    "id": case["id"],
                    "failed_stage": report["failed_stage"],
                    "predicted_oom": predicted,
                    "estimate_bytes": estimate["peak_allocated_bytes"],
                }
            )
            continue
        assert report["status"] == "complete"
        successful += 1
        assert report["processed_columns"] == estimate["processed_columns"]
        for stage in STAGES:
            assert report["peaks"][stage] >= report["model_bytes"]
            assert (
                report["peaks"][stage + "_reserved"] >= report["peaks"][stage]
            )
        actual = max(report["peaks"].values())
        if case["split"] == "validation":
            peak = max(report["peaks"][stage] for stage in STAGES)
            errors[case["size"]].append(
                abs(estimate["peak_allocated_bytes"] / peak - 1)
            )
            if estimate["conservative_peak_bytes"] < actual:
                conservative_misses.append(case["id"])
    summary = {
        size: {
            "cases": len(values),
            "median_absolute_error": float(np.median(values)),
            "maximum_absolute_error": max(values),
        }
        for size, values in errors.items()
    }
    result = {
        "passed": True,
        "successful_cases": successful,
        "oom_cases": len(oom_checks),
        "held_out": summary,
        "conservative_misses": conservative_misses,
        "oom_checks": oom_checks,
        "conservative_bounds_cover_validation": not conservative_misses,
        "all_ooms_predicted": all(
            item["predicted_oom"] for item in oom_checks
        ),
    }
    write_json(run / "verification.json", result)
    return result


def main() -> None:
    """Run fresh-process probes, or refit an existing calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(".cache/kumo-memory")
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--sizes", nargs="+", default=["small", "medium", "large"]
    )
    parser.add_argument("--worker")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.run_dir), indent=2))
        return
    if args.worker:
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        case = next(c for c in manifest["cases"] if c["id"] == args.worker)
        measure(args.run_dir, case)
        return
    if args.calibrate:
        calibrate(args.run_dir)
        return
    if not args.resume:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        checkpoints = {}
        for size in args.sizes:
            path = Path(
                download_checkpoint(
                    repo_id="nvidia/Kumo-Tabular",
                    filename=f"{size}/classifier.pt",
                    revision="v1.0.1",
                    config_filename="config.json",
                )
            )
            checkpoints[size] = {
                "path": str(path),
                "sha256": digest(path),
                "architecture": MODEL_KWARGS[size],
            }
        manifest = {
            "sizes": args.sizes,
            "cases": grid(args.sizes),
            "checkpoints": checkpoints,
            "settings": {
                "query_rows": QUERY_ROWS,
                "num_estimators": 1,
                "precision": "BF16",
                "task": "binary classification",
                "recipe": "default",
                "source_device": "cpu",
                "neural_device": "cuda",
                "allocator_limit_bytes": LIMIT_BYTES,
                "chunk_memory_fraction": float(
                    os.getenv("SDM_CHUNK_MEMORY_FRACTION", ".05")
                ),
                "categorical_cardinalities": [16, 64],
                "missing_fraction": 0.02,
                "seed": SEED,
                "hardware": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code_sha256": {
                name: digest(Path(name))
                for name in [
                    "benchmark/tabular/memory.py",
                    "sdm/models/kumo/tabular/memory.py",
                    "sdm/models/kumo/tabular/model.py",
                ]
            },
        }
        write_json(args.run_dir / "manifest.json", manifest)
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    for i, case in enumerate(manifest["cases"]):
        path = args.run_dir / (case["id"] + ".json")
        if args.resume and path.exists():
            continue
        command = [
            sys.executable,
            "-m",
            "benchmark.tabular.memory",
            "--run-dir",
            str(args.run_dir),
            "--worker",
            case["id"],
        ]
        with (args.run_dir / (case["id"] + ".log")).open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        report = json.loads(path.read_text())
        print(
            f"{i + 1}/{len(manifest['cases'])} "
            f"{case['id']} {report['status']}",
            flush=True,
        )
        if (i + 1) % 20 == 0:
            calibrate(args.run_dir)
    calibrate(args.run_dir)


if __name__ == "__main__":
    main()
