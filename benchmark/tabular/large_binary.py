# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare default XGBoost with maximum, paired and tuned Kumo contexts.

GPU arms run sequentially in isolated processes. See the benchmark README for
split semantics and the distinction between this schedule and public fit.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from benchmark.tabular.large_binary_tuning import train_context
from benchmark.tabular.microsoft_malware_context import (
    digest,
    probabilities,
    synchronized_time,
    verify_direct,
    weights_digest,
    write_json,
)
from sdm import Stype, TableTensor
from sdm.cache import Cache
from sdm.models.kumo.tabular import KumoMemoryEstimator, KumoTabular

DATASETS = {
    "microsoft_malware": ("HasDetections", ["MachineIdentifier"], None),
    "red_hat_business_value": (
        "outcome",
        ["people_id", "activity_id"],
        "people_id",
    ),
    "amex_default_prediction": ("target", ["customer_ID"], None),
    "bosch_production": ("Response", ["Id"], "Id"),
    "avazu_ctr": ("click", ["id"], "hour"),
    "talkingdata_adtracking": (
        "is_attributed",
        ["attributed_time"],
        "click_time",
    ),
}
ARMS = ("xgboost", "maximum", "tuned", "paired")
SOURCE = Path("/mnt/d/tfm-scale-large-data-cache")
CALIBRATION = Path(".cache/kumo-memory/20261014-grid/calibration.json")
LEGACY = Path(".cache/microsoft-malware-context-fit")
LIMIT = 40_000_000_000


def quote(name: str) -> str:
    """Quote a SQL identifier, including names from an input schema."""
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def stratified_sample(
    indices: np.ndarray, labels: np.ndarray, size: int, seed: int
) -> np.ndarray:
    """Select without replacement while retaining both classes."""
    if size >= len(indices):
        return indices.copy()
    selected, _ = train_test_split(
        indices, train_size=size, stratify=labels[indices], random_state=seed
    )
    return selected


def split_rows(
    labels: np.ndarray, key: np.ndarray | None, kind: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve complete groups/time blocks, including unused stopping rows."""
    indices = np.arange(len(labels), dtype=np.int64)
    if kind == "stratified":
        train, holdout = train_test_split(
            indices, test_size=0.1, stratify=labels, random_state=seed
        )
    elif kind == "group":
        assert key is not None
        # pandas hashes are deterministic across processes.
        import pandas as pd  # noqa: PLC0415

        hashed = pd.util.hash_array(np.asarray(key, dtype=object))
        is_valid = hashed % 10 == seed % 10
        train, holdout = indices[~is_valid], indices[is_valid]
    else:
        assert key is not None
        if any(value is None for value in key):
            raise ValueError("Split key contains missing values")
        boundary = np.sort(key)[math.floor(0.9 * len(key))]
        train, holdout = indices[key < boundary], indices[key >= boundary]
    for part in (train, holdout):
        if np.unique(labels[part]).tolist() != [0, 1]:
            raise ValueError(
                "Training and internal holdout must each contain both classes"
            )
    return train, holdout


def context_order(
    train: np.ndarray, labels: np.ndarray, seed: int
) -> np.ndarray:
    """Nested stratified prefixes, with both classes from row two."""
    rng = np.random.default_rng(seed)
    zero = rng.permutation(train[labels[train] == 0])
    one = rng.permutation(train[labels[train] == 1])
    n = len(train)
    # Evenly spread minority rows across context prefixes.
    marks = np.floor(np.arange(1, n + 1) * len(one) / n).astype(np.int64)
    mask = np.diff(np.r_[0, marks]).astype(bool)
    order = np.empty(n, dtype=np.int64)
    order[mask], order[~mask] = one, zero
    # Ensure even the minimum probe includes both labels.
    position = np.flatnonzero(labels[order] != labels[order[0]])[0]
    order[1], order[position] = order[position], order[1]
    return order


def feature_spec(schema: pa.Schema, dataset: str) -> dict[str, str]:
    """Explicitly retain category semantics for integer-coded identifiers."""
    target, excluded, _ = DATASETS[dataset]
    coded = set()
    if dataset == "avazu_ctr":
        coded = set(schema.names) - {"id", "click", "hour"}
    elif dataset == "talkingdata_adtracking":
        coded = {"ip", "app", "device", "os", "channel"}
    elif dataset == "amex_default_prediction":
        coded = {
            "B_30",
            "B_38",
            "D_114",
            "D_116",
            "D_117",
            "D_120",
            "D_126",
            "D_63",
            "D_64",
            "D_66",
            "D_68",
        }
    result = {}
    for field in schema:
        if field.name in {target, *excluded, "__row_id"}:
            continue
        categorical = (
            pa.types.is_string(field.type)
            or pa.types.is_large_string(field.type)
            or pa.types.is_boolean(field.type)
            or field.name in coded
        )
        result[field.name] = "categorical" if categorical else "numerical"
    return result


def prepare_table(origin: Path, destination: Path) -> str:
    """Keep physical row order without sorting or materializing a wide table.

    Reuse unchanged inputs. Calendar expansion and outcome-field exclusion
    are independent per batch, so they need no global SQL operator.
    """
    parquet = pq.ParquetFile(origin)
    schema = parquet.schema_arrow
    calendar = any(
        pa.types.is_date(f.type) or pa.types.is_timestamp(f.type)
        for f in schema
        if f.name != "attributed_time"
    )
    if not calendar and "attributed_time" not in schema.names:
        destination.symlink_to(origin.resolve())
        return "source_parquet"
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=32768):
            arrays, names = [], []
            for field, column in zip(schema, batch.columns, strict=True):
                if field.name == "attributed_time":
                    continue
                if pa.types.is_date(field.type) or pa.types.is_timestamp(
                    field.type
                ):
                    parts = {
                        "year": pc.year(column),
                        "month": pc.month(column),
                        "day": pc.day(column),
                        "dow": pc.day_of_week(column, week_start=7),
                        "hour": pc.hour(
                            pc.cast(column, pa.timestamp("us"))
                            if pa.types.is_date(field.type)
                            else column
                        ),
                    }
                    for part, value in parts.items():
                        names.append(field.name + "_" + part)
                        arrays.append(pc.cast(value, pa.float32()))
                else:
                    names.append(field.name)
                    arrays.append(column)
            table = pa.Table.from_arrays(arrays, names=names)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination, table.schema, compression="zstd"
                )
            writer.write_table(table, row_group_size=32768)
    finally:
        if writer is not None:
            writer.close()
    return "streamed_calendar_features"


def prepare(root: Path, dataset: str, config: dict[str, Any]) -> None:
    """Materialize deterministic tables using bounded-memory SQL."""
    import duckdb  # noqa: PLC0415

    path = root / dataset
    path.mkdir(exist_ok=True)
    metadata_file = path / "data.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        for name, fingerprint in metadata["prepared_sha256"].items():
            assert digest(path / name) == fingerprint, (
                f"Prepared input changed: {name}"
            )
        return
    partial = [
        path / n
        for n in (
            "train.parquet",
            "valid.parquet",
            "selection.npz",
            "sql-temp",
        )
        if (path / n).exists()
    ]
    if partial:
        archive = path / (
            "preparation-attempt-"
            + datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        )
        archive.mkdir()
        for item in partial:
            shutil.move(str(item), archive / item.name)
    source = Path(config["source"]) / dataset / "processed"
    target, _, key = DATASETS[dataset]
    files = [
        source / name
        for name in ("train.parquet", "valid.parquet", "manifest.json")
    ]
    if dataset == "red_hat_business_value":
        files.append(source / "people.parquet")
    prepared_started = time.perf_counter()
    provenance = {str(f): digest(f) for f in files}
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    con.execute(f"SET temp_directory={sql_path(path / 'sql-temp')}")
    bounds = {}
    preparation_methods = {}
    for role in ("train", "valid"):
        origin = source / (role + ".parquet")
        if dataset == "microsoft_malware":
            origin = Path(config["malware_prepared"]) / (role + ".parquet")
            provenance[str(origin)] = digest(origin)
        if dataset not in (
            "amex_default_prediction",
            "red_hat_business_value",
        ):
            preparation_methods[role] = prepare_table(
                origin, path / (role + ".parquet")
            )
            if key:
                values = pq.read_table(origin, columns=[key])[key]
                limits = pc.min_max(values).as_py()
                bounds[role] = (limits["min"], limits["max"])
                if role == "train":
                    split_key_values = values.to_numpy()
            elif role == "train":
                split_key_values = None
            print(
                json.dumps(
                    {
                        "prepared_table": dataset,
                        "role": role,
                        "method": preparation_methods[role],
                    }
                ),
                flush=True,
            )
            continue
        preparation_methods[role] = "sql_join_or_latest_statement"
        query = (
            "SELECT *, row_number() OVER () - 1 AS __row_id "
            f"FROM read_parquet({sql_path(origin)})"
        )
        if dataset == "amex_default_prediction":
            conflicts = con.execute(
                "SELECT count(*) FROM (SELECT customer_ID "
                f"FROM ({query}) GROUP BY customer_ID "
                "HAVING count(DISTINCT target) != 1)"
            ).fetchall()[0][0]
            if conflicts:
                raise ValueError("AMEX has inconsistent customer labels")
            query = (
                f"SELECT * FROM ({query}) QUALIFY row_number() OVER "
                "(PARTITION BY customer_ID ORDER BY S_2 DESC NULLS LAST, "
                "__row_id DESC)=1"
            )
        if dataset == "red_hat_business_value":
            people = source / "people.parquet"
            duplicate = con.execute(
                "SELECT count(*)-count(DISTINCT people_id) "
                f"FROM read_parquet({sql_path(people)})"
            ).fetchall()[0][0]
            if duplicate:
                raise ValueError("People table is not many-to-one")
            names = pq.read_schema(people).names
            projection = ", ".join(
                f"p.{quote(n)} AS {quote('person_' + n)}"
                for n in names
                if n != "people_id"
            )
            query = (
                f"SELECT a.*, {projection} FROM ({query}) a "
                f"LEFT JOIN read_parquet({sql_path(people)}) p "
                "USING (people_id)"
            )
        con.execute(f"CREATE OR REPLACE TEMP VIEW input_table AS {query}")
        if key and dataset != "red_hat_business_value":
            bounds[role] = con.execute(
                f"SELECT min({quote(key)}), max({quote(key)}) FROM input_table"
            ).fetchall()[0]
        schema = (
            con.execute("SELECT * FROM input_table LIMIT 0")
            .to_arrow_table()
            .schema
        )
        expressions = []
        for field in schema:
            n = quote(field.name)
            if pa.types.is_date(field.type) or pa.types.is_timestamp(
                field.type
            ):
                # Never derive features from the post-outcome timestamp.
                if field.name == "attributed_time":
                    continue
                for part in ("year", "month", "day", "dow", "hour"):
                    expressions.append(
                        f"CAST(date_part('{part}', {n}) AS FLOAT) AS "
                        f"{quote(field.name + '_' + part)}"
                    )
            else:
                expressions.append(n)
        destination = path / (role + ".parquet")
        con.execute(
            f"COPY (SELECT {', '.join(expressions)} FROM input_table "
            f"ORDER BY __row_id) TO {sql_path(destination)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
        )
        if role == "train":
            split_key = "__row_id" if key in ("hour", "click_time") else key
            # Preserve the temporal key independently of calendar features.
            if key in ("hour", "click_time"):
                split_key_values = con.execute(
                    f"SELECT {quote(key)} FROM input_table ORDER BY __row_id"
                ).fetchnumpy()[key]
            elif split_key:
                split_key_values = pq.read_table(
                    destination, columns=[split_key]
                )[split_key].to_numpy()
            else:
                split_key_values = None
    con.close()
    if bounds:
        assert bounds["train"][1] < bounds["valid"][0]
    if dataset in ("amex_default_prediction", "red_hat_business_value"):
        entity = (
            "customer_ID"
            if dataset == "amex_default_prediction"
            else "people_id"
        )
        groups = []
        for role in ("train", "valid"):
            groups.append(
                set(
                    pq.read_table(
                        path / (role + ".parquet"), columns=[entity]
                    )[entity].to_pylist()
                )
            )
        assert not groups[0].intersection(groups[1]), "Outer entity leakage"
    labels = (
        pq.read_table(path / "train.parquet", columns=[target])[target]
        .to_numpy()
        .astype(np.int64)
    )
    if np.unique(labels).tolist() != [0, 1]:
        raise ValueError("Expected labels 0 and 1")
    kind = (
        "group"
        if dataset == "red_hat_business_value"
        else "temporal"
        if key
        else "stratified"
    )
    if dataset == "microsoft_malware":
        old = torch.load(
            config["malware_selection"], weights_only=False, map_location="cpu"
        )
        train, holdout = (
            old["train_indices"].numpy(),
            old[
                "val_indices" if "val_indices" in old else "validation_indices"
            ].numpy(),
        )
    else:
        train, holdout = split_rows(
            labels,
            None if split_key_values is None else np.asarray(split_key_values),
            kind,
            config["seed"],
        )
    stop = stratified_sample(
        holdout, labels, config["holdout_cap"], config["seed"]
    )
    order = context_order(train, labels, config["seed"])
    np.savez(
        path / "selection.npz",
        train=train,
        holdout=holdout,
        stopping=stop,
        order=order,
    )
    assert not np.isin(train, holdout).any()
    assert np.isin(stop, holdout).all()
    assert len(np.union1d(train, holdout)) == len(labels)
    stypes = feature_spec(pq.read_schema(path / "train.parquet"), dataset)
    meta = {
        "dataset": dataset,
        "prepare_seconds": time.perf_counter() - prepared_started,
        "preparation_methods": preparation_methods,
        "outer_order_bounds": {
            k: [str(v) for v in value] for k, value in bounds.items()
        },
        "source_sha256": provenance,
        "outer_split": json.loads((source / "manifest.json").read_text())[
            "outer_split"
        ],
        "target": target,
        "features": stypes,
        "train_rows": len(labels),
        "supervised_rows": len(train),
        "holdout_rows": len(holdout),
        "stopping_rows": len(stop),
        "evaluation_rows": pq.ParquetFile(
            path / "valid.parquet"
        ).metadata.num_rows,
        "internal_split": kind,
        "class_counts_train": np.bincount(labels[train], minlength=2).tolist(),
        "class_counts_stopping": np.bincount(
            labels[stop], minlength=2
        ).tolist(),
        "prepared_sha256": {
            n: digest(path / n)
            for n in ("train.parquet", "valid.parquet", "selection.npz")
        },
    }
    write_json(metadata_file, meta)
    print(
        json.dumps(
            {
                "prepared": dataset,
                "rows": meta["train_rows"],
                "features": len(stypes),
            }
        ),
        flush=True,
    )


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Use stable two-class normalization, matching historical Kumo scoring."""
    p = np.clip(np.asarray(prediction, dtype=np.float64), 1e-15, 1 - 1e-15)
    p /= p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or len(p) != len(labels):
        raise ValueError("Incomplete or nonfinite prediction output")
    return {
        "accuracy": float(accuracy_score(labels, p[:, 1] >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, p[:, 1])),
        "average_precision": float(average_precision_score(labels, p[:, 1])),
        "log_loss": float(log_loss(labels, p, labels=[0, 1])),
        "prevalence": float(np.mean(labels)),
    }


class MemoryMonitor:
    """Poll device use for frameworks without allocator statistics."""

    def __init__(self) -> None:
        self.peak = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self) -> None:
        while not self.stopped.is_set():
            try:
                used = (
                    int(
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        )
                        .strip()
                        .splitlines()[0]
                    )
                    * 2**20
                )
                self.peak = max(self.peak, used)
            except (subprocess.SubprocessError, ValueError):
                pass
            self.stopped.wait(0.5)

    def __enter__(self) -> MemoryMonitor:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stopped.set()
        self.thread.join()


def as_table(table: pa.Table, stypes: dict[str, str]) -> TableTensor:
    return TableTensor.from_arrow(
        table.select(list(stypes)),
        stypes={n: Stype(s) for n, s in stypes.items()},
    )


def evaluate(
    path: Path,
    output: Path,
    meta: dict[str, Any],
    predict: Callable[[pa.Table], np.ndarray],
) -> dict[str, Any]:
    """Stream evaluation, separating compute from output/scoring time."""
    prediction = np.lib.format.open_memmap(
        output / "external.npy",
        mode="w+",
        dtype=np.float32,
        shape=(meta["evaluation_rows"], 2),
    )
    labels = np.lib.format.open_memmap(
        output / "labels.npy",
        mode="w+",
        dtype=np.int64,
        shape=(meta["evaluation_rows"],),
    )
    elapsed = 0.0
    offset = 0
    began = time.perf_counter()
    names = list(meta["features"])
    target = meta["target"]
    for record in pq.ParquetFile(path / "valid.parquet").iter_batches(
        batch_size=1024, columns=[*names, target]
    ):
        table = pa.Table.from_batches([record])
        # predict() includes model-specific feature conversion and transfer.
        start = synchronized_time()
        result = predict(table.select(names))
        elapsed += synchronized_time() - start
        if offset == 0:
            np.testing.assert_array_equal(result, predict(table.select(names)))
        prediction[offset : offset + len(table)] = result
        labels[offset : offset + len(table)] = table[target].to_numpy()
        offset += len(table)
        if offset % 131072 == 0:
            print(
                json.dumps({"evaluated_rows": offset, "total": len(labels)}),
                flush=True,
            )
    assert offset == meta["evaluation_rows"]
    prediction.flush()
    labels.flush()
    report = {
        "evaluated_rows": offset,
        "prediction_seconds": elapsed,
        "prediction_end_to_end_seconds": time.perf_counter() - began,
        "rows_per_second": offset / elapsed,
        "repeated_prediction_verified": True,
    }
    report.update(metrics(labels, prediction))
    return report


def capacity(
    limit: int, estimate: int, probe: Callable[[int], bool]
) -> tuple[int, list[dict[str, Any]]]:
    """Approximate maximum with actual probes and a 10%/1024-row bracket."""
    low, high = 0, limit + 1
    candidate = min(limit, max(2, estimate))
    history = []
    while True:
        fits = probe(candidate)
        history.append({"rows": candidate, "fits": fits})
        if fits:
            low = candidate
            if low == limit or high - low <= max(1024, math.ceil(low * 0.1)):
                return low, history
            candidate = (
                min(limit, low * 2) if high == limit + 1 else (low + high) // 2
            )
        else:
            high = candidate
            if candidate == 2:
                raise RuntimeError(
                    "Even a two-class inference context cannot fit"
                )
            if low and high - low <= max(1024, math.ceil(low * 0.1)):
                return low, history
            candidate = (
                max(2, candidate // 2) if low == 0 else (low + high) // 2
            )


def load_kumo_data(
    path: Path, meta: dict[str, Any]
) -> tuple[TableTensor, TableTensor, np.ndarray]:
    table = pq.read_table(
        path / "train.parquet",
        columns=[*list(meta["features"]), meta["target"]],
    )
    labels = table[meta["target"]].to_numpy().astype(np.int64)
    x = as_table(table, meta["features"])
    y = TableTensor.from_arrow(
        table.select([meta["target"]]),
        stypes={meta["target"]: Stype.categorical},
    )
    return x, y, labels


def replay(model: KumoTabular, state: dict[str, Any], *, tuned: bool) -> None:
    """Rebuild prediction caches with the chosen saved adjustment."""
    model.clear()
    encoder = state["encoder_cache"].to("cuda")
    net = model.models["classification"]
    base = state["base"].cuda().float()
    if tuned:
        base += state["adjustment"].cuda()
    cy = state["labels"].cuda()
    cache = Cache(dict(encoder))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        net.icl_block(base.bfloat16(), cy, cache=cache)
    cache.update(
        x_schemas=(state["x_schema"],),
        y_schema=state["y_schema"],
        related_tables_schema=None,
        classes=state["classes"].cuda(),
        class_values=[None],
        categorical_mask=state["categorical_mask"].cuda(),
        finetune_dtype=torch.bfloat16,
    )
    model._cache = Cache(
        {
            "recipe_execution": state["execution"],
            "kwargs": {},
            "num_batches": 1,
            0: cache,
        }
    ).freeze()


def kumo_arm(
    path: Path,
    output: Path,
    meta: dict[str, Any],
    config: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    torch.cuda.set_per_process_memory_fraction(
        config["memory_bytes"]
        / torch.cuda.get_device_properties(0).total_memory
    )
    model = KumoTabular(task="classification", size="small", device="cuda")
    before = weights_digest(model)
    report: dict[str, Any] = {
        "checkpoint": "Kumo-Tabular Small v1.0.1",
        "checkpoint_weights_sha256": before,
        "precision": "bfloat16",
        "estimators": 1,
    }
    selection = np.load(path / "selection.npz")
    context = torch.from_numpy(
        selection["order"][: config["context_rows"]].copy()
    )
    started = synchronized_time()
    if arm == "paired":
        state = torch.load(
            path / "tuned/optimized-context.pt",
            weights_only=False,
            map_location="cpu",
        )
        assert torch.equal(state["context_indices"], context)
        assert not np.isin(context.numpy(), selection["holdout"]).any()
        replay(model, state, tuned=False)
        report["context_rows"] = len(context)
        report["paired_context_identical"] = True
    else:
        x, y, labels = load_kumo_data(path, meta)
        order = selection["order"]
        if arm == "maximum":

            def fit(size: int) -> None:
                model.clear()
                gc.collect()
                torch.cuda.empty_cache()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    indices = torch.from_numpy(order[:size].copy())
                    model.fit(
                        x[indices],
                        y[indices],
                        n_estimators=1,
                        generator=torch.Generator().manual_seed(
                            config["seed"]
                        ),
                        _context_device=torch.device("cuda"),
                    )

            probe_tables = [
                x[torch.from_numpy(selection["stopping"][:1024].copy())]
            ]
            parquet = pq.ParquetFile(path / "valid.parquet")
            for group in sorted(
                {0, parquet.num_row_groups // 2, parquet.num_row_groups - 1}
            ):
                probe_tables.append(
                    as_table(
                        parquet.read_row_group(
                            group, columns=list(meta["features"])
                        ).slice(0, 1024),
                        meta["features"],
                    )
                )

            def probe(size: int) -> bool:
                print(json.dumps({"capacity_probe_rows": size}), flush=True)
                try:
                    fit(size)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for query in probe_tables:
                            probabilities(model, query)
                    return True
                except torch.cuda.OutOfMemoryError:
                    return False
                finally:
                    model.clear()
                    gc.collect()
                    torch.cuda.empty_cache()

            estimator = KumoMemoryEstimator.from_file(config["calibration"])
            numeric = list(meta["features"].values()).count("numerical")
            categorical = len(meta["features"]) - numeric
            lo, hi = 2, len(order)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                predicted = estimator.estimate(
                    "small", mid, numeric, categorical
                )["conservative_peak_bytes"]
                if predicted <= config["memory_bytes"]:
                    lo = mid
                else:
                    hi = mid - 1
            size, history = capacity(len(order), lo, probe)
            report.update(
                capacity_probes=history,
                capacity_seconds=synchronized_time() - started,
            )
            write_json(output / "capacity.json", report)
            report["fit_seconds"] = 0.0
            while True:
                fit_started = synchronized_time()
                try:
                    fit(size)
                    report["fit_seconds"] += synchronized_time() - fit_started
                    result = evaluate(
                        path,
                        output,
                        meta,
                        lambda t: kumo_predict(model, t, meta),
                    )
                    report.update(result)
                    break
                except torch.cuda.OutOfMemoryError:
                    report.setdefault("inference_oom_retries", []).append(size)
                    model.clear()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if size <= 2:
                        raise RuntimeError(
                            "Full evaluation cannot fit a two-class context"
                        ) from None
                    size = max(2, math.floor(size * 0.9))
            report["context_rows"] = size
            torch.save(
                {
                    "context_indices": torch.from_numpy(order[:size].copy()),
                    "generator_seed": config["seed"],
                },
                output / "context.pt",
            )
            del x, y, labels
        else:
            if len(context) != config["context_rows"]:
                raise ValueError(
                    "Requested context exceeds available supervised rows"
                )
            flags = [p.requires_grad for p in model.parameters()]
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            generator = torch.Generator().manual_seed(config["seed"])
            rng = generator.get_state()

            def observer(event: str, data: dict[str, Any]) -> None:
                if event == "restored":
                    snapshot = {
                        k: v.detach().cpu()
                        if isinstance(v, torch.Tensor)
                        else v.to("cpu")
                        if isinstance(v, Cache)
                        else v
                        for k, v in data.items()
                    }
                    torch.save(snapshot, output / "optimized-context.pt")
                elif event in ("epoch", "progress"):
                    record = {"event": event, **data}
                    with (output / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    if event == "epoch":
                        print(json.dumps(record), flush=True)

            def train(probing: bool) -> dict[str, Any]:
                return train_context(
                    model,
                    x,
                    y,
                    context,
                    torch.from_numpy(selection["train"].copy()),
                    torch.from_numpy(selection["stopping"].copy()),
                    recipe=None,
                    generator=generator,
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16,
                    budget=config["memory_bytes"],
                    query_batch_size=config["query_batch_size"],
                    learning_rate=1e-3,
                    min_learning_rate=1e-5,
                    max_epochs=math.ceil(
                        config["steps"]
                        / max(
                            1,
                            math.ceil(
                                len(selection["train"])
                                / config["query_batch_size"]
                            ),
                        )
                    ),
                    patience=5,
                    min_epochs=5,
                    min_improvement=1e-4,
                    shuffle_columns=False,
                    probing=probing,
                    observer=observer,
                    max_steps=config["steps"],
                    validation_interval=config["validation_interval"],
                    validation_batch_size=1024,
                    warmup=50,
                )

            try:
                # Probe twice to include Adam states and gradients.
                train(True)
                model.clear()
                gc.collect()
                torch.cuda.empty_cache()
                generator.set_state(rng)
                report["finetune_result"] = train(False)
                result = report["finetune_result"]
                report["stopping_reason"] = (
                    "patience"
                    if result["completed_steps"] < config["steps"]
                    else "update_budget"
                )
                report["internal_validation_loss"] = result[
                    "best_validation_loss"
                ]
                report["best_step"] = (
                    result["history"][result["best_epoch"] - 1]["steps"]
                    if result["best_epoch"]
                    else 0
                )
                report["context_rows"] = len(context)
            finally:
                for p, flag in zip(model.parameters(), flags, strict=True):
                    p.requires_grad_(flag)
                model.eval()
            report["fit_seconds"] = synchronized_time() - started
            assert flags == [p.requires_grad for p in model.parameters()]
            report["gradient_flags_restored"] = True
            verify_direct(model, x[:32], output)
            report["direct_prediction_verified"] = True
            state = torch.load(
                output / "optimized-context.pt",
                weights_only=False,
                map_location="cpu",
            )
            replay(model, state, tuned=True)
            del x, y, labels
    if arm != "maximum":
        report.setdefault("fit_seconds", synchronized_time() - started)
        report.update(
            evaluate(
                path, output, meta, lambda t: kumo_predict(model, t, meta)
            )
        )
    assert weights_digest(model) == before
    report["weights_unchanged"] = True
    assert model._cache is not None
    report["effective_features"] = len(
        cast(torch.Tensor, cast(Cache, model._cache[0])["categorical_mask"])
    )
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    # Maximum-context ICL caches can consume tens of GB. Save identities and
    # the fitted recipe instead; frozen weights and raw rows reconstruct them.
    torch.save(model._cache["recipe_execution"], output / "preprocessing.pt")
    return report


def kumo_predict(
    model: KumoTabular, table: pa.Table, meta: dict[str, Any]
) -> np.ndarray:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return (
            probabilities(model, as_table(table, meta["features"]))
            .cpu()
            .numpy()
        )


def xgboost_arm(
    path: Path, output: Path, meta: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    import cupy as cp  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415
    import xgboost.callback  # noqa: PLC0415

    selection = np.load(path / "selection.npz")
    names = list(meta["features"])
    categorical = [n for n in names if meta["features"][n] == "categorical"]
    started = time.perf_counter()
    table = pq.read_table(
        path / "train.parquet", columns=[*names, meta["target"]]
    )
    labels = table[meta["target"]].to_numpy().astype(np.int64)
    vocab = {
        n: pc.unique(table[n].take(pa.array(selection["train"]))).drop_null()
        for n in categorical
    }
    write_json(
        output / "categories.json",
        {n: v.to_pylist() for n, v in vocab.items()},
    )

    def encode(table: pa.Table) -> np.ndarray:
        matrix = np.empty((len(table), len(names)), dtype=np.float32)
        for i, name in enumerate(names):
            col = table[name]
            if name in vocab:
                matrix[:, i] = pc.index_in(
                    col, value_set=vocab[name]
                ).to_numpy()
            else:
                matrix[:, i] = col.to_numpy()
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix

    class Iterator(xgb.DataIter):
        def __init__(self, indices: np.ndarray) -> None:
            super().__init__(release_data=True)
            self.indices = indices
            self.offset = 0

        def reset(self) -> None:
            self.offset = 0

        def next(self, input_data: Callable[..., None]) -> bool:
            if self.offset >= len(self.indices):
                return False
            ids = self.indices[self.offset : self.offset + 32768]
            self.batch = cp.asarray(
                encode(table.take(pa.array(ids))), order="C"
            )
            self.target = cp.asarray(labels[ids], dtype=cp.float32)
            input_data(
                data=self.batch,
                label=self.target,
                feature_names=names,
                feature_types=["c" if n in vocab else "q" for n in names],
            )
            self.offset += len(ids)
            return True

    dtrain = xgb.QuantileDMatrix(
        Iterator(selection["train"]), enable_categorical=True, nthread=8
    )
    dvalid = xgb.QuantileDMatrix(
        Iterator(selection["stopping"]),
        ref=dtrain,
        enable_categorical=True,
        nthread=8,
    )
    report: dict[str, Any] = {
        "xgboost_version": xgb.__version__,
        "precision": "float32",
        "prepare_seconds": time.perf_counter() - started,
        "effective_features": len(names),
    }
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda:0",
        "seed": config["seed"],
        "nthread": 8,
        "eval_metric": ["auc", "logloss"],
    }
    history: dict[str, Any] = {}

    class History(xgb.callback.TrainingCallback):
        def after_iteration(
            self, model: Any, epoch: int, evals_log: dict[str, Any]
        ) -> bool:
            if (epoch + 1) % 100 == 0:
                record = {
                    "round": epoch + 1,
                    "validation_loss": evals_log["internal"]["logloss"][-1],
                }
                print(json.dumps(record), flush=True)
                write_json(output / "progress.json", record)
            return False

    started = synchronized_time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config["xgb_rounds"],
        evals=[(dvalid, "internal")],
        evals_result=history,
        callbacks=[
            History(),
            xgb.callback.EarlyStopping(
                rounds=100,
                metric_name="logloss",
                data_name="internal",
                maximize=False,
                save_best=True,
            ),
        ],
        verbose_eval=False,
    )
    report.update(
        fit_seconds=synchronized_time() - started,
        params=params,
        resolved_config=json.loads(model.save_config()),
        best_iteration=model.best_iteration,
        saved_rounds=model.num_boosted_rounds(),
        completed_rounds=len(history["internal"]["logloss"]),
        internal_validation_loss=float(model.best_score),
    )
    assert model.num_boosted_rounds() == model.best_iteration + 1
    write_json(output / "history.json", history)
    model.save_model(output / "model.ubj")
    del dtrain, dvalid
    table = pa.table({})
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    def predict(t: pa.Table) -> np.ndarray:
        p = cp.asnumpy(model.inplace_predict(cp.asarray(encode(t), order="C")))
        return np.stack((1 - p, p), axis=1)

    report.update(evaluate(path, output, meta, predict))
    batch = pa.Table.from_batches(
        [
            next(
                pq.ParquetFile(path / "valid.parquet").iter_batches(
                    batch_size=1024, columns=names
                )
            )
        ]
    )
    expected = predict(batch)
    restored = xgb.Booster()
    restored.load_model(output / "model.ubj")
    restored.set_param({"device": "cuda:0"})
    actual = cp.asnumpy(
        restored.inplace_predict(cp.asarray(encode(batch), order="C"))
    )
    np.testing.assert_array_equal(expected[:, 1], actual)
    report["saved_model_prediction_verified"] = True
    report["model_sha256"] = digest(output / "model.ubj")
    return report


def worker(root: Path, dataset: str, arm: str) -> None:
    torch.set_num_threads(8)
    config = json.loads((root / "config.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    for name, fingerprint in provenance["code_sha256"].items():
        assert digest(Path(name)) == fingerprint, "Worker code changed"
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    path = root / dataset
    output = path / arm
    output.mkdir(exist_ok=True)
    meta = json.loads((path / "data.json").read_text())
    report: dict[str, Any] = {
        "status": "running",
        "dataset": dataset,
        "arm": arm,
        "data_sha256": digest(path / "data.json"),
        "raw_features": len(meta["features"]),
        "seed": config["seed"],
    }
    began = time.perf_counter()
    write_json(output / "result.json", report)
    with MemoryMonitor() as monitor:
        try:
            if arm == "xgboost":
                report.update(xgboost_arm(path, output, meta, config))
            else:
                report.update(kumo_arm(path, output, meta, config, arm))
            prediction = np.load(output / "external.npy", mmap_mode="r")
            labels = np.load(output / "labels.npy", mmap_mode="r")
            for k, value in metrics(labels, prediction).items():
                assert abs(report[k] - value) < 1e-12
            report.update(
                status="complete",
                metrics_verified=True,
                wall_seconds=time.perf_counter() - began,
                prediction_sha256=digest(output / "external.npy"),
            )
        except Exception as error:  # noqa: BLE001
            report.update(
                status="failed", error=f"{type(error).__name__}: {error}"
            )
            traceback.print_exc()
        finally:
            report.update(
                device_peak_polled_bytes=monitor.peak,
                cpu_peak_rss_bytes=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            )
            write_json(output / "result.json", report)
    if report["status"] != "complete":
        raise RuntimeError(report["error"])


def summarize(root: Path, datasets: list[str]) -> None:
    rows = []
    for dataset in datasets:
        for arm in ARMS:
            f = root / dataset / arm / "result.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
            else:
                rows.append(
                    {"dataset": dataset, "arm": arm, "status": "pending"}
                )
    write_json(
        root / "results.json",
        {"results": rows, "requested_arms": len(datasets) * 4},
    )
    lines = [
        "# Six-dataset binary benchmark",
        "",
        "One seed (17), one Kumo estimator. "
        "Outer validation is scored in full. "
        "Inference includes feature conversion/transfer but excludes file "
        "reads, writes and scoring. Device memory polling includes other "
        "GPU processes. Bosch uses an unverified ID-order proxy split.",
        "",
        "| Dataset | Method | Status | Context | AUC | AP | Accuracy "
        "| Log loss | Fit s | Infer s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        values = [
            r["dataset"],
            r["arm"],
            r["status"],
            str(r.get("context_rows", "—")),
        ]
        values += [
            f"{r[k]:.5f}" if k in r else "—"
            for k in (
                "roc_auc",
                "average_precision",
                "accuracy",
                "log_loss",
                "fit_seconds",
                "prediction_seconds",
            )
        ]
        lines.append("| " + " | ".join(values) + " |")
        if r["status"] == "failed":
            lines += [
                "",
                f"Failure ({r['dataset']}/{r['arm']}): {r.get('error')}",
                "",
            ]
    if (root / "imported-results.json").exists():
        lines += [
            "",
            "Completed results retained from a previous code revision; "
            "see imported-results.json for original provenance and hashes.",
        ]
    complete = [r for r in rows if r["status"] == "complete"]
    comparable = [
        d for d in datasets if sum(r["dataset"] == d for r in complete) == 4
    ]
    if comparable:
        lines += [
            "",
            f"Mean AUC ranks across {len(comparable)} datasets "
            "with all four arms complete:",
            "",
        ]
        ranks = {a: [] for a in ARMS}
        from scipy.stats import rankdata  # noqa: PLC0415

        for d in comparable:
            scores = [
                next(
                    r["roc_auc"]
                    for r in complete
                    if r["dataset"] == d and r["arm"] == a
                )
                for a in ARMS
            ]
            for a, rank in zip(ARMS, rankdata(-np.array(scores)), strict=True):
                ranks[a].append(rank)
        lines.extend(
            f"- {a}: {np.mean(v):.3f}"
            for a, v in sorted(ranks.items(), key=lambda kv: np.mean(kv[1]))
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if complete:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        for dataset in {r["dataset"] for r in complete}:
            subset = [r for r in complete if r["dataset"] == dataset]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, metric in zip(
                axes,
                ("roc_auc", "log_loss", "rows_per_second", "fit_seconds"),
                strict=True,
            ):
                ax.bar([r["arm"] for r in subset], [r[metric] for r in subset])
                ax.set_title(metric)
                ax.tick_params(axis="x", rotation=35)
            fig.suptitle(dataset)
            fig.tight_layout()
            fig.savefig(root / dataset / "comparison.png", dpi=150)
            plt.close(fig)


def verify(root: Path) -> None:
    config = json.loads((root / "config.json").read_text())
    checked = []
    for dataset in config["datasets"]:
        path = root / dataset
        if not (path / "data.json").exists():
            continue
        meta = json.loads((path / "data.json").read_text())
        for name, h in meta["prepared_sha256"].items():
            assert digest(path / name) == h
        for name, h in meta["source_sha256"].items():
            assert digest(Path(name)) == h
        ids = np.load(path / "selection.npz")
        assert not np.isin(ids["train"], ids["holdout"]).any()
        assert np.isin(ids["stopping"], ids["holdout"]).all()
        assert np.isin(ids["order"], ids["train"]).all()
        target = pq.read_table(
            path / "valid.parquet", columns=[meta["target"]]
        )[meta["target"]].to_numpy()
        for arm in ARMS:
            output = path / arm
            f = output / "result.json"
            if not f.exists():
                continue
            report = json.loads(f.read_text())
            if report["status"] != "complete":
                continue
            pred = np.load(output / "external.npy", mmap_mode="r")
            labels = np.load(output / "labels.npy", mmap_mode="r")
            np.testing.assert_array_equal(target, labels)
            assert (
                len(pred)
                == meta["evaluation_rows"]
                == report["evaluated_rows"]
            )
            assert (
                digest(output / "external.npy") == report["prediction_sha256"]
            )
            for key, value in metrics(labels, pred).items():
                assert abs(value - report[key]) < 1e-12
            checked.append(f"{dataset}/{arm}")
    write_json(
        root / "verification.json",
        {
            "passed": True,
            "verified_completed_arms": checked,
            "requested_arms": 4 * len(config["datasets"]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--context-rows", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--holdout-cap", type=int, default=100000)
    parser.add_argument("--xgb-rounds", type=int, default=5000)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.worker:
        worker(root, args.dataset, args.worker)
        return
    if args.verify:
        verify(root)
        return
    config = {
        "datasets": args.datasets,
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "context_rows": args.context_rows,
        "steps": args.steps,
        "query_batch_size": args.query_batch_size,
        "validation_interval": args.validation_interval,
        "holdout_cap": args.holdout_cap,
        "xgb_rounds": args.xgb_rounds,
        "memory_bytes": LIMIT,
        "calibration": str(CALIBRATION.resolve()),
        "malware_prepared": str(
            (LEGACY / "20261013-zero-shot-capacity").resolve()
        ),
        "malware_selection": str(
            (LEGACY / "20261014-finetune-32768/7136641/selection.pt").resolve()
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists():
        if (
            not args.resume
            or json.loads((root / "config.json").read_text()) != config
        ):
            raise ValueError(
                "Existing run needs --resume with identical settings"
            )
    else:
        write_json(root / "config.json", config)
    sources = [
        Path(__file__),
        Path(__file__).with_name("large_binary_tuning.py"),
        Path("sdm/models/kumo/tabular/finetune.py"),
        Path("sdm/models/kumo/tabular/model.py"),
        Path("benchmark/tabular/microsoft_malware_context.py"),
    ]
    hashes = {str(p): digest(p) for p in sources}
    provenance = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pyarrow": pa.__version__,
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "code_sha256": hashes,
        "created": datetime.now(UTC).isoformat(),
    }
    if (root / "provenance.json").exists():
        assert (
            json.loads((root / "provenance.json").read_text())["code_sha256"]
            == hashes
        ), "Resume code changed; create a new run"
    else:
        write_json(root / "provenance.json", provenance)
        archive = root / "code"
        archive.mkdir(exist_ok=True)
        for p in sources:
            shutil.copy2(p, archive / (hashes[str(p)] + ".py"))
    summarize(root, args.datasets)
    for dataset in args.datasets:
        try:
            prepare(root, dataset, config)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            write_json(
                root / dataset / "preparation-failure.json",
                {"error": f"{type(error).__name__}: {error}"},
            )
            for arm in ARMS:
                failed = root / dataset / arm
                failed.mkdir(exist_ok=True)
                write_json(
                    failed / "result.json",
                    {
                        "status": "failed",
                        "dataset": dataset,
                        "arm": arm,
                        "error": (
                            f"Preparation failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    },
                )
            summarize(root, args.datasets)
            continue
        for arm in ARMS:
            output = root / dataset / arm
            result = output / "result.json"
            if result.exists():
                report = json.loads(result.read_text())
                if report["status"] == "complete":
                    assert report["data_sha256"] == digest(
                        root / dataset / "data.json"
                    )
                    continue
                if report["status"] == "failed" and not args.retry_failed:
                    continue
                shutil.move(
                    str(output),
                    str(output)
                    + ".attempt-"
                    + datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
                )
            if (
                arm == "paired"
                and not (
                    root / dataset / "tuned/optimized-context.pt"
                ).exists()
            ):
                output.mkdir(exist_ok=True)
                write_json(
                    result,
                    {
                        "status": "failed",
                        "dataset": dataset,
                        "arm": arm,
                        "error": "Tuned context snapshot unavailable",
                    },
                )
                continue
            command = [
                sys.executable,
                "-m",
                "benchmark.tabular.large_binary",
                "--run-dir",
                str(root),
                "--worker",
                arm,
                "--dataset",
                dataset,
            ]
            write_json(root / dataset / (arm + "-command.json"), command)
            print(json.dumps({"starting": dataset, "arm": arm}), flush=True)
            with (root / dataset / (arm + ".log")).open("a") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "8",
                        "MKL_NUM_THREADS": "8",
                        "HF_HUB_OFFLINE": "1",
                    },
                )
            if not result.exists() or (
                completed.returncode
                and json.loads(result.read_text())["status"] == "running"
            ):
                output.mkdir(exist_ok=True)
                write_json(
                    result,
                    {
                        "status": "failed",
                        "dataset": dataset,
                        "arm": arm,
                        "error": (
                            f"Worker exited {completed.returncode}; "
                            f"inspect {arm}.log"
                        ),
                    },
                )
            summarize(root, args.datasets)
    verify(root)
    summarize(root, args.datasets)


if __name__ == "__main__":
    main()
