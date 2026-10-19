# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("sklearn")

from benchmark.tabular.large_binary import (
    capacity,
    context_order,
    feature_spec,
    metrics,
    prepare,
    split_rows,
    stratified_sample,
)


def test_capacity_brackets_approximate_limit() -> None:
    def probe(size: int) -> bool:
        return size <= 12000

    size, history = capacity(100000, 20000, probe)
    assert size <= 12000
    assert size >= 10800
    assert any(not event["fits"] for event in history)
    assert history[-1]["rows"] >= size
    assert capacity(9000, 20000, probe)[0] == 9000


def test_capacity_minimum_failure() -> None:
    with pytest.raises(RuntimeError, match="two-class"):
        capacity(1000, 100, lambda _: False)


def test_stratified_context_is_nested_and_reproducible() -> None:
    labels = np.r_[
        np.zeros(9900, dtype=np.int64), np.ones(100, dtype=np.int64)
    ]
    ids = np.arange(len(labels))
    order = context_order(ids, labels, 17)
    assert len(np.unique(order)) == len(ids)
    for size in (2, 100, 1024, 8000):
        assert np.unique(labels[order[:size]]).tolist() == [0, 1]
        assert abs(labels[order[:size]].mean() - labels.mean()) <= 2 / size
    np.testing.assert_array_equal(order, context_order(ids, labels, 17))


def test_time_blocks_and_holdout_cap_remain_excluded() -> None:
    key = np.repeat(np.arange(20), 20)
    labels = np.tile([0, 1], len(key) // 2)
    train, holdout = split_rows(labels, key, "temporal", 17)
    assert max(key[train]) < min(key[holdout])
    stopping = stratified_sample(holdout, labels, 12, 17)
    assert np.isin(stopping, holdout).all()
    assert not np.isin(train, holdout).any()
    assert len(np.union1d(train, holdout)) == len(labels)


def test_group_split_does_not_leak_entities() -> None:
    keys = np.repeat(np.array([str(i) for i in range(1000)]), 2)
    labels = np.tile([0, 1], 1000)
    train, valid = split_rows(labels, keys, "group", 17)
    assert not set(keys[train]) & set(keys[valid])
    assert np.unique(labels[train]).size == np.unique(labels[valid]).size == 2


def test_metrics_follow_normalized_class_probabilities() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.7, 0.3], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
    result = metrics(y, p)
    assert (
        result["accuracy"]
        == result["roc_auc"]
        == result["average_precision"]
        == 1
    )
    assert result["prevalence"] == 0.5


def test_integer_categories_and_post_outcome_exclusion() -> None:
    schema = pa.schema(
        [
            (n, pa.int32())
            for n in [
                "ip",
                "app",
                "device",
                "os",
                "channel",
                "is_attributed",
                "attributed_time",
            ]
        ]
    )
    spec = feature_spec(schema, "talkingdata_adtracking")
    assert set(spec.values()) == {"categorical"}
    assert "attributed_time" not in spec
    assert "is_attributed" not in spec


def test_amex_uses_latest_statement_and_keeps_customers_disjoint(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")
    source = tmp_path / "source"
    processed = source / "amex_default_prediction/processed"
    processed.mkdir(parents=True)

    for name, start in (("train", 0), ("valid", 100)):
        ids = np.repeat(np.arange(start, start + 40), 2)
        table = pa.table(
            {
                "customer_ID": [str(x) for x in ids],
                "S_2": [datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)]
                * 40,
                "P_2": np.tile([1.0, 2.0], 40),
                "target": ids % 2,
            }
        )
        pq.write_table(table, processed / (name + ".parquet"))
    (processed / "manifest.json").write_text(
        json.dumps({"outer_split": {"strategy": "hash"}})
    )
    root = tmp_path / "output"
    root.mkdir()
    prepare(
        root,
        "amex_default_prediction",
        {"source": str(source), "seed": 17, "holdout_cap": 4},
    )
    path = root / "amex_default_prediction"
    train = pq.read_table(path / "train.parquet")
    valid = pq.read_table(path / "valid.parquet")
    assert len(train) == 40
    assert train["P_2"].to_pylist() == [2.0] * 40
    assert not set(train["customer_ID"].to_pylist()) & set(
        valid["customer_ID"].to_pylist()
    )
    assert "S_2" not in train.column_names
    assert "S_2_year" in train.column_names
    meta = json.loads((path / "data.json").read_text())
    assert "customer_ID" not in meta["features"]
    assert "target" not in meta["features"]
    assert meta["supervised_rows"] == 36
    assert meta["stopping_rows"] == 4


def test_bosch_reuses_parquet_and_preserves_physical_row_splits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    processed = source / "bosch_production/processed"
    processed.mkdir(parents=True)
    for role, start in (("train", 0), ("valid", 1000)):
        # Split by unordered IDs while retaining physical row identity.
        ids = np.arange(start, start + 200)[::-1]
        table = pa.table({"Id": ids, "Response": ids % 2, "F0": ids * 0.25})
        pq.write_table(
            table, processed / (role + ".parquet"), row_group_size=17
        )
    (processed / "manifest.json").write_text(
        json.dumps({"outer_split": {"strategy": "temporal_quantile"}})
    )
    root = tmp_path / "output"
    root.mkdir()
    config = {"source": str(source), "seed": 17, "holdout_cap": 10}
    prepare(root, "bosch_production", config)
    path = root / "bosch_production"
    original = pq.read_table(processed / "train.parquet")
    assert (path / "train.parquet").is_symlink()
    assert pq.read_table(path / "train.parquet").equals(original)
    train, holdout = split_rows(
        original["Response"].to_numpy(),
        original["Id"].to_numpy(),
        "temporal",
        17,
    )
    selection = np.load(path / "selection.npz")
    np.testing.assert_array_equal(selection["train"], train)
    np.testing.assert_array_equal(selection["holdout"], holdout)
    meta = json.loads((path / "data.json").read_text())
    assert list(meta["features"]) == ["F0"]
    assert meta["preparation_methods"]["train"] == "source_parquet"
    assert not (path / "sql-temp").exists()
    before = (path / "selection.npz").stat().st_mtime_ns
    prepare(root, "bosch_production", config)
    assert (path / "selection.npz").stat().st_mtime_ns == before


def test_streamed_calendar_matches_sql_without_changing_row_order(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    from benchmark.tabular.large_binary import prepare_table  # noqa: PLC0415

    # Cross the streaming batch boundary; include missing dates/timestamps.
    size = 33000
    table = pa.table(
        {
            "id": np.arange(size)[::-1],
            "click_time": [datetime.datetime(2020, 1, 5, 12), None]
            * (size // 2),
            "day": [datetime.date(2020, 1, 5), None] * (size // 2),
            "attributed_time": [datetime.datetime(2020, 1, 6)] * size,
            "target": np.arange(size) % 2,
        }
    )
    origin, destination = (
        tmp_path / "input.parquet",
        tmp_path / "output.parquet",
    )
    pq.write_table(table, origin, row_group_size=1000)
    assert prepare_table(origin, destination) == "streamed_calendar_features"
    actual = pq.read_table(destination)
    con = duckdb.connect()
    con.register("input_table", table)
    expressions = ["id"]
    for name in ("click_time", "day"):
        for part in ("year", "month", "day", "dow", "hour"):
            expressions.append(
                f"CAST(date_part('{part}', {name}) AS FLOAT) AS {name}_{part}"
            )
    expressions.append("target")
    expected = con.execute(
        "SELECT " + ", ".join(expressions) + " FROM input_table"
    ).to_arrow_table()
    con.close()
    assert actual.equals(expected)
    assert pq.ParquetFile(destination).num_row_groups == 2
