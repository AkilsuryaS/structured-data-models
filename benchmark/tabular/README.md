# Tabular Benchmarks

This directory contains benchmarks of `structured-data-models` on TabArena/BeyondArena, [ScoringBench](scoringbench/), and [TALENT](talent/).

## TabArena and BeyondArena setup

Run the commands below from the repository root:

```bash
pip install . \
  "autogluon.tabular>=1.6.4b20260924,<=1.7.0" \
  "tabarena[data-foundry,plot,preprocessing]>=0.1.1.dev20260924104714,<=0.2.0"
```

______________________________________________________________________

> [!NOTE]
> Weights of `TabFM` are distributed under the [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TabFM` benchmark, which will download its weights noninteractively.

## TabArena

### Run

- **`TabICLv2`:**

  ```bash
  python -m benchmark.tabular.tabarena.main --model tabiclv2
  ```

- **`KumoTabular`:**

  ```bash
  python -m benchmark.tabular.tabarena.main --model kumo-tabular-large
  python -m benchmark.tabular.tabarena.main --model kumo-tabular-medium
  python -m benchmark.tabular.tabarena.main --model kumo-tabular-small
  ```

- **`TabFM`:**

  ```bash
  python -m benchmark.tabular.tabarena.main --model tabfm
  ```

Pass a dataset name to run only that TabArena dataset:

```bash
python -m benchmark.tabular.tabarena.main \
  --model kumo-tabular-large \
  --dataset blood-transfusion-service-center
```

### Fine-tuning

`--model tabiclv2-ft`, `--model kumo-tabular-large-ft`, `--model kumo-tabular-small-ft`, and `--model tabfm-ft` full fine-tune every parameter of the model on each dataset's training split before evaluating (see `benchmark/tabular/finetune.py`):

```bash
python -m benchmark.tabular.tabarena.main \
  --model tabiclv2-ft \
  --finetune_epochs 75 \
  --finetune_lr 1e-6
```

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.tabarena.evaluate
```

______________________________________________________________________

## BeyondArena

### Run

- **`TabICLv2`:**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model tabiclv2
  ```

- **`KumoTabular`:**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model kumo-tabular-large
  python -m benchmark.tabular.beyondarena.main --model kumo-tabular-medium
  python -m benchmark.tabular.beyondarena.main --model kumo-tabular-small
  ```

- **`TabFM`:**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model tabfm
  ```

By default, each command evaluates the recommended `core` subset.
Repeat `--subset` to combine filters:

```bash
python -m benchmark.tabular.beyondarena.main \
  --model tabiclv2 \
  --subset core \
  --subset grouped
```

Pass a dataset name to run only that BeyondArena dataset.
Use `--subset lite` for its first split:

```bash
python -m benchmark.tabular.beyondarena.main \
  --model tabiclv2 \
  --dataset parkinsons_biomedical_voice_measurements \
  --subset lite
```

Available subset filters include problem types (`classification`, `regression`), size buckets (`tiny`, `small`, `medium`, `large`), split regimes (`iid`, `temporal`, `grouped`), feature groups (`low-dim`, `high-dim`, `text`, `high-cardinality`), and split selections (`core`, `lite`, `all`). Prefix a filter with `!` to negate it.

### Fine-tuning

Same `-ft` model variants and `--finetune_*` flags as TabArena above:

```bash
python -m benchmark.tabular.beyondarena.main --model kumo-tabular-small-ft --subset lite
```

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.beyondarena.evaluate
```

## Microsoft Malware context-fit comparison

Run the cached-data comparison from the repository root:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m benchmark.tabular.microsoft_malware_context
```

This requires the local Microsoft Malware Parquet cache at `/mnt/d/tfm-scale-large-data-cache/microsoft_malware/processed/`, a cached Kumo-Tabular Small v1.0.1 checkpoint, CUDA with BF16 support, and `scikit-learn` for scoring. It uses one estimator and compares explicitly sized tuning (`--finetune`, default `0.02`) with `finetune=0` under a 40,000,000,000-byte CUDA allocator limit. Native-memory checks are separate. The 100k-row pilot evaluates 50k held-out rows; subsequent 1m-row and conditional 2m-row experiments evaluate the complete existing IID held-out split. Positive `finetune` always tunes, even if ordinary inference fits. The `automatic` worker name is retained for historical artifact compatibility; it now uses the explicitly requested context size.

The runner compares the exact initial tuning context and the largest feasible zero-shot context from the same non-validation training pool. A genuine backward/update probe estimates the default 20-epoch training time. Estimates over eight hours stop at preflight without reducing context or changing training settings. Predictions, split identities, provenance, timing events, and a comparison table are saved under `.cache/microsoft-malware-context-fit/` and remain outside source control. Use `--scales 100000` for only the pilot, or `--run-dir <fresh-path>` for an explicit output directory. Each GPU arm runs sequentially in a fresh process; its exact command is recorded for reproduction.

Resume an interrupted run with `--resume --run-dir <existing-path>`; completed arms are reused and failed attempts are archived before retrying. Recompute scores and verify source hashes, split identities, and completed-arm claims with `--verify --run-dir <existing-path>`. Metrics normalize clipped probabilities in float64 to remove float32 row-sum roundoff; `--verify --rescore` preserves previous metric reports and updates scores from saved predictions without GPU inference.

To measure zero-shot capacity using all 7,136,641 training rows before tuning, prepare a fresh run with `prepare(run_dir, [100000, 1000000, 7136641])`, then run `--run-dir <run_dir> --worker capacity --rows 7136641 --full-validation`. This tests the complete training context first, searches approximate capacity within 10,000 rows without the tuning row cap, and scores the selected context on all 1,784,842 held-out rows. Run `--worker full --rows 100000 --full-validation` in the same directory for a baseline on the identical validation split. Capacity probes include first and last validation batches; a result is complete only after the entire validation prediction pass succeeds. These are zero-shot experiments with one estimator, BF16 execution, and the same 40 GB limit.

For a bounded fine-tuning smoke experiment, use `--worker automatic --rows 7136641 --full-validation --max-steps 100 --runtime-gate-hours 0` with the prepared full-data run. This overrides the epoch budget with exactly 100 updates, uses cosine decay over those updates, evaluates the complete internal validation holdout before and after the partial epoch, restores the best context, and scores the complete external validation split. Add `--finetune 163840` to request the previous experiment’s exact context size. A request that exceeds the training pool or GPU budget fails instead of shrinking the context. Run `--worker paired --rows 7136641 --full-validation` afterward to score the same initial context without tuning. These results must be labeled as a smoke run rather than the default 5–20-epoch schedule.

## Context tuning API

`model.fit(x, y)` and `model.fit(x, y, finetune=0)` perform ordinary zero-shot fitting. `finetune=False` remains equivalent to zero. `finetune=True` enables tuning with `min(32_768, training_rows_after_holdout)` context rows per estimator. Positive numeric values request explicit context sizes and always tune; there is no automatic full-context skip or context-capacity search:

```python
model.fit(x, y)  # zero-shot, 8 estimators by default
model.fit(x, y, n_estimators=1)  # zero-shot, one estimator
model.fit(x, y, finetune=True)  # up to 32,768 stratified rows, one tuned estimator
model.fit(x, y, finetune=True, n_estimators=2)  # two independently tuned contexts
model.fit(x, y, finetune=100_000)  # 100,000 context rows, one tuned estimator
model.fit(x, y, finetune=0.1)  # floor(10% of the rows supplied to fit)
model.fit(x, y, finetune=4096, n_estimators=8)  # 8 independently tuned contexts
model.fit(x, y, finetune=True, query_batch_size=2048)  # override the 4,096 default
```

Fine-tuning uses `query_batch_size=4096` by default, capped by the available non-validation training rows for each update. This setting also controls internal validation batch size. Historical benchmark results retain their recorded batch sizes.

Context sampling excludes the fixed stratified 10% internal validation holdout. The requested context must have at least two rows and fit within the remaining training pool and CUDA memory budget. There is no fixed 262,144-row cap; unavailable or oversized requests fail without changing their row count. Counts may be positive integers or whole-valued floats; fractions must be strictly between zero and one. Negative, nonfinite, and nonintegral counts are rejected. `True` automatically sizes the context after reserving the holdout; explicit numeric sizes are not silently reduced. The full non-validation training pool still supplies optimization queries; `finetune` specifies context size, not the number of training updates. Tuning currently supports binary classification. `n_estimators` defaults to eight for zero-shot fitting and one for tuning; the existing `num_estimators` spelling remains an alias. Specify only one spelling, with a positive integer count. Each tuned estimator uses stratified random sampling without replacement, matching class proportions in the non-validation training pool as closely as possible while including both classes; contexts may overlap between estimators. All members share a fixed internal validation holdout, excluded from contexts and optimization queries. Each member independently fits preprocessing and trains a dense embedding adjustment, with its own Adam optimizer, cosine schedule, early stopping, and best-checkpoint restoration. The requested `finetune` size and training budget apply **per estimator**. Training and prediction run members sequentially using shared frozen model weights; prediction averages each member’s postprocessed probabilities. The memory budget includes previously fitted member caches. Requests that cannot retain the ensemble caches fail and discard the partial fit. `finetune_result_` contains per-member reports and aggregate context positions and completed steps for ensembles.

## Synthetic Kumo memory calibration

Run `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -m benchmark.tabular.memory --run-dir .cache/kumo-memory/<run>` to measure Small, Medium, and Large binary classifiers with one estimator, BF16 neural execution, CPU source tables, and a 40 GB allocator limit. The test set always contains 1,024 rows. The calibration grid varies context rows, numeric/categorical composition, and categorical cardinality (16 and 64); missingness is 2%. Held-out shapes include larger contexts and the default recipe's 500-column cap. Each case runs in a fresh CUDA process. Use `--resume` with the same directory to reuse completed cases, or `--calibrate` to refit equations from saved measurements. Use `--verify` to audit checkpoint/code fingerprints, processed-column counts, held-out error, conservative reservation coverage, and OOM predictions. SciPy is required for nonnegative regression.

The benchmark measures loaded model memory, encoder peaks, ICL cache-construction peaks, persistent cache bytes, and cold/warm prediction peaks. Allocated and reserved CUDA bytes are recorded separately. Synchronized timing excludes model loading. OOM observations remain censored and are excluded from regression. Separate phase equations use `1, R, C, R*C` plus capped `R*C` terms at 1,048,576, 4,194,304 and 16,777,216 cells and a capped row-token term derived from the checkpoint width and chunk budget to model automatic workspace batching, where `R` is context rows and `C` is the number of processed columns. Peak fit memory is the maximum of encoding and ICL construction; total required memory also includes the prediction phase. Reserved memory can retain temporary fit allocations during prediction.

Load the resulting estimator without CUDA:

```python
from sdm.models.kumo.tabular import KumoMemoryEstimator

estimator = KumoMemoryEstimator.from_file(".cache/kumo-memory/<run>/calibration.json")
result = estimator.estimate(
    checkpoint="small",  # alternatively, the calibrated checkpoint SHA256
    context_rows=100_000,
    numeric_columns=40,
    categorical_columns=40,
    high_cardinality_columns=3,
)
print(result["conservative_peak_bytes"] / 1e9)
```

Without `high_cardinality_columns`, the estimator assumes every categorical column adds a count feature and also returns the low-cardinality estimate. The default recipe adds counts at cardinality 50 and caps processed columns at 500; constant columns can reduce the actual count. Calibration artifacts record checkpoint hashes, code revision/hashes, precision, allocator settings, hardware, measured peaks, fitted equations, and held-out error. Estimates apply to those execution settings, not arbitrary tasks, precisions, recipes, query sizes or estimator counts. Out-of-range inputs are flagged; memory feasibility still requires an actual inference probe.

## Six large binary classification datasets

Run the local six-dataset comparison with:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m benchmark.tabular.large_binary \
  --run-dir .cache/large-binary/20261016-six-datasets
```

This requires the prepared competition cache at `/mnt/d/tfm-scale-large-data-cache`, CUDA with BF16 support, the cached Small v1.0.1 checkpoint, the existing synthetic memory calibration at `.cache/kumo-memory/20261014-grid/calibration.json`, and the saved Malware split/prepared files from the context experiments. Benchmark-only dependencies include DuckDB, pandas, scikit-learn, SciPy, matplotlib, XGBoost with CUDA support, and CuPy. Source data, prepared tables, fitted artifacts, predictions, and reports remain outside source control. Unchanged tables (including Bosch and Avazu) reference source Parquet files directly. Calendar expansion streams in bounded batches without a global sort. AMEX aggregation and the Red Hat join retain their SQL preparation path; allow space for their temporary files.

The datasets run in this order: Microsoft Malware, Red Hat Business Value, American Express Default Prediction, Bosch Production, Avazu CTR, and TalkingData AdTracking. Four fresh-process GPU arms run sequentially per dataset:

- `xgboost`: installed library defaults with binary logistic objective, CUDA histogram execution, native categorical features, seed 17, eight CPU threads, a 5,000-round limit, and 100-round early stopping on internal log loss. No HPO or holdout refit.
- `maximum`: one zero-shot Kumo-Small estimator, with approximate maximum inference context under a 40,000,000,000-byte PyTorch allocator limit. The calibrated estimate only brackets actual encoding/cache/query probes. Capacity refinement stops within 10% or 1,024 rows. Full evaluation OOM reduces the context by 10% and restarts prediction from the beginning.
- `tuned`: one Kumo-Small estimator with 32,768 stratified context rows, query batches of 4,096, at most 2,000 updates, 50-update warmup, Adam learning rate 0.001, cosine minimum 0.00001, and gradient norm clipping at 1. Score every 100 updates; stop after five insufficient improvements once 500 updates have completed, with minimum improvement 0.0001. Restore the lowest-loss checkpoint, including the initial context. This is a benchmark schedule, not the public API's default epoch schedule.
- `paired`: the exact initial context, fitted preprocessing, and frozen embeddings of the tuned arm, replayed without its learned adjustment. This isolates the effect of optimization from context selection and context size.

The existing outer evaluation partitions remain fixed and are scored in full. AMEX retains the latest statement per customer, with original row order breaking date ties. Red Hat joins the people table with prefixed columns, requiring a many-to-one join. Entity/row IDs and TalkingData's post-outcome `attributed_time` are excluded. Dates become calendar fields. Explicit integer-coded categories retain categorical semantics. Each method receives the same eligible raw features; Kumo's native processed-column selection can retain fewer features than XGBoost.

Reserve an internal 10% holdout before any context selection: stratified customers for AMEX, people-disjoint hashes for Red Hat, complete latest time blocks for Avazu/TalkingData, and latest ID-order blocks for Bosch. Malware reuses its saved split identities and matching reordered source copies. Bosch's split is a provisional production-order proxy, not a verified chronological split. All internal holdout rows stay excluded from supervised training; a fixed stratified subset of at most 100,000 rows is shared for early stopping. Queries may overlap the sampled context but never either held-out partition.

Use `--datasets microsoft_malware red_hat_business_value` to select datasets. Use `--resume` with the original settings to reuse compatible completed arms; `--retry-failed` additionally archives and retries failed arms. Changed settings or code require a new output directory. `--verify --run-dir <path>` independently rechecks source/prepared hashes, split exclusion, full prediction coverage, and saved metrics. Smaller explicit `--context-rows`, `--steps`, `--query-batch-size`, `--validation-interval`, `--holdout-cap`, and `--xgb-rounds` values support pilot runs; these changes are recorded in the run configuration.

Each run writes a live `report.md`, `results.json`, per-arm logs/results, prediction arrays, code provenance, context artifacts, XGBoost models, and per-dataset comparison plots. Metrics include ROC AUC, average precision, accuracy at 0.5, log loss, and prevalence. Inference timing synchronizes CUDA and includes batch feature conversion and transfer, excluding Parquet reads, prediction writes, and scoring; end-to-end prediction time is reported separately. Device memory polling includes other GPU processes; Kumo also records allocator peaks. Mean AUC ranks use only datasets with all four completed arms. Failed arms remain visible, including preparation failures. A completed prediction pass and independently recomputed metrics are required for a successful arm.
