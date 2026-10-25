# GIFT-Eval

Evaluate SDM forecasting models on [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval).

> [!NOTE]
> Weights of `TimesFM3` are distributed under the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TimesFM3` benchmark, which will download its weights noninteractively.

## Setup

Run the commands below from the repository root:

```bash
pip install . pandas gluonts datasets python-dotenv
pip install --no-deps "git+https://github.com/SalesforceAIResearch/gift-eval.git@577a65cfa3acb812405c00d989230bf59f7f8a3d"
```

GIFT-Eval is not released on PyPI and is therefore pinned to a commit. It is installed without dependencies since it pins outdated `numpy` and `datasets` versions.

## Run

```bash
python -m benchmark.timeseries.gift_eval.main --model timesfm3
```

Pass `--dataset` to run only selected GIFT-Eval datasets, *e.g.*, `--dataset hospital --dataset ett1/H`.
Selected datasets are downloaded automatically to the Hugging Face cache and reused on subsequent runs. To use an existing local download, optionally set `GIFT_EVAL` to its root directory.
Batch size and context limits are configured per model in `benchmark/timeseries/model.py`.
Results are written to `benchmark/timeseries/gift_eval_out/<model>/`, and already completed configurations are skipped.

## Evaluate

```bash
python -m benchmark.timeseries.gift_eval.evaluate
```

This compares all available SDM results against published GIFT-Eval results on the same configurations, normalized by a seasonal naive baseline and aggregated by geometric mean.

`TimesFM3` follows the published `TimesFM-3` evaluation with symmetric averaging over two estimators and contexts truncated to 15,360 time steps. Its results can still differ from the published ones:

- Published results batch series of different context lengths together, which shifts their positional encodings on datasets with varying context lengths.
- Published results clip forecasts of non-negative series at zero.
- Published results forecast at most 32 variates jointly, subsampling covariates of `bizitobs_application` and `bizitobs_service`.
