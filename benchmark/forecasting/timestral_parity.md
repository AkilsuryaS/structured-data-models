# Timestral 5M parity

Run from the repository root with the upstream reference installed:

```bash
pip install git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
python -m benchmark.forecasting.timestral_parity --device cpu --repeats 20
python -m benchmark.forecasting.timestral_parity --device cuda --repeats 20
```

The script verifies the installed source revision and uses `mistralai/Timestral-5M` revision `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`. Downloading weights is subject to their [Modified MIT license](../../sdm/models/timestral/WEIGHTS_LICENSE).

It compares upstream `TimestralForecaster.forecast` with SDM's public `Timestral.forward`, using default sign averaging, nonnegative clipping, and normalization refinement. It also compares SDM's fitted `predict` with its one-shot output. Deterministic inputs cover a short patch, univariate and multivariate targets, independent batches, missing and entirely missing targets, constant and positive targets, numerical and binary covariates, categorical past and known-future covariates, unseen and missing categories, missing target history with categorical covariates, a 1,030-step forecast, and truncation of a 32,775-step context.

Recorded on 2026-10-17 with PyTorch `2.14.0+cu130`, a Cortex-X925 CPU using two PyTorch threads, and an NVIDIA GB10 GPU:

| Comparison                    | Result                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| CPU, all 15 cases             | Bit-for-bit equal                                                                         |
| CUDA, horizons up to 65 steps | Maximum absolute error `1.67e-5`                                                          |
| CUDA, 1,030-step horizon      | Maximum absolute error `0.0889` in extreme quantile tails; maximum scaled error `1.64e-5` |
| CUDA, all 15 cases            | Pass `atol=rtol=1e-4`                                                                     |
| Numeric covariates, CPU       | 83.9 ms per public forward                                                                |
| Numeric covariates, CUDA      | 12.0 ms per public forward                                                                |

Scaled error is `abs(actual - reference) / (1 + abs(reference))`. The long-horizon example produces large tail quantiles; its absolute error should be interpreted alongside its scaled error. Timings use one series, 79 context steps, 65 forecast steps, one numerical past covariate, one numerical known-future covariate, three warmups, and 20 timed calls. GPU timing uses CUDA events and synchronizes the end event. These timings characterize this workload and hardware, not a general speedup claim.

Only 5M was checked against pretrained weights. The 28M, 123M, and 209M variants use pinned configurations and checkpoint loading but have not been parity-tested. Parity is scoped to float32; lower precision is unvalidated.

The integration needs no upstream package at runtime. Optional pretrained pytest comparisons skip when upstream or cached 5M weights are unavailable. The ordinary public-contract and neural-block tests use random weights and run offline. SDM represents categorical values with category indices; for multivariate targets those indices become the raw numerical covariates, matching upstream when supplied the same codes.

Separately tensorized string vocabularies are aligned by category value at the model boundary using SDM's relational join. This uses cuDF on CUDA when installed and the existing CPU fallback otherwise. Forecast tensor execution and target encoding stay on the input device.
