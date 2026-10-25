# Time-Series Forecasting Benchmarks

This directory contains time-series benchmark integrations for `structured-data-models`:

- [GIFT-Eval](gift_eval/): General time-series forecasting evaluation.

All benchmarks share one model adapter (see [model.py](model.py)) that maps each forecasting task onto the `ICLModel` interface: time steps become rows, past targets become `y_context`, all covariates and timestamps of the context become `x_context`, and the forecast horizon becomes `x_query` with one row of known covariates and timestamps per future time step. Columns keep their semantic types, so each model decides which ones it consumes. Series in a batch are left-padded with missing values, and multivariate series are flattened for models without multi-target support.
