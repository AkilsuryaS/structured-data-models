# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch

import sdm
from sdm.models.kumo.timeseries.forecast import DARR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sdm.models.KumoForecast(device=device)
darr = DARR(model, k=64, temperature=0.05, alpha=0.01)

url = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv"
frame = pd.read_csv(url)
series = frame[frame["item_id"] == "H170"].sort_values("timestamp")
table = sdm.TableTensor.from_tensor(
    torch.tensor(
        series[["target"]].to_numpy(), dtype=torch.float32, device=device
    ),
    columns=["target"],
)

# Keep the final 72 observations strictly out of both memory and query history.
history, held_out = table[:-72], table[-72:]
darr.fit(history, horizon=72, stride=72)
forecast = darr(history[-512:])
print(
    "DARR MAE:", (forecast.numerical - held_out.numerical).abs().mean().item()
)

# Memory is reused across queries. For rolling evaluation, never fit it using
# observations later than the earliest evaluated query's forecast origin.
