# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sdm.models.KumoForecast(device=device, alpha=0.01)

url = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv"
frame = pd.read_csv(url)
series = frame[frame["item_id"] == "H170"].sort_values("timestamp")
table = sdm.TableTensor.from_pandas(
    series[["target"]], stypes={"target": sdm.Stype.numerical}
).to(device)

# Keep the final 72 observations strictly out of both memory and query history.
history, held_out = table[:-72], table[-72:]
forecast = model(
    x_context=history[:, :0],
    y_context=history,
    x_query=torch.empty(72, 0, device=device),
)
print(
    "DARR MAE:", (forecast.numerical - held_out.numerical).abs().mean().item()
)
