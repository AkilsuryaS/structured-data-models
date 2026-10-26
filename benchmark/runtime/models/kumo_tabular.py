# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections.abc import Callable
from functools import partial

import torch
from torch.utils.benchmark import Compare, Measurement, Timer

import sdm


def _measure(name: str, fn: Callable[[], object], *, rows: int) -> Measurement:
    measurement = Timer(
        stmt="fn()",
        globals={"fn": fn},
        label="KumoTabular",
        sub_label=name,
        description="latency",
    ).blocked_autorange()
    print(f"{name:14} {rows / measurement.median:12.0f} rows/s")
    return measurement


def _preprocess(
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    num_estimators: int,
    generator: torch.Generator,
) -> tuple[sdm.EnsembleTable, ...]:
    recipe = sdm.models.KumoTabular.default_recipe()
    targets = recipe.target.fit_transform_ensemble(
        sdm.EnsembleTable.from_table(y_context, num_members=num_estimators),
        generator=generator,
    )
    context = recipe.features.fit_transform_ensemble(
        sdm.EnsembleTable.from_table(x_context, num_members=num_estimators),
        generator=generator,
    )
    query = recipe.features.transform_ensemble(
        sdm.EnsembleTable.from_table(x_query, num_members=num_estimators),
    )
    return context, targets, query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model",
        choices=(
            "kumo-tabular-small",
            "kumo-tabular-medium",
            "kumo-tabular-large",
        ),
        default="kumo-tabular-small",
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--context-rows", type=int, default=8192)
    parser.add_argument("--query-rows", type=int, default=4096)
    parser.add_argument("--num-columns", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--num-estimators", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    model = sdm.models.KumoTabular(
        task=args.task,
        size=args.model.removeprefix("kumo-tabular-"),
        pretrained=False,
        device=device,
    )

    generator = torch.Generator(device=device).manual_seed(0)
    x_context = sdm.TableTensor.from_tensor(
        torch.randn(
            size=(args.context_rows, args.num_columns),
            device=device,
            generator=generator,
        )
    )
    x_query = sdm.TableTensor.from_tensor(
        torch.randn(
            size=(args.query_rows, args.num_columns),
            device=device,
            generator=generator,
        )
    )
    if args.task == "classification":
        target = torch.arange(args.context_rows, device=device).remainder(2)
        target = target.unsqueeze(-1)
    else:
        target = torch.randn(
            size=(args.context_rows, 1),
            device=device,
            generator=generator,
        )
    y_context = sdm.TableTensor.from_tensor(target)

    print(
        f"\ncontext={args.context_rows} query={args.query_rows} "
        f"columns={args.num_columns}"
    )
    with torch.inference_mode():
        measurements = [
            _measure(
                name="preprocessing",
                fn=partial(
                    _preprocess,
                    x_context=x_context,
                    y_context=y_context,
                    x_query=x_query,
                    num_estimators=args.num_estimators,
                    generator=generator,
                ),
                rows=args.context_rows + args.query_rows,
            )
        ]
        with torch.autocast(
            device.type,
            dtype=dtype,
            enabled=dtype != torch.float32,
        ):
            measurements.append(
                _measure(
                    name="fit",
                    fn=partial(
                        model.fit,
                        x=x_context,
                        y=y_context,
                        num_estimators=args.num_estimators,
                        generator=generator,
                    ),
                    rows=args.context_rows,
                )
            )
            measurements.append(
                _measure(
                    name="predict",
                    fn=partial(model.predict, x_query),
                    rows=args.query_rows,
                )
            )
    Compare(measurements).print()


if __name__ == "__main__":
    main()
