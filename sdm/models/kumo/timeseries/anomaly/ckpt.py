# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D103
    out = {}
    for key, value in ckpt.items():
        if key.startswith("diffmodel."):
            key = "denoiser." + key.removeprefix("diffmodel.")
            for old, new in (
                ("layers.0.self_attn.in_proj_", "attn.qkv_lin."),
                ("layers.0.self_attn.out_proj.", "attn.out_lin."),
                ("layers.0.linear1.", "mlp.0."),
                ("layers.0.linear2.", "mlp.3."),
                ("layers.0.norm1.", "attn_residual_norm."),
                ("layers.0.norm2.", "mlp_residual_norm."),
            ):
                key = key.replace(old, new)
            if key in (
                "denoiser.input_projection.weight",
                "denoiser.output_projection1.weight",
                "denoiser.output_projection2.weight",
            ):
                value = value.squeeze(-1)  # `Conv1d` to `Linear`.
        elif key.startswith(
            ("embed_layer.", "embed_norm.", "time_embed_norm.")
        ):
            key = "side_embedding." + key
        out[key] = value
    return out
