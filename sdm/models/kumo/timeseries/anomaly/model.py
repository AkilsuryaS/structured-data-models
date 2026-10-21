# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

import copy
from collections.abc import Sequence
from functools import partial
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import Recipe, Stype, TableTensor
from sdm._inference import inference_mode
from sdm._warnings import warn_once
from sdm.models import Callback, Model
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.anomaly.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.models.kumo.timeseries.anomaly.embedding import SideEmbedding
from sdm.models.kumo.timeseries.anomaly.recipe import default_recipe
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule


class KumoAnomaly(Model):
    r"""The time series anomaly detection foundation model from the
    `"NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for
    Multivariate Time-Series Anomaly Detection"
    <https://arxiv.org/abs/2610.12754>`__ paper.

    :class:`KumoAnomaly` reconstructs multivariate time series with conditional
    diffusion and uses reconstruction errors to score anomalies. It combines
    sinusoidal time embeddings and learned feature embeddings with
    diffusion-step and masking-strategy embeddings. Its denoiser alternates
    attention over time and features with gated residuals and aggregated skip
    connections that predict noise at each time step and feature.

    At inference, two complementary masks hide alternating temporal blocks.
    Reverse diffusion reconstructs each hidden block from the visible context,
    and the sample median combines repeated reconstructions. Mean absolute
    reconstruction error across features gives a score per time step; larger
    scores indicate less accurately reconstructed observations.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        # Build on the target device, since the checkpoint does not contain
        # the non-persistent sinusoidal lookup of the step embedding:
        self.model = _KumoAnomaly(device=device)

        if pretrained:
            self._load_from_pretrained(device=device)

        self.eval()

    def _load_from_pretrained(
        self,
        device: torch.device | str | None,
    ) -> KumoAnomaly:
        device = torch.get_default_device() if device is None else device

        path = download_checkpoint(
            repo_id="nvidia/Kumo-Anomaly",
            filename="final_model.pth",
            revision="226c5003b0582adc1cac433123f47324bfae39f2",
        )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(remap_ckpt(ckpt["model"]))

        return self

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def forward(
        self,
        x: Tensor | TableTensor,
        *,
        recipe: Recipe | None = None,
        num_samples: int = 30,
        num_steps: int = 20,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        r"""The forward pass.

        Args:
            x: The feature tensor with shape ``[..., R, D]`` with ``R`` rows
                and ``D`` columns.
            recipe: The custom recipe for pre-processing.
            num_samples: The number of reconstruction samples per complementary
                masking strategy. Reconstructions are combined using the
                sample median.
            num_steps: The number of denoiser evaluations per sample and
                masking strategy.
            callbacks: Callbacks applied in sequence to this model call.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.

        Returns:
            The anomaly scores with shape ``[..., R, 1]``, denoting the mean
            absolute reconstruction error across preprocessed features at each
            time step.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = self.training
        requires_grad |= any(callback.requires_grad for callback in callbacks)

        param = next(self.parameters())
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        x = cast(TableTensor, x.to(param.device))

        recipe = (
            self.default_recipe() if recipe is None else copy.deepcopy(recipe)
        )
        with (
            torch.amp.autocast(param.device.type, enabled=False),
            inference_mode("no_grad" if requires_grad else "inference"),
        ):
            x = recipe.features.fit_transform(x, generator=generator)

        for callback in callbacks:
            x, _ = callback.on_query_preprocessing_end(self, x, None)

        invalid = x.active_stypes - self.supported_feature_stypes - {Stype.id}
        if len(invalid) > 0:
            stypes = ", ".join(f"{str(stype)!r}" for stype in invalid)
            warn_once(
                key="model-unsupported-feature-stypes",
                message=(
                    f"{self.__class__.__name__!r} received unsupported "
                    f"feature stypes {stypes}. Columns with unsupported "
                    f"feature stypes will not be consumed by the model."
                ),
            )
        if x.numerical.size(-1) > self.model.num_features:
            raise ValueError(
                f"{self.__class__.__name__!r} supports at most "
                f"{self.model.num_features} features after pre-processing "
                f"(got {x.numerical.size(-1)})"
            )

        with inference_mode("grad" if requires_grad else "inference"):
            out = self.model(
                x.numerical.to(param.dtype),
                num_samples=num_samples,
                num_steps=num_steps,
                generator=generator,
            )
            out = TableTensor(
                columns={Stype.numerical: ["score"]},
                numerical=out.unsqueeze(-1),
            )
            for callback in callbacks:
                out = callback.on_model_forward_end(self, out)
        return cast(TableTensor, out.to(x.dtype))


class _KumoAnomaly(torch.nn.Module):
    def __init__(
        self,
        num_features: int = 40,
        time_channels: int = 256,
        feature_channels: int = 64,
        channels: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        embedding_channels: int = 256,
        num_steps: int = 500,
        window_size: int = 100,
        num_blocks: int = 4,
        batch_size: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_features = num_features
        self.num_steps = num_steps
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.batch_size = batch_size

        self.side_embedding = SideEmbedding(
            num_features=num_features,
            time_channels=time_channels,
            feature_channels=feature_channels,
            **factory_kwargs,
        )
        self.denoiser = DiffusionDenoiser(
            channels=channels,
            side_channels=time_channels + feature_channels,
            num_steps=num_steps,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_channels=embedding_channels,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        num_samples: int,
        num_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:  # [..., R]

        *size, length, features = x.size()
        x = x.reshape(-1, length, features)
        x = F.pad(x, (0, self.num_features - features))

        # Pad the last window by repeating the last row:
        padding = -length % self.window_size
        x = torch.cat([x, x[:, -1:].expand(-1, padding, -1)], dim=1)
        windows = x.reshape(-1, self.window_size, self.num_features)

        schedule = DiffusionSchedule(
            num_steps=self.num_steps,
            beta_start=0.0001,
            beta_end=0.01,
            schedule="linear",
            device="cpu",
        )
        sampler = DPMSolver(schedule.beta.float(), num_steps).to(x.device)

        out = torch.cat(
            [
                self.reconstruct(batch, sampler, num_samples, generator)
                for batch in windows.split(self.batch_size)
            ]
        )
        out = out.reshape(x.size())[:, :length, :features]
        out = (out - x[:, :length, :features]).abs().mean(dim=-1)
        return out.reshape(*size, length)

    def reconstruct(
        self,
        x: Tensor,  # [B, L, K]
        sampler: DPMSolver,
        num_samples: int,
        generator: torch.Generator | None,
    ) -> Tensor:  # [B, L, K]

        x = x.transpose(1, 2)  # [B, K, L]
        batch, _, length = x.size()
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-5
        x = ((x - mean) / std).clamp(-10, 10)

        time = torch.arange(length, device=x.device).expand(batch, -1)
        blocks = time // (length // self.num_blocks)

        out = x.new_zeros((num_samples, *x.size()))
        for strategy in range(2):
            mask = ((blocks % 2) == strategy).to(x.dtype)
            mask = mask.unsqueeze(1).expand_as(x)  # [B, K, L]
            predict_noise = partial(
                self.predict_noise,
                cond=mask * x,
                target_mask=1 - mask,
                side_info=self.side_embedding(time, mask),
                strategy=torch.tensor([strategy], device=x.device),
            )
            for i in range(num_samples):
                noise = torch.empty_like(x).normal_(generator=generator)
                sample = sampler(noise, predict_noise) * std + mean
                out[i] += (1 - mask) * sample

        return out.median(dim=0).values.transpose(1, 2)

    def predict_noise(
        self,
        x: Tensor,  # [B, K, L]
        step: Tensor,  # [1]
        *,
        cond: Tensor,  # [B, K, L]
        target_mask: Tensor,  # [B, K, L]
        side_info: Tensor,  # [B, S, K, L]
        strategy: Tensor,  # [1]
    ) -> Tensor:  # [B, K, L]
        return self.denoiser(
            x=torch.stack([cond, target_mask * x], dim=1),
            side_info=side_info,
            step=step,
            strategy=strategy,
        )
