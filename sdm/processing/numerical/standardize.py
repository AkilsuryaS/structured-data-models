# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import (
    _constant_feature_mask,
    _isfinite,
)


class Standardize(Processor, InvertibleMixin):
    """Center and scale each feature column.

    Constant columns use a unit scale to keep the transform finite and
    invertible. NaN and infinite values are ignored when fitting statistics
    and preserved during the transform.

    Args:
        eps: Value added to each fitted standard deviation.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True
    _packed_state = False

    mean: Tensor
    scale: Tensor

    def __init__(self, *, eps: float = 0.0) -> None:
        super().__init__()
        if eps < 0:
            raise ValueError("epsilon must be non-negative.")
        self.eps = eps
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))
        # Conditional fits use flat buffers; descriptors retain public shapes.
        self._packed_state = False

    def __getattr__(self, name: str) -> Any:
        value = super().__getattr__(name)
        if name in ("mean", "scale") and self._packed_state:
            if not bool(self._fitted_state):
                return self._buffers[f"_{name}_empty"]
            shape = cast(Tensor, self._buffers[f"_{name}_shape"])
            return cast(Tensor, value).reshape(tuple(shape.tolist()))
        return value

    def _set_buffer(self, name: str, value: Tensor) -> None:
        if name in ("mean", "scale") and self._packed_state:
            self.register_buffer(
                f"_{name}_shape",
                value.new_tensor(value.shape, dtype=torch.int64),
                persistent=False,
            )
            self._buffers[name] = value.flatten()
            return
        super()._set_buffer(name, value)

    def _save_to_state_dict(
        self, destination: Any, prefix: str, keep_vars: bool
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._packed_state:
            for name in ("mean", "scale"):
                if name not in self._non_persistent_buffers_set:
                    value = getattr(self, name)
                    destination[prefix + name] = (
                        value if keep_vars else value.detach()
                    )
            if "_fitted_state" not in self._non_persistent_buffers_set:
                destination[prefix + "_fitted_state"] = self._fitted_state.to(
                    self.mean.device
                )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self._packed_state:
            mean, scale = self.mean, self.scale
            self._packed_state = False
            super()._set_buffer("mean", mean)
            super()._set_buffer("scale", scale)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def with_state(self, mean: Tensor | float, scale: Tensor | float) -> Self:
        """Set fixed statistics in place and disable refitting.

        Args:
            mean: The mean to subtract.
            scale: The scale to divide by.
        """
        self._packed_state = False
        self._set_buffer("mean", torch.as_tensor(mean))
        self._set_buffer("scale", torch.as_tensor(scale))
        self.requires_fit = False
        return self

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        mean, scale = self._stats(table.numerical)
        self._packed_state = False
        super()._set_buffer("mean", mean)
        super()._set_buffer("scale", scale)

    def _supports_conditional_fit(self, table: TableTensor) -> bool:
        numerical = table.numerical
        return self.requires_fit and all(
            (
                value.dtype == numerical.dtype
                and value.device == numerical.device
            )
            or value.numel() == 0
            for value in (
                cast(Tensor, self._buffers["mean"]),
                cast(Tensor, self._buffers["scale"]),
            )
        )

    def _stats(self, numerical: Tensor) -> tuple[Tensor, Tensor]:
        finite = _isfinite(numerical)
        count = finite.sum(dim=-2, keepdim=True)
        finite_or_nan = numerical.masked_fill(~finite, torch.nan)
        mean = finite_or_nan.nansum(-2, keepdim=True).div_(count)
        mean.masked_fill_(mean.isnan(), 0.0)
        var = finite_or_nan.sub_(mean).square_().nansum(-2, keepdim=True)
        var /= count
        var.masked_fill_(var.isnan(), 0.0)
        scale = var.sqrt()
        if self.eps == 0:
            mask = _constant_feature_mask(var, mean, num_samples=count)
            scale[mask] = 1.0
        else:
            scale.add_(self.eps)
        return mean, scale

    def _fit_transform_conditional(
        self, table: TableTensor
    ) -> tuple[TableTensor, Tensor]:
        numerical = table.numerical
        self._defer_fitted()
        if not self._packed_state:
            mean, scale = self.mean, self.scale
            self.register_buffer("_mean_empty", mean, persistent=False)
            self.register_buffer("_scale_empty", scale, persistent=False)
            self._packed_state = True
            self._set_buffer("mean", mean)
            self._set_buffer("scale", scale)
        for name in ("mean", "scale", "_mean_shape", "_scale_shape"):
            value = cast(Tensor, self._buffers[name])
            dtype = (
                numerical.dtype if name in ("mean", "scale") else value.dtype
            )
            self._buffers[name] = value.to(
                device=numerical.device, dtype=dtype
            )

        def active(
            values: Tensor,
            mean: Tensor,
            scale: Tensor,
            mean_shape: Tensor,
            scale_shape: Tensor,
            fitted: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            new_mean, new_scale = self._stats(values)
            shape = torch.stack(
                tuple(
                    torch.scalar_tensor(
                        size, dtype=torch.int64, device=values.device
                    )
                    for size in new_mean.shape
                )
            )
            return (
                (values - new_mean).div_(new_scale),
                new_mean.flatten(),
                new_scale.flatten(),
                shape,
                shape.clone(),
                torch.ones_like(fitted),
            )

        def inactive(
            values: Tensor,
            mean: Tensor,
            scale: Tensor,
            mean_shape: Tensor,
            scale_shape: Tensor,
            fitted: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            return (
                values.clone(),
                mean.clone(),
                scale.clone(),
                mean_shape.clone(),
                scale_shape.clone(),
                fitted.clone(),
            )

        out, mean, scale, mean_shape, scale_shape, fitted = torch.cond(
            numerical.size(-1) > 0,
            active,
            inactive,
            (
                numerical,
                self._buffers["mean"],
                self._buffers["scale"],
                self._buffers["_mean_shape"],
                self._buffers["_scale_shape"],
                self._fitted_state,
            ),
        )
        self._buffers["mean"] = mean
        self._buffers["scale"] = scale
        self.register_buffer("_mean_shape", mean_shape, persistent=False)
        self.register_buffer("_scale_shape", scale_shape, persistent=False)
        self._set_buffer("_fitted_state", fitted)
        updated = numerical.new_ones(
            numerical.size(-1), dtype=torch.bool
        ).any()
        return table.replace_blocks(
            numerical=out.reshape(numerical.shape)
        ), updated

    def _supports_conditional_transform(self, table: TableTensor) -> bool:
        return self._packed_state and all(
            value.dtype == table.numerical.dtype
            and value.device == table.numerical.device
            for value in (
                cast(Tensor, self._buffers["mean"]),
                cast(Tensor, self._buffers["scale"]),
            )
        )

    def _transform_conditional(
        self, table: TableTensor, valid_state: Tensor
    ) -> TableTensor:
        numerical = table.numerical

        def active(
            values: Tensor,
            mean: Tensor,
            scale: Tensor,
            mean_shape: Tensor,
            scale_shape: Tensor,
            valid: Tensor,
        ) -> Tensor:
            torch._assert_async(
                valid, "Fitted processor state does not match the input"
            )

            def restore(flat: Tensor, shape: Tensor) -> Tensor:
                dimensions = torch.cat((shape.new_ones(values.ndim), shape))[
                    -values.ndim :
                ]
                size = (*values.shape[:-2], 1, values.size(-1))
                # Broadcast statistics using tensor-valued sizes.
                offset = torch.zeros(
                    size, device=values.device, dtype=torch.int64
                )
                stride = dimensions.new_ones(())
                for dim in range(values.ndim - 1, -1, -1):
                    width = size[dim]
                    coordinate = torch.arange(
                        width, device=values.device
                    ).reshape(
                        (1,) * dim + (width,) + (1,) * (values.ndim - dim - 1)
                    )
                    compatible = (dimensions[dim] == 1) | (
                        dimensions[dim] == width
                    )
                    torch._assert_async(
                        compatible, "Statistics cannot broadcast to the input"
                    )
                    offset = (
                        offset
                        + coordinate.remainder(dimensions[dim].clamp_min(1))
                        * stride
                    )
                    stride = stride * dimensions[dim]
                padded = torch.cat((flat, flat.new_zeros(1)))
                return padded[offset.clamp_max(flat.numel())]

            return (values - restore(mean, mean_shape)).div_(
                restore(scale, scale_shape)
            )

        out = torch.cond(
            numerical.size(-1) > 0,
            active,
            lambda values, mean, scale, mean_shape, scale_shape, valid: (
                values.clone()
            ),
            (
                numerical,
                self._buffers["mean"],
                self._buffers["scale"],
                self._buffers["_mean_shape"],
                self._buffers["_scale_shape"],
                valid_state & self._fitted_state,
            ),
        )
        return table.replace_blocks(numerical=out.reshape(numerical.shape))

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical - self.mean).div_(self.scale)
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        dtype = table.numerical.dtype
        numerical = (table.numerical * self.scale).add_(self.mean)
        return table.replace_blocks(numerical=numerical.to(dtype=dtype))

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}(eps={self.eps})"
