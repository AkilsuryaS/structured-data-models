# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ModuleDict

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.models import ECOC, ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import Callback
from sdm.models.base.model import _Prefetcher
from sdm.models.kumo.tabular.finetune import _context_rows, fit_context
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.models.kumo.tabular.row_embedding import RowEmbedding

MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "small": {
        "cell_channels": 128,
        "num_embedding_layers": 4,
        "num_embedding_heads": 4,
        "num_inducing_points": 128,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 12,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": None,
    },
    "medium": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 24,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": 2,
    },
    "large": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 1024,
        "num_icl_layers": 24,
        "num_icl_heads": 16,
        "num_icl_key_value_heads_for_query": 2,
    },
}


class KumoTabular(ICLModel):
    r"""The tabular foundation model from `"NVIDIA Kumo Tabular Sets a New
    Accuracy-Efficiency Frontier for Tabular Prediction"
    <https://huggingface.co/blog/nvidia/kumo-tabular>`__.

    .. figure:: /images/kumo_tabular.svg
        :width: 100%

    :class:`KumoTabular` processes a table in two stages. First, an interleaved
    row/column encoder transforms raw table cells into fixed-size row
    representations. Numerical and categorical cells use separate learned
    Fourier frequencies and projections, together with a learned missingness
    projection. Each encoder stage consists of:

    * **Column-wise:** Each feature group is processed across rows using
      induced set attention. Both context and query cells attend only to
      context-row keys and values.
    * **Row-wise:** Each row's feature groups and learnable readout tokens
      attend to one another, combining feature interactions into a fixed-size
      row representation.

    Second, a dataset-wise in-context learning transformer processes the row
    representations and predicts each query target from the labeled context
    rows. The ``"large"`` model widens this transformer to 1024 channels and
    16 attention heads. Its four 256-channel readout tokens concatenate
    directly to that width without a projection layer.

    For regression tasks, :class:`KumoTabular` predicts 999 quantiles named
    ``"q001"`` through ``"q999"``.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized.
        size: The model size, one of ``"small"``, ``"medium"``, or
            ``"large"``. Defaults to ``"large"``.
        pretrained: Whether to load pretrained checkpoints.
        device: The device for model parameters. If ``None``, uses PyTorch's
            default device.
        n_estimators: Default number of estimators for :meth:`fit`, including
            embedding tuning. Must be a positive integer; defaults to eight.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_multi_target: ClassVar[bool] = False
    supports_context_only_columns: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
        size: Literal["small", "medium", "large"] = "large",
        pretrained: bool = True,
        device: torch.device | str | None = None,
        *,
        n_estimators: int = 8,
    ) -> None:
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, Integral)
            or n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        super().__init__(task=task)
        self.n_estimators = int(n_estimators)
        if Task.classification in self.tasks:
            self.ecoc = ECOC(max_classes=10)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _KumoTabular(
                num_classes=10 if task == Task.classification else 0,
                num_quantiles=999 if task == Task.regression else 0,
                device="meta" if pretrained else device,
                **MODEL_KWARGS[size],
            )

        if pretrained:
            self._load_from_pretrained(size, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        n_estimators: int | None = None,
        num_estimators: int | None = None,
        context_only_columns: Sequence[str] | None = None,
        estimator_batch_size: int | None = 1,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        finetune: bool | int | float = False,
        query_batch_size: int = 4096,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        max_epochs: int = 1,
        patience: int = 20,
        ema_alpha: float = 0.98,
        min_epochs: int = 0,
        min_improvement: float = 1e-4,
        shuffle_columns: bool = False,
        memory_fraction: float = 0.9,
        **kwargs: Any,
    ) -> None:
        """Fit examples, optionally optimizing a compressed context.

        ``finetune=False`` (or zero) performs ordinary zero-shot fitting.
        ``finetune=True`` chooses up to 32,768 context rows after reserving
        the validation holdout. Classification contexts are sampled within
        each class, preserving training class proportions while
        retaining every class. Regression contexts are sampled uniformly. A
        positive whole number requests exactly that many context rows; a
        fraction in (0, 1) requests floor(fraction * supplied training rows).
        Positive values always tune context embeddings on CUDA. Each
        estimator receives the requested context size and its own optimization
        budget.
        When rows exceed context size plus query batch size, validation holds
        one query batch; otherwise it holds ceil(5% of rows). Context
        sampling and training queries exclude this fixed holdout.
        Validation runs before training and after every update. By default,
        tuning stops when validation loss EMA does not decrease over twenty
        validation checks, or after one sampling epoch. It restores the
        lowest-loss context, including
        the initial context. An epoch is an update budget, not guaranteed
        row coverage.
        Classification uses cross-entropy, including ECOC above ten classes;
        regression uses pinball loss over the 999 normalized-target quantiles.
        Unavailable rows or insufficient memory raise an error without reducing
        the requested count. ``finetune_result_`` reports the tuned outcome.

        Args:
            x: Unbatched training features for tuning.
            y: Training targets.
            related_tables: Related context tables for ordinary fitting.
            recipe: Preprocessing and output recipe.
            n_estimators: Override the constructor's estimator count for this
                fit. By default, use the constructor value (eight). Tuned
                estimators share a fixed validation holdout and average
                postprocessed predictions.
            num_estimators: Alias for n_estimators; specify only one name.
            context_only_columns: Context-only features for ordinary fitting.
            estimator_batch_size: Estimators per ordinary inference batch.
                Tuned estimators always train and predict sequentially.
            callbacks: Callbacks for ordinary fitting.
            generator: Generator for preprocessing and sampling.
            kwargs: Additional ordinary model arguments.
            query_batch_size: Queries sampled per update and validation batch.
                Defaults to 4,096; training batches are capped by the number
                of available training rows.
            learning_rate: Adam learning rate (without weight decay).
            min_learning_rate: Cosine schedule's final learning rate.
            max_epochs: Maximum sampling epochs (default one). Each epoch
                budgets ceil(non_validation_rows / query_batch_size) updates.
            patience: EMA lookback in batch validation checks (default twenty).
                Stop when the current loss EMA is at least its value this many
                checks earlier.
            ema_alpha: Weight of the previous loss EMA (default 0.98). The
                initial validation loss initializes the EMA; each later check
                contributes weight ``1 - ema_alpha``.
            min_epochs: Completed sampling epochs required before early
                stopping is allowed (default zero).
            min_improvement: Retained for compatibility; unused by EMA
                stopping.
            shuffle_columns: Permute processed columns jointly each update.
            memory_fraction: Fraction of available CUDA memory to use.
            finetune: False or zero for ordinary fitting; True to tune up to
                32,768 rows from the training pool after the validation split.
                A positive row count or fraction of supplied training rows
                requests an explicit context size per estimator.

        Other arguments follow :meth:`ICLModel.fit`. Returns ``None``.
        """
        context_rows = _context_rows(finetune, x.size(-2), query_batch_size)
        if n_estimators is not None and num_estimators is not None:
            raise ValueError(
                "Specify only one of n_estimators and num_estimators."
            )
        estimators = (
            n_estimators if n_estimators is not None else num_estimators
        )
        if estimators is None:
            estimators = self.n_estimators
        if (
            isinstance(estimators, bool)
            or not isinstance(estimators, Integral)
            or estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        if context_rows:
            return fit_context(
                self,
                x,
                y,
                observer=kwargs.pop("_finetune_observer", None),
                max_steps=kwargs.pop("_finetune_max_steps", None),
                context_rows=context_rows,
                fit_kwargs=dict(
                    related_tables=related_tables,
                    recipe=recipe,
                    num_estimators=int(estimators),
                    context_only_columns=context_only_columns,
                    estimator_batch_size=estimator_batch_size,
                    callbacks=callbacks,
                    generator=generator,
                    **kwargs,
                ),
                query_batch_size=query_batch_size,
                learning_rate=learning_rate,
                min_learning_rate=min_learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                ema_alpha=ema_alpha,
                min_epochs=min_epochs,
                shuffle_columns=shuffle_columns,
                memory_fraction=memory_fraction,
            )

        return super().fit(
            x=x,
            y=y,
            related_tables=related_tables,
            recipe=recipe,
            num_estimators=int(estimators),
            context_only_columns=context_only_columns,
            estimator_batch_size=estimator_batch_size,
            callbacks=callbacks,
            generator=generator,
            **kwargs,
        )

    def clear(self) -> None:
        """Discard fitted context and its tuning report."""
        super().clear()
        self.finetune_result_: dict[str, Any] | None = None

    def _prefetch(
        self, items: Sequence[Any], device: torch.device
    ) -> _Prefetcher[Any]:
        if isinstance(items[0], Cache) and "finetune_dtype" in items[0]:
            device = next(self.parameters()).device
        elif self._cache is not None:
            device = cast(dict[str, Any], self._cache["kwargs"]).get(
                "_context_device", device
            )
        return super()._prefetch(items, device)

    def _load_from_pretrained(
        self,
        size: Literal["small", "medium", "large"],
        device: torch.device | str | None,
    ) -> KumoTabular:
        device = torch.get_default_device() if device is None else device

        for task, model in self.models.items():
            if task == Task.classification:
                filename = f"{size}/classifier.pt"
            else:
                assert task == Task.regression
                filename = f"{size}/regressor.pt"

            path = download_checkpoint(
                repo_id="nvidia/Kumo-Tabular",
                filename=filename,
                revision="v1.0.1",
                config_filename="config.json",
            )
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt, assign=True)

        return self

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        x_context_only: TableTensor | None,  # [..., R_context, D_only]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        *,
        categorical_mask: Tensor,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 999]
        if x_query is None and x_context is not None:
            x = x_context.numerical
        elif x_context is None and x_query is not None:
            x = x_query.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            classes = cast("Tensor | None", cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if "_context_device" in kwargs:
            x = x.to(kwargs["_context_device"])
            y = y.to(x.device)
            categorical_mask = categorical_mask.to(x.device)

        if cache is not None and "finetune_dtype" in cache:
            device = next(self.parameters()).device
            x = x.to(device)
            y = y.to(device)
            categorical_mask = categorical_mask.to(device)
            with torch.autocast(
                "cuda", dtype=cast(torch.dtype, cache["finetune_dtype"])
            ):
                return self._forward_rows(
                    x=x,
                    y=y,
                    classes=classes,
                    categorical_mask=categorical_mask,
                    cache=cache,
                    generator=generator,
                )

        return self._forward_rows(
            x, y, classes, categorical_mask, cache, generator
        )

    def _forward_rows(
        self,
        x: Tensor,
        y: Tensor,
        classes: Tensor | None,
        categorical_mask: Tensor,
        cache: Cache | None,
        generator: torch.Generator | None,
    ) -> TableTensor:
        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        if classes is None:
            out = self.models[Task.regression](
                x=x,
                y=y,
                categorical_mask=categorical_mask,
                cache=cache,
            )
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out,
            )

        out = self.ecoc(
            model=self.models[Task.classification],
            x=x,
            y=y,
            num_classes=len(classes),
            num_members=x.size(0) if x.dim() > 2 else 1,
            cache=cache,
            generator=generator,
            categorical_mask=categorical_mask,
        )
        if cache is not None and cache.is_replaying:
            columns = cast(tuple[str, ...], cache["class_columns"])
        else:
            columns = tuple(str(i) for i in classes.tolist())

        if cache is not None and cache.is_recording:
            cache["class_columns"] = columns

        return TableTensor(
            columns={"numerical": columns},
            numerical=out,
        )


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        cell_channels: int = 128,
        num_embedding_layers: int = 4,
        num_embedding_heads: int = 4,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 4,
        icl_channels: int = 512,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        num_icl_key_value_heads_for_query: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_frequencies=num_frequencies,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        if cell_channels * num_readout_tokens != icl_channels:
            self.row_project = Linear(
                cell_channels * num_readout_tokens,
                icl_channels,
                **factory_kwargs,
            )
        else:
            self.row_project = Identity()
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=icl_channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            num_key_value_heads_for_query=num_icl_key_value_heads_for_query,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        x = self.encode(x, y, categorical_mask, cache=cache)
        return self.icl_block(x=x, y=y, cache=cache)

    def encode(
        self,
        x: Tensor,
        y: Tensor,
        categorical_mask: Tensor,
        *,
        cache: Cache | None = None,
    ) -> Tensor:
        """Encode table rows before the ICL target embedding."""
        return self.row_project(
            self.row_embedding(x, y, categorical_mask, cache=cache)
        )
