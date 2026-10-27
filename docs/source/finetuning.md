# Fine-Tuning

Tabular foundation models (TFMs) such as {py:class}`~sdm.models.TabICLv2` and {py:class}`~sdm.models.KumoTabular` solve a new task through [in-context learning](icl): they read labeled context rows and predict query rows without updating their weights.
This works well out of the box, but the pretrained weights know nothing specific about your dataset.

**Full fine-tuning** updates every weight of a pretrained model on one dataset.
The result is still an in-context learner with the same interface, and it trains with an ordinary PyTorch loop, since calling {py:meth}`~sdm.models.ICLModel.forward` in train mode enables gradients (see [In-Context Learning](icl)).
Below, the *pretrained model* is the model before fine-tuning, labeled *zero-shot* in the figures.

Fine-tuning does not always pay off.
On standard benchmarks, it helps on more datasets than it hurts, but not on all of them, and gains tend to be small when the pretrained model is already strong.
On datasets with tens of thousands of rows, a run took between 10 and 60 minutes on a single H100 GPU in our experiments.
The procedure below therefore keeps the pretrained weights whenever fine-tuning does not improve the validation score.

## How Full Fine-Tuning Works

Fine-tuning keeps the objective a TFM is pretrained with: predict held-out rows from labeled examples.
What changes is where the tasks come from.
Instead of its pretraining data, each training step draws a task from the training split of your dataset:

```{figure} images/finetuning_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_dark.svg
:figclass: dark-only
:width: 100%
```

1. Sample rows from the training split and divide them into a **context** of *C* labeled rows and a **query** of *Q* rows.
2. Run the model on the labeled context and the query features.
3. Compute the loss of the query predictions against the held-out query labels.
4. Backpropagate and update every weight of the model.

Because the context changes from step to step, the model learns to read its context, as it does at inference, instead of memorizing a fixed set of rows.
After fine-tuning, inference is unchanged: the training split becomes the context, and the model predicts new rows through {py:meth}`~sdm.models.ICLModel.forward` or {py:meth}`~sdm.models.ICLModel.fit` and {py:meth}`~sdm.models.ICLModel.predict`.

```{note}
The next four sections walk through [`examples/tabular/finetune.py`](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabular/finetune.py), which fine-tunes {py:class}`~sdm.models.TabICLv2` or {py:class}`~sdm.models.KumoTabular` on a classification or regression dataset.
```

## Data Splits

Fine-tuning updates the weights on the training rows, so you need separate validation rows to pick the weights and test rows to report results:

- **Training** rows are sampled into context and query rows during fine-tuning, and also serve as the context when the model is evaluated.
- **Validation** rows pick the best weights and never appear in a training step.
- **Test** rows are used once, for the final evaluation.

The snippet below builds a {py:class}`~sdm.tensor.TableTensor` (see [Table Semantics](tensor)) and splits its rows:

```python
import torch
from sklearn.datasets import load_digits

import sdm

df = load_digits(as_frame=True).frame

table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device="cuda",
)

# Split rows into training/validation/test (60/20/20):
index = torch.randperm(table.size(0), device="cuda")
num_train, num_val = int(0.6 * index.numel()), int(0.2 * index.numel())
train_table = table[index[:num_train]]
val_table = table[index[num_train:num_train + num_val]]
test_table = table[index[num_train + num_val:]]
```

## A Training Recipe

Every model ships a default {py:class}`~sdm.processing.recipe.Recipe` (see [Data Processing](processing)) whose output pipeline is built for inference: it combines the outputs of all estimators and turns classification logits into probabilities.
For training, we replace it with {py:class}`~sdm.processing.common.Identity`, so the loss is computed directly on the model outputs:

```python
import sdm.processing as sp

model = sdm.models.TabICLv2(task="classification", device="cuda")

train_recipe = model.default_recipe()
train_recipe.output = sp.Identity()  # No post-processing.
```

The rest of the recipe stays as is.
Its stochastic processors, *e.g.*, {py:class}`~sdm.processing.common.ShuffleColumns` in the features and {py:class}`~sdm.processing.categorical.ShuffleCategories` in the target, draw a new order at every step, which acts as data augmentation.

For regression with {py:class}`~sdm.models.KumoTabular`, also remove random sign flipping ({py:class}`~sdm.processing.numerical.FlipSign`) from the target pipeline.
A flipped target reverses the order of the predicted quantiles, which the default output pipeline restores via {py:class}`~sdm.processing.output.SortQuantiles`, but the identity output does not:

```python
train_recipe.target = sp.StypeDispatch(  # Default target pipeline without FlipSign.
    categorical=[
        sp.AlignCategories(sort_by="value"),
        sp.ShuffleCategories(method="shift"),
    ],
    numerical=sp.Standardize(),
)
```

## The Training Loop

Each step samples a fresh context and query from the training rows and runs a single estimator in train mode with the training recipe:

```python
import torch.nn.functional as F

context_size, query_size, steps_per_epoch = 256, 128, 30
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()
for step in range(steps_per_epoch):
    index = torch.randperm(train_table.size(0), device="cuda")
    context = train_table[index[:context_size]]
    query = train_table[index[context_size:context_size + query_size]]

    out = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
        num_estimators=1,
        recipe=train_recipe,
    )[0]

    logits, y = sdm.evaluation.to_class_indices(
        pred=out,
        target=query["target"],
        missing_score=-torch.inf,
    )
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

With the identity output, the model returns its outputs stacked along a leading estimator dimension, so `[0]` selects the output of the single estimator.

For classification, the output has one logit column per class that appears in the sampled context.
{py:func}`~sdm.evaluation.to_class_indices` aligns these columns with the classes of the query targets and converts the targets to class indices.
Setting `missing_score=-torch.inf` gives classes that are absent from the context zero probability.
If the class of a query row is absent from the context, its loss is infinite, so make sure that every class appears in the context, or compute the loss on probabilities with a finite `missing_score`, as the benchmark harness below does.

For regression, both models predict 999 quantiles (see [In-Context Learning](icl)).
The pinball (quantile) loss trains each output column toward its quantile level, and with it the full predictive distribution rather than only a point estimate:

```python
diff = query["target"].numerical - out.numerical  # [Q, 999]
level = torch.linspace(0.001, 0.999, 999, device="cuda")
loss = torch.maximum(level * diff, (level - 1) * diff).mean()
```

## Model Selection

Fine-tuning can overfit, in particular on small datasets.
We therefore score the model on the validation rows after every *epoch*, here a fixed number of training steps rather than a full pass over the data, and keep the best weights.
Evaluation runs the model as it will be used at inference: in eval mode, with the default recipe, with [estimator ensembling](icl), and with the training rows as context:

```python
import copy

def evaluate(context: sdm.TableTensor, query: sdm.TableTensor) -> float:
    model.eval()
    out = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
        num_estimators=8,
    )
    prob, y = sdm.evaluation.to_class_indices(
        pred=out,
        target=query["target"],
        missing_score=0.0,
    )
    return (prob.argmax(dim=-1) == y).float().mean().item()

best_acc = evaluate(context=train_table, query=val_table)  # Pretrained model.
best_state = copy.deepcopy(model.state_dict())

max_epochs = 10
for epoch in range(max_epochs):
    model.train()
    ...  # Training steps from above.

    acc = evaluate(context=train_table, query=val_table)
    if acc > best_acc:
        best_acc, best_state = acc, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
test_acc = evaluate(context=train_table, query=test_table)
```

Since the first evaluation scores the pretrained model, selection never returns weights that score below it on the validation rows.
For large test sets, replace the one-shot {py:meth}`~sdm.models.ICLModel.forward` call with {py:meth}`~sdm.models.ICLModel.fit` and batched {py:meth}`~sdm.models.ICLModel.predict` calls, optionally with mixed-precision autocasting (see [In-Context Learning](icl)).

## Benchmark Results

The benchmark harness in [`benchmark/tabular/finetune.py`](https://github.com/NVIDIA/structured-data-models/blob/main/benchmark/tabular/finetune.py) runs this procedure on [TALENT](https://github.com/LAMDA-Tabular/TALENT), [TabArena](https://github.com/autogluon/tabarena), and [ScoringBench](https://scoringbench.com).
It differs from the example in one respect: it keeps the default recipe during training, so the loss is computed on class probabilities and sorted quantiles instead of raw outputs.
Its default settings are tuned for larger datasets than the example:

| Setting          | Example         | Harness                   |
| ---------------- | --------------- | ------------------------- |
| Learning rate    | `1e-5`          | `1e-6`                    |
| Context size *C* | 256             | up to 8,000               |
| Query size *Q*   | 128             | up to 2,000               |
| Epochs × steps   | 10 × 30         | 75 × 10                   |
| Validation rows  | 20% of all rows | 20% of the training split |

Each row below counts the datasets on which the fine-tuned model scores better (win), worse (loss), or the same (tie) as the pretrained model, according to each benchmark's own metric:

| Benchmark    | Model                                       | Tasks                     | Wins | Losses | Ties |
| ------------ | ------------------------------------------- | ------------------------- | ---- | ------ | ---- |
| TALENT       | {py:class}`~sdm.models.TabICLv2`            | Classification            | 73   | 51     | 31   |
| TALENT       | {py:class}`~sdm.models.KumoTabular` (small) | Classification            | 51   | 39     | 38   |
| ScoringBench | {py:class}`~sdm.models.TabICLv2`            | Probabilistic regression  | 50   | 26     | 10   |
| TabArena     | {py:class}`~sdm.models.KumoTabular` (large) | Classification/regression | 35   | 12     | 0    |

On every benchmark, fine-tuning helps on more datasets than it hurts.
On TALENT, fine-tuning {py:class}`~sdm.models.TabICLv2` helps on twice as many multiclass datasets as it hurts (36 *vs.* 18), but on binary datasets about as often as it hurts (37 *vs.* 33).
The {py:class}`~sdm.models.KumoTabular` row on TALENT covers fewer datasets because its fine-tuning variant supports at most 10 classes.
On binary tasks, it also fine-tunes for 50 instead of 75 epochs.

To reproduce these results, add `--finetune` to the TALENT and ScoringBench runners (with `--model kumo-tabular-small-ft` for {py:class}`~sdm.models.KumoTabular` on TALENT), or pick a fine-tuned `-ft` model on TabArena:

```bash
python -m benchmark.tabular.talent.main \
  --model tabiclv2 \
  --dataset-path /path/to/talent/data \
  --finetune

python -m benchmark.tabular.tabarena.main --model kumo-tabular-large-ft
```

See the [tabular benchmark instructions](https://github.com/NVIDIA/structured-data-models/tree/main/benchmark/tabular) for setup and all fine-tuning flags.

## Choosing Hyperparameters

On the two datasets we tested, the context size *C* mattered most, learning rates between `1e-6` and `1e-5` worked best, and the query size *Q* mattered least.
In practice, start from the harness defaults, make *C* as large as memory allows, and tune the learning rate on a validation split.
The absolute gains were small, since the pretrained model is already strong on both datasets.

We fine-tuned {py:class}`~sdm.models.TabICLv2` with the benchmark harness on two of the larger TALENT datasets: Airline (`customer_satisfaction_in_airline`; binary classification, 83,123 training rows) and `Diamonds` (regression, 34,521 training rows).
Each run took 750 steps and changed one setting at a time from the harness defaults (learning rate `1e-6`, *C* = 8,000, *Q* = 2,000).
The figure shows test accuracy (higher is better) and RMSE (lower is better), averaged over 3 seeds with the full training split as context; error bars show the standard deviation, and the dashed line is the pretrained model:

```{figure} images/finetuning_sweep_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_sweep_dark.svg
:figclass: dark-only
:width: 100%
```

- **Context size mattered most.** With *C* = 500 or 2,000, fine-tuning did not improve on the pretrained model on either dataset, while *C* = 16,000, the largest size we tried, did best on both. A likely reason is that the model reads the whole training split at inference, so larger training contexts are closer to how it is used. These sizes are specific to datasets of this scale: the example uses *C* = 256 because its dataset has only about 1,000 training rows.
- **Learning rates between `1e-6` and `1e-5` worked best** within the range we tried (`1e-7` to `1e-5`). Smaller rates left Airline at its pretrained accuracy and recovered only part of the gain on Diamonds. On Airline, `3e-6` and `1e-5` improved over `1e-6`; on Diamonds, `1e-6` and `1e-5` performed alike.
- **Query size mattered least.** Airline accuracy rose steadily from *Q* = 128 to *Q* = 8,000, while on Diamonds, *Q* = 2,000 scored best and *Q* = 8,000 was within the variation across seeds.

Selected on validation, the best settings improved test accuracy on Airline from 96.42% to 96.54% (learning rate `1e-5`) and test RMSE on Diamonds from 495.1 to 489.3 (*C* = 16,000).
Memory and time grow with *C* + *Q*: on Airline, fine-tuning peaked at 6 GB of GPU memory for *C* = 500, 13 GB for *C* = 8,000, and 23 GB for *C* = 16,000 (all with *Q* = 2,000), and a run took 46 to 60 minutes on an H100 GPU.

The TALENT runner exposes all three settings: `--finetune-lr`, `--finetune-train-size` (*C* + *Q*), and `--finetune-context-frac` (*C* / (*C* + *Q*)).
For example, the best Diamonds run uses *C* = 16,000 and *Q* = 2,000:

```bash
python -m benchmark.tabular.talent.main \
  --model tabiclv2 \
  --dataset-path /path/to/talent/data \
  --dataset Diamonds \
  --finetune \
  --finetune-train-size 18000 \
  --finetune-context-frac 0.8889
```

## Smaller Contexts After Fine-Tuning

A pretrained TFM knows about your dataset only what its context shows it.
Its accuracy therefore typically keeps improving as more training rows are passed as context, which makes the full training split the default choice.
Inference cost grows with the context as well, both for one-shot calls and for {py:meth}`~sdm.models.ICLModel.fit` and {py:meth}`~sdm.models.ICLModel.predict` (see [In-Context Learning](icl)).

Fine-tuning moves part of that dataset knowledge into the weights.
A fine-tuned model can therefore reach comparable accuracy with fewer context rows, which reduces inference memory and latency, *e.g.*, when serving predictions with a large training split as context.
To use a smaller context, fit the fine-tuned model on a random subset of the training rows:

```python
model.eval()

index = torch.randperm(train_table.size(0), device="cuda")[:4096]
context = train_table[index]

model.fit(
    x=context.drop_columns("target"),
    y=context["target"],
    num_estimators=8,
)
out = model.predict(test_table.drop_columns("target"))
```

To measure the effect, we took the best fine-tuned model per dataset from the previous section (learning rate `1e-5` for Airline, *C* = 16,000 for Diamonds).
We fitted it and the pretrained model on the same random subsets of 256 to 16,384 training rows, and on the full training split.
The figure and table show test accuracy (Airline) and RMSE (Diamonds), averaged over 3 seeds; the dashed line is the pretrained model with the full training split as context:

```{figure} images/finetuning_context_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_context_dark.svg
:figclass: dark-only
:width: 100%
```

| Context rows | Airline, zero-shot | Airline, fine-tuned | Diamonds, zero-shot | Diamonds, fine-tuned |
| ------------ | ------------------ | ------------------- | ------------------- | -------------------- |
| 256          | 89.08%             | 89.88%              | 930.4               | 908.5                |
| 1,024        | 92.67%             | 93.82%              | 630.2               | 621.5                |
| 4,096        | 94.61%             | 95.62%              | 548.3               | 528.3                |
| 16,384       | 95.87%             | 96.30%              | 505.0               | 496.0                |
| All          | 96.42%             | 96.57%              | 495.1               | 488.8                |

- **The fine-tuned model scored better at every context size**, but at the smallest contexts (256 rows on both datasets and 1,024 rows on Diamonds), the difference was within the variation across seeds. The largest clear gaps were at 1,024 rows on Airline (93.82% *vs.* 92.67%) and at 4,096 rows on Diamonds (528.3 *vs.* 548.3 RMSE).
- **With 16,384 rows, the fine-tuned model came close to the pretrained model with the full training split**, using 20% of the Airline training split (96.30% *vs.* 96.42%) and 47% of the Diamonds training split (496.0 *vs.* 495.1).
- **More context still helped.** Both models improved with every increase in context size, and the fine-tuned model with the full training split scored best on both datasets.

The full-context results differ slightly from the previous section because estimator ensembles are randomized at inference.
[Estimator ensembling](icl) also complements smaller contexts by giving each estimator a different subset of the training rows.
