# Fine-Tuning

Tabular foundation models (TFMs) such as {py:class}`~sdm.models.TabICLv2` and {py:class}`~sdm.models.KumoTabular` are pretrained on a broad distribution of tasks and solve a new task through [in-context learning](icl): they read labeled context rows and predict query rows without updating their weights.
This works well out of the box, but the pretrained weights encode a general prior rather than the specifics of your dataset.

**Full fine-tuning** updates every parameter of a pretrained model on a single dataset, so the model adapts to that dataset's features and targets.
The fine-tuned model is still an in-context learner: it keeps the same interface and predicts query rows from labeled context rows.
No dedicated training API is needed either.
Calling {py:meth}`~sdm.models.ICLModel.forward` in train mode (`model.train()`) tracks gradients through the recipe and the model, so a standard PyTorch training loop is all it takes.

## How Full Fine-Tuning Works

Fine-tuning keeps the objective a TFM is pretrained with: predict held-out rows from labeled examples.
What changes is where the tasks come from.
Instead of drawing them from a synthetic prior, each training step draws one from the training split of your dataset:

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

Because the context changes from step to step, the model is trained to read its context, just as it does at inference, instead of fitting a fixed set of rows.
After fine-tuning, inference is unchanged: the training split becomes the context, and the model predicts new rows through {py:meth}`~sdm.models.ICLModel.forward` or {py:meth}`~sdm.models.ICLModel.fit` and {py:meth}`~sdm.models.ICLModel.predict`.

```{note}
The rest of this guide walks through [`examples/tabular/finetune.py`](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabular/finetune.py), which fine-tunes {py:class}`~sdm.models.TabICLv2` or {py:class}`~sdm.models.KumoTabular` on a classification or regression dataset.
```

## Data Splits

Fine-tuning learns from labels, so it needs three disjoint splits:

- **Training** rows are sampled into context and query rows during fine-tuning, and serve as the context when the model is evaluated.
- **Validation** rows select the best checkpoint and never appear in a training step.
- **Test** rows are used once, for the final evaluation.

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

Every model ships a default {py:class}`~sdm.processing.recipe.Recipe` (see [here](processing) for the accompanying tutorial) built for inference: its output pipeline averages the outputs of all estimators and, for classification, turns logits into probabilities.
For training, we want the raw model outputs instead, so the loss can be computed directly on logits and quantiles:

```python
import sdm.processing as sp

model = sdm.models.TabICLv2(device="cuda")

train_recipe = model.default_recipe()
train_recipe.target = sp.StypeDispatch(  # No target flipping.
    categorical=[
        sp.AlignCategories(),
        sp.ShuffleCategories(method="shift"),
    ],
    numerical=sp.Standardize(),
)
train_recipe.output = sp.Identity()  # No post-processing.
```

- With {py:class}`~sdm.processing.common.Identity` as the output pipeline, the model returns its raw outputs stacked along a leading estimator dimension.
  Training steps run a single estimator, so `out[0]` selects its output.
- The target pipeline leaves out random sign flipping of regression targets ({py:class}`~sdm.processing.numerical.FlipSign`, part of the default recipe of {py:class}`~sdm.models.KumoTabular`).
  A flipped target reverses the order of the predicted quantiles, which the default output pipeline restores via {py:class}`~sdm.processing.output.SortQuantiles`, but the identity output does not.
- Feature preprocessing stays as is.
  Stochastic processors such as {py:class}`~sdm.processing.common.ShuffleColumns` and {py:class}`~sdm.processing.categorical.ShuffleCategories` draw a new column or class order at every step, which acts as data augmentation.

## The Training Loop

Each step samples a fresh context and query from the training rows and runs the model in train mode with the training recipe:

```python
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()
for step in range(num_steps):
    index = torch.randperm(train_table.size(0), device="cuda")
    context = train_table[index[:context_size]]
    query = train_table[index[context_size:context_size + query_size]]

    out = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
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

For classification, the output has one logit column per class that appears in the sampled context.
{py:func}`~sdm.evaluation.to_class_indices` aligns these columns with the classes of the target column and converts the query targets to class indices.
Setting `missing_score=-torch.inf` gives classes that are absent from the context zero probability, so the loss is a standard cross-entropy over the aligned logits.

For regression, {py:class}`~sdm.models.TabICLv2` predicts 999 quantiles, named `"q001"` through `"q999"`.
The pinball (quantile) loss trains each output column toward its quantile level, and with it the full predictive distribution rather than only a point estimate:

```python
diff = query["target"].numerical - out.numerical  # [Q, 999]
level = torch.linspace(0.001, 0.999, 999, device="cuda")
loss = torch.maximum(level * diff, (level - 1) * diff).mean()
```

{py:meth}`~sdm.models.ICLModel.predict` raises if the model is in train mode, so training steps always go through {py:meth}`~sdm.models.ICLModel.forward`.

## Model Selection

Fine-tuning can overfit, in particular on small datasets.
We therefore score the model on the validation rows after every epoch and keep the best weights.
Evaluation runs the model exactly as it will be used at inference: in eval mode, with the default recipe, with estimator ensembling, and with the training rows as context:

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

for epoch in range(max_epochs):
    model.train()
    ...  # Training steps from above.

    acc = evaluate(context=train_table, query=val_table)
    if acc > best_acc:
        best_acc, best_state = acc, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
test_acc = evaluate(context=train_table, query=test_table)
```

Since the first evaluation scores the pretrained model, selection never returns weights that score below the zero-shot model on the validation rows.
For large test sets, replace the one-shot {py:meth}`~sdm.models.ICLModel.forward` call with {py:meth}`~sdm.models.ICLModel.fit` and batched {py:meth}`~sdm.models.ICLModel.predict` calls, optionally under mixed-precision autocasting (see [here](icl)).

## Benchmark Results

The benchmark harness in [`benchmark/tabular/finetune.py`](https://github.com/NVIDIA/structured-data-models/blob/main/benchmark/tabular/finetune.py) implements this procedure for TALENT, TabArena, BeyondArena, and ScoringBench.
By default, it trains with AdamW at a learning rate of `1e-6` for 75 epochs of 10 steps, samples up to 10,000 rows per step and splits them 80/20 into context and query rows, and holds out 20% of each dataset's training split for model selection.

Each row below counts the datasets on which the fine-tuned model scores better (win), worse (loss), or the same (tie) as the same model without fine-tuning, according to each benchmark's own metric:

| Benchmark    | Model                                       | Tasks                     | Wins | Losses | Ties |
| ------------ | ------------------------------------------- | ------------------------- | ---- | ------ | ---- |
| TALENT       | {py:class}`~sdm.models.TabICLv2`            | Classification            | 73   | 51     | 31   |
| TALENT       | {py:class}`~sdm.models.KumoTabular` (small) | Classification            | 51   | 39     | 38   |
| ScoringBench | {py:class}`~sdm.models.TabICLv2`            | Probabilistic regression  | 50   | 26     | 10   |
| TabArena     | {py:class}`~sdm.models.KumoTabular` (large) | Classification/regression | 35   | 12     | 0    |

Fine-tuning improves results on more datasets than it degrades on every benchmark.
On TALENT, the gains of {py:class}`~sdm.models.TabICLv2` are larger on multiclass datasets (36 wins, 18 losses, 15 ties) than on binary ones (37 wins, 33 losses, 16 ties).

To reproduce these results, add `--finetune` to the TALENT and ScoringBench runners, or pick a fine-tuned `-ft` model variant on TabArena and BeyondArena:

```bash
python -m benchmark.tabular.talent.main \
  --model tabiclv2 \
  --dataset-path /path/to/talent/data \
  --finetune

python -m benchmark.tabular.tabarena.main --model kumo-tabular-large-ft
```

See the [tabular benchmark instructions](https://github.com/NVIDIA/structured-data-models/tree/main/benchmark/tabular) for setup and all fine-tuning flags.

## Choosing Context Size, Query Size, and Learning Rate

Three settings of the training loop shape the result:

- **Learning rate:** How far each step moves the pretrained weights.
- **Context size *C*:** How many labeled rows the model reads per step.
- **Query size *Q*:** How many rows contribute to the loss per step, *i.e.*, the batch size.

To measure their effect, we fine-tuned {py:class}`~sdm.models.TabICLv2` on two of the larger TALENT datasets, `customer_satisfaction_in_airline` (*Airline* in the figures; binary classification, 83,123 training rows) and `Diamonds` (regression, 34,521 training rows).
Each run uses the benchmark harness configuration from above with a fixed budget of 750 steps and changes one setting at a time, starting from the defaults (learning rate `1e-6`, *C* = 8,000, *Q* = 2,000).
The figure shows test accuracy (higher is better) and RMSE (lower is better) as mean and standard deviation over 3 seeds, using the full training split as context; the dashed line is the zero-shot model:

```{figure} images/finetuning_sweep_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_sweep_dark.svg
:figclass: dark-only
:width: 100%
```

- **Context size has the clearest effect.** With *C* = 500 or 2,000, fine-tuning does not improve on the zero-shot model on either dataset. Gains start at *C* = 8,000 and are largest at *C* = 16,000, the largest size we tried (96.53% *vs.* 96.42% accuracy and 489.3 *vs.* 495.1 RMSE). At inference, the model reads the whole training split, so larger training contexts are closer to the setting it is evaluated in.
- **Learning rates between `1e-6` and `1e-5` work best.** With `1e-7` or `3e-7`, 750 steps leave accuracy on `customer_satisfaction_in_airline` at its zero-shot level and recover only part of the gain on `Diamonds`. On `customer_satisfaction_in_airline`, `3e-6` and `1e-5` improve further over `1e-6`; on `Diamonds`, `1e-6` and `1e-5` perform alike. We did not observe degradation up to `1e-5`, the largest rate we tried.
- **Query size matters less.** Accuracy on `customer_satisfaction_in_airline` rises steadily from *Q* = 128 to *Q* = 8,000, while on `Diamonds`, *Q* = 2,000 scores best and *Q* = 8,000 is within the variation across seeds.

Memory and time grow with *C* + *Q*.
On an H100 GPU, fine-tuning on `customer_satisfaction_in_airline` peaks at 6 GB of memory for *C* = 500, 13 GB for *C* = 8,000, and 23 GB for *C* = 16,000 (all with *Q* = 2,000), and takes 46 to 60 minutes per run.

In practice, start from the harness defaults, increase *C* as far as memory allows, and tune the learning rate between `1e-6` and `1e-5` on a validation split.
Note that the absolute gains on these two datasets are small, since {py:class}`~sdm.models.TabICLv2` is already strong on them zero-shot: the best settings, selected on validation, improve test accuracy from 96.42% to 96.54% (learning rate `1e-5`) and test RMSE from 495.1 to 489.3 (*C* = 16,000).

The TALENT runner exposes all three settings: `--finetune-lr`, `--finetune-train-size` (*C* + *Q*), and `--finetune-context-frac` (*C* / (*C* + *Q*)).
For example, the best `Diamonds` run uses:

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

A zero-shot TFM knows about your dataset only what its context shows it.
Its accuracy therefore typically keeps improving as more training rows are passed as context, which makes the full training split the default choice.
Inference cost grows with the context as well: {py:meth}`~sdm.models.ICLModel.forward` processes all context rows on every call, and {py:meth}`~sdm.models.ICLModel.fit` keeps per-row key/value state in memory for subsequent {py:meth}`~sdm.models.ICLModel.predict` calls.

Fine-tuning moves part of that dataset knowledge into the model weights.
A fine-tuned model can therefore reach comparable accuracy with fewer context rows, which reduces inference memory and latency, *e.g.*, when a model serves predictions for a large training split.
To use a smaller context, fit the fine-tuned model on a random subset of the training rows:

```python
index = torch.randperm(train_table.size(0), device="cuda")[:4096]
context = train_table[index]

model.fit(x=context.drop_columns("target"), y=context["target"])
out = model.predict(test_table.drop_columns("target"))
```

To measure the effect, we took the best fine-tuned model per dataset from the previous section, selected on validation (learning rate `1e-5` for `customer_satisfaction_in_airline` and *C* = 16,000 for `Diamonds`), and fitted it and the zero-shot model on the same random subsets of 256 to 16,384 training rows, as well as on all of them.
The figure and table show test accuracy (*Airline*) and RMSE (*Diamonds*) as the mean over 3 seeds, with the standard deviation as error bars; the dashed line is the zero-shot model with all training rows as context:

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

- **The fine-tuned model is better at every context size.** The gap is largest between 1,024 and 4,096 rows, *e.g.*, 93.82% *vs.* 92.67% accuracy at 1,024 rows and 528.3 *vs.* 548.3 RMSE at 4,096 rows. At 256 rows, it is within the variation across seeds.
- **With 16,384 rows, the fine-tuned model comes close to the zero-shot model with all rows**, using 20% of the `customer_satisfaction_in_airline` training split (96.30% *vs.* 96.42%) and 47% of the `Diamonds` training split (496.0 *vs.* 495.1).
- **More context still helps.** Both models improve with every increase in context size, and the fine-tuned model with all rows scores best on both datasets.

The full-context results of the fine-tuned models differ slightly from the previous section because estimator ensembles are randomized at inference.

Estimator ensembling complements smaller contexts: each estimator can receive a different subset of the training rows, so an ensemble covers more of the training split while each estimator keeps a small context (see the ensembling section [here](icl)).
