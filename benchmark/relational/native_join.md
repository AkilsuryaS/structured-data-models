# Native integer join prototype

The implementation is in `native_join.py`. It is an experimental standalone join, not connected to SDM's default join backend.

It hashes integer keys, sorts the right hashes, finds candidate matches, and checks every original key column. Hash collisions cannot create incorrect matches. Results contain original row indices and preserve duplicate match multiplicity; output order is unspecified.

## Run

From the repository root, in an SDM environment:

```sh
python -m benchmark.relational.native_join
python -m benchmark.relational.native_join --compile
```

The first command runs 145 correctness cases, including forced hash collisions, nulls, empty inputs, duplicate keys, and integer extrema. The second also runs 18 full-graph Inductor cases with dynamic input lengths on CPU. Both commands passed on PyTorch 2.7.1 and 2.14.0; the script enables dynamic-output capture for compilation.

## Limits

- Inputs must be two-dimensional `int64` tensors with the same nonzero number of key columns and the same device. Optional validity masks may be per-row or per-column.
- Strings, floating-point keys, and the existing backend's mixed-type safe-cast behavior are not implemented.
- Collision candidates can require quadratic memory even when few final matches remain. A bounded-memory strategy is needed before production use.
- The compilation check runs outside `torch.inference_mode()`. An additional PyTorch 2.7 check inside inference mode hit an internal symbolic-shape assertion; compiled inference under that mode needs further investigation.
- CPU prototypes were faster than Arrow at 1,000 rows per side but slower on shuffled 10,000-row inputs. GPU and complete-model performance have not been validated.
