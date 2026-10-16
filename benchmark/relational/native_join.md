# Native integer join prototype

`native_join.py` contains two standalone experimental joins. They are not connected to SDM's default join backend. Both accept two-dimensional `int64` keys, optional row/column validity masks, and return original row-index pairs with duplicate multiplicities preserved.

## Implementations

- `native_join`: sorts right hashes, finds candidate ranges, and verifies the original keys. When the right input has no more rows than the left, a group-end scan replaces the upper-bound search; larger right inputs retain both searches. Hash collisions can still create quadratic candidate storage.
- `native_join_bounded`: checks whether right rows sharing a hash have identical original keys. Pure groups use an exact representative comparison before expansion. Impure groups use exact lexicographic grouping. `torch.cond` selects fixed-size metadata, then only true matches are expanded. Temporary storage is `O((N + M) * K + Z)`, where `K` is key width and `Z` is the number of actual matches. True duplicate outputs can themselves be quadratic.

Hash equality alone never establishes a match. Strings, floating-point keys, mixed-type safe casting, and complete SDM model integration remain outside this prototype.

## Run

From the repository root, in an SDM environment:

```sh
python -m benchmark.relational.native_join --compile
python -m benchmark.relational.native_join --compile --bounded
```

Each command checks 151 eager cases and 48 Inductor cases under `torch.inference_mode()`, with graph breaks allowed and forbidden. Cases cover nulls, empty/singleton inputs, duplicates, extrema, forced collisions, and different real keys sharing the production hash. The script enables dynamic-output capture and increases the validation-only compilation budget for its deliberately varied shapes. Importing either function does not change compiler settings; PyTorch 2.7 callers compiling it must enable dynamic-output capture themselves.

Both commands passed on PyTorch 2.7.1 and 2.14.0. Additional production-path checks changed match counts and collision branches repeatedly within one compiled graph, using noncontiguous keys and masks.

## CPU measurements

Milliseconds per join; shuffled two-column nullable keys, 10,000 rows per side. The published baseline is commit `360973cae`.

| PyTorch | Published | Adaptive candidates | Bounded | Arrow |
| ------- | --------: | ------------------: | ------: | ----: |
| 2.7.1   |     1.585 |               1.157 |   1.269 | 1.177 |
| 2.14.0  |     1.525 |               1.128 |   1.087 | 1.196 |

Real Inductor, `fullgraph=True`, `dynamic=True`, inference mode, one Torch/Arrow CPU thread. Compilation and warmup excluded; medians of 15 balanced-order blocks of four calls. The 14-case matrix also covers unequal table sizes, foreign-key joins, duplicates on both sides, and one/four key columns. Outputs matched Arrow; no recompilation occurred during timing.

The adaptive implementation had no material regression against the published baseline across this matrix. The bounded implementation was faster when the left table was at least as large as the right, but 4–25% slower when the right table was larger. Neither always beats Arrow. GPU performance, complete-model speed, and prediction metrics remain unvalidated.

## Collision-memory stress check

Compiled full-graph inference with 2,000 rows per side, a deliberately constant hash, and zero actual matches:

| PyTorch | Candidate expansion | Bounded |
| ------- | ------------------: | ------: |
| 2.7.1   |            68.06 MB | 0.31 MB |
| 2.14.0  |            68.06 MB | 0.36 MB |

These are peak CPU tensor-storage bytes recorded by the PyTorch profiler, using separate workers after compilation and warmup. They are not total process RAM; some preexisting input storage, allocator caches, and runtime/compiler memory are outside that accounting. Separate process-RSS checks had different baselines and do not establish lower absolute process RAM. The bounded algorithm avoids expanding the four million false candidates in this case.
