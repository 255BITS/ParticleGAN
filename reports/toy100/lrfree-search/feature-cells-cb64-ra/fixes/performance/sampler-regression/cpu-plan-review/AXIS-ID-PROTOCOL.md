# Exact cache and canonical sampled-ID correction

Use a new private package copied from frozen pkg-CB64-RA3. Previous packages,
patches, checkpoints, test receipts and quality evidence stay frozen.

Cache selected latent coordinate axes as Python integers after one batched
CPU transfer during `_orders` construction. The same stable sort, selected
axes, unique values, representative rows, distance arithmetic and table-version
invalidation remain in use. No randomness or semantic state is added. Warm
queries avoid converting CUDA scalar axes into Python indices once per axis
per chunk. Cold cache construction still computes the original geometry.

Expose `_generate(..., indices=None, *, rows=None)`: canonical screen.py detects
the `indices` name and supplies a fifth positional argument. The existing rows
keyword is an alias. Reject both nonnull ID arguments before randomness. Four
argument behavior and normal trainer row forwarding retain RA3 behavior.

The CPU suite reuses saved geometry input/noise and the existing seed90229
focused trainer fixture. It runs the frozen canonical harness's option resolver
and exact native argument-building/call statements through AST extraction.
It does no harness trajectory, evaluator call or seed experiment. All 7 new
checks and the existing 11 lineage checks must pass with one CPU thread and
CUDA uninitialized. Two trainer updates compare frozen RA3 changed-function
implementations with the new package, excluding only observational eval time.

Root alone may execute the paired cache profile on physical GPU0 after obtaining
its serial queue slot. No CPU timing is used to assign the native grid100
trajectory regression to this hotspot. CUDA cached/cold displacement must be
bit-identical on the same saved tensors, and the optimized warm scalar-axis
count must be zero. The paired script accepts either this private package or
a later exact combined package:

```bash
/tmp/pr38-default-env/bin/python profile_axis_pair_gpu.py \
  --package-root /absolute/path/to/candidate-package \
  --output /absolute/path/to/fresh-axis-pair-result.json
```

The inherited stats, mass policy, graph state, model schema and gates are
unchanged. Per-group allocation batching is a separate future proposal.
