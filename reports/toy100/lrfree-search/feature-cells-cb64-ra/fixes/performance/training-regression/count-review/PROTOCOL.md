# Independent count and lineage review

This directory is the only editable area for this review. Existing packages,
configuration, READY files, validation sources, saved inputs, checkpoints and
the completed parent training-regression diagnosis are read only. Root owns
GPU execution. CUDA is hidden before Torch import, one CPU thread and one
interop thread are used, and deterministic algorithms are enabled.

`audit_counts.py` uses a specified 64-row prefix of the existing toy1000 real
feature artifact and its saved CPU planning RNG bytes. It compares v4 and the
one support count proposal's geometry, original support law, fitting RNG and
even boundary under odd/fake mutations. A separate 29-allocation exact pooled
null uses rational arithmetic to check the retained-empty-category family.
Specified constant/minimum fixtures cover ties, empty cells and degenerate
metrics. No learned action response or quality trajectory is rerun.

`audit_lineage.py` uses specified small row/copy fixtures and fixed latent/noise
tensors to check overwrites, reciprocal eviction, simultaneous parents, empty
updates, degree bounds, malformed graphs, empty-graph kernel equivalence,
indexed identity derivatives and graph resume. Backend state checks reuse the
existing toy1000 seed and an explicit table without model or optimizer updates.
The first-copy failure log is retained; only the concrete corrected risk is
retested. No CUDA RNG state is installed in a CPU generator.

Both scripts hash their reviewed sources before and after execution and write
receipts here. Implementation owners' broader focused receipts are inspected
read only. There is no additional seed experiment, optimizer update, Q change,
gate change or score/boundary search.

```
CUDA_VISIBLE_DEVICES='' /tmp/pr38-default-env/bin/python -B audit_counts.py > count-review.log 2>&1
CUDA_VISIBLE_DEVICES='' /tmp/pr38-default-env/bin/python -B audit_lineage.py > lineage-review.log 2>&1
```

Root's later request to inspect the completed grid100 validation uses only its
saved JSON/JSONL metrics and frozen scorer/source rules. Any receipt for that
analysis is separate from the frozen count/lineage evidence.
