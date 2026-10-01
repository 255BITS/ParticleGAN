# Bounded semantic lineage correction

This package changes the latent sampling kernel only. It retains the frozen
RA2 mass allocation, isolation policy, support test and gates. The original
saved-checkpoint diagnosis and failed quality evidence remain authoritative
and unchanged. The radius miss is one cause in a copied table; the saved RA2
toy radius error is too small to explain its quality failure on its own.

Keep all 64 sorted-coordinate candidates. Add at most
`min(metric_rank, 64, N-1)` current endpoints from a symmetric row-copy graph;
the declared rank 8 recipe has at most 72 candidates. Full-vector distances
are evaluated against the current live or EMA table using shared row IDs and
the same graph topology. The graph does not store geometric distances.

Register copies after the inherited latent, EMA, optimizer and history copy.
Overwrite invalidation removes all old incident edges through the copied
rows' bounded adjacency lists. Endpoint eviction removes the reciprocal edge.
New children are newest first; a repeated manual parent keeps its newest
bounded children. A parent overwritten in the same simultaneous batch refers
to its previous row incarnation, so it receives no link to its replacement.
Mutation uses batched tensors and touched degree squared neighborhoods, with
no full population scan or scalar loop over copies.

The graph is semantic checkpoint state. Validate shape, integer bounds,
self-edge exclusion, unique neighbors and symmetry before loading anything.
Then copy it onto the current table device. Sorted-coordinate caches and
geometry work counters remain derived and are discarded on load. Feature-cell
backend schema changes from 3 to 4 and the kernel setting becomes
`bounded_local_dv12_lineage`. GANTrainer schema was already 4. An RA2 checkpoint
with no graph or the old kernel is rejected; historical links are not inferred
from a saved table. Same-law save/resume preserves the graph exactly.

CPU tests use one thread, no CUDA context and the original seed 90229 focused
fixture. Saved table and noise inputs are reused. No seed comparisons, quality
acceptance rerun, or gate tuning occurs here. Two update continuations check
the actual trainer integration. Only the observational
`birth_death.last.eval_seconds` field is excluded from replay fingerprints.

The root coordinator owns physical GPU0 and the serial queue. Its focused
runner accepts an independent combined package and a fresh output path:

```bash
/tmp/pr38-default-env/bin/python gpu_lineage_check.py \
  --package-root /absolute/path/to/combined-package \
  --output /absolute/path/to/new-gpu-lineage-receipt.json
```

Run this only when the coordinator releases its queue slot. CUDA checks use
the same focused suite and require the frozen physical GPU0 UUID, deterministic
execution, disabled TF32 and a 20 percent process memory limit. They do not
replace the original quality gates.
