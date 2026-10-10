# Exact integer group counts and bounded birth profile

CPU mechanical PASS; GPU parity/performance pending. This is a separate private
proposal based on frozen RA6. Only `FeatureCellSnapshot._group_counts` changes;
replacing that method restores the complete RA6 file bytes. The other 28 modules,
config, training/controller/copy/solver laws, schemas and serialized state remain
exact. No frozen RA6 or prior evidence is edited.

The original method uses one boolean advanced index per group. For matching 1D
int32/int64 vectors, the proposal broadcasts the counts into a G×K matrix, masks
other groups to zero, and sums each row. It has at most K²=4096 entries. Both
methods accumulate integer counts in int64, so the result is exact, including
modular int64 overflow. Other dtypes and invalid/multidimensional shapes execute
the original expression. No new cache or state key is introduced.

## Proof and profile

`cpu-group-contract.json` passes 224 cases: groups up to 25/64, zero counts,
20k+ counts, noncontiguous vectors, int32/int64 extrema/overflow, float32/float64
and NaN fallback, other dtypes, shape errors and multidimensional fallback.
Two saved clean-table reactions at RA4 toy checkpoints 1250/2000 each produce
47 ordinary copies and four paired novel births. Complete plans, certificates,
ledgers, bookkeeping/work, planning RNG, live/EMA coordinates, saved optimizer
moments/history, row evidence, anchors, graph and final private RNG match exactly.
No optimizer step, new seed or quality trajectory is run.

| Saved case | nonzero operations, old → proposed | scalar reads | Jacobian/SVD | feature forwards |
|---|---:|---:|---:|---:|
| 1250 | 355 → 55 | 386, unchanged | 8, unchanged | 24, unchanged |
| 2000 | 347 → 47 | 355, unchanged | 6, unchanged | 20, unchanged |

The 300 removed operations correspond to twelve repeated `_group_counts` calls
over 25 groups. Boolean advanced indexing requires data-dependent output sizes
on CUDA. The GPU synchronization cost still needs the serial root probe; CPU
counts/timings do not assign the observed training slowdown to a specific CUDA
operator. The proposal leaves scalar reads and solver arithmetic unchanged.

The unchanged solver also recomputes full-prior coordinate std eight times per
reaction, evaluates features separately for scoring and Jacobian construction,
and repeatedly transforms/scores the same candidate for support and count
categories. Per-iteration progress records, acceptance branches, finite/rank
checks and terminal diagnostics issue scalar reads. Those remain outside this
narrow patch. Worst case is four candidate cells × two models × four Jacobian/SVD
linearizations = 32; these saved cases exercise eight/six. Their measured costs
are not a reconstruction of early RA6 reactions or historical GPU copy actions.

## Root-only paired GPU probe

`profile_anchor.py` accepts `--package-root`, `--reference-package-root`,
`--inputs`, `--device` and `--output`. The command in `READY.json` compares the
proposal to the **same RA6 law**, on copied saved CPU fixtures, and checks the
complete planner outputs plus unchanged RNG. It measures callbacks, Jacobian,
SVD, item/scalar/nonzero/std and matrix operators, with warmed repeat timings
reported separately. Fixture/model transfer is outside the measured planner.
It uses deterministic algorithms, no TF32 and memory fraction 0.2 on CUDA0.
Only root may run CUDA, after its active quality child finishes.

The fixtures preserve original checkpoints and use their stored CPU RNG state
only for private reference fitting/planning. Global RNG stays unchanged. The
CPU profiler's device discovery sees no exposed GPU; the checks confirm no CUDA
context is initialized. No learned/native quality PASS or CUDA speed is claimed.
