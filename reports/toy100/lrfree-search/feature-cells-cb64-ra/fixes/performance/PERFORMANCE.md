# CB64-RA synchronization diagnosis and private patch

The completed frozen learned runs measured CB64-RA training 11.7% slower than E22 on toy and 13.5% slower on MNIST, despite lower allocated GPU memory. This diagnostic isolates implementation overhead; it introduces no model, recipe, seed, quality gate, statistical null or sampling-law changes. The baseline studies remain read only. Root alone executes CUDA diagnostics in serialized slots that pause the next original-harness launch after its current child completes.

The feature backend launches many tiny per-cell operations and reads their results on the host. Scalar extraction waits for CUDA results; boolean indexing invokes a variable-size nonzero result that also requires synchronization. These operations interrupt asynchronous kernel dispatch. This behavior is documented in the [official PyTorch CUDA synchronization discussion](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/), and the measured CUDA trace below establishes the actual synchronization counts for this implementation.

## Conditional count comparison: verified on CUDA

| Measurement for K64, 512 calibration draws and 1024 fake draws | Original | Batched |
|---|---:|---:|
| Synchronized milliseconds per call, 20 repetitions | 29.669 | 1.088 |
| Speed ratio | 1 | 27.28 |
| CPU scalar extraction events | 132 | 1 |
| Variable nonzero events | 64 | 0 |
| CUDA stream synchronization events | 196 | 2 |
| CUDA memcpy events | 260 | 3 |

The original loops over every cell, extracts two CUDA counts per cell, builds its support, then boolean-selects the probability tail. The patch batches all support rows, masks padding before logsumexp and transfers just the support extent/work metadata together. Validation is also batched. The factorial expression, two-sided probability ordering, `+1e-10` equality rule, Q/cells rejection threshold and logical count-test work are unchanged.

CUDA parity covers 196 exhaustive tiny two-cell tables; uniform64, rare64, one-cell and many-empty-cell tables; and seven malformed count tables. The maximum absolute probability difference is 1.11e-16 across tiny tables. Special tables are bit-exact; every categorical rejection and logical enumeration work receipt matches. The same CPU cases are entirely bit-exact. Device identity is guarded to the declared physical GPU0. See gpu-count-diagnostic.json and the two CUDA count traces.

The measured 28.58-ms per-call reduction would remove about 7.15 seconds across 250 comparisons if this microbenchmark cost held throughout a 2000-update learned run. That estimate gives a plausible scale for the observed regression; it is not a measured whole-training speedup. Shared GPU conditions and actual count distributions affect the full run.

Integration patch: COUNT-ONLY.diff, SHA256 `3fca6295393ffee675e009c1d66b18e74abbd179936b33ccfc7f023864535066`. It changes only conditional_count_pvalues and can be applied independently of the policy fixes.

## Fitting and parent pools: CPU and CUDA parity verified

Stable integer grouping replaces per-cell boolean row gathers during Lloyd fitting, SSE collection and representative selection. It retains the original within-cell row order and per-cell mean/sum operations, avoiding floating scatter atomics. Farthest-first center selections stay as device tensors and use index_select, preserving the original argmin/argmax ties while avoiding host scalar extraction. Counts/offsets are copied to the host once per grouping pass.

Parent pools now sort one unchanged random-priority vector stably by priority and then stably by cell. This produces the exact original per-cell priority order, original row-index tie order, same 64-slot cap, same eligibility counts and same generator state. It replaces K variable nonzero selections with fixed-shape integer gather/where operations.

CPU checks passed bit-exactly for six full snapshots: learned widths128/64 at N1024, constant/rank0 features, tiny odd N7, repeated rows and rare patterns. Assignment/support/count-comparison decisions, all saved snapshot fields/work receipts and RNG states match. Ten pool fixtures include no eligible rows, a saturated reservoir, rare cells and tied priorities; three transports include inaccessible birth cells and excluded deaths. Four malformed feature fixtures reject correctly.

| One CPU N1024/width128 snapshot fit | Original | Grouped |
|---|---:|---:|
| Scalar extraction events | 199 | 6 |
| Variable nonzero events | 385 | 1 |
| Recorded fit milliseconds, single observation | 17.03 | 11.71 |

The CPU fit timing is one observation. Root also executed the frozen snapshot check in a second serialized CUDA slot:

| One CUDA N1024/width128 snapshot fit | Original | Grouped |
|---|---:|---:|
| Synchronized fit milliseconds | 56.905 | 33.824 |
| Scalar extraction events | 201 | 12 |
| Variable nonzero events | 385 | 1 |

CUDA fit speed ratio is 1.68 for this single timed observation. All six fitted snapshots, ten parent pool fixtures and three quota transports match bit-exactly on CUDA, including every floating fit field, assignment/support/count decisions, parent/child indices, work counters and RNG state. Maximum snapshot difference is zero. Constant/rank0, odd/tiny, repeated and rare feature cases remain covered. Private feature_cells.py SHA256 stays `53f86ed38337c8875c374ebbf1e4abf4eb0e561e71d77b5d3c9c87cc03eb3615` across both GPU checks. See gpu-snapshot-parity.json and the two CUDA fit traces. These function measurements remain separate from whole-training throughput.

PERFORMANCE-CORE.diff composes count comparison, fitting and pools, with SHA256 `c5eaef39b2672dd633ad3334ba1e965081d8e4de0a4cb4a95008dd931a31237f`. All three functions now have CUDA reference-parity evidence; root is integrating this core. The stability owner confirmed the new pool API/output semantics compose with its mass policy. The core patch excludes ordinary_transport and _integer_allocate so it preserves the stability owner's capacity, parent-reuse and isolation changes. The full PERFORMANCE-PATCH.diff retains those separate metadata optimizations for reference, not as the integration patch.

## Scope and remaining checks

The batched count matrix pads to the widest cell support. Extreme unequal cell counts can therefore do more arithmetic than the original variable support loop, bounded by K times the widest support. It avoids reference-by-reference arrays and retains the original logical work receipt; those receipts count statistical terms, not padding arithmetic. Memory and timing should be read from the actual CUDA traces for each declared case.

CPU diagnostics used an empty CUDA visibility mask and finished with Torch CUDA uninitialized. The CPU profiler emitted a masked-device discovery warning from its library, but created no CUDA context or GPU work. No seed sweep, learned training, quality evaluation or GPU launch was executed by this agent.

Recommendation: integrate the CUDA-verified count/fit/pool core while retaining the stability owner's transport logic. Keep the original quality gates and evaluate the independently motivated policy/kernel fixes through root's declared tests. The count and fit speedups establish that repeated synchronization caused substantial overhead in those functions. They do not resolve CB64-RA's quality regressions or certify a whole-training speedup.
