# CB64-RA failure diagnostics and corrections

Updated 2026-09-30T08:09:04.123083+00:00. Status: IN_PROGRESS.

No package is recommended for the toy/grid target until both unchanged gates and required validity/replay checks pass. E22 remains the broad reference; RA4 has the best measured MNIST feature distance.

See [the current toy/grid leaderboard](quality/REPORT.md) for the required joint target. A failed toy leaves that candidate's grid unrun; runtime errors have no quality verdict.

## Matched learned CUDA results

N=1024, z=128, batch=128, 2000 updates, seed=314159; identical saved data/evaluator and initial G/D/prior tensors.

| Variant | Toy precision | Modes /25 | Toy mass TV | MNIST active FD ↓ | MNIST precision | MNIST recall |
|---|---:|---:|---:|---:|---:|---:|
| E22 | 71.545% | 25 | 0.284546 | 0.544488 | 86.914% | 84.717% |
| CB64-RA | 62.329% | 24 | 0.377969 | 1.488828 | 85.254% | 32.275% |
| CB64-RA2 | 15.662% | 1 | 0.843384 | 0.814142 | 84.180% | 71.387% |
| CB64-RA3 | 40.991% | 17 | 0.592935 | 0.476611 | 90.771% | 75.830% |
| CB64-RA4 | 75.806% | 21 | 0.263540 | 0.393402 | 89.551% | 78.613% |
| CB64-RA5 | ERROR | — | — | PENDING | — | — |
| CB64-RA6 | 51.697% | 23 | 0.483032 | PENDING | — | — |
| CB64-RA7 | 68.164% | 25 | 0.318359 | PENDING | — | — |
| CB64-RA8 | 96.533% | 25 | 0.052114 | PENDING | — | — |

The frozen toy gate requires precision ≥90%, all 25 modes with at least 1% supported mass each, and mass TV ≤0.10. MNIST has comparative metrics rather than an absolute promotion gate. These measurements do not establish broad architectural scalability.

## Canonical CUDA screens

| Variant | Portability passes /13 | Native passes /3 | Completed /16 |
|---|---:|---:|---:|
| CB64-RA | 9 | 0 | 16 |
| CB64-RA2 | 13 | 2 | 16 |
| CB64-RA3 | 0 | 0 | Withheld: indexed API mismatch |
| CB64-RA4 | 4 | 0 | 5 |
| CB64-RA5 | 0 | 0 | 0 |
| CB64-RA6 | 0 | 0 | 0 |
| CB64-RA7 | 0 | 0 | 0 |
| CB64-RA8 | 0 | 0 | 1 |

All native gates require the original 7000-update budget, 34 observations, five passing terminal 20k evaluations and an independent 100k holdout. A passing final cloud alone does not pass the full stability gate.

RA4 exposes the declared indexed generation API. The copied strict collector hardcodes the old candidate's `evaluation_generate=plain`, so its original INVALID/ERROR metadata receipts are preserved. A separate hash-bound checker changes only that expected option to `indexed`; it retains the original source/data/init/stream/schedule/native evidence and quality checks. RA4 acceptance is reported explicitly under this declared API correction. Counts in the table are accepted, audited results; raw primary PASS values alone are insufficient. The original strict monitor and its ERROR receipts remain alongside the separate indexed-API review.

## Measured cost corrections

Paired CUDA contracts preserve outputs, actions and RNG while reducing synchronization. The exact count kernel measured 29.669 → 1.088 ms with scalar reads 132 → 1. Warm indexed sampling at N=20,000 measured 13.78 → 9.39 ms for 2048 queries and 217.99 → 75.37 ms for 20,000 queries; corresponding axis scalar reads fall to zero. The final planner reduces warm saved-toy scalar reads 2175 → 106 and nominal-fixture reads 851 → 339. These are focused measurements on shared GPU0; cold timings and small fixtures vary, and they do not establish faster end-to-end training or an architectural scaling law.

## Reproduced causes and fixes

- Finite count-test resolution prevents actuation below N=800. The corrected package routes these populations to the established E22 backend.
- With-replacement parents amplified a two-row rare group into 29 rows. Real-reference cell/group targets, unique parents and post-action supported ledgers prevent that reproduced mass inflation.
- A fixed latent noise scale collapsed folded support. A bounded local sampler restores that fixed-input support contract; coordinate candidates alone miss some copied neighbors, addressed by a bounded serialized lineage graph.
- Widespread support flags blocked both isolation and ordinary mass recovery. V4 allows count-certified flagged-only ordinary recovery within its existing 5% budget.
- Nearest-cell counts conceal support holes within cells. The prospective count family combines original cell mass, fixed even-fit inside/outside cell regions and aggregate support counts, with one correction across all hypotheses and one shared action ledger.
- CUDA coordinate tensor indices force repeated device-to-host scalar reads. Python axis metadata removes that overhead without changing sampler values or RNG. Exact planner batching preserves quotas, action order, certificates and the shared budget.
- Canonical indexed generation requires a fifth argument named indices. RA3 exposes rows instead, so its native screens are withheld; the subsequent API fix restores indexed lineage sampling.

RA5/RA6 add separately checked live/EMA copy geometry, bounded real-anchor latent births and population stationarity after replacement. RA5 stopped on diagnostic JSON serialization; RA6 corrected that error and failed the final toy. RA7 lowers G/sigma base rates and adds exact integer group-count reduction; its final toy also fails, with adaptive scaling compensating for the lower base rate. RA8 preserves RA7 training and checks current paired averaged geometry before serving the average. Its final toy passes and its full grid fails. See the target leaderboard for exact gates.

## Validation scope and limitations

RA2 executes the original 16 canonical screen budgets. RA3 is a learned ablation with focused GPU contracts and exact checkpoint replay; its native suite is not run because the frozen API mismatch was identified first. RA4 receives the same learned and canonical CUDA tests after source/config/commands are frozen.

The standalone high-dimensional support-detector proposals remain unqualified. Bounded parent selection requires an existing eligible parent in an accessible target region; it cannot reliably recover an absent mode. Training and served averages can have different mode counts. Finite fixed-partition count-test checks do not provide cumulative error control for repeatedly adapted learned features/FIFO decisions. GPU0 is shared with other jobs; timing variation is not a scaling law. Source integrity, restored optimizer/RNG state and checkpoint replay are audited separately from model quality.

## Artifacts

- [RA2 validation](validation/), [RA3 learned ablation](validation-ra3/) and [RA4 validation](validation-ra4/).
- [RA4 source freeze](integration/iteration-4/READY.json), [declared indexed API checker](performance/sampler-regression/cpu-plan-review/indexed-metadata/), and [separate acceptance audit](integration/review/ra4-indexed-api-monitor/).
- [Lineage tests](geometry/training-regression/LINEAGE-REPORT.md), [training-state diagnosis](performance/training-regression/REPORT.md), and [independent reviews](performance/training-regression/count-review/).
- [Kernel/count profiling](performance/PERFORMANCE.md) and [sampler API/cache proof](performance/sampler-regression/cpu-plan-review/AXIS-ID-REPORT.md).
- [Paired CUDA sampler receipts](integration/axis-gpu/result.json), [exact planner receipts](integration/iteration-4/gpu-plan.json), and [missing-mode diagnosis](integration/review/training-regression/post-ra4-mode-diagnosis/REPORT.md).
- Local experiment root and raw-artifact paths are recorded in SOURCE-ARCHIVE.json when archived. Raw checkpoints, datasets, clouds and large traces are retained locally.
