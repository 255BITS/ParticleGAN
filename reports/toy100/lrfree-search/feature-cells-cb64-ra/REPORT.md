# CB64-RA configuration and validation

**Result: the recommended combined config fails overall quality acceptance.** The implementation and checkpoint checks pass, and reaction cost improves. Mass preservation, folded geometry and learned-model coverage remain unresolved. Keep this config experimental; it is not ready to become the base package default.

All prescribed measurements are complete.

## Configuration and implementation

Created [overrides-CB64-RA.json](configs/overrides-CB64-RA.json) and its [matching package](pkg-CB64-RA/particlegan/feature_cells.py). Select this package before importing ParticleGAN; [USAGE.md](USAGE.md) shows the loader. The JSON preserves every E22 option and adds:

```json
{
  "birth_death_backend": "feature_cells",
  "birth_death_cells": 64,
  "birth_death_metric_rank": 8,
  "birth_death_chunk": 256,
  "birth_death_parent_policy": "real_anchor"
}
```

Ordinary reaction and isolation both execute. A real-reference-only projection and 64 cells replace global neighbor scans. Isolation scores use the held-out real half. Parents come from bounded reservoirs near real cell representatives; each repair examines at most 256 eligible parents. Ordinary moves use conditional cell-count tests and mass quotas. This changes the categorical transport law and has the statistical limits recorded in the [implementation protocol](implementation/PROTOCOL.md), fixed before acceptance.

Training, fake-pool generation, serving and row copies use latent Gaussian jitter with standard deviation .025 and norm cap .05. This replaces the candidate's adaptive DV12 bandwidth and nearest-table clipping; the other DV12 training controls remain. EMA, Adam/AMSGrad buffers, A2 history, row-evidence reset and stationarity rebase follow copied rows.

The generic JSON retains E22's N=12/z=4 structural defaults. Harnesses set their declared dimensions; users must set dimensions for their networks. The feature-cell backend requires the matching code. The legacy `knn` backend remains the default.

## Acceptance summary

| Check | CB64-RA result | Meaning |
| --- | --- | --- |
| Integration | 8/8 pass | Counts, bounded plans, row state, refresh and checkpoint correctness |
| Learned checkpoint replay | 4/4 pass, including E22 controls | Ten-update semantic state and loss equality |
| Nominal reaction speed | PASS: 20.48× at N8192 | Full first reaction, fixed toy CPU scope |
| Cost quality | 0/12 native-full passes | Ordinary and isolation both enabled |
| Trained-critic geometry | 1/14 native-full passes | No complete family qualifies |
| Frozen-critic geometry diagnostic | 1/14 native-full passes | Separate representation diagnostic |
| Learned 25-mode toy | FAIL | Precision, mode coverage and mass-TV gate |
| MNIST | Coverage regresses | Comparison; no numerical pass gate was declared |
| Portability CPU scorers | 10/13 pass | Full-budget CPU diagnostics |
| Native CPU scorers | 0/3 pass | Full-budget CPU diagnostics |
| Canonical GPU acceptance | UNAVAILABLE | CUDA device access lost in managed environment |

The independent static lane ran 40 unchanged fixtures against both packages (80 rows), plus six separately declared latent-sampling probes. A failed E22 reference does not qualify the candidate. Each family has its existing fixed seed; there were no seed sweeps or post-outcome source, threshold or gate changes.

## Reaction cost and scaling

| Particles | CB64-RA ms | E22 ms | Speedup |
| --- | --- | --- | --- |
| 1024 | 40.42 | 51.82 | 1.28× |
| 2048 | 44.45 | 145.96 | 3.28× |
| 4096 | 74.28 | 533.76 | 7.19× |
| 8192 | 108.30 | 2218.33 | 20.48× |

The fitted time slope is 0.501 for CB64-RA and 1.813 for E22. These are four single timings at fixed width, rank, cell count and model size. They establish this benchmark's speed gate, not a universal sublinear scaling law.

Source bounds controller work by linear terms plus sorting, roughly `O(N*h*r + N*K*r + N*h*log(N))` for fixed caps. G/D forwards still cost `O(N*(C_G+C_D))`; wider models increase these costs. Native projected distance blocks never exceed 256×256. Raw-real FIFO memory remains `O(N*output_size)` and feature arrays remain `O(N*h)`. The largest retained snapshot receipt is 643,584 bytes, which is not a process peak. Cumulative mixed-package process RSS reaches 1,078 MiB and cannot be attributed to candidate peak memory.

## Why quality does not pass

**Supported parents can still have the wrong mass allocation.** All nominal isolation parents are supported, but intended-mode agreement is only 60.9–72.1%. Even exact cloning produces TV .01184–.01709 against E22's zero, exceeding the original E22+.01 limit at all four particle counts. The high-dimensional fixture has zero unsupported-row recall at every N, leaving about 4.5% unsupported mass. Rare-hole isolation passes the broad original support gates while inflating rare mass 4.375–23.75×; full population receipts make that failure visible.

**A correct parent does not make a safe perturbation.** On the trained folded 2D fixture, parent intended-mode validity is 100% conditional on detected unsupported children. Native four-turnover centers nevertheless retain only 60.8% of target rare mass. A separate actual sampling-kernel probe takes exactly repaired centers from 100% support to 46.14% support; the same 128D fold retains 99.32%. The dimension-independent norm cap changes coordinate noise with dimension and does not account for local generator geometry.

**Calibration also fails fixed rare-tail gates.** The trained N2048 critic detects all 82 planted rows but flags three supported rows, including one rare row. The frozen-initialization critic flags four supported rows, giving FPR .002035 against a .002 limit. Finite shared calibration imposes a roughly 40-discovery BH floor, and the 5% isolation guard disables action when many rows are flagged. The evolving critic and reused FIFO do not supply an unconditional guarantee for repeated adaptive decisions.

Static native quality evaluates clean G(table centers) after actual copy perturbations. The six probes additionally jitter every row for serving. Their oracle evaluates latent jitter only; output-noise support is undefined. Full support-repair precision includes ordinary relocations of already supported rows, so a low combined precision alone is not evidence that those moves are harmful. Population TV, rare retention and unsupported mass are separate evidence.

## Learned-model comparisons

Both models ran 2,000 updates at N1024/z128/batch128 against the same data and initial model/prior hashes. Timings count trainer updates and exclude evaluation and checkpoint writing. These paired CPU measurements are contemporary.

| 25-mode toy | Precision | Modes /25 | Mass TV | Train seconds | Ordinary / isolation moves |
| --- | --- | --- | --- | --- | --- |
| E22 | 0.6808 | 25 | 0.3192 | 77.05 | 1981 / 0 |
| CB64-RA | 0.5160 | 23 | 0.4840 | 35.64 | 3268 / 0 |

Both fail the toy gate: precision ≥.9, all 25 modes, mass TV ≤.1. CB64-RA is 2.16× faster at equal updates but has worse precision, coverage and TV. At the common 35.64-second training budget, E22's latest eligible checkpoint is update 750 (29.61 seconds): precision .5262, 25 modes, TV .4738. Candidate update 2000 is .5160/23/.4840. The 6.03 unused E22 seconds reflect checkpoint granularity; this comparison also supplies no quality gain.

| MNIST | Active embedding precision | Active embedding recall | Active FD | Class TV | Train seconds |
| --- | --- | --- | --- | --- | --- |
| E22 | 0.8345 | 0.8774 | 0.6651 | 0.0631 | 339.80 |
| CB64-RA | 0.9297 | 0.1519 | 1.1627 | 0.0914 | 278.36 |

MNIST uses the previous real-only CNN evaluator (98.08% held-out accuracy), with 39 training-active standardized embedding coordinates; raw 64-dimensional metrics are retained too. These are embedding FD and manifold metrics. Both packages cover ten confident classes, but candidate recall falls from 87.7% to 15.2%, indicating substantially less coverage within the evaluator's feature space. Higher precision accompanies recall and FD regression. Candidate ordinary/isolation moves are 5,386/85 versus E22's 398/0.

At the common 278.36-second training budget, E22's eligible update 1500 (255.20 seconds) has active FD .7900 and class TV .0685, compared with candidate 1.1627/.0914 at update 2000. E22 has 23.16 unused seconds. No numerical image acceptance gate was declared. These are single-seed comparisons, and the combined change has not been ablated; the results do not identify a sole cause.

## Full-budget CPU scorer results

CUDA became unavailable after the environment changed. The pre-outcome [CPU amendment](validation/CPU-AMENDMENT.md) preserves budgets and scorer gates in an owned host-runner copy. CPU RNG/device and native initial tensor hashes differ from the canonical GPU fixture and are recorded. Original hosts/scorers remain unchanged. These CPU verdicts cannot certify canonical GPU acceptance; historical GPU successes are references only.

| Family | Task | Completed / required updates | CPU scorer verdict |
| --- | --- | --- | --- |
| portability | mode_hold | 1200 / 1200 | PASS |
| portability | img_intensity2 | 600 / 600 | PASS |
| portability | img_blobs4 | 600 / 600 | PASS |
| portability | img_stripes2 | 600 / 600 | PASS |
| portability | img_bars4 | 600 / 600 | FAIL |
| portability | vector_two_broad | 1200 / 1200 | PASS |
| portability | vector_unequal_mass | 1200 / 1200 | FAIL |
| portability | vector_unequal_width | 1200 / 1200 | PASS |
| portability | vector_anisotropic | 1200 / 1200 | PASS |
| portability | vector_overlap | 1200 / 1200 | PASS |
| portability | vector_spiral | 1600 / 1600 | PASS |
| portability | ring_shift | 4600 / 4600 | PASS |
| portability | stationary | 7500 / 7500 | FAIL |
| native | grid100 | 7000 / 7000 | FAIL |
| native | rotated100 | 7000 / 7000 | FAIL |
| native | staggered100 | 7000 / 7000 | FAIL |

Native runs include all 7,000 updates, 34 observations, five final 20,000-sample checks and the 100,000-sample holdout. Official noisy/live coverage and accuracy scorers decide acceptance; clean and EMA outputs are diagnostics. The [validation report](validation/REPORT.md) retains primary scorer verdicts and detailed receipts.

| Native task | Coverage/accuracy | Holdout precision | Holdout TV | Absolute covariance trace bias | Radial KS |
| --- | --- | --- | --- | --- | --- |
| grid100 | FAIL/FAIL | 0.9384 | 0.0393 | 0.3763 | 0.1542 |
| rotated100 | FAIL/FAIL | 0.9385 | 0.0231 | 0.3729 | 0.1490 |
| staggered100 | FAIL/FAIL | 0.9382 | 0.0289 | 0.3345 | 0.1415 |

All three native runs cover 100 modes at the end, but fail their final five accuracy checks and independent holdouts. Absolute covariance trace bias exceeds its .10 limit and radial KS exceeds .04. Mode count alone does not satisfy the coverage gate, which also requires quality. These timings are concurrent one-thread CPU diagnostics, not comparisons with archived GPU runtime.

`img_bars4` fails with only 3/4 qualifying modes. `vector_unequal_mass` meets final metric thresholds but fails stable convergence: one passing final check versus five required. Final snapshots alone are insufficient for its gate.

`stationary` passes many early and middle checks, then drops below its quality gate late in the 7,500-update run. Final high-quality fraction is .2949 against the required .9, with zero consecutive passing checks at the end. It fails both noisy and clean versions. Stopping at an earlier passing observation would have missed this result.

## Correctness, reproducibility and recommendation

All eight integration checks pass, including actual ordinary actuation, parent eligibility, EMA/AMSGrad/A2 row copies, stale refresh, malformed FIFO rejection, legacy defaults and exact checkpoint replay with moves. Four learned checkpoint replays pass over ten continued updates. Candidate branches include an actual cell evaluation and 28 toy / 51 image ordinary moves. All semantic state and losses match; only observational evaluation duration is excluded.

The [source freeze](review/source-freeze.json) and [audit](review/audit.json) identify tested bytes. The [patch](review/candidate.patch) changes four source files: the new feature-cell module, recipe validation, trainer routing and exports. Reference package bytes remain unchanged. Scheduling amendments only transfer exclusive CPU task ownership; collector before/after hashes are recorded. They do not alter candidate code, quality runners or gates.

**Recommendation:** retain the bounded compute design as a research candidate. Resolve mass allocation, local jitter geometry and support calibration before promoting it for general users. This run supplies no evidence that the combined config improves large-model quality. The inherited row-evidence gate remains inactive at z128 (effective-sample cap 99 versus a required 384), and raw FIFO memory still scales with output size. These package issues remain open.

Artifacts: [machine-readable leaderboard](leaderboard.json), [static metrics](geometry/case_metrics.csv), [static report](geometry/REPORT.md), [learned and scorer results](validation/results.json), [implementation notes](implementation/README.md), [artifact manifest](artifact_manifest.json).
