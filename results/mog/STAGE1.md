# Fixed-sigma MoG: Stage 1 optimizer pilot

Stage 1 only: N=100/400, nominal r=1/8, 7k updates, three seeds per cell. Standardization on; fixed σ calibrated at initialization. LR multipliers are relative to the shipped particle LR 0.006. All other training settings retain the shipped recipe.

## Revised pass criterion

Frozen before the first pilot result: 100 modes; HQ/real ≥ 0.99883228; width/real in [0.79091586, 1.20908414]; KL ≤ 0.03750414.
This accepts the observed three-seed C0 envelope (C0 passes 3/3), not statistical equivalence. It allows width to move toward real data without rewarding excessive broadening. The original strict pass and comparison with C0 means are also recorded. HQ remains high because the owner requested original-or-better quality; the width and KL requirements are relaxed.

## Leaderboard

| N | LR ×0.006 | β1 | Pass | Strict | Modes | HQ/real | Width/real | KL | r_eff | Purity | Empty allocations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 0 | 0/3 | 0/3 | 90.7 | 0.29137 | 4.400 | 0.3768 | 0.1642 | 0.980 | 13.3 |
| 100 | 10 | 0.5 | 0/3 | 0/3 | 90.0 | 0.28825 | 4.284 | 0.3558 | 0.1539 | 0.985 | 12.0 |
| 100 | 3 | 0 | 0/3 | 0/3 | 91.7 | 0.26122 | 4.735 | 0.3606 | 0.1486 | 0.963 | 12.7 |
| 100 | 1 | 0 | 0/3 | 0/3 | 90.3 | 0.24171 | 5.011 | 0.4148 | 0.1320 | 0.931 | 16.0 |
| 100 | 0.3 | 0 | 0/3 | 0/3 | 93.0 | 0.18589 | 5.711 | 0.3861 | 0.1284 | 0.904 | 15.7 |
| 100 | 0.1 | 0 | 0/3 | 0/3 | 93.0 | 0.15951 | 6.100 | 0.3709 | 0.1211 | 0.908 | 17.0 |
| 400 | 10 | 0.5 | 0/3 | 0/3 | 100.0 | 0.38016 | 3.373 | 0.1563 | 4.6297 | 1.000 | 0.0 |
| 400 | 10 | 0 | 0/3 | 0/3 | 100.0 | 0.37668 | 3.451 | 0.1383 | 4.5357 | 1.000 | 0.0 |
| 400 | 3 | 0 | 0/3 | 0/3 | 100.0 | 0.30486 | 3.890 | 0.1559 | 2.6593 | 1.000 | 0.0 |
| 400 | 1 | 0 | 0/3 | 0/3 | 100.0 | 0.20841 | 4.810 | 0.1386 | 0.1455 | 1.000 | 0.0 |
| 400 | 0.3 | 0 | 0/3 | 0/3 | 100.0 | 0.17343 | 5.139 | 0.1377 | 0.1227 | 0.999 | 0.0 |
| 400 | 0.1 | 0 | 0/3 | 0/3 | 99.7 | 0.16879 | 5.220 | 0.1360 | 0.1208 | 0.998 | 0.3 |

Ranking: pass rate, then mean HQ, then distance of mean width/real from one. Cells use three runs; no best-seed selection. Seed standard deviations are in stage1_leaderboard.csv and every run is in stage1_results.csv.

## Selected settings

- N=100: LR multiplier **10** (initial LR 0.06), particle β1 **0**, **0/3 baseline-relative passes**, 0/3 strict passes. HQ/real 0.29137, width/real 4.400, KL 0.37679.
- N=400: LR multiplier **10** (initial LR 0.06), particle β1 **0.5**, **0/3 baseline-relative passes**, 0/3 strict passes. HQ/real 0.38016, width/real 3.373, KL 0.15626.

## Interpretation and diagnostics

- N=100: failure counts across 18 optimizer runs: {'coverage': 18, 'hq': 18, 'width': 18, 'balance': 18}. These overlap; they are not independent categories.
  Selected cell: purity 0.9803, components below 0.9 purity 7.0, empty majority allocations 13.3, r_eff 0.1642; r_eff drift flags 1/3, raw-scale drift flags 3/3.
- N=400: failure counts across 18 optimizer runs: {'coverage': 1, 'hq': 18, 'width': 18, 'balance': 18}. These overlap; they are not independent categories.
  Selected cell: purity 1.0000, components below 0.9 purity 0.0, empty majority allocations 0.0, r_eff 4.6297; r_eff drift flags 3/3, raw-scale drift flags 3/3.

The deterministic component-center audit is diagnostic only; it does not replace any noise-on quality metric. It evaluates G(means()) on CPU, using the saved EMA state.

- N=100, selected cell: fraction of component centers mapping within a data mode's 3σ radius 0.9100; fraction of noisy samples HQ 0.2881; nearest neighbor has the same majority mode for 23.667% of evaluable components.
- N=400, selected cell: fraction of component centers mapping within a data mode's 3σ radius 0.9942; fraction of noisy samples HQ 0.3759; nearest neighbor has the same majority mode for 91.206% of evaluable components.

## Recommendation

No original-or-better result at nominal r=1/8 and 7k updates. The relaxed threshold is not the main issue: noisy outputs remain substantially too broad and HQ is far below C0. Do not launch the full Stage 2 grid on the assumption that this pilot succeeded.
Recommended next experiment, pending owner direction: prioritize the N=400 standardized atoms control (r=0), r=1/32, and r=1/16 at the selected optimizer setting. The r=1/32 cell would be an explicit addition to the original Stage 2 design. This isolates the small-table limitation from noise-induced spread before spending the full 120-run budget. No such runs were launched.
Both LR winners are at the tested upper boundary, and their raw-scale drift needs to remain visible. They are the best tested optimizer settings, not established optima. Same-mode component clumping also makes the prescribed median-NN r_eff hard to interpret as separation between different output modes; retain its drift flag and the additional diagnostic, without silently redefining the independent variable.

## Prediction status

| Prediction | Status after Stage 1 | Evidence / missing comparison |
|---|---|---|
| 1: N=100 atoms | Inconclusive | No r=0 small-N control was trained. |
| 2: N=400 plateau | Inconclusive | 0/18 pilot runs pass at nominal r=1/8; other r values are untested and r_eff drifts. |
| 3: bridge heuristic | Inconclusive as a wall-sharpness test | Mode widths are too broad to attribute non-HQ mass solely to walls. Per-run bridge and heuristic values are retained in the CSV. |
| 4: balance by tearing | Inconclusive overall; little HQ tearing in selected N=400 cell | Mean HQ-conditioned purity 1.00000, KL 0.15626; allocation-null KL 0.13208. |
| 5: N-dependent allocation | Inconclusive for pass rates | Selected empty-allocation means: N=100 13.33, N=400 0.00; no r sweep. |
| 6: lower plateau edge | Inconclusive | No noise-radius sweep. |
| 7: r=2 versus C1 | Inconclusive | No nominal r=2 run. A large clumping-driven r_eff is not that control. |

## Execution and changes from the original design

- The owner explicitly revised the primary criterion to original-or-better. configs/mog/stage1_criteria.json freezes the baseline CSV hash, seed list, exact thresholds, and C0-mean thresholds.
- 30 LR runs plus six new β1=0.5 runs; the six selected β1=0 comparisons reuse identical completed LR runs. That is 42 planned comparisons but 36 unique training runs.
- The β1 round follows the winning LR independently for each N. No other hyperparameter was retuned.
- All final metrics use 200k EMA samples with noise enabled and equal-size reals. Traces use 20k samples every 100 completed updates. Both pass definitions are saved without overwriting Stage 0 history.
- An additional final-checkpoint diagnostic measures nearest neighbors with different majority output modes, to interpret clumping. It never replaces the prescribed r_eff or changes any selection criterion.
- Component purity excludes components with no HQ samples, which are counted separately. Width uses the unchanged median-radius estimator. r_eff and raw scale are reported, including drift flags.
- Optimizer plots use LR as the swept x-axis; r_eff is a diagnostic panel. The planned log-r_eff curves require Stage 2.
- All seven preregistered predictions remain conditional on their specified comparisons; Stage 1 only probes N=100/400 at nominal r=1/8. It does not test the full plateau, N=100 atoms, r=2, or the no-standardization control.

## Timing

Measured mean 74.6 seconds/run under the recorded GPU concurrency; total accumulated run time 44.8 minutes. Runner logs contain actual elapsed wall time.

## Artifacts

- stage1_results.csv: one row per unique run, all config and final scalar metrics, seed and git SHA.
- stage1_leaderboard.csv, stage1_lr_winners.json, stage1_winners.json.
- stage1_optimizer.png, stage1_width_hq.png, stage1_component_centers.csv.
- stage1/<run>/metrics.jsonl, log.txt, components.json, final.pt, final_samples.npz, source.zip and provenance.json.
