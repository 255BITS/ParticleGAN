# Fixed-sigma MoG prior: Stage 0 gate

Only Stage 0 has run. Stages 1–3 require the owner’s go/no-go. Reference code: `af1843a`; run code SHA is recorded per row in `results.csv`, together with per-run source hashes and archives.

## Regression

- Seed 1: 70 original log entries identical: **True**; every final EMA generator tensor and the particle table identical: **True**.
- Seed 2: 70 original log entries identical: **True**; every final EMA generator tensor and the particle table identical: **True**.
- Seed 3: 70 original log entries identical: **True**; every final EMA generator tensor and the particle table identical: **True**.

## Leaderboard

| Arm | Seed | Modes | HQ | HQ/real | Width/real | KL balance | Pass | t_cover |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| C0 | 1 | 100 | 0.99011 | 1.00127 | 0.79092 | 0.037504 | no | 5500 |
| C0 | 2 | 100 | 0.98805 | 0.99925 | 0.89654 | 0.033769 | no | 5500 |
| C0 | 3 | 100 | 0.98796 | 0.99883 | 0.84284 | 0.033068 | no | 5500 |
| C1 | 1 | 100 | 0.06409 | 0.06482 | 10.14276 | 0.425967 | no | None |
| C1 | 3 | 100 | 0.05925 | 0.05990 | 10.24265 | 0.357353 | no | None |
| C1 | 2 | 99 | 0.05670 | 0.05734 | 10.18208 | 0.401190 | no | None |

Cells are ranked by pass rate, then HQ, then closeness of width to real. Individual seeds above are shown for diagnosis, not model selection.

- **C0: 0/3 pass**; HQ/real 0.99978 ± 0.00107; width/real 0.84343 ± 0.04312; KL 0.034780 ± 0.001947. Spread is population std across seeds.
- **C1: 0/3 pass**; HQ/real 0.06069 ± 0.00310; width/real 10.18916 ± 0.04109; KL 0.394837 ± 0.028369. Spread is population std across seeds.

## C0e deterministic mass balance

| Seed | All-table HQ | All-table KL | Min share ×100 | Max share ×100 |
|---|---:|---:|---:|---:|
| 1 | 0.98975 | 0.036844 | 0.4648 | 1.7984 |
| 2 | 0.98785 | 0.033613 | 0.5264 | 1.6500 |
| 3 | 0.98810 | 0.033031 | 0.4908 | 1.7660 |

This audit evaluates each of the 20,000 atoms once, so repeated sampling cannot explain its imbalance. The ordinary final metrics use 200,000 draws, matched to 200,000 independent real samples.

Historical hist_kl uses all nearest-assigned samples, including bridge samples. Applying both estimators directly to the saved pre-change 20k-sample outputs:

| Seed | Historical definition | HQ-only definition |
|---|---:|---:|
| 1 | 0.034977 | 0.036270 |
| 2 | 0.032215 | 0.033291 |
| 3 | 0.032404 | 0.032937 |

These are measurements of freshly reproduced pre-change outputs; they are not recovered scores from the old five-seed FINDINGS study, whose raw summaries were not present in the checked result locations.

## Allocation null (1,000 simulated draws each)

- N=100: mean KL **0.56891**, mean empty modes **36.557**.
- N=200: mean KL **0.28368**, mean empty modes **13.526**.
- N=400: mean KL **0.13208**, mean empty modes **1.829**.

## Timing

- C0: 70.2 seconds/run including evaluation and artifacts (range 69.8–70.8).
- C1: 59.3 seconds/run including evaluation and artifacts (range 58.8–59.7).

## Predictions and next decision

All seven preregistered predictions remain **inconclusive**: no small-N or positive-r 7k-step experiment has run. Specifically: (1) N=100 atoms, (2) N=400 plateau, (3) bridge versus heuristic, (4) tearing versus migration, (5) N-dependent allocation failures, (6) low-r conditioning, and (7) r=2 versus C1 all await their specified comparisons. C0 and C1 are references, not tests of those predictions.

Recommendation: proceed to the Stage 1 optimizer pilot, subject to owner approval, while retaining the preregistered pass threshold. C0 has a reproducible mass-balance problem and compressed width; matching its HQ alone would not establish success. C1 supplies a clearly separated failure reference. No positive-noise performance claim is supported yet. At roughly 70 seconds/run and two GPUs, budget about 25 minutes for the 42-run pilot, 70 minutes for the 120-run main grid, and 3–4 minutes per control block, before re-timing a small-N run.

## Definitions and deviations

- Accepted correction: pilot multipliers are relative to shipped particle LR 0.006; G LR remains 0.0006. The exposed particle beta1 is independent of G/D beta1.
- Accepted correction: mode_coverage already uses sample(); no direct-indexing fix was needed there. Fixed scatters cap the sample count at N and reuse a seeded epsilon buffer.
- Accepted correction: raw VICReg is scale-dependent and its variance term is not redundant. Standardization removes the adversarial scale escape approximately, not full-loss scale dependence.
- Accepted correction: bridge_heuristic is not a rigorous bound. It is recorded but does not decide PASS.
- Width exactly reuses median-centered radial core sigma, averaged equally over nearest-assigned modes with at least 50 samples; no HQ truncation. Width audited-mode counts accompany it.
- The wrapper’s older final coverage threshold differed from >=10 HQ samples; the new suite uses >=10 throughout. Existing training log lines are preserved for the regression comparison.
- Metrics run at completed updates 100, 200, …, 7000; t_cover has 100-step resolution. Legacy console logs retain their original zero-based labels.
- Reference samples are seeded independently of latent samples. Noise remains enabled at evaluation for positive-r runs. C0e separately evaluates every atom exactly once.
- Prior geometry and component identity have no counterpart for real data or a fresh Gaussian prior: those comparisons are not defined. C1 component/geometry fields are null, and those C1 reference lines are omitted from plots.
- Components with zero HQ samples have undefined purity and no majority allocation; they are counted separately as components_no_hq. Purity mean excludes them. components_unsampled records missing observations.
- Per-run components.json contains purity_i, bridge_i, majority_mode and alloc. For C0 these use the exact all-atom audit; final aggregate component statistics use the 200k sample.
- Geometry is measured on the EMA table; raw_std_live and its ratio also track the live table. Drift flags are symmetric for increases/decreases beyond the specified factor.
- Final eval is batched in chunks of 20k after drawing all latent codes, bounding generator memory. The C0 checkpoint audit establishes exact training preservation.
- Stage 0 plots use training step, since C0 has r_eff=0 and C1 has no particle-distance ratio. Log-r_eff plots, pass-rate curves and the Stage 2 scatter await Stage 2; no placeholder results are claimed.
- Three explicitly requested seeds override the repository’s general no-seed-experiments preference for this study. C0e reuses C0; no separate training runs.
- The optional rotated-grid control remains unimplemented and unrun at this gate. FINDINGS.md does not already resolve the lattice confound.

## Artifacts

- results.csv: one row per C0/C1 run, configuration, seed, SHA and all scalar final metrics.
- stage0/<arm>_s<seed>/metrics.jsonl: 70 metric rows per run; log.txt is tail-friendly.
- regression.json, allocation_null.json, stage0_traces.png, stage0_width_hq.png.
- Each run retains source.zip, provenance.json, final.pt, final_samples.npz, and components.json.
