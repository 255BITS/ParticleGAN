# More MoG components and training

Exploratory single-seed configuration screen (seed 1), not a seed sweep or pass-rate estimate. All final metrics use 200k noise-on EMA samples and the same-size real reference. The C0 acceptance envelope remains frozen; passing it is weaker than beating C0 on every metric.

Completed 21 new configurations, including 14 positive-noise MoGs. 4 MoGs pass the envelope. 0 MoGs dominate seed-1 C0 on measured coverage, HQ, width error and KL. That comparison ignores training and table costs and is not a statistical claim.

Historical C0 across three seeds: HQ/real 0.99978, width/real 0.84343, KL 0.03478. The table uses the matching seed-1 C0 (width 0.79092, KL 0.03750), which was the weakest historical C0 width/balance result. The per-run CSV also records passed_c0_mean as a sensitivity check; no new repetitions were run.

## Outcome and recommendation

**MoG now has passing settings at both 400 and 20,000 components.** The 400-component r=1/40 model passes at 28k steps: HQ/real 0.999535, width/real 0.92636, KL 0.02888. This is 50 times fewer components and four times the training steps of original C0. It improves width and balance relative to original C0, but HQ is lower. Its raw scale grows 4.11 times, a recorded optimizer/gauge warning even though standardized reads keep output latent scale controlled. There is no matched 400-atom 28k control yet.

**At 20k components, extra training helps both priors.** At 28k, the unstandardized r=1/16 MoG has HQ/real 1.000046, width/real 0.95429, KL 0.026566; matched atoms have 0.999110, 0.94375, 0.026554. MoG has better measured HQ and slightly better width, with effectively tied balance (KL difference +0.000012). This is not evidence of universal superiority from one training seed. r=1/40 also passes, with width 0.95669 and KL 0.02742.

The 20k r=1/16, 28k result is the only new MoG that also clears all historical C0-mean thresholds. It uses four times the training budget. The unstandardized 20k r=1/40 setting already passes the frozen envelope at 7k, but not all C0-mean thresholds. Raw scale grows only about 11–13% in the longer 20k runs; effective sigma/spacing increases rather than collapsing to zero. Their effective-radius drift flags are retained.

**More components alone are not the solution.** At 7k, the fast small-table optimizer fails as N grows; switching to the shipped optimizer restores width but the standardized 1,600/6,400-component models still miss balance. Unstandardized and standardized 20k settings differ substantially on this seed. These observations do not establish a monotonic count law or imply standardization is always harmful.

Recommendation: keep both prior types. Use C0 as the simple 7k reference; use 400-component MoG at 28k when a compact prior with continuous support is valuable. For a larger-table quality comparison, retain 20k r=1/16 at 28k and matched 28k atoms. The next useful training comparison is a 400-atom 28k control, followed by lower-cost schedule refinements of the passing 400-component MoG. Avoid further increases in count before establishing a benefit at matched optimizer and budget. No additional training is left running.

## Leaderboard

| Setting | N | Steps | HQ/real | Width/real | KL | Pass | Failed | Seconds |
|---|---:|---:|---:|---:|---:|:---:|---|---:|
| C0_s1 | 20000 | 7000 | 1.001274 | 0.7909 | 0.03750 | yes | — | 70.8 |
| n20000_r1over40_shipped_std0_7k_s1 | 20000 | 7000 | 1.000066 | 0.8258 | 0.03570 | yes | — | 79.5 |
| n20000_r1over16_shipped_std0_28k_s1 | 20000 | 28000 | 1.000046 | 0.9543 | 0.02657 | yes | — | 302.7 |
| n20000_r1over40_shipped_std0_28k_s1 | 20000 | 28000 | 0.999778 | 0.9567 | 0.02742 | yes | — | 307.3 |
| n400_r1over40_28k_s1 | 400 | 28000 | 0.999535 | 0.9264 | 0.02888 | yes | — | 292.9 |
| C0_28k_s1 | 20000 | 28000 | 0.999110 | 0.9438 | 0.02655 | yes | — | 302.8 |
| n1600_r1over40_fast_7k_s1 | 1600 | 7000 | 1.003590 | 0.7308 | 0.05867 | no | width,balance | 76.4 |
| n6400_r0_fast_7k_s1 | 6400 | 7000 | 1.001684 | 0.6944 | 0.11355 | no | width,balance | 77.9 |
| n20000_r0_shipped_std1_7k_s1 | 20000 | 7000 | 1.001219 | 0.8913 | 0.05008 | no | balance | 81.1 |
| n6400_r1over40_fast_7k_s1 | 6400 | 7000 | 1.000521 | 0.6888 | 0.13411 | no | width,balance | 77.8 |
| n1600_r0_shipped_std1_7k_s1 | 1600 | 7000 | 1.000091 | 0.7332 | 0.05681 | no | width,balance | 76.2 |
| n6400_r0_shipped_std1_7k_s1 | 6400 | 7000 | 0.999611 | 0.8116 | 0.03816 | no | balance | 76.9 |
| n1600_r0_fast_7k_s1 | 1600 | 7000 | 0.999014 | 0.6135 | 0.08115 | no | width,balance | 75.6 |
| n20000_r1over40_shipped_std1_7k_s1 | 20000 | 7000 | 0.997654 | 0.8721 | 0.05203 | no | HQ,balance | 81.7 |
| n1600_r1over40_shipped_std1_7k_s1 | 1600 | 7000 | 0.997558 | 0.9178 | 0.05421 | no | HQ,balance | 76.1 |
| n6400_r1over40_shipped_std1_7k_s1 | 6400 | 7000 | 0.996764 | 0.8933 | 0.04664 | no | HQ,balance | 78.1 |
| n400_r1over40_14k_s1 | 400 | 14000 | 0.994721 | 0.9310 | 0.03576 | no | HQ | 145.0 |
| n400_r0_s1 | 400 | 7000 | 0.993720 | 0.3636 | 0.03222 | no | HQ,width | 75.1 |
| n400_r0p03125_14k_s1 | 400 | 14000 | 0.984037 | 1.0360 | 0.03950 | no | HQ,balance | 144.7 |
| n20000_r1over16_shipped_std0_7k_s1 | 20000 | 7000 | 0.975198 | 1.0686 | 0.04094 | no | HQ,balance | 79.1 |
| n400_r1over32_s1 | 400 | 7000 | 0.971775 | 1.1180 | 0.05011 | no | HQ,balance | 67.2 |
| n1600_r1over16_fast_7k_s1 | 1600 | 7000 | 0.931744 | 1.2864 | 0.05252 | no | HQ,width,balance | 77.8 |
| n6400_r1over16_fast_7k_s1 | 6400 | 7000 | 0.910840 | 1.2874 | 0.17693 | no | HQ,width,balance | 79.2 |
| n20000_r0_fast_7k_s1 | 20000 | 7000 | 0.608169 | 2.0041 | 0.08470 | no | coverage,HQ,width,balance | 82.8 |
| n20000_r1over40_fast_7k_s1 | 20000 | 7000 | 0.577158 | 2.8085 | 0.12431 | no | HQ,width,balance | 74.9 |
| n20000_r1over16_fast_7k_s1 | 20000 | 7000 | 0.510398 | 2.6420 | 0.07845 | no | HQ,width,balance | 82.5 |

Rank: frozen pass first, then HQ, then distance of width from real, then balance. Use the separate columns and quality frontier when a different tradeoff matters.

## Interpretation limits

- N>1,024 uses sampled-row raw VICReg, as shipped; N=400 references use the full table. At fixed r, increasing N also changes initial spacing and absolute sigma.
- Fast optimizer means particle LR multiplier 10 (initial LR 0.06), beta1=0.5. Shipped means multiplier 1 (LR 0.006), beta1=0. No generator/discriminator retuning.
- Standardized reads are used except explicitly named std0 controls and C0. Raw scale and effective-radius drift flags remain in the per-run CSV.
- Longer runs restart from the same seed and scale the delayed cosine schedule to their budget; they do not resume EMA checkpoints. Matched-budget controls are needed before crediting noise for budget gains.
- Purity is HQ-conditioned. With 200k samples and 20k components, each component gets only about ten evaluation draws on average; per-component diagnostics are less precise than at N=400.
- Times include metrics and output artifacts; concurrent execution changes elapsed time per run. These are observed experiment costs, not isolated throughput benchmarks.

- Adding positive noise consumes additional training RNG draws; zero-noise follows the original RNG contract. Fixed seed does not mean identical subsequent random streams.

## Matched noise versus atoms comparisons

Same N, budget, standardization and optimizer. Positive HQ delta favors MoG; negative width-error and KL deltas favor MoG.

| MoG run | Atoms control | Δ HQ/real | Δ absolute width error | Δ KL |
|---|---|---:|---:|---:|
| n400_r1over32_s1 | n400_r0_s1 | -0.021945 | -0.51841 | +0.01789 |
| n1600_r1over16_fast_7k_s1 | n1600_r0_fast_7k_s1 | -0.067270 | -0.10007 | -0.02864 |
| n1600_r1over40_fast_7k_s1 | n1600_r0_fast_7k_s1 | +0.004576 | -0.11730 | -0.02249 |
| n1600_r1over40_shipped_std1_7k_s1 | n1600_r0_shipped_std1_7k_s1 | -0.002533 | -0.18466 | -0.00260 |
| n20000_r1over16_fast_7k_s1 | n20000_r0_fast_7k_s1 | -0.097771 | +0.63791 | -0.00624 |
| n20000_r1over16_shipped_std0_7k_s1 | C0_s1 | -0.026076 | -0.14047 | +0.00344 |
| n20000_r1over40_fast_7k_s1 | n20000_r0_fast_7k_s1 | -0.031011 | +0.80448 | +0.03962 |
| n20000_r1over40_shipped_std0_7k_s1 | C0_s1 | -0.001208 | -0.03493 | -0.00180 |
| n20000_r1over40_shipped_std1_7k_s1 | n20000_r0_shipped_std1_7k_s1 | -0.003565 | +0.01922 | +0.00195 |
| n6400_r1over16_fast_7k_s1 | n6400_r0_fast_7k_s1 | -0.090843 | -0.01824 | +0.06337 |
| n6400_r1over40_fast_7k_s1 | n6400_r0_fast_7k_s1 | -0.001163 | +0.00565 | +0.02056 |
| n6400_r1over40_shipped_std1_7k_s1 | n6400_r0_shipped_std1_7k_s1 | -0.002847 | -0.08167 | +0.00847 |
| n20000_r1over16_shipped_std0_28k_s1 | C0_28k_s1 | +0.000935 | -0.01054 | +0.00001 |
| n20000_r1over40_shipped_std0_28k_s1 | C0_28k_s1 | +0.000667 | -0.01294 | +0.00086 |

## Geometry audit

| Run | sigma | d0 | r_eff | Raw std final/initial | r flag | Scale flag | Purity |
|---|---:|---:|---:|---:|:---:|:---:|---:|
| C0_s1 | 0 | 0.19914 | 0 | 1.093 | False | False | 1.00000 |
| n400_r0_s1 | 0 | 0.52574 | 0 | 3.242 | False | True | 1.00000 |
| n400_r1over32_s1 | 0.0164294 | 0.52574 | 0.034645 | 3.192 | False | True | 1.00000 |
| n400_r0p03125_14k_s1 | 0.0164294 | 0.52574 | 0.042463 | 3.847 | False | True | 1.00000 |
| n400_r1over40_14k_s1 | 0.0131435 | 0.52574 | 0.027332 | 3.853 | False | True | 1.00000 |
| n1600_r0_fast_7k_s1 | 0 | 0.37582 | 0 | 1.367 | False | False | 1.00000 |
| n1600_r0_shipped_std1_7k_s1 | 0 | 0.37582 | 0 | 1.066 | False | False | 1.00000 |
| n1600_r1over16_fast_7k_s1 | 0.0234887 | 0.37582 | 0.27313 | 1.358 | True | False | 1.00000 |
| n1600_r1over40_fast_7k_s1 | 0.0093955 | 0.37582 | 0.088453 | 1.322 | True | False | 1.00000 |
| n1600_r1over40_shipped_std1_7k_s1 | 0.0093955 | 0.37582 | 0.024845 | 1.065 | False | False | 1.00000 |
| n20000_r0_fast_7k_s1 | 0 | 0.19899 | 0 | 2.211 | False | True | 1.00000 |
| n20000_r0_shipped_std1_7k_s1 | 0 | 0.19899 | 0 | 1.095 | False | False | 1.00000 |
| n20000_r1over16_fast_7k_s1 | 0.0124371 | 0.19899 | 0.070824 | 2.298 | False | True | 1.00000 |
| n20000_r1over16_shipped_std0_7k_s1 | 0.012446 | 0.19914 | 0.092903 | 1.096 | False | False | 1.00000 |
| n20000_r1over40_fast_7k_s1 | 0.00497484 | 0.19899 | 0.028794 | 2.223 | False | True | 1.00000 |
| n20000_r1over40_shipped_std0_7k_s1 | 0.00497839 | 0.19914 | 0.037309 | 1.094 | False | False | 1.00000 |
| n20000_r1over40_shipped_std1_7k_s1 | 0.00497484 | 0.19899 | 0.041486 | 1.096 | True | False | 1.00000 |
| n6400_r0_fast_7k_s1 | 0 | 0.26702 | 0 | 1.458 | False | False | 1.00000 |
| n6400_r0_shipped_std1_7k_s1 | 0 | 0.26702 | 0 | 1.087 | False | False | 1.00000 |
| n6400_r1over16_fast_7k_s1 | 0.0166888 | 0.26702 | 0.12493 | 1.486 | True | False | 1.00000 |
| n6400_r1over40_fast_7k_s1 | 0.00667551 | 0.26702 | 0.050418 | 1.469 | True | False | 1.00000 |
| n6400_r1over40_shipped_std1_7k_s1 | 0.00667551 | 0.26702 | 0.030942 | 1.088 | False | False | 1.00000 |
| C0_28k_s1 | 0 | 0.19914 | 0 | 1.112 | False | False | 1.00000 |
| n20000_r1over16_shipped_std0_28k_s1 | 0.012446 | 0.19914 | 0.10326 | 1.131 | True | False | 1.00000 |
| n20000_r1over40_shipped_std0_28k_s1 | 0.00497839 | 0.19914 | 0.040615 | 1.113 | True | False | 1.00000 |
| n400_r1over40_28k_s1 | 0.0131435 | 0.52574 | 0.034587 | 4.106 | False | True | 1.00000 |

## Validation and artifacts

Every new result was checked against its generated config, completion certificate, summary SHA-256, source provenance, frozen criteria, and expected metric-trace count. No training implementation changed. Historical results are reused.

Checked 2310 new trace entries. All 4 longer runs have identical original-log prefixes to their shorter parents before the shorter schedule starts annealing; only budget, output path and (for historical C0) evaluation pass thresholds differ in config. Details: component_scale_prefix_check.json.

- [Design and launch plan](COMPONENT_SCALE_PLAN.md)
- [All per-run metrics](component_scale_results.csv) and [leaderboard](component_scale_leaderboard.csv)
- [Component-count plot](component_scale_count.png) and [quality tradeoffs](component_scale_tradeoffs.png)
- [Longer-budget trajectories](component_scale_training.png)
- Raw outputs and JSONL traces: component_scale/<run>/ and scale_longer/<run>/.
