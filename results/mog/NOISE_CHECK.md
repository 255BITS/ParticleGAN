# N=400 MoG: smaller noise and longer training

Three seeds per cell, the Stage 1 selected particle LR multiplier 10 (initial LR 0.06), particle beta1=0.5, standardized reads, fixed sigma calibrated at initialization. The C0-relative acceptance thresholds remain unchanged. All new runs use the existing trainer and grid runner.

## Results

| Steps | Nominal r | Pass | Original rule | Modes | HQ/real | Width/real | KL | r_eff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7000 | 0 | 0/3 | 0/3 | 100.0 | 0.99171 | 0.393 | 0.02754 | 0.0000 |
| 7000 | 0.03125 | 0/3 | 0/3 | 100.0 | 0.96220 | 1.154 | 0.04966 | 0.0395 |
| 7000 | 0.0625 | 0/3 | 0/3 | 100.0 | 0.65407 | 2.113 | 0.08248 | 0.9994 |
| 7000 | 0.125 | 0/3 | 0/3 | 100.0 | 0.38016 | 3.373 | 0.15626 | 4.6297 |
| 14000 | 0.025 | 0/3 | 0/3 | 100.0 | 0.99641 | 0.926 | 0.03588 | 0.0319 |
| 14000 | 0.03125 | 0/3 | 0/3 | 100.0 | 0.97851 | 1.066 | 0.03591 | 0.0651 |

The r=1/8, 7k row reuses the three selected Stage 1 runs. The other rows comprise nine new 7k runs, three selected 14k runs, and three targeted r=1/40 refinement runs at 14k. All final metrics use 200k noisy EMA samples with matched reals; traces use 20k samples every 100 updates. Per-seed results and population standard deviations are saved in the CSV artifacts.

The r=0 control is not C0: it has 400 particles, standardized reads, and the selected particle optimizer. Its comparison to positive-r cells isolates the addition of noise at those settings.

## Budget comparison

The selected positive-noise setting was nominal r=0.03125, selected across all three seeds by pass rate, then HQ, then width closeness to real. The atoms control was excluded from this selection because the requested longer run tests the MoG's continuous neighborhoods.
At 7k: 0/3 pass, HQ/real 0.96220, width/real 1.1535, KL 0.04966. At 14k: 0/3 pass, HQ/real 0.97851, width/real 1.0661, KL 0.03591.
The 14k runs start from the same seeds and initialization. They are fresh training runs, not continuation from EMA checkpoints. The delayed cosine schedule scales with total budget (annealing starts at 8,400 instead of 4,200 updates), so this tests the longer-budget recipe, not extra updates at an unchanged LR schedule.

## Targeted refinement

After inspecting the r=1/32 longer-budget result, we tested r=1/40 at 14k with the same optimizer and seeds. This is an exploratory follow-up chosen to reduce remaining spread; it was not part of the original grid. The fixed acceptance thresholds were not changed.

## Geometry and component diagnostics

- r=0, 7000 steps: component-center HQ 0.9808, noisy HQ 0.9807, HQ-conditioned purity 1.0000, empty majority allocations 0.00; r_eff 0.0000, r_eff flags 0/3, raw-scale flags 3/3.
- r=0.03125, 7000 steps: component-center HQ 0.9967, noisy HQ 0.9515, HQ-conditioned purity 1.0000, empty majority allocations 0.00; r_eff 0.0395, r_eff flags 0/3, raw-scale flags 3/3.
- r=0.0625, 7000 steps: component-center HQ 0.9967, noisy HQ 0.6468, HQ-conditioned purity 1.0000, empty majority allocations 0.00; r_eff 0.9994, r_eff flags 3/3, raw-scale flags 3/3.
- r=0.125, 7000 steps: component-center HQ 0.9942, noisy HQ 0.3759, HQ-conditioned purity 1.0000, empty majority allocations 0.00; r_eff 4.6297, r_eff flags 3/3, raw-scale flags 3/3.
- r=0.025, 14000 steps: component-center HQ 0.9975, noisy HQ 0.9854, HQ-conditioned purity 1.0000, empty majority allocations 0.00; r_eff 0.0319, r_eff flags 0/3, raw-scale flags 3/3.
- r=0.03125, 14000 steps: component-center HQ 0.9992, noisy HQ 0.9677, HQ-conditioned purity 1.0000, empty majority allocations 0.00; r_eff 0.0651, r_eff flags 2/3, raw-scale flags 3/3.

The center-only diagnostic is explicit and supplementary; it never replaces noise-on evaluation. Large median-NN ratios can reflect neighboring components that map to the same output mode. Component-center CSV records that diagnostic separately. The r=0 deterministic all-table audit has only 400 points: its >=10 coverage and >=50 width thresholds are not suitable for judging that control; use its primary 200k-sample metrics.

## Changes and validation

- r=1/32 was the explicitly proposed addition to the original noise grid. r=0 and r=1/16 were already planned values. Only N=400 is tested here.
- The owner authorized further experiments and a longer-budget experiment. Budget doubled to 14k for the most promising positive-noise cell. A subsequent r=1/40 refinement changed only the fixed noise radius; model, optimizer, regularizer and evaluation settings remain fixed.
- Pass thresholds are frozen from Stage 0, with the old design criterion retained in passed_strict. Stage 0 and Stage 1 reports are preserved as historical results.
- Stage 1 references are reused rather than retrained. No seed-only search or best-seed selection.
- For r=0, sigma consumes no noise RNG. Thus r=0 and r>0 training streams differ after sampling, as required by the zero-noise regression contract.

## Artifacts

- noise_check_results.csv, noise_check_leaderboard.csv, noise_check_winner.json.
- noise_check_metrics.png, noise_check_width_hq.png, noise_check_component_centers.csv.
- noise_check/<run>/ and longer/<run>/: complete run configs, source archives, final checkpoints/samples, component diagnostics, JSONL traces, and logs.

## Completed outcome

The r=1/40 refinement passes 0/3: HQ/real 0.996408, width/real 0.9263, KL 0.03588. All three runs meet coverage, width and balance; only HQ fails the frozen envelope. It uses 50 times fewer components and twice the training steps of C0. Width is closer to real, but this is not an equal-budget match or a full pass.

The saved longer_prefix_check.json verifies that the first 42 original log entries (through step 4100) agree exactly between the 7k and 14k r=1/32 runs for all three seeds.

## Overlap audit

Independent post-training CPU diagnostic: 20k latent samples per run. Compute the exact equal-weight shared-Gaussian posterior, then estimate E[1-max posterior] for component identity and for components grouped by their observed HQ-majority destination. These learned labels are not external ground truth.

| r | Steps | Component ambiguity | Destination ambiguity | Generated non-HQ |
|---:|---:|---:|---:|---:|
| 0.03125 | 7000 | 0.129824 | 7.57542e-18 | 0.048465 |
| 0.0625 | 7000 | 0.2738 | 2.25567e-11 | 0.353177 |
| 0.125 | 7000 | 0.522684 | 0.000198769 | 0.624052 |
| 0.025 | 14000 | 0.13036 | 7.61428e-18 | 0.0146367 |
| 0.03125 | 14000 | 0.199106 | 1.01715e-17 | 0.0323367 |

Different-destination ambiguity is much smaller than generated non-HQ mass. Overlap between components serving the same mode can be harmless. The results point toward difficulty shaping the noisy neighborhoods, rather than destination ambiguity explaining most failures. This is a diagnostic interpretation, not a causal isolation; near-zero Monte Carlo estimates do not prove zero overlap. The bridge metric includes over-wide within-mode tails, not only samples in inter-mode walls.

Recommendation at this checkpoint: test a slightly smaller radius, more training, or more components with matched atoms controls. Keep the acceptance thresholds fixed and distinguish small-table parameter savings from training cost.
