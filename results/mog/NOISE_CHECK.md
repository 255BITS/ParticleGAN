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
