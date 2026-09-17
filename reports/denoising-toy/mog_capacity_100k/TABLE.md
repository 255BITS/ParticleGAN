# Small-generator MoG/DDGAN comparison

Four certified runs, one shared training seed; no seed sweep. G width 32, D width 128, depth 3, 400 learned components, 100,000 updates, constant LR. Atoms and MoG share standardization and optimizer settings; sigma_rel is 0 versus .025.

Sorted by final conditional SW1 (lower is better), evaluated on 20,000 draws. HQ, coverage, balance, width and tails must also be considered. Width ratio should approach 1.

| Rank | Run | Conditional SW1 ↓ | HQ % ↑ | Modes ↑ | Conditional TV ↓ | Core width | Tail % ↓ | Train seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ddgan_atoms | 0.1172 | 77.86 | 100 | 0.1016 | 0.989 | 8.25 | 905.4 |
| 2 | ddgan_mog | 0.1190 | 79.57 | 100 | 0.1072 | 0.889 | 7.95 | 893.9 |
| 3 | gan_atoms | 0.7595 | 93.46 | 77 | 0.4212 | 0.854 | 2.99 | 782.5 |
| 4 | gan_mog | 0.9025 | 95.19 | 77 | 0.4969 | 0.701 | 1.70 | 811.8 |

Equal-time comparison at 782.5 training seconds. These are interpolated 8,192-draw checkpoint metrics, not the 20k-draw final scores.

| Run | Equal-time conditional SW1 ↓ | Last 4k SW1 change ↓ | G parameters | Prior parameters | Sampling ms / 1k |
|---|---:|---:|---:|---:|---:|
| ddgan_mog | 0.1168 | -0.0041 | 2658 | 1600 | 0.09 |
| ddgan_atoms | 0.1235 | -0.0168 | 2658 | 1600 | 0.08 |
| gan_atoms | 0.7583 | -0.0004 | 2466 | 1600 | 0.02 |
| gan_mog | 0.8635 | +0.0557 | 2466 | 1600 | 0.02 |

Mean checkpoint metrics over the final 20% of updates (8,192 draws per checkpoint). These summarize the training trajectory, not independent trials or seed uncertainty.

| Run | Mean HQ % | Mean SW1 ↓ | Mean conditional TV ↓ | Mean width |
|---|---:|---:|---:|---:|
| gan_atoms | 95.24 | 0.7678 | 0.4255 | 0.495 |
| gan_mog | 94.84 | 0.7493 | 0.4362 | 0.670 |
| ddgan_mog | 76.80 | 0.1200 | 0.1121 | 1.014 |
| ddgan_atoms | 74.03 | 0.1363 | 0.1199 | 1.203 |

The GAN/DDGAN comparison changes the training objective, conditioning inputs, and sampling computation. It does not isolate representational capacity. A small-G screen also needs a matched wider-G comparison before claiming a capacity interaction. A fixed-budget loss does not rule out a later crossover.

Artifacts: `leaderboard.csv`, `comparison.json`, `curves.png`, `samples.png`. Raw logs/checkpoints: `results/denoising/mog_capacity_100k/<run>/`.

All non-timing checkpoint metrics exactly reproduce the shorter runs through 14,000 updates. See `prefix_audit.json`.

| Run | HQ before → after | Modes before → after | Width before → after | SW1 before → after |
|---|---:|---:|---:|---:|
| gan_atoms | 96.40% → 93.46% | 96 → 77 | 0.559 → 0.854 | 0.5784 → 0.7595 |
| gan_mog | 94.21% → 95.19% | 99 → 77 | 1.120 → 0.701 | 0.5984 → 0.9025 |
| ddgan_atoms | 5.17% → 77.86% | 28 → 100 | 9.317 → 0.989 | 0.1884 → 0.1172 |
| ddgan_mog | 5.24% → 79.57% | 24 → 100 | 9.456 → 0.889 | 0.1817 → 0.1190 |