# Small-generator MoG/DDGAN screen

Four certified runs, one shared training seed; no seed sweep. G width 32, D width 128, depth 3, 400 learned components, 14k updates, constant LR. Atoms and MoG share standardization and optimizer settings; sigma_rel is 0 versus .025.

Sorted by final conditional SW1 (lower is better), evaluated on 20,000 draws. HQ, coverage, balance, width and tails must also be considered. Width ratio should approach 1.

| Rank | Run | Conditional SW1 ↓ | HQ % ↑ | Modes ↑ | Conditional TV ↓ | Core width | Tail % ↓ | Train seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ddgan_mog | 0.1817 | 5.24 | 24 | 0.5483 | 9.456 | 57.49 | 127.2 |
| 2 | ddgan_atoms | 0.1884 | 5.17 | 28 | 0.5555 | 9.317 | 56.87 | 124.9 |
| 3 | gan_atoms | 0.5784 | 96.40 | 96 | 0.2885 | 0.559 | 1.78 | 110.6 |
| 4 | gan_mog | 0.5984 | 94.21 | 99 | 0.2809 | 1.120 | 1.10 | 113.0 |

Equal-time comparison at 110.6 training seconds. These are interpolated 8,192-draw checkpoint metrics, not the 20k-draw final scores.

| Run | Equal-time conditional SW1 ↓ | Last 4k SW1 change ↓ | G parameters | Prior parameters | Sampling ms / 1k |
|---|---:|---:|---:|---:|---:|
| ddgan_atoms | 0.1909 | +0.0007 | 2658 | 1600 | 0.08 |
| ddgan_mog | 0.2085 | -0.0098 | 2658 | 1600 | 0.09 |
| gan_atoms | 0.5778 | -0.0582 | 2466 | 1600 | 0.02 |
| gan_mog | 0.6056 | -0.0030 | 2466 | 1600 | 0.03 |

The GAN/DDGAN comparison changes the training objective, conditioning inputs, and sampling computation. It does not isolate representational capacity. A small-G screen also needs a matched wider-G comparison before claiming a capacity interaction. A 14k loss does not rule out a later crossover.

Artifacts: `leaderboard.csv`, `comparison.json`, `curves.png`, `samples.png`. Raw logs/checkpoints: `results/denoising/mog_capacity/<run>/`.
