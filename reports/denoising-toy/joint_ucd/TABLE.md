# Denoising GAN comparison

Completed 2 / 2 runs.

Sorted by conditional sliced W1 for inspection, not an automatic winner selection. Report coverage, class fidelity, shape, tails, and posterior error together. ± denotes sample standard deviation across seeds.

| Model | D | Latent | Step noise | Updates | T | Classes | n | Joint HQ | Modes | Class acc | Conditional SW1 | Mode TV | Core | Cov min / max | Tail | Posterior SW1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ddgan | ucd/class | learned | gaussian | 56000 | 4 | 4 | 1 | 0.912 | 100.000 | 0.962 | 0.104 | 0.059 | 0.512 | 2.682 / 6.004 | 0.034 | 0.190 |
| ddgan | ucd/time_class | learned | gaussian | 56000 | 4 | 4 | 1 | 0.919 | 100.000 | 0.963 | 0.119 | 0.062 | 0.523 | 2.613 / 5.518 | 0.032 | 0.190 |

Full configs, paired effects, timing, and missing runs are in `comparison.json`.
