# Denoising GAN comparison

Completed 9 / 9 runs.

Sorted by conditional sliced W1 for inspection, not an automatic winner selection. Report coverage, class fidelity, shape, tails, and posterior error together. ± denotes sample standard deviation across seeds.

| Model | D | Latent | Step noise | Updates | T | Classes | n | Joint HQ | Modes | Class acc | Conditional SW1 | Mode TV | Core | Cov min / max | Tail | Posterior SW1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ddgan | concat | gaussian | gaussian | 28000 | 4 | 4 | 1 | 0.721 | 100.000 | 0.877 | 0.088 | 0.135 | 1.013 | 9.192 / 19.195 | 0.114 | 0.210 |
| ddgan | concat | learned | gaussian | 28000 | 4 | 4 | 1 | 0.712 | 100.000 | 0.868 | 0.108 | 0.170 | 1.097 | 9.491 / 20.646 | 0.118 | 0.216 |
| ddgan | concat | learned | gaussian | 28000 | 4 | 1 | 1 | 0.105 | 51.000 | 1.000 | 0.114 | 0.141 | 9.530 | 57.093 / 91.694 | 0.566 | 0.188 |
| ddgan | ucd | learned | gaussian | 28000 | 4 | 4 | 1 | 0.781 | 100.000 | 0.898 | 0.122 | 0.134 | 0.838 | 7.536 / 15.907 | 0.090 | 0.207 |
| gan | concat | learned | N/A | 7000 | 0 | 1 | 1 | 0.992 | 100.000 | 1.000 | 0.141 | 0.112 | 0.873 | 2.665 / 21.281 | 0.004 | — |
| ddgan | concat | learned | gaussian | 14000 | 2 | 4 | 1 | 0.430 | 77.000 | 0.744 | 0.171 | 0.316 | 4.658 | 19.493 / 54.287 | 0.246 | 0.362 |
| gan | concat | learned | N/A | 28000 | 0 | 4 | 1 | 0.995 | 100.000 | 1.000 | 0.229 | 0.104 | 0.943 | 0.662 / 1.562 | 0.000 | — |
| ddgan | concat | learned | gaussian | 7000 | 1 | 4 | 1 | 0.992 | 100.000 | 0.998 | 0.234 | 0.116 | 0.755 | 1.499 / 7.964 | 0.003 | 0.332 |
| gan | concat | fixed | N/A | 7000 | 0 | 4 | 1 | 0.193 | 46.000 | 0.599 | 0.269 | 0.506 | 7.544 | 33.010 / 93.663 | 0.400 | — |

Full configs, paired effects, timing, and missing runs are in `comparison.json`.
