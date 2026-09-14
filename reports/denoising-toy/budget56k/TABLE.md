# Denoising GAN comparison

Completed 6 / 6 runs.

Sorted by conditional sliced W1 for inspection, not an automatic winner selection. Report coverage, class fidelity, shape, tails, and posterior error together. ± denotes sample standard deviation across seeds.

| Model | D | Latent | Step noise | Updates | T | Classes | n | Joint HQ | Modes | Class acc | Conditional SW1 | Mode TV | Core | Cov min / max | Tail | Posterior SW1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ddgan | concat | learned | gaussian | 56000 | 4 | 4 | 1 | 0.889 | 100.000 | 0.952 | 0.082 | 0.072 | 0.693 | 3.803 / 7.546 | 0.043 | 0.198 |
| ddgan | ucd | learned | gaussian | 56000 | 4 | 4 | 1 | 0.912 | 100.000 | 0.962 | 0.104 | 0.059 | 0.512 | 2.682 / 6.004 | 0.034 | 0.190 |
| ddgan | concat | gaussian | gaussian | 56000 | 4 | 4 | 1 | 0.891 | 100.000 | 0.956 | 0.114 | 0.065 | 0.641 | 3.644 / 7.599 | 0.044 | 0.187 |
| ddgan | ucd | learned | fixed | 56000 | 4 | 4 | 1 | 0.909 | 100.000 | 0.961 | 0.136 | 0.065 | 0.507 | 3.008 / 6.014 | 0.035 | 0.198 |
| ddgan | ucd | learned | learned | 56000 | 4 | 4 | 1 | 0.934 | 100.000 | 0.968 | 0.140 | 0.070 | 0.439 | 2.160 / 4.786 | 0.027 | 0.199 |
| gan | concat | learned | N/A | 56000 | 0 | 4 | 1 | 0.998 | 100.000 | 1.000 | 0.220 | 0.103 | 0.640 | 0.284 / 0.663 | 0.000 | — |

Full configs, paired effects, timing, and missing runs are in `comparison.json`.
