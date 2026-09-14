# Denoising GAN comparison

Completed 3 / 3 runs.

Sorted by conditional sliced W1 for inspection, not an automatic winner selection. Report coverage, class fidelity, shape, tails, and posterior error together. ± denotes sample standard deviation across seeds.

| Model | D | Latent | Step noise | Updates | T | Classes | n | Joint HQ | Modes | Class acc | Conditional SW1 | Mode TV | Core | Cov min / max | Tail | Posterior SW1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ddgan | ucd | learned | gaussian | 28000 | 4 | 4 | 3 | 0.778 ± 0.016 | 100.000 ± 0.000 | 0.898 ± 0.007 | 0.107 ± 0.014 | 0.123 ± 0.011 | 0.865 ± 0.042 | 7.399 ± 0.473 / 15.554 ± 1.306 | 0.090 ± 0.007 | 0.205 ± 0.003 |

Full configs, paired effects, timing, and missing runs are in `comparison.json`.
