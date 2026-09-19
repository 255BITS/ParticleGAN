# Learning-rate forks from the 80k residual CNN

| Arm | FID85k | FID90k | FID95k | FID100k | Coverage100k |
|---|---:|---:|---:|---:|---:|
| unchanged (existing) | — | 15.7901 | — | 16.4609 | 63.36% |
| half_g | 15.9338 | 16.3924 | 16.3796 | 16.1118 | 64.19% |
| half_all | 17.2142 | 17.5723 | 18.6474 | 19.2482 | 61.01% |

Starting parent FID15.7527. Full-state resume, original sigma and moving16k prior. All scores use FID50k.
Fixed-input drift uses256 latent draws from the parent EMA prior; it measures G movement independently of center movement. Pixel drift is not a quality metric.
Per-tensor update/weight RMS is one actual Adam step at each log, not a cumulative update. Zero/small weights can make ratios large; inspect absolute RMS too.
Control checkpoints have snapshot diagnostics only, not historical per-step updates. Previous-probe drift spans10k for controls and5k for interventions; compare parent drift at matching90k/100k.
This tests late learning rates, not whether residual connections themselves cause the plateau. Older1k lower-rate trials failed; no seed-only reruns or automatic extension.
