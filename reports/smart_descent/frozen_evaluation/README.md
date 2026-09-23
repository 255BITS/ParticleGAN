# Smart descent v2 — frozen comparison

The nine familiar toys are development data. Three transfer task/architecture combinations were declared before search and first evaluated after the selected policy was frozen. All training uses seed 0. Live weights determine success; EMA is separate.

| Controller | Live bounds | Sustained toys | Mean confirmation / budget | Ring modes / HQ | Ring confirmation | Sum confirmation seconds | Full suite seconds | Controller seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cosine | 29/29 | 9/9 | 0.585 | 8/8 / 100.00% | 1050 | 13.95 | 21.24 | 0.27 |
| feedback | 29/29 | 9/9 | 0.571 | 8/8 / 100.00% | 1150 | 13.73 | 21.18 | 0.32 |
| bias_only | 29/29 | 8/9 | 0.710 | 7/8 / 100.00% | — | — | 20.58 | 0.31 |
| lr_only | 29/29 | 9/9 | 0.571 | 8/8 / 100.00% | 1150 | 13.33 | 20.63 | 0.31 |
| reg_only | 29/29 | 9/9 | 0.585 | 8/8 / 100.00% | 1050 | 13.48 | 21.29 | 0.32 |
| constant_feedback | 27/29 | 8/9 | 0.687 | 5/8 / 66.77% | — | — | 20.56 | 0.29 |

Confirmation needs five consecutive final passing observations in the complete 24-point curve. The ring requires 8/8 modes and HQ≥90%. Mean confirmation fractions use 2 for a non-converged toy; they are meaningful for comparing speed after checking every toy passed.

| Fresh transfer task | Controller | Live modes / total | Live HQ | Confirmed step | EMA modes / HQ | Seconds |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| ring6_width64 | cosine | 5/6 | 74.12% | — | 5 / 90.99% | 6.43 |
| ring6_width64 | feedback | 6/6 | 90.99% | 1400 | 6 / 100.00% | 6.43 |
| ring6_width64 | bias_only | 6/6 | 92.16% | — | 5 / 83.74% | 7.11 |
| ring6_width64 | lr_only | 6/6 | 90.99% | 1400 | 6 / 100.00% | 6.40 |
| ring6_width64 | reg_only | 5/6 | 74.12% | — | 5 / 90.99% | 6.75 |
| ring6_width64 | constant_feedback | 1/6 | 8.42% | — | 6 / 100.00% | 6.34 |
| grid16_width128 | cosine | 12/16 | 77.56% | — | 12 / 77.61% | 11.09 |
| grid16_width128 | feedback | 13/16 | 76.76% | — | 13 / 74.78% | 11.16 |
| grid16_width128 | bias_only | 16/16 | 90.06% | 1467 | 16 / 96.51% | 11.25 |
| grid16_width128 | lr_only | 13/16 | 76.76% | — | 13 / 74.78% | 11.47 |
| grid16_width128 | reg_only | 12/16 | 77.56% | — | 12 / 77.61% | 11.20 |
| grid16_width128 | constant_feedback | 14/16 | 93.99% | — | 15 / 93.26% | 10.82 |
| ellipse8_r1r2 | cosine | 7/8 | 100.00% | — | 7 / 93.92% | 7.59 |
| ellipse8_r1r2 | feedback | 8/8 | 93.33% | — | 8 / 79.88% | 7.53 |
| ellipse8_r1r2 | bias_only | 6/8 | 81.88% | — | 5 / 63.11% | 7.52 |
| ellipse8_r1r2 | lr_only | 8/8 | 93.33% | — | 8 / 79.88% | 7.82 |
| ellipse8_r1r2 | reg_only | 7/8 | 100.00% | — | 7 / 93.92% | 7.59 |
| ellipse8_r1r2 | constant_feedback | 8/8 | 87.04% | — | 8 / 87.70% | 7.24 |

The cosine arm performs no gradient feature extraction. All other arms retain observation overhead, including ablations. Times are single CPU observations with setup and metric evaluation; concurrency and system load can affect them. Controller seconds count feature/action work, except the fixed arm records the full lightweight host callback. Sum confirmation seconds adds each host's measured time to its fifth final passing observation, including setup/measurement; later observations were still run to check stability. No early-stop rule is implemented. No wall-time speedup is established by update counts alone.

`bias_only` zeroes four feedback inputs, preserving learned constant offsets and cosine. `lr_only` suppresses regularization actions; `reg_only` suppresses LR actions. `constant_feedback` keeps the frozen coefficients but removes cosine; it is not a refit for that setting.

Full curves, actions, errors, exact configs and source hashes: [evaluation.json.gz](evaluation.json.gz).

All ten independent shared checks pass. The selected policy has zero regularization coefficients: `lr_only` is numerically identical to feedback, and `reg_only` reproduces cosine. Their timing spread exceeds the small observed timing difference between feedback and cosine.

[Runnable policy](policy.json) · [Frozen selection](frozen.json.gz) · [Exact source](source.tar.gz).
