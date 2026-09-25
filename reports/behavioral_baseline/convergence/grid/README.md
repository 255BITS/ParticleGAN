# Actual 100-Gaussian comparison

Runs the existing `examples/100gaussians.py` trainer: seed 0, 7,000 updates, 20,000 particles, batch 256, Fourier 2. Arms run sequentially on the same device. Each checkpoint evaluates 20,000 fresh fixed-seed draws separately for live and EMA weights.

**100-Gaussian coverage/HQ PASS means all 100 modes have at least 10 HQ samples and HQ ≥90% at the final step. It does not certify the nine-toy suite or distributional calibration.** A stable suffix requires at least five consecutive passing observations through step 7,000, with the entire 250-step observation schedule present.

| Arm | Live modes | Live HQ | Last-five worst HQ | Live coverage/HQ | EMA modes | EMA HQ |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `stock` | 100/100 | 98.21% | 94.15% | PASS | 100/100 | 98.73% |
| `toy_transfer` | 100/100 | 98.45% | 97.37% | PASS | 100/100 | 98.74% |
| `penalty_only` | 100/100 | 98.25% | 92.27% | PASS | 100/100 | 98.79% |

First-pass, stable-start and confirmation are observed checkpoint steps, not interpolated convergence times. Stable-start is retrospective: every later observation must pass. Wall time includes setup and measurement; training time excludes callbacks and maintenance. Throughput uses training time.

| Arm / weights | First pass | Stable from | Confirmed | Stable wall / train sec | Final train sec | Updates/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock` / live | 6000 | 6000 | 7000 | 57.0 / 56.0 | 65.4 | 107.1 |
| `stock` / ema | 5500 | 5500 | 6500 | 52.3 / 51.2 | 65.4 | 107.1 |
| `toy_transfer` / live | 6000 | 6000 | 7000 | 59.9 / 59.7 | 69.5 | 100.8 |
| `toy_transfer` / ema | 5250 | 5250 | 6250 | 52.6 / 52.5 | 69.5 | 100.8 |
| `penalty_only` / live | 6000 | 6000 | 7000 | 57.3 / 57.1 | 66.7 | 104.9 |
| `penalty_only` / ema | 5500 | 5500 | 6500 | 52.6 / 52.4 | 66.7 | 104.9 |

Final distribution diagnostics use independent fixed-seed 20,000-sample fake and real draws. Width and core ratios near 1 indicate matching scale; covariance ratios expose collapsed axes. TV and SW1 are lower-is-better. These measurements have no tuned PASS threshold. Width/core summaries omit modes with fewer than 20 samples; audited counts are shown.

| Arm / weights | Width / core ratio | Covariance min / max | Audited modes | Mode TV | SW1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stock` / live | 2.696 / 0.934 | 1.913 / 41.655 | 100 | 0.111 | 0.144 |
| `stock` / ema | 2.647 / 0.868 | 1.839 / 41.294 | 100 | 0.111 | 0.147 |
| `toy_transfer` / live | 2.241 / 0.844 | 1.772 / 24.261 | 100 | 0.106 | 0.147 |
| `toy_transfer` / ema | 2.188 / 0.773 | 1.738 / 24.288 | 100 | 0.106 | 0.151 |
| `penalty_only` / live | 2.400 / 1.009 | 0.838 / 38.378 | 100 | 0.117 | 0.132 |
| `penalty_only` / ema | 2.365 / 0.928 | 0.720 / 38.765 | 100 | 0.117 | 0.133 |

`stock` keeps the trainer's recipe. `toy_transfer` uses cap κ=1.25, coefficient=3, LR=0.00051 and prior regularization=0.05. `penalty_only` changes only cap κ/coefficient. The existing 60%-delay cosine schedule and 5% floor remain shared.

Exact resolved arguments, source hashes, curves, convergence times and raw diagnostics are in [results.json](results.json). Nonfinite values become null and cannot pass. Missing observations cannot certify stability.
