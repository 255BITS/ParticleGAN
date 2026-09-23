# Actual 100-Gaussian comparison — public training API

Runs with `--training-api` at implementation commit `fdc6e00`.
[Parity evidence](parity.json) compares all 28 live/EMA checkpoints per arm
with the [original loop](../grid/README.md): coverage and HQ match exactly;
distribution diagnostics differ by at most 1.24×10⁻¹³.

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
| `stock` / live | 6000 | 6000 | 7000 | 54.9 / 53.9 | 63.0 | 111.2 |
| `stock` / ema | 5500 | 5500 | 6500 | 50.0 / 49.0 | 63.0 | 111.2 |
| `toy_transfer` / live | 6000 | 6000 | 7000 | 56.4 / 56.2 | 65.3 | 107.3 |
| `toy_transfer` / ema | 5250 | 5250 | 6250 | 49.1 / 49.0 | 65.3 | 107.3 |
| `penalty_only` / live | 6000 | 6000 | 7000 | 55.6 / 55.4 | 64.7 | 108.1 |
| `penalty_only` / ema | 5500 | 5500 | 6500 | 50.8 / 50.6 | 64.7 | 108.1 |

Final distribution diagnostics use separate fixed-seed 20,000-sample fake and real draws, isolated from training. Width and core ratios near 1 indicate matching scale; covariance ratios expose collapsed axes. TV and SW1 are lower-is-better. These measurements have no tuned PASS threshold. Width/core summaries omit modes with fewer than 20 samples; audited counts are shown.

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
