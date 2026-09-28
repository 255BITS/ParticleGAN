# Frozen-generator critic refit: st5 rotated100 at 14k

Both warm-start and fresh-QR critics switched the covariance-repair directional derivative from adverse to favorable by the first 250-update observation. This supports critic tracking/optimization as a plausible source of the saved adverse signal. The same objective and regularizer can produce favorable local signals on the frozen generator distribution. No generator updates or new frozen-gate evaluations were performed.

Each arm used candidate RpGAN D loss plus KA2 penalty/optimizer, fixed saved D LR, and a frozen saved data-drift controller. G, prior, output noise, latent bandwidth/radii and controller state were frozen. D optimizer history and its EMA/anchor mechanism remained active; the warm arm restored them and the fresh arm initialized them anew. Thus fresh versus warm does not isolate weight initialization alone.

Negative derivatives favor the proposed correction under the local generator loss, with fixed membership/jitter. Intervals below are normal 95% intervals across 128 independent batches, conditional on this checkpoint and fixed geometry calibration. Terminal draws were independent of the monitoring draws.

| Arm/check | D updates | Center derivative | Covariance derivative | log-sigma derivative |
|---|---:|---:|---:|---:|
| warm: initial | 0 | -2.71e-06 [-2.75e-06, -2.67e-06] | 7.07e-07 [6.57e-07, 7.58e-07] | 1.27e-07 [1.17e-07, 1.38e-07] |
| warm: first check | 250 | -1.38e-05 [-1.39e-05, -1.37e-05] | -3.39e-06 [-3.47e-06, -3.3e-06] | 4.69e-08 [3.41e-08, 5.97e-08] |
| warm: independent terminal | 5000 | -3.22e-05 [-3.24e-05, -3.2e-05] | -2.12e-05 [-2.14e-05, -2.1e-05] | -1.8e-07 [-2.13e-07, -1.47e-07] |
| fresh_qr: initial | 0 | 5.59e-05 [5.32e-05, 5.86e-05] | 2.19e-06 [-1.43e-06, 5.81e-06] | -9.72e-09 [-5.85e-07, 5.65e-07] |
| fresh_qr: first check | 250 | -3.76e-05 [-3.79e-05, -3.72e-05] | -5.61e-05 [-5.67e-05, -5.56e-05] | -1.02e-06 [-1.1e-06, -9.35e-07] |
| fresh_qr: independent terminal | 5000 | -3.04e-05 [-3.06e-05, -3.02e-05] | -3.37e-05 [-3.4e-05, -3.35e-05] | -4.88e-07 [-5.26e-07, -4.5e-07] |

## Per-mode covariance direction at the terminal critic

| Arm | Favor correction | Oppose correction | Uncertain |
|---|---:|---:|---:|
| warm | 89 | 6 | 5 |
| fresh_qr | 92 | 6 | 2 |

These per-mode intervals are unadjusted and descriptive. A favorable aggregate score does not establish accurate shape gradients in every mode, convergence, or a passing continuation. HQ membership and jitter are held fixed for derivatives; the local transport field is diagnostic rather than an exact gate derivative.

## Stopping and integrity

{
  "arms": {
    "warm": {
      "updates": 5000,
      "stop_reason": "cap"
    },
    "fresh_qr": {
      "updates": 5000,
      "stop_reason": "cap"
    }
  },
  "source_checkpoint_unchanged": true,
  "frozen_G_prior_sigma_controller_verified": true,
  "elapsed_sec": 286.6028633117676
}

The plateau rule required three consecutive checks where paired changes in held-out D logistic loss and shape derivative both had intervals including zero. Repeated-look intervals are descriptive. Reaching the 5000-update cap does not establish critic convergence, but the early persistent sign reversal answers the narrower question: favorable shape gradients are available within the same critic/regularizer family on this frozen law.

The full read-only artifacts are under
`/ml2/hypergan/lrfree-20260926/reports/st5-14k-critic-refit/`:
`manifest.json`, `observations.jsonl`, `training.jsonl`, independent
`*-terminal.json`, and `*-critic.pt`. The reproduction script is
`/ml2/hypergan/lrfree-20260926/reports/st5-14k-critic-refit.py`.
The original checkpoint SHA-256 and frozen-state integrity checks are
recorded in `manifest.json` and `completion.json`. This summary is adapted
from the harness's `summary.md` to make the artifact paths explicit.
