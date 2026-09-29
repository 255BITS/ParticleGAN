# 100 Gaussians: PacGAN-8 without regularization

This experiment uses 8-point packing on the existing 10×10 Gaussian grid
(target σ=0.03). It keeps the learnable 20,000-particle prior, 2D latent space,
MLP generator, Fourier critic, deterministic initialization, RpGAN logistic
loss, learning rates/schedules, and EMA evaluation from the existing example.
It runs 7,000 updates at seed 1234; there is no seed sweep.

[PacGAN packing](https://arxiv.org/abs/1712.04086) concatenates independent
samples from the same distribution before the discriminator. Here 2,048
individual points become 256 packs of 8×2 coordinates. Real and fake packs
are formed separately; G still emits individual 2D points. The critic's input
layer grows; hidden widths stay at 128. Gradients flow through all eight points.

`no_regularizer = true` uses plain Adam for G, D, and the prior and bypasses:

- Critic gradient/anchor penalties (`reg_coeff = 0`).
- Particle spread penalty (`lambda_ep = 0`).
- K3P critic step guarding and latent-row damping.
- Critic input noise and generator output noise.

There is no weight decay, gradient clipping, or spectral normalization. EMA is
retained for evaluation. Learning-rate scheduling is retained, including the
recipe's 1,600-update network horizon cap and 0.01 network floor; the prior
anneals over all 7,000 updates to its 0.05 floor. This is a packing ablation of
the repository's RpGAN objective. The regularized defaults remain available
with `pack_size=1` and `no_regularizer=false`.

## Run

From the repository root, with the CUDA training environment installed
(`.venv/bin/python -m pip install -e '.[experiments]'` if needed):

```bash
.venv/bin/python experiments/run_grid.py \
  --configs configs/100gaussians/pacgan8_no_reg.toml \
  --trainer experiments/train_100gaussians.py \
  --python .venv/bin/python --gpus 0 --workers_per_gpu 1
```

Change `--gpus 0` to select another GPU. The runner writes unbuffered output;
from a second terminal:

```bash
tail -F results/100gaussians/pacgan8_no_reg/log.txt
# Machine-readable progress, at the first update, every ~100 updates and the end:
tail -F results/100gaussians/pacgan8_no_reg/metrics.jsonl
```

Completed runs with matching source/config fingerprints are reused. Add
`--force` to archive an earlier attempt and rerun. The config disables plots.

## Read results

```bash
.venv/bin/python experiments/analyze_100gaussians.py
```

This prints and writes `results/100gaussians/LEADERBOARD.md`. Rank by final
coverage, then HQ, then sliced W1; inspect width and balance alongside the rank.

| Metric | Interpretation |
|---|---|
| `modes` | Higher is better, maximum 100; at 20,000 evaluation points each mode needs ≥10 HQ samples. |
| `hq` | Higher is better; fraction within 3σ=0.09 of a target center. |
| `sw1` | Lower is better; sliced Wasserstein distance to fresh real samples. |
| `mode_tv` | Lower is better; deviation of nearest-mode mass from uniform. |
| `per_mode_core_ratio` | Near 1 is best; below 0.8 flags overly narrow modes. |
| `train_seconds` | Training time excluding periodic evaluation and plotting. |

`summary.json` contains the final metrics, a real-vs-real reference floor,
configuration, source provenance, and timing. `final_samples.npz` contains
20,000 clean EMA samples; `final.pt` contains EMA generator and prior weights
(evaluation checkpoint, not a resumable training state). `metrics.jsonl` tracks
coverage, HQ, losses and training time during the run.

A useful initial bar is 100/100 modes and HQ≥0.90; inspect core width and mode
balance before interpreting good coverage as distribution matching. A one-run
leaderboard does not establish improvement. The next recommended experiment
is the same config/seed with `pack_size=1` and a new `out_dir`, to isolate the
effect of packing. A subsequent comparison with the regularized default also
changes noise and optimizer stabilization. Do not vary seeds as experiments.
Pass the completed summaries explicitly to compare:

```bash
.venv/bin/python experiments/analyze_100gaussians.py \
  results/100gaussians/pacgan8_no_reg/summary.json \
  results/100gaussians/pacgan1_no_reg/summary.json
```
