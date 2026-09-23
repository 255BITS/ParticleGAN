# README 100-Gaussian visualization

The root `100gaussians.gif` shows one run of the current `get_recipe("gan")`
defaults through the explicitly imported `GANTrainer`, using the existing
`examples/100gaussians.py` generator and Fourier-feature discriminator. There
are no hyperparameter overrides and no seed comparison. Seed: 1234; target:
100 equally weighted Gaussian modes on a 10×10 grid, each with σ = 0.03.

The animation displays 4,096 fixed target samples and outputs from the same
4,096 particle indices at each snapshot. Numeric metrics use 20,000 random
prior draws from the EMA generator and EMA prior, resetting a separate
evaluation RNG at each checkpoint for a consistent comparison. A covered mode has at least
10 generated samples within Euclidean distance 3σ of its center.

## Single-run leaderboard

| Configuration | Updates | HQ modes | Within 3σ | Mode TV ↓ | Sliced W1 ↓ | Beyond 10σ ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current GAN defaults + example Fourier MLP, EMA | 7,000 | 99/100 | 75.08% | 0.1692 | 0.1301 | 4.385% |

Training took 59.9 seconds on an NVIDIA RTX A6000. The delayed learning-rate
anneal materially changed the outcome: HQ rose from 2.69% at step 5,000 to
56.01% at 6,500 and 75.08% at 7,000. The animation includes each 500-update
snapshot; it is not a selection of favorable checkpoints.

This is an illustration and a quality-control observation, not a multi-run
estimate or evidence of complete Gaussian calibration. It does **not** reach
100/100 modes or 90% HQ. The learned prior has 20,000 zero-noise particles,
so generated support is finite. The 20,000 evaluation draws contain 12,662
distinct output points. The per-mode core width ratio is 1.116, while tail
mass and full covariance distortion remain material. These EMA results do
not establish live-model performance or contradict the transfer suite's
separate metrics and discriminator choices.

Recommendation: assess the benchmark's discriminator and training budget
against the intended shared defaults before treating this example as a
convergence demonstration. Keep the public defaults unchanged until a
controlled configuration comparison justifies a change; no seed sweep was
performed for this image.

## Reproduce

Install the repository's experiment dependencies, then run from its root:

```bash
python -u reports/readme-100gaussians/generate.py > /tmp/readme-image.log 2>&1
tail -f /tmp/readme-image.log
```

The script replaces the root GIF and this directory's `summary.json`. Frames
and evaluation samples are saved under `/tmp/particlegan-develop-qc/readme-image/`.
The recorded run log is `/tmp/particlegan-develop-qc/readme-image.log`.
[summary.json](summary.json) records all resolved recipe fields, checkpoint
metrics, source hashes, checkout commit, and environment. Source hashes cover
the working tree used for this run, including the uncommitted API cleanup;
the checkout commit alone does not describe those changes.

The previous GIF was generated in [commit 8664db9](https://github.com/255BITS/ParticleGAN/commit/8664db993f718e7ba290becc352224a46d619313)
using the older 0.1.2 recipe: 7,000 updates, seed 1, cap coefficient 1.0,
and generator learning rate 0.0006. Its final frame explicitly displays
`step 7000/7000`, `modes 100/100`, and `hq 0.992`; the commit also records
the same budget and EMA readout. It did not use a longer training budget.
Different hyperparameters and seeds make this an uncontrolled comparison,
and the old GIF has no equivalently audited shape-metric protocol here.
