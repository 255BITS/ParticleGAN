# README 100-Gaussian visualization

The root `100gaussians.gif` shows one run of
`get_recipe("gan", total_steps=20000)` through the explicitly imported
`GANTrainer`, using the existing `examples/100gaussians.py` generator and
Fourier-feature discriminator. Only the training budget changes; all other
current GAN defaults and seed 1234 are retained. There is no seed comparison. Target:
100 equally weighted Gaussian modes on a 10×10 grid, each with σ = 0.03.

This run starts from the same initialization as the prior 7,000-update run;
it is not a continuation of that run's final checkpoint. Increasing the
budget also moves the delayed cosine anneal from step 4,200 to step 12,000.
Both schedules hold the initial learning rates for 60% of their budgets,
then decay to a 5% floor. The comparison therefore changes both duration
and absolute anneal timing; the library's default budget remains 7,000.

The animation displays 4,096 fixed target samples and outputs from the same
4,096 particle indices at each snapshot. Numeric metrics use 20,000 random
prior draws from the EMA generator and EMA prior, resetting a separate
evaluation RNG at each checkpoint for a consistent comparison. A covered mode has at least
10 generated samples within Euclidean distance 3σ of its center.

There are 200 frames, captured every 100 updates from step 100 through
20,000. Playback is 25 frames per second with a one-second final hold.
Metrics are evaluated at step 100 and every 500 updates; intermediate
frames explicitly label the last metric evaluation step. A shared palette
keeps colors consistent across the animation.

## Budget comparison

| Configuration | Updates | HQ modes | Within 3σ | Mode TV ↓ | Sliced W1 ↓ | Beyond 10σ ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k budget, current GIF, EMA | 20,000 | 100/100 | 93.27% | 0.1644 | 0.1457 | 2.010% |
| 7k default budget, previous GIF, EMA | 7,000 | 99/100 | 75.08% | 0.1692 | 0.1301 | 4.385% |

The 20k run took 167.7 seconds of training on an NVIDIA RTX A6000, versus
59.9 seconds for 7k. In the 20k run, HQ reached 95.99% at step 19,000,
dropped to 91.25% at 19,500, and ended at 93.27%. The animation includes
every 100-update snapshot and ends at the prescribed budget rather than
selecting the best checkpoint. The longer schedule improved final HQ,
mode coverage, and tail mass; sliced W1 was slightly worse.

For the prior 7k run, HQ rose from 2.69% at step 5,000 to 56.01% at 6,500
and 75.08% at 7,000. Both runs sharpen late in their respective annealing
schedules. Their matching early metrics precede the schedule divergence
at step 4,200.

This is an illustration and a quality-control observation, not a multi-run
estimate or evidence of complete Gaussian calibration. The learned prior has 20,000 zero-noise particles,
so generated support is finite. The 20,000 evaluation draws contain 12,662
distinct output points in each run. The 20k per-mode core width ratio is
0.769 (target 1), compared with 1.116 at 7k; its sharper modes are narrower
than the target. Tail mass, unequal mode weights, and full covariance
distortion remain material. These EMA results do
not establish live-model performance or contradict the transfer suite's
separate metrics and discriminator choices.

Recommendation: use the explicit 20k budget to reproduce this visualization.
It improves mode matching for this example, but the mixed distribution
metrics do not justify promoting it to a new universal default. Assess
budget or discriminator changes across the intended research tasks before
changing shared defaults. No seed sweep or additional tuning was performed.

## Reproduce

Install the repository's experiment dependencies, then run from its root:

```bash
python -u reports/readme-100gaussians/generate.py > /tmp/readme-image-20k.log 2>&1
tail -f /tmp/readme-image-20k.log
```

The script replaces the root GIF and this directory's `summary.json`. Frames
and evaluation samples are saved under `/tmp/particlegan-develop-qc/readme-image-20k/`.
The recorded run log is `/tmp/particlegan-develop-qc/readme-image-20k.log`.
[summary.json](summary.json) records all resolved recipe fields, checkpoint
metrics, source hashes, checkout commit, and environment. Source hashes cover
the working tree at the start of this run, including the renderer changes;
the checkout commit alone does not describe those changes. The earlier run's
complete summary and provenance are preserved in [summary-7000.json](summary-7000.json).

The previous GIF was generated in [commit 8664db9](https://github.com/255BITS/ParticleGAN/commit/8664db993f718e7ba290becc352224a46d619313)
using the older 0.1.2 recipe: 7,000 updates, seed 1, cap coefficient 1.0,
and generator learning rate 0.0006. Its final frame explicitly displays
`step 7000/7000`, `modes 100/100`, and `hq 0.992`; the commit also records
the same budget and EMA readout. It did not use a longer training budget.
Different hyperparameters and seeds make this an uncontrolled comparison,
and the old GIF has no equivalently audited shape-metric protocol here.
