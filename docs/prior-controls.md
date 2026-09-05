# Matched prior controls

The 100-Gaussian example and its Gaussian counterpart now call the same training
function. Architecture, discriminator loss and penalty, batch size, generator
learning rate, discriminator learning rate, run length, schedule, and generator
EMA are identical. The prior controls are:

| Config / CLI value | Training and metric samples | Learned prior parameters |
| --- | --- | --- |
| `particles` | Uniform draws from a learned finite table | 20,000 × 4 = 80,000 by default |
| `frozen_gaussian` | Uniform draws from one fixed Gaussian table | None |
| `fresh_gaussian` | New independent Gaussian noise on every call | None |
| `gaussian` | Compatibility alias for `frozen_gaussian` | None |

The particle arm also optimizes its table at 10× the generator learning rate,
applies VICReg, and averages its learned positions for EMA readout. The two
Gaussian controls have no prior optimizer or VICReg gradient. Generator and
discriminator parameter counts match; total trainable parameter counts differ.
This isolates the learned-prior intervention, not a fixed total capacity budget.

Fresh Gaussian sampling keeps a fixed reference buffer exclusively for plots
requested with `fixed_first_n=True`. Ordinary `sample()` calls, coverage, sliced
W1, and final metrics all draw new Gaussian noise. Re-seeding an evaluation
generator makes the evaluation latent values repeat across checkpoints without
restricting the training distribution to a table. `sample()` returns `None`
for fresh-sample indices because those samples are not particle rows.

The table variants can produce at most 20,000 distinct outputs through this
deterministic generator. The trainer records `unique_samples` among the 100,000
final draws. That limit matters when extrapolating toy-benchmark results to
continuous or higher-dimensional generative modeling.

## Reproduce a comparison

```bash
python experiments/compare_priors.py \
  --study-dir runs/prior_comparison --run --device cuda:0
```

The default comparison is nine runs: each of the three priors on seeds 23001,
23002, and 23003, with 7,000 steps, Rp logistic loss, Fourier-2 discriminator,
`b_cap` coefficient 1.0, generator LR 6e-4, discriminator LR multiplier 1.5,
Adam beta1=0, delayed cosine annealing, and EMA 0.995. These seeds are separate
from the historical study's standard seeds. They are specified in advance;
report all of them, including failed runs, instead of selecting the best seed.

Omit `--run` to generate configs and a manifest for an external scheduler. The
manifest includes resolved config paths, source revision, whether the checkout
was dirty, Python and dependency versions, and the PyTorch CUDA build version.
Use a clean committed checkout for reported runs. Generate into a new directory
for each study; `--collect` collects completed runs from an existing manifest
without rewriting its provenance. A small `--total-steps` value is useful for
smoke tests, but is not evidence about convergence under the full recipe.

Each run saves its exact final sample cloud, checkpoint, evaluation time series,
and summary. `comparison.json` gathers final metrics for all seed/prior pairs.
Use coverage together with high-quality fraction, transport distance, histogram
balance, radial core spread, and both covariance eigenvalue ratios. A radial
median alone can hide anisotropy or tails. Three seeds provide a modest
replication check, not a broad guarantee of stability.

## RNG isolation and historical results

Training now owns separate generators for real data, latent samples, and
penalty interpolation. Evaluation has its own re-seeded generator. Changing
evaluation frequency no longer changes the learned parameters; a bounded CPU
regression exercises the actual trainer for all three priors and compares every
saved model tensor and final sample array across evaluation intervals.

This correction changes trajectories relative to historical code, which drew
training latents and diagnostic samples from the same global stream. New
summaries and comparison manifests identify the RNG scheme as
`separate_data_latent_penalty_v1`. Historical GIFs and results retain their
original meaning; they should not be relabeled as results of the corrected
matched comparison. In particular, the older no-particle GIF also used a
different architecture and stabilization recipe and does not isolate the prior.

Checkpoints include `prior_kind`; reconstruct a prior with
`lib.particle_prior.make_prior(prior_kind, num_particles=..., z_dim=...)` before
loading its state to preserve fresh-noise versus finite-table semantics.
