Ring16's full SVD update amplifies an observed `1.03e-7` relative gradient
difference at the checkpoint boundary into a `.252` relative direction change.
This adds an experiment-only float32 numerical-rank truncation rule and compares
a single intervention during update 401 with applying the same rule on every
update. Both arms train fresh live objects through the public ParticleGAN API,
with unchanged architecture, prior, seed, target batches, learning rates and full
quality bounds.

The frozen protocol reserves two 1,600-update CUDA arms (600 seconds total), plus
one zero-update saved-gradient CUDA probe (30 seconds). Smoke requires a full
passing scheduled state and independent same-state confirmation; the historical
five-terminal reducer remains a separate diagnostic. The runner retains full
state on errors and produces actual-training GIFs from saved samples. Production
defaults and qualification are unchanged.

Validation: syntax/source bindings, static protocol checks, and fail-closed
unavailable-CUDA guards. Training and algebra are unmeasured because this host
has no CUDA device. GitHub publication is pending network access. See the
[prospective report](reports/forge/ring16-truncation/README.md) and its verification
receipt for the exact scope and reproduction commands.
