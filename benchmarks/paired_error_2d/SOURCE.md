# Paired-error transport extraction

This benchmark extracts the 2D subset of the model-glue cap comparison:

- model-glue commit `c89476d29d64bd4b9fe23c735b9148691ddcd13d`,
  `configs/particle-cap-comparison-20260922.json` and
  `model_glue/particle_cap_comparison.py`.
- particle-sliders commit `8e2ea7ee617883ee8135692130614dd07d119607`,
  `conceptmod/textsliders/particle_bridge_gan.py`.
- Candidate settings from ParticleGAN PR #38 at
  `6c499e6ce5d05af3673c4636142c39f8cc0ffb37`.

The routed MLP and absolute-target normalization are adapted from particle-sliders
(MIT; original notice included in LICENSE). Model-glue's residual host, affine/swirl
maps, splits, parameter initialization order and experiment settings are retained.
The numerical losses, gradient penalty, VIC and cosine schedule call this checkout's
public ParticleGAN implementations. Neither application is imported by the benchmark.

Differences from the nine-host behavioral winner remain explicit: 128×4 routed
particles, no output reconstruction/cover loss, a noise-corrupted paired-error
critic, every-fourth-step cap with compensation, and EMA-based checkpoint selection.
This is an additional transfer benchmark; its results do not change the existing
29-bound suite or production recipe defaults.

The reproduction holds seed 0 fixed. It does not add a seed search. Both historical
2D tasks, all three declared recipes and both cloud controls are retained. The
already opened historical test set is used only to check reproduction, not marketed
as new independent held-out evidence.

All runtime sources are hashed in `protocol.json`. Optional reference comparison
reads the historical artifacts and checks tensors and scalars without tolerances,
including full final optimizer/EMA/RNG states and complete validation curves. A
mismatch produces a failing exit code and a machine-readable list of differences.
