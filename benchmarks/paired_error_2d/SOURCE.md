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
maps, splits and paired-error data-noise rule are retained as the problem definition.
Neither application is imported by the benchmark.

Training no longer follows the extraction: the problem runs on `benchmarks.toy_runner`
under the shipped recipe (recipe-built optimizers and schedule, RpGAN loss, the
recipe's critic penalty, noise and EMA; explicit `particlegan.init`). The particle
table is the recipe's particle prior. The three historical hyperparameter arms
(baseline, cap-cosine, cap-cosine-vic005) collapse into that one recipe, and the
exact artifact audit against model-glue was removed because exactness cannot hold.
Historical results stay in `reports/paired_error_2d/`.
