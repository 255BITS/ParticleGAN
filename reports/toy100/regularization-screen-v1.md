# Shared prior and critic-cap screen

The [frozen manifest](regularization-screen-v1-manifest.json) declared four
single-field changes to the production affine-square/H1600 shared config
before training (SHA-256
`db6f56c6da31ac520cb99b946c1f8f6781e3320d324d00ca4bedae99ee3e5d01`).
Every row retained β₂=0.999, prior LR multiplier 2, output σ=0.029 warmed over
20% of the host budget, input σ=0.5 ending at 10%, and the same global LR
schedule. Each used seed 0, one CPU thread, nine fixed canonical hosts, and
the source epoch at commit `0d96796`. The model policy is recorded on every
episode: `toy100_model=affine_square_v1` and
`network_lr_horizon_cap=1600`. Host architectures, data, budgets, and gates
remain frozen.

| Global change from prior weight 0.05, cap κ=1.25 | Strict live passes | Failed hosts |
| --- | ---: | --- |
| Prior weight 0.02 | 3/9 | trajectory, mode hold, anisotropic, overlap, stripes, bars |
| Prior weight 0.10 | 5/9 | mode hold, unequal mass, overlap, bars |
| Cap threshold κ=1.0 | **7/9** | overlap, stripes |
| Cap threshold κ=1.5 | 5/9 | mode hold, overlap, stripes, bars |

Lowering κ to 1.0 repairs mode hold and four-blob image quality within this
subset while retaining rare mass, unequal width, anisotropy, trajectory, and
four-bar image. Overlap and stripes still miss their sustained live gates.
The original κ=1.25 run had zero sampled cap penalties at all 24 overlap
checkpoints and at 23 of 24 stripes checkpoints; changing the activation
threshold tests more than rescaling a usually inactive penalty. Checkpoint
penalties do not prove whether the cap was active between checkpoints.

Every episode passes independent integrity regrading, including the exact
common noise and optimizer policy receipts. None meets the predeclared 9/9
threshold, so no row was promoted to all 19 transfer hosts or claimed to
work across 22 toys. The numeric ranking, per-host misses, source archive,
configs, compressed episodes, and tail-friendly log are retained locally at
`artifacts/toy100-accuracy/compatibility/regularization-screen-v1/` and the
four declared `regscreen_v1_*` artifact directories. Raw `artifacts/` paths
are local workspace evidence, not GitHub links. The [shared search ledger](shared-recipe-search.md)
also includes all four rows and their source/config hashes.
