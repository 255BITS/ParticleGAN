# BCAP search winner as the named API preset

After the 96-configuration search concluded, the owner requested promoting its
winner. `get_recipe("bcap")` now selects the winning global trainer settings:
zero-momentum DualNorm, non-saturating loss, smoothing **0.001**, per-offset
convolution updates, G/E step **0.012**, D step **0.018**, and sampled-prior
step **0.030**, with constant rates and BCAP coefficient/cap **1** every update.
No research configuration file is loaded by the package at runtime.

| Trainer setting | Previous public BCAP preset | Selected preset |
| --- | --- | --- |
| Loss | relativistic | non_saturating |
| DualNorm smoothing | 0 | 0.001 |
| Convolution updates | none | per_offset |

The [selected configuration](../../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json)
passed **6/6 Tier 1** and **7/21 Tier 2**, tying two other configurations.
The newly measured relativistic/smoothing-1e-5 control scored **6/21** under
the same source, runtime and task contracts. This improves the finite search
score but misses its **10/21** target. Word retention and stripes pass; Gaussian
stability and broader coverage still fail. The [readout](README.md) retains all
96 configurations, metrics, actual-training GIFs and recommendations.

This is an owner-directed API preset choice. Calibrated scientific promotion,
independent confirmation and scale transfer remain unestablished. The original
study's `default_adoption=false`, qualification, source-bound receipts and
byte-verified archive retain their publication-time identities. No unchanged
experiment is rerun to make this selection.

The single [current leaderboard](../technique-inventory.md) already selects this
complete recipe as Forge's BCAP configured standard. The package's default
`get_recipe()` continues to select KA2. `bcap_adam` and Halloween retain their
explicit historical Adam configurations. Forge API v1 still resolves its
historical `bcap` base through `bcap_adam`, so saved cards and configuration IDs
keep their meaning. Task/caller-owned architecture, prior, initialization,
sampling and budget remain explicit rather than being copied from a test host.

Use the complete saved recipe to restore historical checkpoints:

```python
from particlegan import Recipe, get_recipe

recipe = get_recipe("bcap")
historical_recipe = Recipe(**checkpoint["recipe"])
```

For the previous unsmoothed public configuration, explicitly select
`get_recipe("bcap", loss="relativistic", optimizer_smoothing=0.0,
optimizer_convolution="none")`. Checkpoint restoration rejects substituting
the new recipe for an old packet; constructing from its saved fields preserves
its actual training law. [The selection receipt](default-selection.json) records
the trainer delta, source/evidence identities and software verification.
