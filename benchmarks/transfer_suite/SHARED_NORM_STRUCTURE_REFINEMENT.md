# Frozen normalization structure refinement

The [16-card screen](SHARED_NORM_STRUCTURE_RESEARCH.md) produced no live PASS.
Fixed center-only normalization finished with minimum component eigen ratio
.1296 versus the .15 gate; its late readings ranged .1022–.1385 and every
other late metric passed. The all-layer LayerNorm plus half-strength raw
activation bypass reached .1507 at the final observation, but its preceding
four readings ranged .0307–.0791. The screen's exact curves and failures are
retained in `reports/transfer_suite/unadjusted/runs/shared-norm-structure-search/screen`.

The [12-card refinement plan](plans/shared_norm_structure_refinement_rare.json)
is frozen before this stage runs. Five cards center features and divide by a
fixed fractional power of their per-example feature standard deviation,
between center-only and full LayerNorm. Three cards use GroupNorm within one
example, with two or four feature groups. Three hold the screen's raw bypass
strength at .5 but apply it to selected hidden layers. One combines fixed
centering with a .25 raw activation bypass. This probes whether preserving a
raw path or part of the feature amplitude can stabilize within-mode gradients.

All cards use the same raw input, width 96, three hidden layers, Softplus β4,
and the unchanged `shared_c6` recipe. No target statistics, labels, batch
statistics, or evaluation feedback enter training. Original generator,
prior, initialization, resources, 1200-step budget, 24 live/EMA observations,
and gates are unchanged. This rare-only screen does not establish full-suite
support. Every result is retained through the canonical runner with its
source archive, applied optimizer settings, and full observations.

```sh
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_norm_structure_search \
  --plan benchmarks/transfer_suite/plans/shared_norm_structure_refinement_rare.json \
  --output /tmp/shared-norm-structure-refinement > /tmp/shared-norm-structure-refinement.log 2>&1
```
