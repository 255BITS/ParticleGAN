# Frozen center-only discriminator follow-up

Fixed center-only normalization at width 96 and Softplus β4 was the most
stable spread result in the [structural screen](SHARED_NORM_STRUCTURE_RESEARCH.md):
all non-spread metrics pass across the final five observations, while the
minimum normalized eigen ratios span .1022–.1385 against the .15 gate.
The subsequent power-norm, grouped-norm, and bypass-placement refinement
produced no sustained PASS.

Before running any card in this stage, the [six-card plan](plans/shared_norm_structure_center_followup_rare.json)
is frozen: width 96 with Softplus β3/5/6/8 and width 128 with β4/8. This is a
focused activation/capacity check of the stable center-only structure. All
cards retain three hidden layers, raw 2D input, fixed per-example centering
before each activation, and the unchanged `shared_c6` recipe. No labels,
target statistics, batch statistics, or evaluation feedback enter training.
The original generator, prior, seed, resources, 1200-step budget, 24 live/EMA
observations, and behavioral gates remain unchanged.

All six unequal-mass episodes are retained with source archive, original and
effective specs, actual optimizer settings, full curves, and verdicts. This
rare-only screen is incomplete for the full 19-case profile.

```sh
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_norm_structure_search \
  --plan benchmarks/transfer_suite/plans/shared_norm_structure_center_followup_rare.json \
  --output /tmp/shared-norm-structure-center-followup > /tmp/shared-norm-structure-center-followup.log 2>&1
```
