# Frozen LayerNorm Softplus refinement

The [pointwise normalization screen](SHARED_POINTNORM_RESEARCH.md) found one
useful lead: raw, three-layer, width-96 Softplus(beta=5) with LayerNorm after
every hidden linear layer gave final minimum component eigen ratio .11905 on
unequal mass, versus .01891 for the earlier raw SiLU witness. It remained a
FAIL: the .15 eigen-ratio gate was unmet, and the last five checkpoints also
showed intermittent covariance error and sample-quality failures. No
normalization card passed either remaining test.

Before this stage runs, the [six refinement cards](plans/shared_pointnorm_refinement_rare.json)
are fixed. Three vary LayerNorm Softplus width (64, 128, 160) at beta=5. Two
hold width 96 and vary activation curvature through beta=2 or beta=10. The
last holds width 96/beta=5 and adds a zero-initialized raw linear score skip
to preserve an unnormalized coordinate gradient. These are generic
pointwise architecture parameters. They use no task labels, target-derived
features, batch statistics, or evaluation feedback during training. Original
generator, prior, seed, resources, budgets, gates, and unchanged `shared_c6`
recipe are retained.

All six cards first run the unequal-mass case. If one or more earns a sustained
live PASS, run the unequal-width case for every passing card. If none passes,
run unequal width for the two closest failures, selected by longest final
passing suffix, then lower final normalized shortfall, then name. This
advancement rule is fixed before running the six cards. Both stages retain
every attempted failure with all 24 live/EMA observations and exact source and
optimizer receipts. Neither stage alone is a complete 19-case profile.
