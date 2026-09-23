# Pointwise normalization follow-up

The initial 12-card [smooth ensemble screen](SHARED_ENSEMBLE_RESEARCH.md)
produced no live PASS on either remaining `shared_c6` case. All 24 attempts
remain archived. Its best final minimum component eigen ratio was .03546 on
unequal mass and .11250 on unequal width, below the .15 threshold. The best
prior raw SiLU width run met final bounds but held them for only one
observation. This follow-up asks whether pointwise feature normalization or
weight parameterization stabilizes the raw smooth critic's local gradients.

The [six cards](plans/shared_pointnorm_screen.json) were declared before this
stage's training. All use raw 2D inputs and three hidden layers. `layer_first`
normalizes only the first hidden vector; `layer_all` and `rms_all` normalize
each hidden vector, within one input example. They use learnable per-feature
scale (and LayerNorm offset), epsilon 1e-5, then Softplus(beta=5) or SiLU.
`weight_all` applies PyTorch's differentiable WeightNorm parameterization to
all linear weights, including the score head; it has no activation
normalization. There is no batch dependence, running state, target-derived
feature, evaluation feedback, or extra training update. The parameterization
changes are explicit discriminator architecture choices.

The recipe remains `shared_c6` without any per-task adjustment: Rp logistic,
b_cap coefficient 6 and kappa 1.25, prior spread .05, no particle L2, Adam
(0, .99), absolute G/D LR .00425 and particle LR .0085, cosine annealing from
60% to a 5% floor. The generator, prior, initialization seed, target, resources,
budget, live/EMA measurements, and gates stay fixed. The original host spec
and actual optimizer receipts are stored for every trial.

Run with a new output directory:

```sh
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_pointnorm_search \
  --plan benchmarks/transfer_suite/plans/shared_pointnorm_screen.json \
  --output /tmp/shared-pointnorm-screen > /tmp/shared-pointnorm-screen.log 2>&1
```

The runner snapshots and verifies source, archives all complete episodes and
failures, and emits importer-ready `index.json`, `protocol.json`, and
`source.tar.gz`. A live PASS requires all 24 checkpoints and at least five
final passing observations. EMA is separate. This two-case screen does not
establish a full 19-test profile.
