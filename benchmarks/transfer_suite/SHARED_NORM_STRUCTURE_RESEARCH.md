# Frozen pointwise normalization structure screen

The prior LayerNorm-all, Softplus β4, width-96 three-layer critic finishes with
minimum normalized component eigen ratio .12525, short of the .15 gate. Its
other final metrics pass, but none of the final five observations passes all
gates. The 16 cards in the [frozen plan](plans/shared_norm_structure_rare.json)
test structural changes to this critic on `vector_unequal_mass`.

All cards retain width 96, three hidden layers, raw 2D input, and Softplus β4.
They vary normalization before/after activation; learnable versus fixed
per-feature scale and offset; all versus selected layers; LayerNorm, RMSNorm,
or centering alone; and fixed strengths of raw or residual paths. `raw_blend`
adds a fraction of the unnormalized activation at each hidden layer. The
`input_injection` card adds raw coordinates through an ordinary learned linear
map at the second and third hidden layers. `rms_mix` averages LayerNorm and
RMSNorm outputs before activation. All operations are pointwise within one
example. No batch statistics, target information, labels, or evaluation
feedback enter training.

The unchanged recipe is `shared_c6`: Rp logistic, b_cap 6, κ 1.25, spread .05,
no particle L2, Adam (0, .99), absolute G/D LR .00425 and particle LR .0085,
60% hold then cosine to a 5% floor. The generator, prior, seed, particle and
batch counts, 1200 steps, 24 live/EMA observations, and thresholds remain
fixed. A live PASS requires at least five consecutive final observations
meeting every original gate. This rare-only screen is incomplete for 19/19.

Run from the repository root with a new output directory:

```sh
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_norm_structure_search \
  --plan benchmarks/transfer_suite/plans/shared_norm_structure_rare.json \
  --output /tmp/shared-norm-structure-rare > /tmp/shared-norm-structure-rare.log 2>&1
```

The canonical runner snapshots exact source before training, verifies source
hashes throughout, and retains each complete live/EMA curve, original/effective
spec, D card, actual optimizer settings, actions, verdict, and failures.
