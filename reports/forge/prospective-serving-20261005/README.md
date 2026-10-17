# Explicit public serving observations

This opt-in Forge helper observes the output selected by the public
`GANTrainer.served_model()` interface, then calls `ServedModel.sample()` once
with a separate evaluation RNG and explicit `output_noise=True`. Its receipt
records the selected fast/averaged source, actual noise scale, completed update
and a caller-supplied complete semantic-state fingerprint before and after.

The public sampling default is `output_noise=False`. Existing task laws retain
their own observations. A caller must declare this new law before collecting
new evidence, preserve the task's metrics, thresholds, horizon and holdout, and
bind a complete state reader. The initial helper supports unconditional
independent-row sampling; enumerated image rows and contextual generation need
their own declared interfaces.

The installed helper passed 16 software tests with 28 focused subtests. These
check one selection/draw, RNG separation, state purity and unsupported-law
refusal. Run them with:

```sh
python -m unittest discover -s tests -p test_forge_canonical_serving_measurement.py -v
```

These are software controls. Numerical Atlas validation is a separate fresh
candidate with its own representation, source, admission and complete original
observation schedule. One terminal draw cannot recover a past convergence
curve. The earlier measured results remain in
[the common26 diagnostic report](../common26-full-original-diagnostic-20261005/README.md).

[PROTOCOL.md](PROTOCOL.md) explains the caller contract; [PROTOCOL.json](PROTOCOL.json)
is its machine-readable declaration. The protocol's source-only status means
that this publication assigns no numerical benchmark verdict or default winner.
