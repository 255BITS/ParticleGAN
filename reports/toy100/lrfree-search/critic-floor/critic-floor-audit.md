# Applied critic-rate floor: source and smoke audit

Audited isolated candidate
`candidate-prior-handoff-critic-floor/package/particlegan/training.py`, SHA-256
`b3d472d155702e9d49ea9fc514a145ad5b8a13f80d50bcb16c66dc52de2c8c7b`.
The package diff from `candidate-prior-geomean-handoff/package` is confined to
this file. Its only additions calculate the already-used geometric applied
prior scale, take `max(generator_scale, *prior_scales)`, and floor the critic
tester scale at that value. The floor happens before the existing DV12 critic
payoff damping. It uses optimizer/tester state only; no task name, target
geometry, evaluator metric, or random draw enters the rule. Overrides are
byte-identical (SHA-256
`bccb5824e4c4e5cf1c39ff202f53fa00c7f2f7e4a8c73ba3ebb125d2ed9ba77c`).

The 1-update frozen grid100 host smoke in
`runs/h2-handoff-critic-floor/smoke-grid100/` completed with expected `INVALID`
budget verdict, zero stream deviations and no warnings. `native-fixture.json`
matches the handoff grid100 fixture exactly, including its initialization
hashes. The smoke does not activate the floor because all tester scales are 1.
Frozen host source hashes remained unchanged and passed the harness check.

I independently loaded the saved rotated100 handoff schema-4 checkpoint into
the new package. Checkpoint top-level and D state keys, optimizer-group counts
and KA2 calls (7,000) match. An in-memory activation check set tester scales
G=.25, prior=1, D=.0625, then performed one unlabelled update. It produced G
LR .0010625, prior LR .00425 (applied scale .5), and D LR .002124848
(base × .5 × ~.99993 payoff damping). The resulting checkpoint remains schema 4.
No saved source or checkpoint was modified by this audit.
