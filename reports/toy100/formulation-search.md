# GPU formulation search

The user authorizes a faster search for a better GAN formulation, starting from
`constraints_simple_regularization` with CPU initialization and CUDA training.
The full native GPU reference remains 16/22; no new formulation is qualified.

Three independent fresh Astra/max attempts cover critic regularization/objective,
particle preconditioning, and adversarial game updates. Each has one GPU worker,
a 60-minute ceiling and at most six meaningful proposals. Adapt after measured
failures; coefficient grids and seed sweeps remain excluded. Declared changes to
the adversarial objective, regularizer or optimizer are allowed. Preserve the
original decay/noise schedules and auxiliary host losses as the starting defaults.
This formulation scope supersedes earlier instructions freezing the exact loss
and optimizer. Keep architectures, data, evaluation and training budgets fixed.

Run both ring and unequal-mass gates on each proposal. A candidate passing both
advances to trajectory, intensity, bars and blobs, then its own full 22-toy GPU
suite. Qualify stability from its own converged state with its declared schedule.
Report any extra gradient evaluations separately; never silently spend additional
optimizer updates. Never combine passes from different candidates or weaken gates.

## Completed work to avoid repeating

- CPU initialization alone passes 4/6 previously failing hosts; ring and rare
  component remain failing. This is not a measured 20/22 result.
- Full CPU random streams with CUDA training also pass only 4/6.
- PyTorch 2.14 reproduces all four 2.13 blocker controls exactly. Precision changes
  (FP64 linear accumulation, split FP32 linear, FP64 Fourier) did not fix ring.
- Multiplicity-averaged particle gradients pass unequal mass (eigen ratio .48714),
  but fail ring (7 modes/HQ .9165).
- Post-Adam displacement cap passes ring (8 modes/HQ 1), but loses rare mass.
- Gaussian stream isolation passes ring but fails unequal mass. Composing it with
  multiplicity averaging was measured separately and still fails ring.
- Dedicated CPU streams pass unequal mass but fail ring; prior-index isolation
  adds no success. Gradient row clipping fails both sustained gates.

The prior round has nine agent proposals (15 primary gates) plus one direct
combination (one failed ring gate): no joint winner. Two instrumentation replays
are not independent proposals. Full receipts remain in the local prior batch
`/ml2/hypergan/gan-attempts/cpu-recipe-gpu-port-20260924T225604Z`.
Use those implementation references read-only; do not repeat unchanged proposals.

Reuse `cpu-recipe-gpu-port/probe.py` and its verified prepared sources. Snapshot
candidate code and declarations before execution. Retain every FAIL/ERROR with
raw metrics, source hashes, exact commands, CUDA proof and actual update counts.
No target-derived corrections or non-adversarial substitute for the GAN.

## Completed formulation round and current steering

The initial three lanes finished all 18 proposals and 36 primary gates: 7 PASS,
29 FAIL, no joint winner. See [retained code and independent audit](formulation-round-20260924/README.md).
Direct follow-ups then found Ra + R1(real)/cap(fake), and Ra + symmetric b-cap .5,
pass both initial blockers but fail trajectory. The hybrid also fails intensity.
They are not promoted. Further objective ablations restore original Rp while
retaining each regularizer. Keep trajectory as an early transfer regression.

The user explicitly wants promising candidates to keep advancing unless they
fail. R1-containing mechanisms remain eligible; avoid rerunning unchanged failed
R1/R2 proposals. Decide by the measured frozen gates. Do not infer a candidate's
success or failure from the regularizer's name or history alone.

The direct follow-ups are now complete: seven proposals, 18 gates (9 PASS, 9 FAIL).
The original-Rp hybrid also fails trajectory, so restoring Rp alone is insufficient.
The next round uses trajectory as the first cheap gate, then ring/unequal mass.
Keep promising regularizers eligible; fix measured transfer failures rather than
repeating already-failed variants or discarding mechanisms based on their names.
