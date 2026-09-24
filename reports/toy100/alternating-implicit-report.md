# Alternating-map implicit response: warm stability, failed acquisition

**No fixed-target solution.** One bounded arm replaces the old implicit
adapter's simultaneous field with a precisely defined alternating-map field.
It passes all 200 warm checks, then fails the original cold trajectory at
MSE .2523977, with zero passing observations. The controller stops before
cold ring or extended hold. No seed change, parameter scan, new penalty,
time-based attenuation, distribution shift or gate relaxation was used.

The [derived summary](continuous-evidence/alternating-implicit-round3/summary.json)
and [source/raw manifest](continuous-evidence/alternating-implicit-round3/manifest.json)
preserve the exact executed adapters, drivers, transformed hosts, declarations,
controls and results. All optimizer evidence is scratch-only and carries
`shared_gate_eligible=False` in its declaration and dynamics receipt.

## Why retest the implicit method

PR82 establishes that the earlier recorder captured G before D moved,
unlike the actual host. Ordinary alternating Adam can acquire the trajectory
task. Earlier simultaneous full-J implicit response passed warm stability
but failed cold trajectory at approximately .253794. Correcting the field
ordering is therefore a distinct causal test, with the old solver settings
held fixed. This does not invalidate results from direct optimizer wrappers
that already preserved alternation.

[`alternating_implicit_scratch.py`](alternating_implicit_scratch.py) subclasses
the archived implicit solver. At the first evaluation of each outer update,
it executes the **actual** ordinary Adam D step before evaluating G, then
executes G/prior Adam. Each player's moments advance once. Save the base
`(D0,G0)`, first gradient `a = FD(D0,G0)`, rounded D proposal `D1`, and the
resulting bias-corrected Adam preconditioner `P`, including nominal rates.

For every later query at `(D,G)`, replay the same host minibatch/noise and use

```text
Ftilde_D(D,G) = FD(D,G)
Dvirtual     = round(D1 + (D-D0) - PD * (FD(D,G)-a))
Ftilde_G(D,G) = FG(Dvirtual,G)
```

Restore queried D after capturing G. This anchor reproduces the base
alternating gradients bit-for-bit, including actual Adam/dtype rounding.
A mandatory base replay verifies that fact before every nonzero solve.
The combined G role includes trainable prior particles at their own rate.

This is an implicit response to an **LR-dependent alternating-map field**.
It is not implicit Euler on the unmodified simultaneous GAN field, nor a
claim of converging to a true game equilibrium. Field zeros are preserved
only up to the explicit rounding anchor `D1-D0+PD*a`; its maximum parameter
norm is 3.39e-7 warm and 2.02e-7 cold. Inner virtual D uses the full frozen
base metric, even when the outer solver reduces its accepted scale.

## Unchanged solver and measured limits

In scaled coordinates `u = P^(-1/2) delta`, finite-difference GMRES solves

```text
(I + alpha * sqrt(P) * J_Ftilde * sqrt(P)) u
    = -alpha * sqrt(P) * Ftilde(base).
```

The settings are unchanged: at most eight Krylov directions, linear
relative residual .1, actual rounded nonlinear relative residual .5,
finite-difference physical scale `1e-4*(1+||base||)`, correction norm limit
2, and at most eight halvings per update. The next update starts at twice
the previous accepted scale, capped at 1; there is no age or horizon input.
The guard rejects on the measured residual, not on an evaluation target.
Exact zero fields can rest without queries or parameter movement.

| Test | Result | Cost and scale |
| --- | --- | --- |
| Warm updates 1001–1200 | PASS 200/200; minimum 8 modes/HQ .993896; final 8/.994873 | 2,379 field evaluations/player for 200 moment updates; mean alpha .306719 |
| Original cold trajectory, 400 updates | FAIL MSE .252398; 0/24 passing, limit .02 | 3,743 fields/player for 400 moment updates; mean alpha .032617 |
| Cold ring / fixed-target 2,400-step hold | Not run after acquisition failure | — |

The warm scheduled identity reproduces the complete cold scheduled state.
An additional **active** ordinary-control fork executes the adapter's first
D-then-G callbacks and retains their actual rounded Adam proposal. Its
complete model/Adam/EMA/RNG hash and all 24 production observations match
the untouched constant host exactly. Both ordinary constant controls pass
only 6/200 dense warm checks. The inherited warm-state hash is
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`.

Cold acquisition fails despite 400 exact base replays. Of 405 rejected
trials, 395 fail the nonlinear residual, seven fail the linear residual
and three exceed the correction norm bound. Median accepted alpha is
.0078125; its mean over the last 100 updates is .0051172, and its minimum
is .00390625. The final own-nearest fraction is zero and set-cover score
is .414309. This is failure to acquire a learnable discrepancy, not an
objection to resting at a matched target. Correct field ordering alone
does not cure the implicit method's overly constrained acquisition path.

## Accounting and reproducibility

All 2,179 warm and 3,343 cold replay queries verify the complete host/global
and noise RNG progression. Nominal G/D rates remain .00425, prior .0085;
warm final Adam counters are exactly 1,200 and cold counters exactly 400.
Callbacks include all repeated gradient evaluations, while moments advance
once per outer update. The original frozen hosts determine noise clocks:
mode-hold and any extension use 1,200, trajectory uses its original 400.
The executed declaration's scalar `noise_horizon=1200` was shorthand for
the mode-hold rule; the actual trajectory receipt correctly records 400.
The current driver makes this distinction explicit. Archived original
driver/declaration bytes are unchanged; this metadata correction changes
no training behavior or result.

Five new tests pass: closed-form bilinear alternating implicit solve,
zero-field rest, active ordinary-control exact host/optimizer/noise/RNG
parity, delayed activation parity, and active host exact base replay plus
query/moment accounting. The bilinear check uses tight solve accuracy to
test the math; the experimental solver retains its original settings.
Independent source review found no ordering or anchoring defect in this
declared field. Scope is the two deterministic frozen CPU MLP hosts.

The exact executed adapter SHA256 is
`31c17fd179ea02a2461ec75731ac7a5eef323aea4f51c34dd477d9c9b3390d94`.
The source references the [CGD author explanation](https://f-t-s.github.io/projects/cgd/)
only for strategic-response motivation; this anchored full-J variant has
no claimed theorem for the neural, nonsmooth host.

With the handoff's one-thread CPU/AVX2 environment, reproduce using fresh paths:

```bash
python -m pytest -q tests/test_alternating_implicit_scratch.py
python reports/toy100/alternating_implicit_probe.py --phase warm --output /tmp/ai-warm
python reports/toy100/alternating_implicit_probe.py --phase cold \
  --previous /tmp/ai-warm/run/summary.json --output /tmp/ai-cold
```

The current cold result deliberately blocks `--phase hold`. Neither warm
stability nor a verified nonlinear residual qualifies this arm as a
replacement for LR decay.
