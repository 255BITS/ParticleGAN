# Alternating positive own-secant G response

This scratch controller passed the 200-update passing-state filter but failed
the next, full-budget cold trajectory gate. It was stopped before cold mode
hold, extended fixed-target hold, or the common 22-task suite. Its low final
trajectory MSE does **not** meet the sustained requirement.

PR #82's alternating adapter keeps the frozen host's D-then-G update order.
Here D retains its own-curvature bound at 2. After G's ordinary Adam proposal
against the realized D, one same-batch and same-noise replay measures
`s = theta_proposal - theta_base` and `y = g_proposal - g_base`. Let `P` be
the diagonal Adam metric. If `s·y > 0`, use the positive-semidefinite
one-pair curvature `J_hat = yyᵀ/(s·y)` and solve

```text
(P⁻¹ + J_hat) delta = P⁻¹ s
delta = s - P y (s·y)/(s·y + yᵀ P y).
```

The Sherman–Morrison form damps the measured stiff direction without a dense
matrix or a global scalar shrink. Its Adam-metric norm cannot exceed that of
`s`. If `s·y <= 0` or the calculation is nonfinite, the deterministic
fallback is PR #82's G own-curvature factor `min(1, .25/rho_G)`. This is a
G-own-field response, distinct from the earlier simultaneous joint-game
secant extragradient. It is not a convergence claim for the nonlinear GAN.
No target labels, quality scores, elapsed-time rates, or zero pull enter the
controller. D/G nominal Adam rates stay `.00425` and learned prior `.0085`.

| Stage | Result | Decisive observation |
| --- | --- | --- |
| Scheduled identity / constant warm controls | 200/200 / 6/200 | Identity final full-state hash matches uninterrupted control |
| Positive-secant warm | **200/200 PASS** | Minimum HQ .92944; final 8 modes/HQ .99976 |
| Cold trajectory | **FAIL** | Final identity MSE .001159, but only 4 consecutive passing checkpoints; five required |
| Cold mode hold and later gates | Skipped | First failed stage stops advancement |

The cold trajectory MSE is .1282 at checkpoint 334, then approximately
.0014/.00115/.00119/.00116 at 350/367/384/400. Thus the candidate learned
the identity too late under the frozen 400-update budget. The warm pass used
the positive rank-one rule in 162/200 updates, with 38 scalar fallbacks;
the D bound never fired in that passing continuation. The cold trajectory
used 397/400 positive rank-one updates and 3 fallbacks, with D bound active
171/400. These are observations, not tuned thresholds.

The raw receipts record every one of the 600 warm and 1,200 cold gradient
calls **per player** at fixed applied rates, with one Adam moment advance per
outer update. Warm moment steps end at 1200 and cold trajectory at 400 for
both players. Noise and full-state control checks pass. Four focused tests
verify the analytic two-dimensional contraction, fallback/zero-field rule,
D-then-G order, exact moment count and RNG replay. A same-seed rerun added
rate/moment observations only and reproduced the original numeric verdict.

The [manifest](continuous-evidence/alternating-positive-secant/manifest.json)
binds compressed warm and cold raw evidence, source copies/hashes, effective
configurations, per-update records, and tailable logs. Reproduce using new
output directories:

```bash
python -u reports/toy100/alternating_positive_secant_probe.py --phase warm --output NEW_WARM
python -u reports/toy100/alternating_positive_secant_probe.py --phase cold --previous NEW_WARM/summary.json --output NEW_COLD
```
