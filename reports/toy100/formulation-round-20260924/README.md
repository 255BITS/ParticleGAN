# Formulation round: 18 proposals, no joint winner

**36 GPU training gates: 7 PASS, 29 FAIL, 0 ERROR.** No candidate passes both
ring and unequal mass, so none advances to the full 22 or post-convergence tests.
The full native GPU reference remains 16/22.

| Candidate | Ring modes / HQ | Ring verdict | Unequal eigen ratio | Unequal verdict |
|---|---:|---|---:|---|
| c01_r1r2 | 8 / 0.9014 | FAIL (1/5 suffix) | 0.02234 | FAIL (0/5 suffix) |
| c02_ra_cap | 7 / 0.9956 | FAIL (0/5 suffix) | 0.57846 | FAIL (0/5 suffix) |
| c03_smooth_cap | 2 / 0.2595 | FAIL (0/5 suffix) | 0.00000 | FAIL (0/5 suffix) |
| c04_hinge_cap | 5 / 1.0000 | FAIL (0/5 suffix) | 0.00689 | FAIL (0/5 suffix) |
| c05_ra_r1r2 | 8 / 1.0000 | PASS (5/5 suffix) | 0.38229 | FAIL (0/5 suffix) |
| c06_ra_r1_only | 5 / 1.0000 | FAIL (0/5 suffix) | 0.33534 | PASS (6/5 suffix) |
| shared_coordinate | 6 / 1.0000 | FAIL (0/5 suffix) | 0.62283 | PASS (11/5 suffix) |
| exposure_trust | 7 / 1.0000 | FAIL (0/5 suffix) | 0.66360 | PASS (6/5 suffix) |
| exposure_mean | 6 / 0.9802 | FAIL (0/5 suffix) | 0.82172 | PASS (7/5 suffix) |
| shared_geometry | 6 / 1.0000 | FAIL (0/5 suffix) | 0.62920 | PASS (5/5 suffix) |
| visit_isotropic | 5 / 0.8333 | FAIL (0/5 suffix) | 0.62519 | PASS (9/5 suffix) |
| density_mobility | 5 / 0.9966 | FAIL (0/5 suffix) | 0.16980 | FAIL (3/5 suffix) |
| optimistic_adam | 8 / 1.0000 | FAIL (3/5 suffix) | 0.02351 | FAIL (0/5 suffix) |
| optimistic_gradient | 7 / 1.0000 | FAIL (0/5 suffix) | 0.03398 | FAIL (0/5 suffix) |
| predictive_critic | 0 / 0.0000 | FAIL (0/5 suffix) | 0.51876 | FAIL (4/5 suffix) |
| joint_lookahead | 4 / 0.9968 | FAIL (0/5 suffix) | 0.02156 | FAIL (0/5 suffix) |
| extragradient | 1 / 0.0850 | FAIL (0/5 suffix) | 0.02901 | FAIL (0/5 suffix) |
| optimistic_critic | 1 / 0.0608 | FAIL (0/5 suffix) | 0.04697 | FAIL (0/5 suffix) |

A suffix over five is sufficient only when all frozen metrics pass. Displayed
final values do not override sustained verdicts. In particular optimistic Adam
reaches eight ring modes but has only three final passing checks.

Ra logistic + R1+R2 (`c05_ra_r1r2`) passes ring but over-spreads the rare component
(covariance error 1.34110 versus the .85 gate). R1 only fixes unequal mass but loses
ring coverage. Five particle updates pass unequal mass and fail ring. These are
partial results from different formulations, not interchangeable passes.

All training uses audited CPU initial parameters with CUDA models, gradients and
optimizer states, fixed seeds and frozen 1200-step budgets. Original schedules and
auxiliary host losses are retained. ExtraAdam performs two gradient blocks per
outer step: its result is equal-step, not equal-compute. All other proposals add no
model forward/backward evaluations, though tensor-operation costs differ.

The three agent audits reported 12 passing mechanism/audit gates (not 12 toy
passes). An independent review additionally regraded all 36 runs and checked their
frozen specifications, exact candidate sources, initial parameters, schedules,
CUDA updates and random receipts. [Audit](independent-audit.json) · [Results](summary.json).

Candidate code, prelaunch declarations, raw results compressed as JSON, original
command receipts and lane reports are retained here. Prepared baseline sources
are reconstructed using the existing checksum-verifying source preparer, avoiding
another copy of the archive. Failed final-state checkpoints remain in the original
local attempt directories; no continuation or qualification depends on them.

Next measured follow-ups: Ra + real-point R1 / fake-point cap; and Ra + R1+R2
combined separately with exposure-mean and shared-coordinate particle updates.
Each composition must earn its own two blocker passes. No formulation is promoted.

Replay one recorded candidate in a new work directory with the original 2.13
CUDA environment (the helper verifies the source archives and candidate hashes):

```bash
/tmp/pr38-default-env/bin/python reports/toy100/formulation-round-20260924/replay.py \
  --lane critic_formulation --candidate c05_ra_r1r2 --task mode_hold \
  --workdir /tmp/ra-r1r2-ring-replay-new --gpu 1
```

[Measured follow-ups](FOLLOWUPS.md) retain subsequent candidate results separately.
