# Global cosine schedule follow-up

**All four frozen global schedule variants fail the rare unequal-mass case.**
The best supported shared-c6 discriminator profile and every non-schedule
Recipe field were identical across trials. Each candidate changed only the
global cosine hold fraction and floor. No candidate passed a single complete
live checkpoint, so the predeclared remaining-18 completion condition was not
met. These four rows are 1/19 **INCOMPLETE**, not new complete 19-case scores.

The [center-fixed β6 discriminator follow-up](BETA6_FOLLOWUP.md) crosses these
same four schedules with an alternative rare-case D. Its .50/.01 candidate
reaches a three-check final passing streak, below the required five.
[Six nearby global schedule cards were declared but not run](INTERPOLATION_UNRUN.md)
after an independent unchanged-c6 rare-case PASS took priority.

| Global hold / floor | Final minimum normalized eigenvalue | Final covariance error | Final HQ | Passing observations / suffix |
| --- | ---: | ---: | ---: | ---: |
| .30 / .01 | .01026 | .6920 | .9834 | 0 / 0 |
| .30 / .05 | .00445 | .8282 | .9729 | 0 / 0 |
| .40 / .01 | .08926 | .7121 | .9873 | 0 / 0 |
| .50 / .01 | .01833 | .8011 | .9971 | 0 / 0 |

The variance gate is **≥.15**, and a PASS requires all metrics to pass for
five consecutive final observations of the complete 24-checkpoint live curve.
The existing .60/.05 shared-c6 profile reaches .12525 at the final checkpoint
and also has zero passing observations. The earlier schedule hypothesis is not
supported by this screen; all four candidates have less final within-mode
variance than the current profile. EMA is retained separately.

The scoped schedule bridge changes the arguments passed to the public
`particlegan.learning_rate_scale` function from the existing
`FixedSchedule`. It does not alter the optimizer recipe, host update clock,
loss, discriminator, generator, particle count, data, metrics, thresholds,
budget or seed. Every episode retains the full Recipe including
`lr_anneal_start` and `lr_floor`, plus a `schedule` receipt with the hold,
floor, controller call count, actual trace roles and equation check. The 120
rare-case action entries in each candidate match the public schedule equation
with maximum absolute error **0**. Focused tests cover mixed generator/AE
prior groups and direct particle optimizers under the same global schedule.

A full-step default .60/.05 control on `two_pole` (legacy) and
`vector_unequal_mass` exactly matches the current archived profile in recipe,
candidate, original/effective specs, discriminator, actual optimizer receipts,
all live/EMA observations, all actions and both verdicts. Only timing and
source-manifest fields differ. The control's [parity checks](control/checks.json)
and [source archive](control/source.tar.gz) are validation evidence, not
additional selection attempts.

The [frozen screen plan](screen_plan.json), [screen index](screen/index.json),
[protocol](screen/protocol.json), [exact source archive](screen/source.tar.gz),
[all curves](screen/README.md) and [tail-able log](screen.log) retain every
candidate. A dry-run primary import of the four screen entries validated all
519 episodes with zero errors; each new recipe remains 1/19 INCOMPLETE.
The [identity plan](control_plan.json) and [control log](control.log) are
separate from the selection screen.
