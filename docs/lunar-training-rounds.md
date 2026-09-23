# Lunar training rounds

All development comparisons below reuse the fixed expert training cohort
`24000:24096` and controller validation cohort `34000:34020` in the declared
bidirectional Box2D variant. The old `84000:84030` test result is now a
**diagnostic**, because it was inspected before the calibration fix. The new
full pipeline evaluated its predeclared, previously untouched `94000:94030`
test cohort once after validation selection. [Failure analysis](lunar-failure-analysis.md)
explains the mechanism and limits of these measurements.

The first pilot exposed an actuator problem: any positive main command ignites
the stock engine at at least half power. A `0.12` policy deadband corrected the
initial network's near-zero commands; applying it to the **same** first-pilot
checkpoints changed slow validation from 17/20 to 20/20 and fast from 0/20 to
20/20. A later independent pilot used 8,000 behavior-cloning updates and 400
conditional paired RpGAN updates. Restricting fast training to successful,
faster same-reset expert flights gave slow 20/20, fast 20/20 and a 1.161×
paired speedup on validation. These were development results, not a new test.

The original single-command run used 2,500 world updates, 8,000 cloning and
400 RpGAN updates per policy round. It selected fast round 1 by validation
success then successful-flight steps. Its [archived report](../reports/lunar_fast/audit/original_report.json)
shows slow 19/20 and fast 20/20 validation landings; on the former test cohort
slow landed 28/30 and fast 29/30. Both landed on 27 identical test resets, and
fast won all 27 at 1.172× paired speed. This is a genuine measurement of the
**old policy** in the named simulator variant, but the subsequent diagnosis
shows that its slow controller fails on successful expert training resets.

## Fixed-data mechanism and budget comparisons

The audits held expert data, simulator variant, policy architecture, 8,000
cloning updates, paired Rp logistic loss, `b_cap` cadence, optimizer settings,
and training seed fixed. The old raw-action world had too little data around
the off-to-up ignition jump. A training-only collector replayed exact expert
prefixes on the world-model training seeds and took six one-step main-action
branches per anchor. These 5,472 real Box2D transitions trained **only the
world model**. Policy targets stayed the original successful expert actions.
The calibrated model also maps raw commands to their known engine-power
features. Its predicted mean off-to-up velocity jump was 0.02496 against
0.02731 measured, compared with 0.00468 from the original world model. The
[calibration audit](../reports/lunar_fast/audit/calibrated_world.json) records
the model comparison; scratch runs remain under
`results/gym/lunar_systematic_audit/calibrated_durability/`.

| Calibrated policy | Fixed validation landings | Mean successful steps | Expert training-world landings | Wrong upward commands on 42 high/rising expert-off states |
| --- | ---: | ---: | ---: | ---: |
| Slow, 400 RpGAN updates | 20/20 | 211.60 | 91/92; one off-pad landing | 0/42 |
| **Slow, 1,200 updates** | **20/20** | **210.80** | **92/92** | **1/42** |
| Slow, 2,400 updates | 20/20 | 214.00 | 91/92; one crash | 39/42 |
| Fast, 400 updates from slow-1,200 | 20/20 | 184.40 | 90/91; one incomplete landing | n/a; fast expert commands down on the 29 analogous rows |

The [slow-1,200](../reports/lunar_fast/audit/calibrated_slow_1200.json) →
[fast-400](../reports/lunar_fast/audit/calibrated_fast_400.json) pair landed on
all 20 matched validation resets.
Fast won all 20, saving 26.4 steps on average for a 1.143× paired speedup.
The [paired comparison](../reports/lunar_fast/audit/calibrated_pair.json)
records those per-controller results. The
[2,400-step slow checkpoint](../reports/lunar_fast/audit/calibrated_slow_2400.json)
still scored 20/20 validation, yet fired the
upward engine on 39/42 high/rising training states where the expert kept it
off and lost one training-world landing. More adversarial updates are therefore
not an established durability improvement. The bounded 1,200/400 budget is
supported by the fixed-data audit; it is **not** a claim that 2,400 updates
are stable.

## Calibrated single-command result

The [calibrated report](../reports/lunar_fast/report.json) records the
predeclared `94000:94030` test cohort after selecting the slow-1,200 →
fast-400 path on validation. Both controllers landed 20/20 validation worlds;
mean successful flight time was 210.80 steps for slow and 184.40 for fast.
On the new test cohort, slow landed **28/30** at 217.43 mean successful steps
and fast landed **29/30** at 186.45. Both landed on 27 identical resets;
fast finished sooner on all 27, with a 32-step median saving and **1.180×**
paired speedup. The remaining outcomes were two slow and one fast
`incomplete_landing`; there were no classified crashes or flyaways.

This passes the declared success and speed gate on the new cohort. The old
84000 and new 94000 results come from different resets and different policy
versions, so their numerical difference is **not** a paired before/after
improvement estimate. A 29/30 result also leaves one observed incomplete
landing; it does not establish universal reliability.

Earlier scratch recipe metadata showed `ema_decay=0.995`, but none of these
training loops ever updated or selected EMA weights. The live policy weights
produced every reported flight. Current code sets unused recipe EMA metadata
to zero so the saved configuration describes that fact accurately.
