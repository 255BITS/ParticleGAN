# Lunar training rounds

All development comparisons below reuse the fixed expert training cohort
`24000:24096` and controller validation cohort `34000:34020` in the declared
bidirectional Box2D variant. Both `84000:84030` and `94000:94030` have now
been inspected and are **diagnostic** cohorts. A fresh single-command run on
clean revision `194fe91` completed with corrected contact scoring and the
same training hyperparameters. [Failure analysis](lunar-failure-analysis.md)
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

| Calibrated policy | Fixed validation, old flag score | Mean flagged-success steps | Training worlds, old flag score | Wrong upward commands on 42 high/rising expert-off states |
| --- | ---: | ---: | ---: | ---: |
| Slow, 400 RpGAN updates | 20/20 | 211.60 | 91/92; one `off_pad_landing` flag | 0/42 |
| **Slow, 1,200 updates** | **20/20** | **210.80** | **92/92** | **1/42** |
| Slow, 2,400 updates | 20/20 | 214.00 | 91/92; one generic crash classification | 39/42 |
| Fast, 400 updates from slow-1,200 | 20/20 | 184.40 | 90/91; one `incomplete_landing` flag | n/a; fast expert commands down on the 29 analogous rows |

The [slow-1,200](../reports/lunar_fast/audit/calibrated_slow_1200.json) →
[fast-400](../reports/lunar_fast/audit/calibrated_fast_400.json) pair landed on
all 20 matched validation resets.
Fast won all 20, saving 26.4 steps on average for a 1.143× paired speedup.
The [paired comparison](../reports/lunar_fast/audit/calibrated_pair.json)
records those per-controller results. The
[2,400-step slow checkpoint](../reports/lunar_fast/audit/calibrated_slow_2400.json)
still scored 20/20 under the old validation flag rule, yet fired the upward
engine on 39/42 high/rising training states where the expert kept it off. Its
one old-score training miss has not been checked against physical contacts.
More adversarial updates are therefore not an established durability
improvement. The bounded 1,200/400 budget is supported by the fixed-data
action-calibration audit; it is **not** a claim that 2,400 updates are stable.

## Contact-score correction and fresh full run

The first calibrated full run selected the slow-1,200 → fast-400 path, then
scored `94000:94030` with a cached leg-contact flag. Its **historical flag
scores** were 20/20 for both on validation; on test, slow was 28/30 at 217.43
flagged-success steps and fast was 29/30 at 186.45. The three misses were
labeled `incomplete_landing`. These numbers remain in the
[archived cached-flag report](../reports/lunar_fast/audit/contact_correction/report.json)
and must not be read as physical landing failures.

The terminal contact audit found both legs actually touching terrain in all
three cases: slow seeds 94006 and 94029, and fast seed 94020. On fast 94020,
Box2D `EndContact` cleared leg 0's cached ground flag at step 169 while an
enabled terrain contact remained. The
[terminal event trace](../reports/lunar_fast/audit/contact_correction/event_trace.json)
records that mismatch. A
[frozen-checkpoint replay](../reports/lunar_fast/audit/contact_correction/frozen_rescore.json)
of the same 60 flights scored **30/30 physical contacts for slow and 30/30 for
fast**, with 1.222× paired speed. This is a scoring correction on existing
policies, **not** a retrained full-pipeline result. The old 94000 cohort is
now inspected, so the fresh full run treats it as a regression cohort.

The fresh [corrected report](../reports/lunar_fast/report.json) comes from a
new end-to-end run, not the frozen replay. Correct contact labels made all
96 slow and 96 fast expert training flights successful. All 96 same-reset
pairs met the faster-fast criterion; policy training used 21,727 slow and
18,162 fast physical transition rows. The optimizer, loss, 1,200 slow and
400 fast RpGAN update budgets, and live-weight policy selection stayed the
same. Fast round 3 won validation selection. The run finished in 71.8 seconds
on CPU.

| Fresh corrected controller | Validation landings / mean steps | 94000 regression landings / mean steps |
| --- | ---: | ---: |
| Slow | 20/20 / 210.35 | 30/30 / 219.83 |
| Fast | 20/20 / 184.35 | 30/30 / 186.77 |

On all 30 matched 94000 resets the newly trained fast policy landed sooner:
**1.177×** paired speedup and 32 median steps saved. This is a completed
full-pipeline measurement with corrected scoring. Because 94000 was already
inspected during diagnosis, it is a fixed regression cohort here, not a new
untouched test. The older frozen-checkpoint rescore also reached 30/30 for
both policies, but involved different weights and a 1.222× paired speedup;
it must not be substituted for this fresh-run result.

The old 84000 and 94000 results use different resets and policy versions;
their numerical difference is not a paired before/after improvement estimate.

Earlier scratch recipe metadata showed `ema_decay=0.995`, but none of these
training loops ever updated or selected EMA weights. The live policy weights
produced every reported flight. Current code sets unused recipe EMA metadata
to zero so the saved configuration describes that fact accurately.
