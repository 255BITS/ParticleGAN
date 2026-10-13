# Quality and retention stages

Tier 2 measures fixed-target quality. Tier 3 measures continued retention or target-change recovery. These independent measurement contracts share the original runs, budgets and numerical bounds.

Original combined tasks and recorded qualification grades remain intact. Stage cells confer no new suite qualification. An aggregate PASS or FAIL cannot fill missing stage evidence; insufficient or incompatible source-bound submetrics are UNMEASURED. Counts below are receipt cells, not independent runs.

| Stage | Tier | Original source tasks | Exact criterion | Stage cells |
| --- | ---: | --- | --- | --- |
| Gaussian fixed-target quality | 2 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | Final update 1,000: all original distribution bounds plus independent same-state confirmation. | 12 UNMEASURED |
| Gaussian stationary retention | 3 | [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | All 72 original observations from the candidate's own smoke state through update 4,000 must pass. | 2 FAIL, 10 UNMEASURED |
| Gaussian shift recovery and hold | 3 | [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | Original mean shift 2 to 3; five passing deadline-terminal checks by update 5,000 and all 24 subsequent checks. Original matched frozen control remains diagnostic. | 2 FAIL, 10 UNMEASURED |
| 100G fixed-target distribution quality | 2 | [grid100](../../configs/forge/tasks/grid100.json), [rotated100](../../configs/forge/tasks/rotated100.json), [staggered100](../../configs/forge/tasks/staggered100.json) | Original independent 100,000-draw final live holdout: complete coverage and all four fidelity limits, with a valid target oracle. | 6 FAIL, 30 UNMEASURED |
| 100G terminal retention | 3 | [grid100](../../configs/forge/tasks/grid100.json), [rotated100](../../configs/forge/tasks/rotated100.json), [staggered100](../../configs/forge/tasks/staggered100.json) | All five original 20,000-draw terminal coverage/fidelity checks at updates 6,000, 6,250, 6,500, 6,750 and 7,000. This is a limited within-run window, not a new long continuation. | 36 UNMEASURED |
| Five-word confirmed joint quality | 2 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | An independently confirmed scheduled state satisfies every original generation and inverse bound; report the earliest confirmed state. | 8 UNMEASURED |
| Five-word own-checkpoint retention | 3 | [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | Original 4,000-update continuation: all 25 primary/confirmation pairs, including the exact restored acquired state, satisfy every original joint bound. | 8 UNMEASURED |
| 100G rotation recovery | 3 | Protocol not frozen | Unmeasured: rotated100 is a fixed rotated target, not a target-change event. Freeze an own-checkpoint rotation event, paired control, recovery deadline, observation cadence and continuation budget before admission. | UNMEASURED |

The JSON drilldown keeps every recipe, source, protocol/RNG/runtime cohort and receipt hash separate. A stage result never selects a different recipe or pools evidence across cohorts. Costs belong to the original run once; stages add no updates and must not be summed as new run costs.

100G terminal retention covers only the original five-check window within 7,000 updates. It does not establish long continuation or recovery from a rotation. `rotated100` is a fixed rotated target; its quality does not prove rotation recovery. That separate stage remains unmeasured until the event, own-checkpoint pairing, deadline and continuation budget are frozen.

The conditional `unipolar`, `cover_leftover` and `mid_scale_identity` tests preserve neutral/content/identity constraints across control inputs during a fresh fit. They have no later target event or acquired-state continuation and are outside this temporal split. `mode_hold` is also a fresh fit, despite its name.

[Stage contracts](../../configs/forge/test-stages.json) · [Source-bound stage drilldown](test-stages.json) · [Original combined performance](test-performance.md)

Stage input digest `781507b98352045ade029638e5c191077df97be898ddd3832e872c5b2675dc83`.
