# DRAFT: delayed-arm extra D TTUR

Not solved. Do not merge. No 22/22 claim and no production-readiness claim.

One mechanism on the PR #107 stall-reach pin. Host: neural, seed 0, one thread, PyTorch 2.14.0+cpu. Ranked runs use

`ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2`

with one thread. CPU capability is on every log line. Runner: [gan_followup_probe.py](gan_followup_probe.py) method `delayextrad`. Code: [pr84_delayed_arm_extra_d.py](pr84_delayed_arm_extra_d.py). Logs: [continuous-evidence/delayed-arm-extra-d](continuous-evidence/delayed-arm-extra-d/).

## Mechanism

Stall reach is unchanged until the arm (G own-curvature .25, G Adam lr, D rates, losses, D curvature bound 3). The arm sticks on the first diagnostic with **update_index ≥ 1200**, **modes ≥ 8**, and **HQ ≥ 0.9**. An 8/.9 before 1200 does not arm. After the arm, each training turn runs **+2 host D Adam steps** on that turn's batch, after the bounded host D* and before G. Pre-arm D step count is unchanged.

Not used: G Adam lr scaling, a curvature-bound change, a mode-count latch, a stall predicate, G×0.5, common-mode null, EMA-D, freeze.

Every apply logs `armed`, `update_index`, `modes`, `HQ`, `extra_d_steps_this_turn`.

## Gates

Identity warm is 200/200, so the warm rank stands. An earlier run that set only `ATEN_CPU_CAPABILITY=avx2` reproduced the stored AVX512 prefix (identity 0/200, min modes 6, min HQ .953125) and was not ranked.

| Gate | PR84 | #107 stall reach | **Delayed-arm +2 D** |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .920654. Arm at 1200, **0 fires** |
| Identity warm | 200/200 | 200/200, min HQ .990 | **200/200**, min HQ .989990 |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (3.4s), not armed, 0 fires |
| Cold ring (AVX2) | 8, suffix 5 | 8, 10/24, suffix 8 | **8, 10/24, suffix 8**, final HQ .999. First 8/.9 is step 650 (HQ .917) and does not arm. Arm at 1200, **0 fires** |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25, min 0, recovers | **54/120**, final **0 / 0**, suffix **0**, min 0. Last pass 2080. **0 modes from 2120 through 2400** |

## Fires

Arm at update 1200. Applies are updates 1201–2400 only (1200 fires, 2400 extra D steps). None before 1200.

| Window | Fires |
| --- | --- |
| Acquire (before 1690) | 489 |
| Dropout window 1690–2300 | 611 |
| After 2300 | 100 |

#123's stall gate fired 30 times by update ~651 and never inside 1690–2300. This arm does the opposite: zero fires before 1200, and 611 fires inside the dropout window. The cloud still dies there.

## Why this is a kill

Warm and cold match stall reach, including the warm min HQ float .920654 and the ring's 10/24 with suffix 8. The 8/.9 at step 650 does not arm. Pre-arm dynamics are the #107 path.

Stay is not. Extra D during the acquire window already cuts the ring: step 1220 is 4 modes, step 1330 is 0. The shared dropout does not recover. From 2120 to 2400 every check is 0 modes. #107's continued run returns to 8 / .971 with no failures after 2150. #84 ends that run at 6 / .904. This run ends at 0 / 0, with a pass count (54/120) next to #84's 53/120 and well under #107's 97/120.

Putting +2 D steps into the dropout window does not hold the ring. The same terminal 0/0 showed up under #135's mode-count latch. The calendar arm avoids murdering the cold ring, which the early 8/.9 arm did, and it still does not stay.

## Keep / kill

- **Kill** delayed-arm extra D TTUR. Do not retune +2 or the 8/.9/1200 arm.
- **Keep #107** stall reach as the acquisition reference. This bet does not beat it on stay and does not change warm or cold.
- **Not solved.** Do not call the ring held. Do not claim 22/22 or production readiness.
