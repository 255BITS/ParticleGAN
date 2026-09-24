# Delayed-arm critic width held at .15

One mechanism on the PR #140 delayed-arm pin (itself on PR #107 stall reach). Draft only. Not a 22/22 result and not a production claim.

Host: neural, seed 0, one thread. Pin `ATEN_CPU_CAPABILITY=avx2` (torch 2.14.0+cpu, capability AVX2 on every receipt). AVX512 is a build check. The acquire floats match the #140 cu130 receipts (warm min HQ .920654296875, cold AVX2 tail unchanged). Runner: [gan_followup_probe.py](gan_followup_probe.py) method `holdw15`. Code: [pr84_delayed_arm_width_hold.py](pr84_delayed_arm_width_hold.py). Logs: [continuous-evidence/delayed-arm-width-hold](continuous-evidence/delayed-arm-width-hold/).

## Mechanism

#140 is unchanged until the arm: first diagnostic with **update_index ≥ 1200**, **modes ≥ 8**, and **HQ ≥ .9**. The arm sticks. Later G Adam steps stay at 0.5 × the pre-arm rates (network 0.00425 → 0.002125, prior 0.0085 → 0.00425). D is not scaled. The curvature bound stays .25.

Pre-arm stall reach is unchanged. Width is `max(.15, .5·min(s, 1/s))`, and it is **.5** when D's slope is ≥ .6 and G's own-trust average over the last 50 updates is ≤ .1.

After the arm, G's five-point critic width is **.15 on every phase**. That replaces both the stall override of .5 and the peak opening. The #140 step-2190 dip was the peak opening, not the stall override: slope ~.89 and trust ~.21, so `_stalled` was false and the recorded width was ~.446. Holding only the stall branch would have left that opening in place. The hold sticks for the rest of training.

Every post-arm decision logs `width_hold_apply` with `armed`, `update_index`, `width_before`, `width_after`, `fires`, and `phase`. `grep width_hold_apply` tails the new lines; `g_lr_apply` is unchanged.

## Gates

Identity warm is 200/200, so the warm rank stands. CPU capability is on every receipt line. Cold ring and warm have **0 width fires** (the arm is the diagnostic at step 1200, after that update's width was chosen).

| Gate | PR84 | #107 stall reach | #140 delayed-arm G lr ×0.5 | **Width hold .15** |
| --- | --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | 200/200, min HQ .921 | **200/200**, min HQ .921 (same float .920654). Arm at 1200, **0 width fires**, 0 G-lr fires |
| Identity warm | 200/200 | 200/200 | 200/200, min HQ .990 | **200/200**, min HQ .990 |
| Cold trajectory (AVX2) | PASS | PASS | PASS (3.6s) | **PASS** (3.5s), not armed |
| Cold ring (AVX2) | 8, suffix 5 | 8, 10/24, suffix 8 | 8, 10/24, suffix 8, HQ .999 | **8, 10/24, suffix 8**, final HQ .9988. **0 width fires** |
| Cold ring (AVX512, build check) | 7 | 8, 8/24, suffix 8 | 8, 8/24, suffix 8, HQ .998 | **8, 8/24, suffix 8**, final HQ .9980. **0 width fires** |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25, min 0 | 114/120, final 8 / 1.0, suffix 14, min modes 5, min HQ .363 | **115/120**, final **8 / 1.0**, suffix **2**, min modes **7**, min HQ **.622** |

Stay fires from update 1201. Width applies: 3600 (3 phases × 1200 updates). G-lr fires: 1200. Post-arm recorded width is .15 on all 1200 updates. Of the 3600 applies, 646 replaced a wider stall-reach value, including 95 exact-.5 stall overrides, all on the curvature phase. The Adam phase's pre-hold width never exceeded .187.

Failing checks: 1560 (8 / .806), 1810 (7 / .622), 2110 (8 / .764), 2210 (8 / .867), 2380 (8 / .895). Each recovers by the next check. #140's 2190 collapse (5 / .363) is 8 / .933 here. No check under 7 modes.

## Why the stay moved

#140's deep dip was the stencil opening to ~.45 while the half Adam rate was already on. After the arm that opening is gone, and the run no longer loses the cloud down to 5 modes. The pass count moves 114 → 115 and the worst check moves from 5 / .363 to 7 / .622. The run still ends on a perfect ring.

The new misses are single checks. At all five, the Adam phase was already width .15. The curvature phase was wider on four of them (1810 would have been .211, 2110 .257, 2210 .262, 2380 the stall value .5) and was held at .15. Step 1560 was already .15 on every phase, so that miss is the changed path, not a clamp on that update. The passing suffix falls from 14 to 2 because 2380 slips to HQ .895 and the next two checks pass.

## Keep / kill / next bet

- **Keep.** Warm does not regress (identity 200/200, method 200/200, same min HQ as #140). Cold trajectory and both cold rings match stall reach / #140, with 0 width fires. Stay is a full ring at the end, and the degeneracy is shallower (min modes 5 → 7, min HQ .363 → .622, the 2190 five-mode drop is gone). It is not a fake hold of an incomplete cloud.
- **Not solved.** Five of 120 stay checks still fail. Do not call the ring held, and do not claim 22/22 or production readiness.
- **Do not retune** .15, ×0.5, or the 1200 index. Do not stack a curvature bound, extra D, mean restore, common-mode null, or a mode-drop freeze.
- **Next bet, one mechanism:** keep this .15 hold on G's Adam read, and stop applying it to the curvature phase. That phase is the one that still wanted to open (2380 would have been .5; 2110 and 2210 were ~.26), and the Adam phase was already .15 at every remaining miss.
