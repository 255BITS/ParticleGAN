# Delayed-arm G Adam lr ×0.5

One mechanism on the PR #107 stall-reach pin. Draft only. Not a 22/22 result and not a production claim.

Host: neural, seed 0, one thread. Pin `ATEN_CPU_CAPABILITY=avx2` (torch 2.14.0+cu130, capability AVX2). AVX512 is a build check. Runner: [gan_followup_probe.py](gan_followup_probe.py) method `delayg05`. Code: [pr84_delayed_arm_g_lr.py](pr84_delayed_arm_g_lr.py). Logs: [continuous-evidence/delayed-arm-g-lr](continuous-evidence/delayed-arm-g-lr/).

## Mechanism

Stall reach is unchanged (width, D, losses, G own-curvature bound .25, pre-arm Adam rates) until the first diagnostic with **update_index ≥ 1200**, **modes ≥ 8**, and **HQ ≥ .9**. That arm sticks. Each later G Adam step sets both G-optimizer param groups to **0.5 × the pre-arm rate** (network 0.00425 → 0.002125, prior 0.0085 → 0.00425). D is not scaled. The curvature bound is not changed. The post-bound placement is not scaled again.

The half rate is what Adam and the own-curvature metric read. It is restored when that call returns, so the host schedule does not compound it. Every apply logs `armed`, `update_index`, `g_lr_before`, `g_lr_after`. The cold ring's first 8/.9 is at step 650 and does not arm.

## Gates

Identity warm is 200/200, so the warm rank stands. CPU capability is on every receipt line.

| Gate | PR84 | #107 stall reach | **Delayed-arm G lr ×0.5** |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921 (same float .920654). Arm at 1200, **0 fires** |
| Identity warm | 200/200 | 200/200 | **200/200**, min HQ .990 |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (3.6s), not armed |
| Cold ring (AVX2) | 8, suffix 5 | 8, 10/24, suffix 8 | **8, 10/24, suffix 8**, final HQ .999. Arm at 1200, **0 fires** |
| Cold ring (AVX512, build check) | 7 | 8, 8/24, suffix 8 | **8, 8/24, suffix 8**, final HQ .998. **0 fires** |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25, min 0 | **114/120**, final **8 / 1.0**, suffix 14, min modes **5**, min HQ .363 |

Stay fires: 1200, updates 1201–2400 only. Failing checks: 1820 (8 / .881), 1850 (8 / .881), 2180 (7 / .850), 2190 (5 / .363), 2200 (8 / .782), 2260 (8 / .891). No check at or under 4 modes. #107 had whole-cloud dropouts to 0 and 23 failing checks, and no failures after 2150.

## Why the stay moved

#107's 1755 fork was G-led translation: G steps ×.5 passed 28/29, D steps ×2 passed 0/29. #125 armed this same G Adam half at the first 8/.9 (~650) and lost the cold ring. Delaying the arm to 1200 leaves acquire on the stall-reach path (warm min HQ matches #107; cold fires stay 0) and spends the half rate only after the ring exists.

The half rate cuts the dropout's depth (min modes 0 → 5) and raises the pass count (97 → 114), and the run ends on a perfect ring. It does not clear the episode. The only mode-losing check is step 2190 (5 modes, HQ .363). There G's own-trust factor is ~.21 and D's slope is ~.89, so stall reach opens the critic stencil from .15 to ~.45 while the half Adam rate is already on. The episode also shifts later: the passing suffix is 14, against #107's 25.

## Keep / kill / next bet

- **Keep.** Warm does not regress against #107 (200/200) or PR84 (196/200). Cold acquire matches stall reach. Stay is the best constant-rate continuation on this line (114/120, final 8/1.0, no ≤4-mode dip).
- **Not solved.** Six of 120 stay checks still fail. Do not call the ring held, and do not claim 22/22 or production readiness.
- **Do not retune** ×0.5 or the 1200 index. Do not stack #138's post-arm curvature bound .125, extra D, mean restore, trust reject, mode freeze, or common-mode null.
- **Next bet, one mechanism:** after this same 1200 / 8 / .9 arm, hold G's critic width at .15. The residual dip is the stencil opening (.15 → ~.45) under a trust factor ~.21, and acquire no longer needs that opening. Skip it if #138 already clears the late dip.
