# Post-arm G Adam β1 → 0 on stall reach — KILL

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Rankable warm pin: `ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. Logged capability on that pin: AVX2. The warm identity fork is 200/200 (min HQ .990, final 8 / .999), the same prefix as #107, so this warm is ranked. AVX512 is a build check only. Receipts: [continuous-evidence/post-arm-g-beta1](continuous-evidence/post-arm-g-beta1/).

**Not solved. Not production. Not 22/22. Draft only.**

## Mechanism

#107 stall reach is unchanged until an existing offline ring check reports 8 modes and HQ ≥ 0.9. That check is the live mode counter already used for ring/HQ logging. It is not a loss. The flag stays armed. The next real Generator Adam step sets β1 to 0 and zeroes G's `exp_avg` once (9 buffers: G network and prior). Later G steps keep β1 at 0 and do not zero again. β2, eps, lr, D Adam, both curvature bounds, and the losses are untouched.

Trajectory never logs ring modes, so it never arms.

## Why this does not move the game

The declared #107 recipe is Adam β = (0.0, 0.999) for G, the prior, and D (`prior_betas` is null). Every apply logged `g_beta1_before = 0.0`. Setting β1 to 0 writes the value the step already used. With β1 = 0 the Adam first-moment update replaces `exp_avg` with the current gradient, so wiping that buffer once, between updates, does not change the next step. The gates below match the #107 stall-reach pin, including the shared 1720–2150 dropout.

## Gates

| Gate | PR84 pin | #107 stall reach | **This bet** |
| --- | --- | --- | --- |
| Warm (AVX2) 1001–1200 | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921, final 8 / .981 |
| Cold trajectory | PASS | PASS | **PASS**, MSE .000943, never armed |
| Cold ring AVX2 | 8, suffix 5 | 8, 10/24, suffix 8 | **8**, 10/24, suffix 8, final HQ .999 |
| Cold ring AVX512 | 7 | 8, 8/24, suffix 8 | **8**, 8/24, suffix 8, HQ .918–1.0 (build check) |
| Stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, 0-mode dips | **97/120**, final 8 / .971, suffix 25, min 0 modes |

Warm did not regress against #107 (200/200) or PR84 (196/200). The fail-fast stop did not fire. Identity on the AVX2 pin is 200/200, so the warm row is rankable.

Stay still drops the cloud to 0 modes at updates 1770 and 1820. The 23 failing checks run through 2150 and then the ring returns (suffix 25). That is the #107 dropout, not a new degeneracy and not a fix. Holding a broken partial ring is not the claim here either: the run does reacquire 8 modes, and it still fails the stay.

## Arm timing

| Run | Armed | Arm update | Apply update | Phase | exp_avg zeroed | β1 before |
| --- | --- | --- | --- | --- | --- | --- |
| Warm continuation | yes | 1000 | 1001 | warm | yes, 9 buffers, once | 0.0 |
| Cold trajectory | no | — | — | — | no | — |
| Cold ring AVX2 | yes | 650 | 651 | cold_acquire | yes, once | 0.0 |
| Cold ring AVX512 | yes | 850 | 851 | cold_acquire | yes, once | 0.0 |
| Stay AVX2 | yes | 570 | 571 | cold_acquire | yes, once | 0.0 |

## Keep / kill

**Kill.** One mechanism, no retune of the arm (8 modes, HQ 0.9) and no re-enabling of β1. There is no nonzero G momentum on this pin to remove, and the one-shot `exp_avg` wipe does not change the update. #107 stall reach stays the reference: warm 200/200, trajectory PASS, cold ring 8 on AVX2 and AVX512, stay 97/120 ending on the ring, with the 1720–2150 episode still open.

Do not merge.
