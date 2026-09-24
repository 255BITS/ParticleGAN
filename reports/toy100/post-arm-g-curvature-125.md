# KILL: post-arm G own-curvature bound .125

Draft only. Not solved. Not a 22/22 claim. Not production-ready.

Host: neural, seed 0, one thread, PyTorch 2.14.0+cpu. Valid runs use the repo AVX2 pin
`ATEN_CPU_CAPABILITY=avx2` plus `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`,
`DNNL_MAX_CPU_ISA=AVX2`, and one BLAS thread. AVX512 is a build check. Receipts:
[continuous-evidence/post-arm-g125](continuous-evidence/post-arm-g125/).

## Mechanism

Stall reach (#107) is unchanged until the first logged ring check with modes ≥ 8 and HQ ≥ 0.9.
After that sticky arm, G's own-curvature cap in `min(1, c/ρ)` is 0.125. Before the arm it stays 0.25.
D's cap, the stall-reach width rule, and the losses are untouched. This is the trust-region radius,
not an Adam step scale.

Every G placement logs `G_BOUND` with `armed`, `g_bound_before`, `g_bound_after`, the host update,
and the running `post_arm_fires` count.

## Why this was the bet

The #107 continuation dropout is G-led. From the shared 1755 state, G steps ×0.5 passed 28/29 checks
and D steps ×2 passed 0/29. An always-on G cap of 0.125 on that same table killed the cold ring
(AVX2 ended at 2 modes; stay 30/120) because it also slowed acquisition. The arm was supposed to
apply 0.125 only after the ring existed.

## Gates

| Gate | PR84 pin | #107 stall reach | **This bet** |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200, min HQ .961** |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (never arms; cap stays .25) |
| Cold ring (AVX2) | 8, suffix 5 | 8, terminal HQ 1.0, suffix 8 | **FAIL: live 6 / .444, 4/24, suffix 0** |
| Cold ring (AVX512, build check) | 7 | 8, suffix 8 | PASS: live 8 / .937, 8/24, suffix 8 |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25, min modes 0 | **107/120, final 8 / .805, suffix 0, min modes 6** |

Identity on the valid warm fork is 200/200 (min HQ .990), matching the #107 AVX2 control.
A first warm with only `ATEN_CPU_CAPABILITY=avx2` reproduced the AVX512 identity artifact
(0/200, min modes 6, min HQ .953) and was refused. It is not a rank.

## What the arm actually did

The first 8/.9 check is not a settled ring.

- AVX2 cold ring, 50-step log: step 600 is 4 modes / HQ .20, step 650 is 8 / .917. The arm sticks at 650. The next 550 G steps use cap .125 (`post_arm_fires` 550). The live ring ends at 6 / .444. EMA is 8 / .837, so the running mean looks better than the particles G is training.
- Stay uses the probe's 10-step checks, so the same graze arms earlier, at update 570 (`post_arm_fires` 1830). By step 1200 the cloud is 7 modes / HQ .47. Later, 107 of 120 checks pass and nothing falls to 0 modes (worst check is 6 / .66 at 2210). The run does not finish on the ring: the last check is 8 / .805, suffix 0.
- AVX512 arms later (update 850) and the ring holds (suffix 8). That does not repair the AVX2 pin.
- Warm arms at the step-1000 prefix check, when the scheduled run is already solved. All 200 continuation steps use .125. That is why warm matches the old always-on-.125 warm (min HQ .961) and still clears 200/200.

So the cap does what the 1755 fork suggested once the cloud is already up: the empty-cloud dropout is gone, and more stay checks pass than #107 (107 vs 97). It does that by clamping G during the acquire transient, and on AVX2 that transient never becomes the #107 ring.

## Leaderboard (this question only)

1. **#107 stall reach** — still the reference. AVX2 and AVX512 rings both end at 8, and the stay ends at 8 / .971.
2. **This bet** — warm holds and the stay is shallower, but the AVX2 ring fails and the stay suffix is 0.
3. **PR84 pin** — warm 196/200, stay ends degenerate at 6 modes.

## Call

**Kill.** Do not sweep 0.125. Do not stack it with G×0.5, extra D, mean restore, or a mode freeze.
Neural acquire on the AVX2 pin does not clear, so the continuous-learning bet is not solved.

**Next bet (one):** keep the #107 cap of .25 through update 1200, and clamp G's own-curvature cap to .125
only after that budget. The dropout that G×0.5 removed sits in 1720–2150, which is after #107 has
already acquired. A mode/HQ arm is the wrong trigger: this run shows the first 8/.9 tick is still
an acquire graze. No Adam step scale, no coefficient change, no extra critic steps.
