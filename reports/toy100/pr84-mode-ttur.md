# Post-arm mode-count TTUR on #107

One mechanism. After the first ring read with 8 modes and HQ ≥ 0.9, any later update whose current mode count is ≤ 6 takes two extra discriminator Adam steps on that same batch, at the same D loss, D learning rate, and Adam betas, after D* and before G. Otherwise the update is #107. Reach, both curvature bounds, and G's step are unchanged. G is not frozen, rejected, or halved.

The ring read is the host diversity score on a fixed 4096-draw through the unwrapped generator. It is a cadence gate, not a coverage loss, anchor, assignment, or clip. Train RNGs are not advanced by the read.

Host: this VM, seed 0, one thread. Torch 2.14.0+cpu. The rankable pin is

`CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2`.

`torch.backends.cpu.get_cpu_capability()` reported AVX2 on the ranked runs and AVX512 on the ring build check. Receipts: [continuous-evidence/pr84-mode-ttur](continuous-evidence/pr84-mode-ttur/).

## Why this and not a retune

#107 still drops to 0 modes in the continued run (about 1770 and 1820) and comes back. #123's extra critic step used the stall predicate (D slope ≥ .6 and mean G trust ≤ .1) and never fired in 1690–2300. This bet uses the mode count after arm instead. It does not freeze G (#129), halve G (#125/#132), or reject G on trust (#130/#133).

Thresholds were not retuned. ≤ 6 and +2 D are the only settings that were run.

## Gates

| Gate | PR84 pin | #107 stall reach | **Mode-count TTUR** |
| --- | --- | --- | --- |
| Warm control (AVX2) | 200/200 | 200/200 | **200/200**, min HQ .990, final 8 / .999 |
| Warm 1001–1200 | 196/200 | 200/200, min HQ .921, final 8 / .981 | **200/200**, min HQ .921, final 8 / .981. Armed at 1000. **0 fires** |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (identity MSE .00094). Never armed. 0 fires |
| Cold ring (AVX2) | 8, suffix 5 | 8, terminal HQ 1.0, suffix 8 | **PASS**, 8 modes, suffix 11, terminal HQ .965, confirmed at 900. Armed at 569. **102 fires** (204 extra D steps), last at 729 |
| Cold ring (AVX512) | 7 | 8, suffix 8 | **PASS**, 8 modes, suffix 8, terminal HQ .998. Armed at 811. **0 fires** |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25, 0-mode at 1770 and 1820 | **82/120**, final **0 / 0**, suffix **0**, min modes 0. **377 fires** (754 extra D steps) through update 2399 |

Warm is the #107 continuation because modes stayed at 8, so the extra steps never ran. Cold trajectory has no ring, so the cadence never arms.

## What the fires did

The AVX2 cold ring arms at 569, the same step #129 armed on. Modes then fall through 6 and the gate fires on 102 updates, ending at 729, including reads of 0–2 modes. The official checkpoints from 700 on are 8 modes. The ring still passes, with a longer suffix than #107 and a lower terminal HQ (.965 vs 1.0).

On the continued run the same arm is in force. Fires:

| Window | Fires |
| --- | --- |
| 569–1200 | 102 |
| 1210–1689 | 89 |
| 1690–2300 | 87 |
| 1770–1820 | 0 |
| 2200–2400 | 186 |

The old 1770/1820 checks are not 0-mode here (they read 8 modes; HQ fails the .9 bar at 1780 and 1800). The gate is quiet there because the count stays above 6. From 2210 the cloud leaves the ring, hits 0 modes at 2250, and is still at 0 at 2400. Extra D steps are on for that whole collapse (186 fires, many of them at 0 modes) and do not bring it back.

AVX512 never sees ≤ 6 modes after arming at 811, so that ring is stall reach again.

## Keep / kill

**Kill.** Do not promote. Do not retune the ≤ 6 threshold or the +2 D count.

Keep #107 stall reach as the GAN-native reference. Reach .5 stays the fallback (103/120 stay, no AVX512 ring).

## Leaderboard (this line only)

| Rank | Entry | Warm | Cold ring | Stay |
| --- | --- | --- | --- | --- |
| 1 | #107 stall reach | 200/200 | 8, suffix 8 | 97/120, final 8 / .971, suffix 25 |
| — | Reach .5 (fallback) | 200/200 | 8 on AVX2, 7 on AVX512 | 103/120, final 8 / 1.0 |
| kill | Post-arm mode-count TTUR | 200/200 (idle) | 8, suffix 11, terminal HQ .965 | 82/120, final 0 / 0, suffix 0 |

Not a 22/22 result and not a production candidate.

## Recommendation

The mode-count predicate does fire in the continued episode, which the stall predicate did not. Two extra critic steps at that moment do not reopen a ring G can stay on. The run that fires through a 0-mode cloud ends on that cloud. Another D-step count or a different ≤ N cutoff would be a retune of a failed response, not a new signal.
