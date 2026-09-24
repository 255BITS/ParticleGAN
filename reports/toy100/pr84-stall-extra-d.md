# Stall-gated extra critic step on #107

One mechanism. When #107's stall predicate is true (critic slope ≥ .6 and the generator's own-curvature trust averages ≤ .1 over the last 50 updates), take one more discriminator Adam step on that same batch and the same D learning rate, after the curvature bound has set D* and before the generator step. Otherwise the update is #107. Reach, the .5 fallback, and both curvature bounds are unchanged.

Host: this VM, seed 0, one thread. Torch 2.14.0+cu130. The rankable pin is

`CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2`.

`torch.backends.cpu.get_cpu_capability()` reported AVX2 on both runs below. Receipts: [continuous-evidence/pr84-stall-extra-d](continuous-evidence/pr84-stall-extra-d/).

## Why this and not a retune

#107 widens G's critic read under stall. #121 and #122 change G's step under the same signal. This bet strengthens the critic instead, with a real Adam update, not a virtual look-ahead. It is not #105 (different predicate), not #102 (not always on), and not #120 (no separate acquire detector).

## Gates

| Gate | PR84 pin | #107 stall reach | **Extra D** |
| --- | --- | --- | --- |
| Warm control | 200/200 on AVX2 | 200/200 | **200/200**, min HQ .990, final 8 / .999 |
| Warm 1001–1200 | 196/200 | 200/200, min HQ .921, final 8 / .981 | **200/200**, min HQ .921, final 8 / .981, **0 extra steps** |
| Cold trajectory | PASS | PASS | **PASS** (identity MSE .00094), 0 extra steps |
| Cold ring | 8 | 8, suffix 8, terminal HQ 1.0 | **PASS**, 8 modes, suffix 7, terminal HQ .973, confirmed at 1100. 30 extra steps, last at update 651. Dip to 4 modes at 700–850, then back |
| Stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25, 0-mode at 1770 and 1820 | **97/120**, final 8 / .996, suffix 10, min modes 1, **no 0-mode check**. Nine checks at ≤4 modes (1720–1860). Failures through 2300. Still 30 extra steps, none after 651 |

Warm is bit-identical to #107 because the predicate never fired on that solved continuation. Cold trajectory is the non-2D host, so the 2D stall read never arms and the extra step never runs.

## Unrankable warm

With only `ATEN_CPU_CAPABILITY=avx2`, capability still printed AVX2, but the unchanged control was 0/200, min modes 6, min HQ .953, final 6 / .999. That is the AVX512 6-mode warm, not the pin. It was not used to rank. Adding the MKL and oneDNN AVX2 caps reproduced the 200/200 control.

## Why it does not move the stay

The dropout the board is chasing does not sit inside this predicate. The #107 trace has D's slope at .29–.43 while the common-mode translation grows, and the slope hits the .6 line only once the cloud has already dropped. In this run the extra critic step fired 30 times, all by update 651 (12 in the first 100 updates, 10 around 300–400, 8 around 500–650). From 652 through 2400 the predicate was false, including the whole 1690–2300 episode. The stay difference versus #107 is the downstream of those early critic steps, not a reaction during the collapse.

Same number of failed checks (23/120). The exact 0-mode samples are gone, and the final HQ is higher (.996 vs .971), but the clean suffix shrinks from 25 to 10 and failures continue past 2150 (2270 and 2300). Min modes is 1, at update 1860. That is the same unsolved episode, not a shorter one.

## Keep / kill

**Kill.** Do not promote. Do not retune the step count, the learning rate, or the predicate.

Keep #107 stall reach as the GAN-native reference. Reach .5 stays the fallback (103/120 stay, no AVX512 ring).

## Leaderboard (this line only)

| Rank | Entry | Warm | Cold ring | Stay |
| --- | --- | --- | --- | --- |
| 1 | #107 stall reach | 200/200 | 8, suffix 8 | 97/120, final 8 / .971, suffix 25 |
| — | Reach .5 (fallback) | 200/200 | 8 on AVX2, 7 on AVX512 | 103/120, final 8 / 1.0 |
| kill | Stall-gated extra D | 200/200 (idle) | 8, suffix 7 | 97/120, final 8 / .996, suffix 10 |

Not a 22/22 result and not a production candidate.

## Recommendation

Do not spend another bet on a D-side reaction to this stall predicate. It is off during dropout onset, and turning it on during acquisition (where it does fire) does not buy a cleaner stay. The open G-side bets (#121, #122) are the ones aimed at the fast common-mode motion. A new detector would repeat #120.
