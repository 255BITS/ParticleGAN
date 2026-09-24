# Mode-drop G freeze on stall reach — KILL

Host: stall reach from `cursor/gan-native-followup-f60e` (PR #107). Seed 0, one thread, PyTorch 2.14.0+cpu. Warm and the AVX2 gates use `ATEN_CPU_CAPABILITY=avx2`. The AVX512 cold ring is the same run with the capability pin set to `avx512`, not a second seed. Logs: [continuous-evidence/pr84-mode-drop](continuous-evidence/pr84-mode-drop/).

This is not a 22/22 result and it is not production-ready.

## Mechanism

One change. After the run first reads an 8-mode ring at HQ ≥ 0.9 (armed), each later update with **≤ 6 modes** skips the G+prior parameter write. D still steps. Writes resume only at **≥ 7 modes**. Before arming, the update is stall reach: same reach width, stall predicate, curvature bounds, rates, and D step count.

The read is the host diversity score on a fixed 4096-draw through the unwrapped generator. It does not enter the loss. Adam's moment counter still advances so the probe's G/D counters stay locked; the weights are copied back to the pre-update snapshot.

## Gates

Unchanged warm control (identity, AVX2) is **200/200**, min HQ .990. The comparison is valid.

| Gate | PR84 pin | #107 stall reach | Mode-drop G freeze |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921. Armed at step 1000. **0 G skips** |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (final identity MSE .00094, suffix 18). Never armed |
| Cold ring (AVX2) | 8, terminal HQ .988–.999, suffix 5 | 8, terminal HQ 1.0, suffix 8 | **FAIL 0/24**. Final 8 / .645. Armed at 569 (HQ .917), freeze from update 574 through 1200 (**627 skips**, no resume) |
| Cold ring (AVX512) | 7 | 8, terminal HQ .92–1.0, suffix 8 | **PASS 8/24, suffix 8**, final 8 / .998. Armed at 811. **0 G skips** (freeze never fired) |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, dips to 0, suffix 25 | **0/120**. Final 8 / .643, min HQ .637. **1827 G skips**, updates 574–2400, no resume. All **611** updates in 1690–2300 are skips |

## What happened

Warm matches #107 because the continuation never left 8 modes, so the freeze never fired.

On AVX2 the cadence armed during acquisition. At update 569 the clean draw read 8 modes at HQ .917. Four updates later it read 5 modes at HQ .588, and G+prior weights were restored on every update after that. The official 50-step checkpoints never pass: the run ends at 8 modes with HQ stuck near .64. The resume rule (≥ 7) never fired. With G and the particles frozen, the quantity the rule watches cannot climb.

The 1690–2300 window is not a recovery test on this run. G had already been frozen since 574, so those 611 skips are the same latch, not a response to #107's 0-mode dips.

On AVX512 the same rule armed at update 811 and then never saw ≤ 6 modes, so it did not change the ring. That ring matches #107's AVX512 result. The mechanism did not improve it.

## Call

**KILL.** Do not retune the ≤ 6 / ≥ 7 cut. #107 stall reach stays #1 on the GAN-native board. Its stay is still the unresolved 1720–2150 episode; this cadence traded that episode away by freezing acquisition on AVX2.
