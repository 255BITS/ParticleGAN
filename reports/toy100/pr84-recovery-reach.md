# Recovery reach: KILL

Draft only. Not a 22/22 result and not production-ready. Host is #107 stall reach.
CPU pin: `ATEN_CPU_CAPABILITY=avx2`, PyTorch capability `AVX2`, torch 2.14.0+cpu, seed 0, one thread.
Receipts: [continuous-evidence/pr84-recovery](continuous-evidence/pr84-recovery/).

## Mechanism

Stall reach is the width except on a hard dip. Support is the mean critic gap between the clean particles and one fixed real probe (seed 0, 128 draws from the ring the critic already trains on). A hard dip is that gap falling at least 0.15 below a short EMA (alpha 0.2), updated once per outer step. On those steps G's stencil width is the fixed recovery reach **0.75**. When the gap is no longer that far under the EMA, the width returns to stall reach. G's curvature and trust factor are not inputs. No coverage loss, Chamfer assignment, mode quota, clip ladder, or rho gate.

The 0.15 drop was read once from the unchanged stall-reach trace, where it led every post-1200 zero-mode step and did not fire on late healthy 8-mode windows. It was not retuned after this kill. Local critic maxima were measured and discarded: they sit near zero on a full ring.

## Gates

Unchanged warm control is 200/200, so the warm gate is rankable.

| Gate | #84 pin | #107 stall reach | Recovery reach |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921 |
| Cold trajectory | PASS | PASS | **PASS** |
| Cold ring (AVX2) | 8, suffix 5 | 8, terminal HQ 1.0 ×5, suffix 8, 10/24 | **FAIL** 6/24, suffix 4. Terminal 8 / .997, but update 1000 is 8 / **.572** |
| Continued stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, hard 0-mode dips | not run (cold ring failed) |

## Why kill

Recovery fired on **115 of 1200** cold-ring updates, including updates 3–7, before any ring exists. The wider stencil is therefore not confined to the continued-training collapse. The acquire gate loses the sustained suffix: six passing checks and a suffix of four, against stall reach's ten and eight. Update 1000 is the counterexample (full mode count, HQ .572). Warm matching #107 does not offset that.

Stay was not ranked. A cold ring that already misses the sustained HQ bar is not a better acquire, and the width was not retuned.

## Leaderboard (this bet only)

1. **#107 stall reach** — still the GAN-native reference. Warm 200/200, trajectory pass, ring 8 on the AVX2 pin, stay 97/120 ending 8 / .971. Hard 0-mode dips around 1720–2150 remain.
2. **#84 smoothed critic** — fail-fast pin. Weaker warm and a degenerate continued cloud.
3. **Recovery reach** — killed on the cold ring.

## Recommendation

Do not sweep recovery width or the 0.15 drop. A critic-gap fall is common while the cloud is still acquiring, so this on/off switch spends its wider reach too early. The unsolved stay problem is still #107's hard mode loss after a full ring, and it needs a signal that stays off until that ring is actually held.
