# Post-acquire G Adam ×0.5 on stall reach — KILL

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Pin: `ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. Logged CPU capability: AVX2. The warm identity fork is 200/200 (min HQ .990, final 8 / .999), the same prefix as #107, so this warm is ranked. Receipts: [continuous-evidence/post-acquire-ghalf](continuous-evidence/post-acquire-ghalf/).

**Not solved. Not production. Draft only.**

## Mechanism

#107 stall reach is unchanged until an existing offline ring check reports 8 modes and HQ ≥ 0.9. That check is the live mode counter already used for ring/HQ logging. It is not a loss. The flag then stays armed, and each later Generator Adam displacement is multiplied by 0.5 (the fork-winning G step at update 1755). D steps, both curvature bounds, and the losses are untouched. Before the flag, the update is stall reach.

Trajectory never logs ring modes, so it never arms.

## Gates (AVX2)

| Gate | PR84 pin | #107 stall reach | **This bet** |
| --- | --- | --- | --- |
| Warm 1001–1200 | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .934, final 8 / .999 |
| Cold trajectory | PASS | PASS | **PASS**, MSE .00094, never armed |
| Cold ring | 8, suffix 5, terminal HQ .988–.999 | 8, suffix 8, terminal HQ 1.0 | **FAIL**, 1/24 pass, suffix 0, terminal 6 / .653 |
| Stay 1210–2400 | 53/120, final 6 / .904 | 97/120, final 8 / .971, 0-mode dips ~1770/1820 | not run |

Warm identity was 200/200, so the run is not an AVX512 6-mode prefix. AVX512 was not run. Stay was not run.

## Arm timing

| Run | Armed | Update | Phase | Post-arm G steps |
| --- | --- | --- | --- | --- |
| Warm continuation | yes | 1000 | warm | 200 (every method step) |
| Cold trajectory | no | — | — | 0 |
| Cold ring | yes | 650 | cold_acquire | 550 |

The cold ring's only passing check is step 650 (8 modes, HQ .917). Step 550 was already 8 modes at HQ .841, under the 0.9 line. The next check, step 700, is still 8 modes but HQ .549. The run then leaves the ring and ends at 6 modes, HQ .653. Halving G from that first logged full-ring check is what the kill rule predicted: the arm flips during cold acquire, and the ring regresses against #107.

The warm fork is the other side of the same rule. The scheduled prefix is already on the ring at update 1000, so all 200 continuation steps are halved and the hold is 200/200, slightly above #107's min HQ (.934 vs .921). That does not rescue acquisition.

## Keep / kill

**Kill.** One mechanism, no retune of 0.5, of HQ 0.9, or of the arming rule. No falling-trust, slope, or translation predicate on top.

#107 stall reach stays the reference: warm 200/200, trajectory PASS, cold ring 8 on the AVX2 pin, stay 97/120 ending on the ring. This bet does not acquire the full ring from scratch, so it does not test a post-acquire stay. The 1755 fork's G×0.5 result does not transfer onto the first logged 8 / ≥0.9 check.
