# Post-arm reject on a 50-step trust opening

Host: neural. Draft only. Not solved. Not 22/22. Not production. Stall reach (PR #107) stays the reference. This bet does not displace it. Production still uses scheduled decay.

One mechanism, added to the stall-reach adapter. The 0.10 delta and the 50-step lag were not retuned.

## Mechanism

Stall reach is unchanged until an already-scheduled ring check reports 8 modes and HQ ≥ 0.9. That log only arms the mechanism. It is not a loss, and mode count is not the reject predicate.

After the arm, before each G Adam step the trainer snapshots G parameters and Adam moments, takes the normal stall-reach step (curvature bounds stay .25 / 3), and recomputes the rolling mean of G's own-curvature trust factor `factor = min(1, 0.25/ρ)` over the same 50-step window #107 uses. The step is rejected, and the snapshot restored, when that mean is at least **0.10 higher** than the 50-step mean from **50 updates earlier**. That is a window-to-window opening, not the one-step collapse ratio from #130.

D always trains. The reject is inactive before the arm. No half-step, coverage, anchor, forward-KL training signal, mode quota, or Chamfer assignment.

## Gates

Seed 0, one thread, PyTorch 2.14.0+cpu. Ranked pin: `ATEN_CPU_CAPABILITY=avx2` plus `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. Every receipt logs `cpu` and `cpu_env`. The unchanged warm control is 200/200, min HQ .989990234375, so the warm result is rankable. AVX512 is a build check.

Receipts: [continuous-evidence/g-trust-open-reject](continuous-evidence/g-trust-open-reject/).

| Gate | PR84 pin | #107 stall reach | 50-step trust-open reject |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2, control 200/200) | 196/200, min HQ .866 | 200/200, min HQ .921 | **191/200**, min modes 6, min HQ .705. Final 8 / .9995. Armed at 1000. **85 fires** |
| Cold trajectory (AVX2) | PASS, MSE .000943 | PASS | **PASS**, MSE .000943. Never armed |
| Cold ring (AVX2) | 8, suffix 5 | 8, 10/24, suffix 8 | **8, 10/24, suffix 8.** Armed at 650. **0 fires** / 550 checks. Max open .051 |
| Cold ring (AVX512, build check) | 7 | 8, suffix 8 | **8, suffix 8.** Armed at 850. **0 fires** / 350 checks |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, min 0, final 8 / .971, suffix 25 | **95/120, min 0, min HQ 0, final 8 / .9998, suffix 1.** Armed at 570. **2 fires** |

## Fire counts

| Phase | Checks after arm | Rejects |
| --- | --- | --- |
| Warm continuation 1001–1200 | 200 | 85 (continuation updates 100–184, host steps 1100–1184) |
| Cold ring after arm | 550 | 0 |
| Cold ring AVX512 after arm | 350 | 0 |
| Stay, steps 571–1200 | 630 | 0 |
| Stay, steps 1201–2400 | 1200 | 2 |
| Dropout window 1720–2150 | 431 | 2 (steps 2132 and 2137) |

The warm opening that tripped the rule is not the dropout shape. By continuation update 100 the 50-step mean is already .503 against a lag mean .238 (open .265), and it keeps climbing toward 1. Those 85 restored G steps are what drops the hold to 6 modes at host steps 1186–1194.

In the dropout window the onset itself stays under the fixed delta. From 1720 through 1900 the largest 50-step open is .091 (update 1808). The mean moves about .06 → .16 while the lag window is already rising, so the gap never reaches .10 during the collapse (0 modes on the 1770 check, factor .133, open only .023). The two rejects are at the tail, deltas .1001 and .1005, after the failing checks have already started at 1720. Stay then keeps failing, including 2200, 2350, and 2390, which #107 did not.

## Call

**KILL.** Warm regresses versus the PR84 pin (191/200 against 196/200) and versus #107 (200/200). Cold acquisition does not die: trajectory passes and both rings match #107, with 0 rejects, because that phase never opens the 50-step mean by .10. The dropout window is not a zero-fire miss, but the two fires are late and the continued run is worse than stall reach (95/120, min 0, suffix 1, failures after 2150). Holding that run is not closer.

Do not lower 0.10 and do not change the 50-step lag. On this same run the warm open is already .265 at the first full lag, and the dropout onset peaks at .091, so either direction of a delta change hits the gate that already failed.

**Next bet:** leave G's own-curvature trust factor. #130's one-step ratio never fired, and this 50-step opening fires on the warm hold while missing the 1756–1808 onset. The next single adversarial bet should arm the same way, then reject a G step on a signal that is not a statistic of `min(1, 0.25/ρ)`.
