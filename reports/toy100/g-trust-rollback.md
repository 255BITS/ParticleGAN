# G-step rollback on own-curvature trust collapse

Host: neural. Draft only. Not solved. Not 22/22. Not production. Stall reach (PR #107) stays the reference. This bet does not displace it.

One mechanism, added to the stall-reach adapter. Thresholds were not retuned after the run.

## Mechanism

Stall reach is unchanged until an already-scheduled ring check reports 8 modes and HQ ≥ 0.9. That log only arms the mechanism. It is not a loss, and mode count is not the rollback predicate.

After the arm, before each G Adam step the trainer snapshots G parameters and Adam moments and records G's own-curvature trust: the mean of `factor = min(1, 0.25/ρ)` over the last 50 updates, the same window #107 uses for stall. The normal stall-reach step runs, including both curvature bounds. Trust is recomputed with that same probe. The step is rejected, and the snapshot restored, when trust collapsed across the step:

- pre ≥ 0.5 and post ≤ 0.1, or
- post/pre ≤ 0.2

D always trains. The rollback is inactive before the arm. No coverage, anchor, forward-KL training signal, mode quota, or Chamfer assignment.

## Gates

Seed 0, one thread, PyTorch 2.14.0+cu130 on CPU. Ranked pin: `ATEN_CPU_CAPABILITY=avx2` plus `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. Every receipt logs `cpu` and `cpu_env`. The unchanged warm control is 200/200, min HQ .989990234375, so the warm result is rankable. AVX512 is a build check.

Receipts: [continuous-evidence/g-trust-rollback](continuous-evidence/g-trust-rollback/).

| Gate | PR84 pin | #107 stall reach | Trust-collapse rollback |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2, control 200/200) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921, final 8 / .981. Armed at 1000. **0 fires** |
| Cold trajectory (AVX2) | PASS, MSE .000943 | PASS | **PASS**, MSE .000943. Never armed |
| Cold ring (AVX2) | 8, suffix 5 | 8, 10/24, suffix 8 | **8, identical curve**, 10/24, suffix 8. Armed at 650. **0 fires** / 550 checks |
| Cold ring (AVX512, build check) | 7 | 8, suffix 8 | **8, identical curve**, suffix 8. Armed at 850. **0 fires** / 350 checks |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, min 0, final 8 / .971, same 23 fails, suffix 25 | **97/120, min 0, final 8 / .971**. Failing steps match #107, including 1720–2150. Armed at 570. **0 fires** / 1830 checks |

## Fire counts

| Phase | Checks after arm | Rollbacks |
| --- | --- | --- |
| Warm continuation (acquire hold) | 200 | 0 |
| Cold ring after arm | 550 | 0 |
| Stay, steps 571–1200 | 630 | 0 |
| Stay, steps 1201–2400 | 1200 | 0 |
| Dropout window 1720–2150 | 431 | 0 |

In that dropout window the pre-step window mean stays in .048–.180, so the 0.5/0.1 clause cannot fire. The sharpest post/pre is .274 at update 1983, above 0.2. The published onset is a gradual opening of the factor as ρ falls from about 5 to .9, not a one-step collapse.

## Call

**KILL.** The rollback never fires. Warm, cold ring, and stay reproduce stall reach, including the shared 1720–2150 dropout and the recovery to 8 / .971. There is no recovery improvement. Do not lower 0.2 and do not change 0.5/0.1.

**Next bet:** after the arm, reject a G step when the 50-step mean trust factor opens across that window (the actual .05→.27 onset), not on a one-step ratio.
