# Post-arm critic-advantage collapse → reject the G Adam step

Host: neural. Draft only. Not solved. Not 22/22. Not production. Stall reach (PR #107) stays the reference. This bet does not displace it.

One mechanism, added to the stall-reach adapter. The 0.10 drop and the arm HQ were not retuned after the run.

## Mechanism

Stall reach, both curvature bounds, and the losses are unchanged until an already-scheduled ring check reports 8 modes and HQ ≥ 0.9. That log only arms the mechanism. It is not a loss, and mode count is not the reject predicate.

After the arm, each update appends the trainer's existing critic advantage, `log(2) − D loss` on the phase-0 sharp critic (the recorded `critic_advantage`). A ring of those values supplies the 50-step mean `m_t` and the mean 50 updates earlier, `m_{t-50}`. When `m_{t-50} − m_t ≥ 0.10`, that update's G Adam step is skipped and G's gradients are zeroed. D still steps. G's Adam moments do not advance on a reject. Before the ring holds 100 post-arm values, G steps as in stall reach.

No coverage objective, anchor, forward-KL training signal, mode quota, Chamfer assignment, clip ladder, or trust-ratio predicate.

## Why this signal

#104's leave was a fully accepted G step with positive alignment and a collapsing critic advantage. Shrinking on alignment did not catch it. #130's one-step trust ratio never fired: the dropout opens gradually. This bet uses the advantage, not trust or ρ. The #107 dropout around 1720–2150 is a G-led translation; D's slope spikes after the cloud has already moved. The hope was that D's margin would fall first, and rejecting G there would leave D updating.

## Gates

Seed 0, one thread, PyTorch 2.14.0+cu130 on CPU. Ranked pin: `ATEN_CPU_CAPABILITY=avx2` plus `MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`. Every receipt logs `cpu` and `cpu_env`. The unchanged warm control is 200/200, min HQ .989990234375, so the warm result is rankable. AVX512 is a build check.

Receipts: [continuous-evidence/adv-collapse](continuous-evidence/adv-collapse/). Logs are plain JSON lines.

| Gate | PR84 pin | #107 stall reach | Advantage-collapse G reject |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2, control 200/200) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921. Armed at 1000. **0 fires** |
| Cold trajectory (AVX2) | PASS, MSE .000943 | PASS | **PASS**, MSE .000943. Never armed. **0 fires** |
| Cold ring (AVX2) | 8, suffix 5 | 8, 10/24, suffix 8 | **8, identical curve**, 10/24, suffix 8. Armed at 650. **0 fires** / 451 checks |
| Cold ring (AVX512, build check) | 7 | 8, suffix 8 | **8**, 8/24, suffix 8, terminal HQ .918–1.0. Armed at 850. **0 fires** / 251 checks |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, min 0, final 8 / .971, suffix 25 | **97/120, min 0, final 8 / .971, suffix 25**. Same 23 failing steps, including 1720–2150. Armed at 570. **0 fires** / 1731 checks |

The stay still drops to 0 modes inside 1720–2150 and recovers onto the ring. That is the same degeneracy as stall reach. It is not a clean stay.

## Fire counts

| Phase | Checks after the ring was full | Rejects |
| --- | --- | --- |
| Warm continuation | 101 | 0 |
| Cold ring after arm | 451 | 0 |
| Stay, post-arm | 1731 | 0 |
| Dropout window 1720–2150 | inside the stay checks | 0 |

On this stay the sharpest post-arm drop of the 50-step mean is 0.075 at update 671, while the ring is still being acquired. Inside 1720–2150 the sharpest drop is 0.027 at update 1960. The advantage is already small and flat (about 0.02–0.03) before the translation; it rises when D's slope spikes after the cloud has moved. A falling-advantage rule does not lead the G step that leaves.

## Call

**KILL.** The reject never fires. Warm, cold ring, and stay reproduce stall reach, including the shared 1720–2150 dropout and the recovery to 8 / .971. There is no stay improvement. Do not lower 0.10 and do not change the arm HQ. Do not stack this with trust-open ρ, G×0.5, or mode-drop freeze.

**Keep** stall reach as the GAN-native reference. This candidate does not acquire-and-stay any better than that pin, because it never changes a step.
