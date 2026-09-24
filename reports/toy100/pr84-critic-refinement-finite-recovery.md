# Bounded numerical recovery of the cold critic fit

The later-nonfinite-trial policy recovers the exact failed inner fit and runs
updates 472–491 with finite accepted state. This is **PASS_NUMERICAL_RECOVERY**,
not a cold acquisition or stationary-quality pass. All twenty observations
still contain only three modes; no quality threshold was relaxed or used to
select the critic, a retry, a scale, or an update.

The original cold run stopped during update 472 after 471 completed updates.
Its first 48 L-BFGS closures were finite. Strong-Wolfe then proposed a NaN
step after a very large finite trial, making all eight critic parameter
tensors nonfinite at closure 49. The best previously evaluated penalized
training loss was 0.2382626235, compared with 0.2839560509 initially. The exact
failure payload SHA256 is
`19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965`.

The separate [finite policy](pr84_critic_refinement_finite.py) changes only
what happens after a later evaluated loss or gradient is nonfinite. It ends
the existing attempt, restores its best finite evaluated training-loss point,
and clears the invalid leaf gradients. An invalid initial point remains an
error. The original 40-iteration and hard 80-attempted-closure budgets remain;
failed trials count, and there is no second optimization attempt. Finite-path
L-BFGS arithmetic, gradient state and all original receipt fields are unchanged.
The original fit, warm adapter and cold adapter remain untouched on disk.

The [recovery driver](pr84_critic_refinement_recovery.py) first reproduced the
old error and all 48 original finite closure records exactly. The guarded fit
then used 49 attempted calls: 48 finite and one rejected trial. Its restored
critic is bitwise identical to the best critic in the failure payload; all
parameters are finite, invalid gradients are cleared, and RNG is unchanged.

The saved `pre_step` boundary is **after** `noise_policy.set_step`, before D
sampling or gradients. The separate [after-clock replay helper](pr84_critic_refinement_after_step.py)
restores every model, Adam moment, EMA tensor, random stream and NoisePolicy
counter/history, then skips exactly the first already-consumed clock call.
Later clocks execute normally. The host budget and noise horizon remain
1200; the outer loop alone covers 472–491. Replay stops after checkpoint 491,
including its G update and EMA, before optional extra diagnostics or final
evaluation. The first replayed accepted-D full snapshot, cached bank and
Adam metric all match the failure payload exactly.

| Observed property | Updates 472–491 |
|---|---:|
| Finite accepted model/Adam/EMA state | 20/20 |
| D/G Adam moment increments | 20 each |
| D/G phase callbacks | 60 each |
| Bank and fit RNG checks | 20 each |
| Attempted fit gradient evaluations | 1,128 |
| Finite / rejected nonfinite evaluations | 1,127 / 1 |
| Modes | 3 at every update |
| HQ minimum / maximum / final | 0.167480 / 0.954346 / 0.841797 |
| Largest accepted clean support RMS / row movement | 0.460472 / 0.503190 |
| Critic parameter norm, first → last | 15.2178 → 15.3582 |

The numerical failure is repaired without parameter blowup in this window,
but sizable functional oscillation and incomplete mode acquisition remain.
In particular, HQ falls to 0.16748 at update 478 and recovers to 0.95435 at
479. These are retained observations, not a passed quality gate. A complete
new cold trajectory/ring gate and subsequent own-acquired continuation are
still required; the earlier borrowed-state warm/hold result alone cannot
establish acquisition or indefinite training.

Seven focused finite-policy tests pass, including exact old/new finite
L-BFGS behavior, the hard attempted-call budget, fatal invalid initialization,
and real two-update host parity. Three additional after-clock tests pass:
global and isolated output RNG with simultaneous nonzero input/output noise
match uninterrupted three-update runs bitwise for full learning state,
observations, update records and complete noise histories; a post-update
snapshot is rejected as the wrong boundary. Independent review repeated all
ten checks and the exact failed-bank recovery.

The [committed gate](continuous-evidence/critic-refinement-finite-recovery/gate.json)
and [manifest](continuous-evidence/critic-refinement-finite-recovery/manifest.json)
bind exact source bytes, declaration, input hash, configuration, every finite
fit record, all twenty quality/movement observations and final full state.
Large raw and source files are deterministic gzip archives. Reproduction
requires the exact failure payload; it does not rerun its 471-update prefix:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
/tmp/pr38-default-env/bin/python reports/toy100/pr84_critic_refinement_recovery.py \
  --capture /path/to/failed-fit.pt \
  --config reports/toy100/continuous-evidence/critic-refinement-finite-recovery/config.json \
  --output /tmp/new-refinement-recovery
```

The reported CPU execution uses the established PyTorch 2.13 environment.
All nominal rates remain D/G 0.00425 and prior 0.0085. Every result remains
scratch evidence with `shared_gate_eligible=False`.
