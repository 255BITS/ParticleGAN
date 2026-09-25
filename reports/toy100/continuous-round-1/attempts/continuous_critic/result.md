No candidate qualified. K3P remains the measured parent: **22/22 toys, 1200/1200 hold, 300/300 extension, recovery FAIL 28/81**. Its eleven selected source hashes remain unchanged; it was not retrained. All three proposal slots were used, and the search stopped after first-stage failures.

| Formulation | Own hold | Canonical extension | Recovery deadline | Worst hold / deadline HQ | Own toy qualification |
|---|---|---|---|---|---|
| K3P, measured parent | PASS 1200/1200 | PASS 300/300 | FAIL 28/81; delay 1130 | .90723 / .66846 | 22/22 PASS |
| C3 stationary innovation | FAIL 26/27 checked; 1200 required | NOT_RUN | FAIL 18/81; no sustained recovery | .76807 / .70728 | 22 NOT_RUN |
| C1 static cap + anchor | FAIL: NOT_CONVERGED | NOT_RUN | FAIL 0/81; no sustained recovery | N/A / .41553 | 22 NOT_RUN |
| C2 innovation damping | FAIL: NOT_CONVERGED | NOT_RUN | FAIL 0/81; no sustained recovery | N/A / .10376 | 22 NOT_RUN |

Ranking uses hold + extension + timely recovery first. K3P remains ahead; C3 has the strongest acquisition diagnostic among the new candidates but fails qualification. C1/C2 both fail before entering the hold. No modified candidate inherits a parent toy pass.

C1 and C2 each failed all 4,800 settling checks and finished their 6,300-update hold-driver runs at six and three modes respectively. Their worst settling HQ was .74854 and .16138. C3 confirmed convergence at update 4012, then failed at 4039 with eight modes but HQ .76807. Updates 4040 and 4041 also failed (.83301 and .85889). The following 300-update diagnostic windows scored C1 0/300, C2 0/300, C3 298/300 (C3 minimum HQ .83301); these are not canonical extension passes because the standard hold failed.

All shift runs used the frozen change at 2400, deadline 2800, and endpoint 3600. Their continued-hold prerequisite scored 0/120. Thus their recovery-window scores also do not represent adaptation from a qualified pre-shift state. C3 ended with eight modes and HQ .82544. None has a sustained recovery delay within the 1200-update post-shift window. No deadline was moved or later passing interval selected.

The critic rules use the parent definitions: A is dimension-normalized real R1 plus the fake RMS gradient cap; B is the real/fake L2 gradient cap; P is the real-data gradient gap to the EMA critic, divided by dimension. All penalty coefficients and cap thresholds remain one, with the overall factor 0.5.

| Candidate | Penalty and mixing | LR and noise policy |
|---|---|---|
| C1 | `0.5*(B+P)`; `s=0` always | Original K3P schedules; intermediate diagnostic |
| C2 | `0.5*(s*A+(1-s)*B+P)`; `s=stopgrad(P/(P+E))`, `E=mean(||grad D_ema||²/d)` | Original K3P schedules; intermediate diagnostic |
| C3 | Same critic rule as C2 | Fixed G/D .000425; prior .00085; constant input noise 0 and output noise .029 |

C2/C3 use denominator clamp 1e-30 and initialize s=1 before an EMA measurement exists. The anchor retains full inner coefficient one independently of s. All candidates preserve EMA decay .999, initialization on call 2, critic guard ratio 5 after 200 prior steps, Adam betas (0,.999) for G/D/registered prior and the existing history rules, learned prior, direct response, sparse-latent damping, architecture, data, seeds, host auxiliary losses, and fixed quality thresholds. The anchor adds an evaluation from call 3 onward. Direct particle response retains its [1,2] gain and (0,.9) betas where scoped; the ring uses a registered prior and does not exercise that scope.

C1/C2 retain base LR .00425, prior multiplier 2, network/prior floors .01/.05, anneal start .6, network horizon cap 1600, input noise .5 decaying over .1 of the horizon, and output noise .029 warming over .2. The LR multiplier is cosine decay from one to the floor after .6 of the relevant horizon; network horizon is min(T,1600), prior horizon T. Ring T remains the frozen 1200. Input noise decays linearly to zero and output noise rises linearly to .029 over the stated fractions. These are explicitly horizon-dependent diagnostics. C3 is a joint stationary rate/noise ablation, with a single declared rate choice, not a coefficient grid. Its stationary hooks ignore horizon/update arguments; its critic constraint is explicitly independent of the LR clock. The legacy cap field is inert API metadata in C3.

Actual shift-run mixing samples (every 50 calls, excluding initialization): C1 exactly0; C2 .08351–.86369, final .28490; C3 .11630–.79493, final .14695. C3 hold samples span .26619–.79493. The EMA anchor is active with constant positive rates. Each shift run recorded **3,598 added EMA critic forward/input-gradient evaluations**, compared with **2,636 inferred for the unchanged parent** from its exact source and initialization receipt. This is +962 evaluations at the same 3600 updates, with no added optimizer steps. The parent legacy extra-forward counter is not maintained.

All shift runs advanced both optimizer counters from 2400 to 3600. Sampled post-shift parameter displacement was strictly positive. C3 critic displacement RMS ranged 1.38190e-4–4.47290e-4, and combined G/prior displacement 8.56091e-5–5.90857e-4. Actual C3 rates remained .000425/.000425/.00085; C1/C2 post-shift rates were .0000425/.0000425/.000425. C3 therefore moved more actively without preserving the required quality. Training and floating history state are CUDA FP32, deterministic algorithms enabled, TF32 off, one CPU thread, one assigned GPU worker.

Horizon regression: C1 and C2 **FAIL** exact-prefix invariance at 64 updates when declared horizon changes 1200→2400; retained noise changes models, optimizer/EMA state, and noise values. C3 **PASS**: after 1024 updates the complete model, gradients, Adam, generator/prior/critic EMA, controller/history, and RNG state hashes match exactly. Both hashes are `292b935b92df2c0e79669a93eaf33b7dc0e453556a6c90acb704b61e96f2e240`. This finite ring-prefix check includes guard warmup; native/transfer prefix qualification is NOT_RUN. Remaining C3 budget use is in frozen host loop termination, evaluation/receipt bookkeeping and API metadata, not the learning rates, critic rule or noise values. No checkpoint-restart equivalence or indefinite-learning claim is made.

Test totals: **six canonical gates: 0 PASS / 6 FAIL / 0 ERROR**. Corrected horizon audits: **1 PASS / 2 FAIL**. Three earlier audit-only ERROR records remain visible: the serializer could not byte-view a scalar Adam counter. Only audit serialization was fixed (`reshape(-1)`); candidate source hashes stayed identical. The ledger has **1 PASS / 8 FAIL / 3 ERROR / 15 SKIPPED** rows. There were six canonical GPU runs plus nine audit-prefix processes, including the three serialization failures. Setup metadata is not counted as a candidate.

All four sensitive transfer screens, full 22 (including each full 7000 native coverage AND accuracy gate), matched frozen controls, and long-hold/second-change stress are **NOT_RUN**. The first-stage failures blocked advancement; no successful recovery claim needs confirmation. The separate stress protocol was frozen before outcomes: 9600 updates, changes at 6000 and 8400, deadline 400 after each, unchanged eight-mode/HQ≥.90 criterion. No stress result is claimed.

Recommendation: retain K3P. For the next critic-focused proposal, keep C3 rates/noise fixed and test a reversible damping signal sensitive to successive critic-gradient direction changes, while keeping the full EMA anchor and decay fixed. C3 exposes brief precision failures after acquisition; C1/C2 show that static caps or this innovation-energy ratio alone do not preserve cold acquisition under the inherited schedules. Step size, noise and anchoring interactions remain unresolved; the joint C3 ablation does not identify a sole cause. Run own hold + extension and fixed-deadline recovery again before transfer qualification.

Artifacts root: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/repo/reports/toy100/continuous-critic-3603364`. Each candidate directory contains exact `config.json`, `mechanism.py`, optional `stationary_policy.py`, copied drivers/history files, `declaration.json`, and `hashes.json`. Raw gates are `out/<candidate>-<hold|shift>/result.json`; actual counters/traces are in `mechanism-receipt.json`, `training-trace.csv`, and `quality-trace.csv`. `summary.json`, `training-audit.json`, `parent-compute-reference.json`, `qualification-matrix.json`, `final-audit.json`, and `regression/*/result.json` summarize the evidence.

Ledger: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/tests.jsonl`. This report: `/ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/result.md`.

Exact gate commands and explicit environment are saved in `out/*-command.json`. Replay into a fresh directory using:

```sh
/tmp/pr38-default-env/bin/python /ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/repo/reports/toy100/continuous-critic-3603364/replay.py \
  /ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/repo/reports/toy100/continuous-critic-3603364/out/c3_stationary_innovation-shift-command.json \
  --output /ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/repo/reports/toy100/continuous-critic-3603364/out/c3-stationary-shift-replay
```

The replay helper preserves `/tmp/pr38-default-env/bin/python`, the frozen runtime/fixture paths, GPU UUID `GPU-72c1b506-891d-b8bc-b353-e020585e1c47`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one-thread variables, deterministic FP32 controls, and disabled TF32. Audit commands and immutable audit-driver snapshots are under `regression/*-scalarfix/`.

```sh
tail -F /ml2/hypergan/gan-attempts/formulations-20260925T154444Z/continuous_critic/20260925T154444Z-3603364/repo/reports/toy100/continuous-critic-3603364/logs/current.log
```

Canonical wall time including process startup: 1211.6s; audit time including failed serializers: 137.5s. Canonical totals: 27,739 G/D update pairs and 27,727 extra critic evaluations. Hold runs stop by the unchanged gate protocol; failed C1/C2 used 6300 updates versus the measured parent’s2900, so hold elapsed costs are not equal-step comparisons. Shared-GPU wall times are not an isolated performance benchmark.
