No full-suite winner. The native GPU reference remains 16/22.

Three distinct proposals completed 13 CUDA training gates: 10 PASS, 3 FAIL. These totals describe experiment outcomes, not a combined candidate score. All six regression gates passed. The ledger retains one repaired receipt-audit ERROR and 56 SKIPPED downstream gates. No warm probes ran.

| Measured rank | Candidate | Trajectory MSE / suffix | Ring modes, HQ / suffix | Unequal eigen ratio / suffix | Furthest measured result |
|---|---|---:|---:|---:|---|
| 1 | dimension_rms_hybrid | 0.000738294 / 22 | 8, 1.000000 / 15 | 0.388526 / 10 | 6 PASS, then two_pole FAIL |
| 2 (tie) | dimension_rms_bcap | 0.000758296 / 23 | 7, 0.914307 / 0 | 0.754127 / 14 | 2 PASS; ring FAIL |
| 2 (tie) | detached_fake_scale_cap | 0.000736976 / 11 | 5, 0.911133 / 0 | 0.866018 / 8 | 2 PASS; ring FAIL |

Ranking reflects reached gates and measured verdicts. The two ring failures are tied; their unequal-mass improvements do not compensate for missing modes. Each candidate earned its own trajectory, ring and unequal-mass measurements.

Best candidate: `dimension_rms_hybrid`, from the exact retained `rp_r1_fake_cap` recipe. With d equal to the per-example input tensor element count, its penalty is:

`lambda/2 * (mean(||grad D(real)||² / d) + mean(relu(||grad D(fake)|| / sqrt(d) - kappa)²))`

Rp logistic, lambda=1 and kappa=1 remain as declared. It passed trajectory, ring, unequal mass, intensity, bars and blobs, with terminal passing suffixes 22, 15, 10, 7, 20 and 17 respectively. Image final HQ was 1.0, 0.96875 and 0.96875. It then failed the first remaining full-suite gate, `two_pole`: mean_abs=0.104394905 versus >=0.3; grad_med=0.066183008 versus <=1.0; zero passing observations in 24. Its full-suite attempt stopped at 6/7 measured, leaving 15 toys unmeasured. Own-state continuation is SKIPPED.

`dimension_rms_bcap` starts from the exact retained Ra symmetric b-cap .5 partial winner and divides both gradient norms by sqrt(d). It passed trajectory and unequal mass, but ring ended at seven modes, HQ=0.914306641, and zero passing suffix. Four earlier passing ring observations did not satisfy the sustained gate.

`detached_fake_scale_cap` keeps the first proposal’s Rp objective and RMS fake cap, and changes the real term to `relu(u_real - stop_gradient(sqrt(mean(u_fake²))))²`, with `u=||grad D||/sqrt(d)`. It passed trajectory and unequal mass, but ring ended at five modes, HQ=0.911132813, with no passing observations. Both capped alternatives stop at their measured ring failures.

The first result supports dimension normalization as a useful transfer change: d=16 on trajectory and d=64 on images passed while d=2 retained both coverage gates. In d=1, the hybrid still has its original real R1 pressure. The small two_pole movement and low critic gradient are consistent with excessive suppression there; the experiment does not isolate that explanation. Relaxing real pressure through the two tested cap formulations lost ring coverage.

Recommendation: retain the dimension-normalized hybrid as an unqualified six-gate partial winner. A future focused proposal could test a small real-gradient dead zone while preserving its fake RMS cap; use two_pole as an additional cheap failure screen. That formulation is untested here. Avoid rerunning either failed cap formulation unchanged. R1 remains eligible; its measured failure is specific to this proposal and gate.

Validation: the existing source preparer verified the archives and all 1,614 CUDA source files. Expected specs were rebuilt from each declared recipe, including image regularizer aliases. Frozen data, architecture, thresholds, evaluations, budgets, rate/noise actions and auxiliary host losses were preserved. Original CPU fixtures were reused on the six early gates; the missing two_pole fixture was captured with the retained CPU preparer/probe and zero optimizer updates. All 12 runs with retained CUDA random-stream receipts matched exactly; two_pole records its native CUDA draws separately.

Execution: physical GPU `GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69` (RTX A6000), visible as cuda:0; FP32, deterministic algorithms, TF32 off, one worker and one CPU thread. Model parameters, gradients and all Adam state tensors, including step counters, were CUDA. A 25-step regression checked bitwise equality against original Adam arithmetic. The 13 gates executed 20,560 Adam updates (10,280 D plus 10,280 G/prior), and 286.526 recorded training seconds. Regularization used the existing two critic forwards per application; no extra gradient blocks. Original schedules and optimizer hyperparameters remain unchanged.

The first launch was before five minutes (see commands.jsonl). Candidate cap reached: 3/3. Finished before the hard stop; no workers remain.

One intensity receipt audit initially rejected the permitted image field `gradient_penalty`. The result already matched the declared recipe spec. The alias check was corrected and the saved result re-audited without training rerun. The original ERROR, raw result, error traceback and PASS correction are preserved; it is not an extra GPU gate.

Artifacts:

- Attempt ledger: [/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/tests.jsonl](/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/tests.jsonl)
- Leaderboard: [/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/leaderboard.json](/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/leaderboard.json)
- Final receipt audit: [/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/final-audit.json](/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/final-audit.json)
- Launch commands/environment/hashes: [/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/commands.jsonl](/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/commands.jsonl)
- Code/config/declaration hashes: [/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/artifact-hashes.json](/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/artifact-hashes.json)
- Frozen prepared source: `/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/prepared/repos/cuda`
- Source manifest: `/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/prepared/prepared-sources.json`
- Candidate `dimension_rms_hybrid`: `/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/candidates/dimension_rms_hybrid`; `probe.py`, `mechanism.py`, `config.json`, `declaration.json`; each executed gate has `<gate>/result.json`, `<gate>/audit.json`, and `<gate>.log`.
- Candidate `dimension_rms_bcap`: `/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/candidates/dimension_rms_bcap`; `probe.py`, `mechanism.py`, `config.json`, `declaration.json`; each executed gate has `<gate>/result.json`, `<gate>/audit.json`, and `<gate>.log`.
- Candidate `detached_fake_scale_cap`: `/ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/candidates/detached_fake_scale_cap`; `probe.py`, `mechanism.py`, `config.json`, `declaration.json`; each executed gate has `<gate>/result.json`, `<gate>/audit.json`, and `<gate>.log`.

Exact candidate hashes:

| Candidate | Mechanism SHA256 | Config SHA256 |
|---|---|---|
| dimension_rms_hybrid | `e21d7ea1b107ca3e7d5f7a92a259d9d5d0343f071288da158a23b04dbbfad628` | `a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2` |
| dimension_rms_bcap | `6b7b833ad30840fd239d8c443c0cbd082ffeab0a8e48dae1cbe2596157823bca` | `833b3f0011ba4c8aabfd2f682f96bddcad1576b5a72b6a132c785b51a250d11f` |
| detached_fake_scale_cap | `1df562e03b9f70be3061619d624f830eeee733932576795e5c0bd261516bdb85` | `a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2` |

Common executed probe SHA256: `d4375bab9e711028efcda8232d5debc3029fcada1d2197df9cfa20e5cc9d7ffe`.

Replay an exact recorded gate into a fresh output directory using the checksum-validating replay helper:

```bash
cd /ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo
/tmp/pr38-default-env/bin/python reports/toy100/critic-scale-attempt/replay_gate.py \
  dimension_rms_hybrid trajectory \
  reports/toy100/critic-scale-attempt/replay/hybrid-trajectory
```

Replace the candidate/gate as needed; use a fresh output directory. Existing results are protected. Exact original commands are retained in commands.jsonl.

Inspect logs:

```bash
tail -f /ml2/hypergan/gan-attempts/formulations-20260925T002058Z/critic_scale/20260925T002058Z-2110671/repo/reports/toy100/critic-scale-attempt/progress.log
```

All verdicts use the sustained frozen gate. No full-22 score or continuation qualification is claimed.
