# Game dynamics formulation search

Measured candidates: 6. No candidate has cleared both frozen blockers; no release or stability qualification.

Ranking uses sustained blocker passes, then mean frozen final-metric shortfall (offline only). A favorable final point never overrides the five-observation passing suffix.

| Rank | Candidate | Ring: sustained / modes / HQ / suffix | Unequal: sustained / mass ratio / min eigen ratio / suffix | Four follow-ups | Full 22 | Own-state stability |
|---:|---|---|---|---|---|---|
| 1 | optimistic_adam | FAIL / 8 / 1 / 3 | FAIL / 0.19531 / 0.023511 / 0 | NOT_RUN | NOT_RUN | NOT_RUN |
| 2 | optimistic_gradient | FAIL / 7 / 1 / 0 | FAIL / 0.58594 / 0.033979 / 0 | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 | joint_lookahead | FAIL / 4 / 0.99683 / 0 | FAIL / 0.36621 / 0.021563 / 0 | NOT_RUN | NOT_RUN | NOT_RUN |
| 4 | predictive_critic | FAIL / 0 / 0 / 0 | FAIL / 0.45166 / 0.51876 / 4 | NOT_RUN | NOT_RUN | NOT_RUN |
| 5 | optimistic_critic | FAIL / 1 / 0.060791 / 0 | FAIL / 0.54932 / 0.046968 / 0 | NOT_RUN | NOT_RUN | NOT_RUN |
| 6 | extragradient | FAIL / 1 / 0.084961 / 0 | FAIL / 0.3418 / 0.029013 / 0 | NOT_RUN | NOT_RUN | NOT_RUN |

Executed GAN gates: 12; FAIL=12.
Mechanism/regression checks: 7; PASS=7.

All blocker budgets are 1,200 outer steps. The one-evaluation mechanisms perform 1,200 D and 1,200 G Adam calls. ExtraAdam performs 1,200 predictor and 1,200 corrector calls per player; predictor moments are discarded, leaving 1,200 accepted updates per player. It costs two model/gradient blocks per outer step. These are equal-step comparisons, and ExtraAdam is **not** an equal-compute result. Wall times and full update receipts are in each result.

## Declarations and evidence

- Executed code and prelaunch declarations: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/candidates/<candidate>/`.
- Raw metrics, complete frozen curves and CUDA/randomness proofs: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/runs/<candidate>/<gate>/result.json`.
- Prepared baseline (one prepare invocation): `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/prepared/repos/cuda/`; hash manifest `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/prepared/prepared-sources.json`.
- Retained initialization fixtures reused: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/cpu-recipe-gpu-port/initialization-fixtures/`.
- Exact commands, environments, declaration hashes: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/commands.jsonl`.
- Machine-readable leaderboard: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/leaderboard.json`.
- Append-only gate ledger: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/tests.jsonl`.
- Small CUDA mechanism checks: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/check_mechanisms.py`, receipts `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/mechanism-checks.json`.

## Replay

From `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo`:

```bash
/tmp/pr38-default-env/bin/python -u reports/toy100/game-dynamics/run_batch.py CANDIDATE mode_hold vector_unequal_mass
tail -F reports/toy100/game-dynamics/batch.log
tail -F reports/toy100/game-dynamics/runs/CANDIDATE/mode_hold.log
```

Existing run destinations are intentionally never overwritten. For exact standalone replay, take the command from commands.jsonl and change only `--output` to a fresh directory inside this checkout. Sources are already prepared; do not prepare again. The driver enforces one foreground worker, configured physical GPU, deterministic FP32 and one CPU thread; the probe disables TF32.

No architecture, data, seeds, thresholds, evaluation, host auxiliary terms, original learning-rate decay or noise schedule was changed. Model parameters, gradients, Adam moments and correction tensors use CUDA; scalar Adam counters retain the original control behavior. The new mechanisms receive no evaluation metrics, mode labels, centers or target statistics.

Planned gate/stage entries explicitly SKIPPED (NOT_RUN): 36. These are not executed tests.

## Interpretation and next experiments

The attempt stopped at the **six-proposal cap**, not the time limit. No formulation qualifies for promotion. Both blockers were run for every proposal; all 12 full-budget GAN gates failed, with no training errors. The four follow-up gates, full 22 suite, qualification regressions and own-state continuation are **NOT_RUN**. This attempt provides no new 22-toy score.

1. **Optimistic Adam directions:** the best final-metric shortfall, but ring only reaches 8 modes/HQ 1 at step 1100. Its passing suffix is 3, below the frozen 5. Unequal mass ends with mass ratio 0.19531 (<0.25) and minimum eigen ratio 0.02351 (<0.15). Ring recovery is late acquisition, not sustained qualification.
2. **Optimistic raw gradients:** moving correction before Adam's moments gives ring 7 modes/HQ 1. Unequal mass passes three observations at steps 550–650, then loses shape; final minimum eigen ratio is 0.03398. Adapting the preconditioner to the corrected gradient is insufficient.
3. **Joint Lookahead:** averaging both players every five G updates ends at only four ring modes. Unequal-mass eigen ratio is 0.02156. This coarse averaging loses coverage without retaining rare shape.
4. **Predictive critic:** exposing an extrapolated critic to G ends with zero ring coverage. It improves final unequal-mass shape (eigen ratio 0.51876), but only the last four checks pass. Its improvement on one final metric is not a joint or sustained success.
5. **Critic-only optimism:** preserving ordinary G/prior Adam does not fix ring (1 mode/HQ 0.06079). Unequal mass passes eight intermediate observations between steps 500 and 1000, with longest passing run four, then ends at eigen ratio 0.04697. This is direct measured loss of previously passing quality within the frozen budget.
6. **Explicit simultaneous ExtraAdam:** two model/gradient blocks per step still end at 1 ring mode/HQ 0.08496 and unequal-mass eigen ratio 0.02901. Its combined runtime is 185.0 s versus 75.2–83.1 s for the one-block proposals. Extra computation did not solve either blocker.

**Recommendation:** promote none of these rules. Compare the other formulation lanes' independently qualified results before allocating more update-rule trials. If another game-update proposal is authorized, test a joint correction whose size is controlled by local gradient/curvature information, rather than another fixed-strength optimism coefficient. The measured large predictive updates damage ring acquisition, while weaker trajectory averaging loses modes. Any such follow-up still needs cold coverage early enough to sustain the final five checks and retention of rare-component eigen ratio ≥0.15. These are recommendations for a future budget, not additional tested claims.

## Validation and retained states

Seven checks passed: scheduled-direction correction and optimizer-state replay; corrected-gradient moment estimation; temporary-critic restoration; coupled Lookahead synchronization; extragradient mathematics/RNG replay/accepted clocks; critic-only isolation and memory replay; and the executed-gate receipt audit. The audit regraded every complete 24-observation curve with the original protocol, verified all 1,614 prepared files, all prelaunch declarations/code hashes, initialization fixtures and actual update counts. All five one-block proposals have identical complete ordered random-draw hashes for each blocker. ExtraAdam's additional random evaluations replay the same oracle and are declared separately.

`optimistic_critic` retains its own **failed** final `final-state.pt` for each blocker, including model parameters, Adam/correction state and RNG. These are evidence artifacts; no continuation was run or claimed. Original controls, fixtures and prior checkouts remain unchanged. No seed experiments, extra baseline/version controls, images, agents or concurrent benchmark workers were used.

The base config SHA-256 is `4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`. Candidate directories contain the exact copied `probe.py`, `mechanism.py` and `declaration.json`; `extragradient/sources/` additionally contains the exact transformed legacy function bodies. Code is retained under `repo/reports/toy100/game-dynamics/`, not installed as a new public default.

## Standalone replay example

This reuses the exact measured code/config/CPU fixture, with a fresh output directory. The other 11 exact launch commands are preserved in `commands.jsonl`; no benchmark control needs to be rerun.

```bash
env -u LD_PRELOAD -u PYTHONPATH CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 /tmp/pr38-default-env/bin/python -u /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/candidates/optimistic_adam/probe.py --repo /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/prepared/repos/cuda --config /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/configs/toy100/constraints_simple_regularization.json --task mode_hold --backend cuda --initial-state /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/cpu-recipe-gpu-port/initialization-fixtures/mode_hold/initial-values.pt --output /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/game_dynamics/20260924T233426Z-2040502/repo/reports/toy100/game-dynamics/replay/optimistic_adam/mode_hold
```

Mechanism and receipt-check commands (run serially, after reading `../supervisor.md`):

```bash
CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/game-dynamics/check_mechanisms.py
CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/game-dynamics/audit_attempt.py
```

