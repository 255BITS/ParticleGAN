# Adversarial mobility attempt: seven gates cleared; full22 stopped on width fidelity

`direct_particle_response` improves two-pole movement from the selected base’s **0.104395 to 0.644163** and independently passes all seven required first gates. It then reaches **15 PASS / 1 FAIL / 6 NOT_RUN** on its own full22 attempt. `vector_unequal_width` fails component covariance error **0.981321 > 0.85**, with zero terminal passing checks. Qualification stopped immediately. **Own-state stability: NOT_RUN.**

Three proposals were used; no further training is authorized under this attempt’s cap. The 45-minute cap was not treated as a quota. First candidate execution began at approximately 00:55:59 UTC, within five minutes of the 00:53:24 start. Training ended at approximately 01:14 UTC; reporting and audit followed.

## Leaderboard

Counts below are measured gates, not inferred full22 scores. No passes were borrowed.

| Proposal | PASS / measured | Two-pole movement | Stop reason | CUDA Adam updates |
|---|---:|---:|---|---:|
| `direct_particle_response` | 15/16 | 0.644163 | `vector_unequal_width` FAIL | 21,060 |
| `coherent_particle_response` | 2/3 | 0.644163 | `mode_hold` FAIL | 3,360 |
| `prior_recent_rms` | 0/1 | 0.264529 | `two_pole` FAIL | 160 |

## Exact formulations and evidence

All candidates retain the selected dimension-RMS critic exactly: `lambda/2 * (mean(||grad D(real)||²/d) + mean(relu(||grad D(fake)||/sqrt(d)-kappa)²))`, with lambda=kappa=1. Rp logistic GAN losses, architecture, fixed seeds, data, original decay/noise schedules, auxiliary AE/token losses, budgets and frozen scoring remain unchanged. Training, gradients, Adam moments and counters are CUDA FP32. Noncapturable original Adam arithmetic is preserved. No extra model forwards, optimizer steps, target forces, seed sweeps or coefficient grids were added.

1. **prior_recent_rms:** only `prior_betas=(0,.9)` replaces the parent’s `(0,.999)` for particle priors. It improves movement to .264529 but fails .30, so no later gates run.
2. **coherent_particle_response:** retain the short memory and multiply the one scheduled particle Adam step by `1 + max(0, cos(center(g_t), center(g_previous)))`. Centering subtracts the particle-wise mean only for the alignment measurement; the actual gradient is untouched. This passes two-pole (.644163; gradient median .130398) and trajectory, then fails ring (6 modes, HQ .922852). Its mean gain is 1.62 on two-pole, versus 1.05 on trajectory. The ring failure does not isolate short memory from the gain.
3. **direct_particle_response:** exact parent `config.json` bytes. Apply `(0,.9)` and the same coherence gain only to direct sample-particle groups: the existing `_comparison_prior` role must be true and parameters must not belong to a registered `ParticlePrior`. All learned latent priors retain original Adam settings. Scheduled LR and group betas are restored after each call. This is a structural parameter-ownership rule; it has no toy-name, target or metric input.

The raw baseline trace first moves out to .05255, contracts to .00859, then reaches .10439. Actual first-proposal gradients show strong late directional agreement (cosine approximately .99), while scheduled final displacement RMS shrinks to .000473. The coherence response rises from roughly 1.13 during the first 20 updates to roughly 2 once relative particle gradients align. These observations motivated the proposals; the critic-regularization cause of the original delay has not been isolated.

For the final candidate, the response is active on two-pole and inactive on all other measured hosts. All six retained regression results match the parent’s non-timing metrics exactly in fresh executions. The unequal-width failure therefore occurs on the unchanged latent-prior path; it is not evidence that the direct-particle gain damaged this host.

## Final candidate measured gates

| Gate | Result | Key metric | Terminal passing checks |
|---|---|---|---:|
| `two_pole` | PASS | movement 0.644163; gradient 0.130398 | 10 |
| `trajectory` | PASS | MSE 0.00073829 | 22 |
| `mode_hold` | PASS | 8 modes; HQ 1.00000 | 15 |
| `vector_unequal_mass` | PASS | minimum eigen ratio 0.388526 | 10 |
| `img_intensity2` | PASS | 2 modes; HQ 1.00000 | 7 |
| `img_bars4` | PASS | 4 modes; HQ 0.96875 | 20 |
| `img_blobs4` | PASS | 4 modes; HQ 0.96875 | 17 |
| `residual_student` | PASS | MSE 0.00074016; success 1.0 | 19 |
| `unipolar` | PASS | cover 0.998514; neutral hold 0.995765 | 19 |
| `ae_gan_hold` | PASS | reconstruction 0.017293; hold 0.004051 | 22 |
| `cover_leftover` | PASS | u-kept 0.975444; leak 0.000358 | 13 |
| `unused_token_hold` | PASS | hold 0.999083; movement 0.942948 | 11 |
| `mid_scale_identity` | PASS | identity at mid 0.997460 | 17 |
| `img_stripes2` | PASS | 2 modes; HQ 0.96875 | 9 |
| `vector_two_broad` | PASS | component covariance error 0.075460 | 23 |
| `vector_unequal_width` | FAIL | component covariance error 0.981321 > .85 | 0 |

Unequal-width still passes normalized SW1 .03585 ≤ .18, mass TV .02441 ≤ .15, HQ .95581 ≥ .85, and minimum eigen ratio .28422 ≥ .15. It has 11 passing observations out of 24, but no passing terminal suffix. Component covariance errors are [2.72220, .53838, .39402, .27068]; the first component dominates the failure.

Not run after this failure: `vector_anisotropic`, `vector_overlap`, `vector_spiral`, `grid100`, `rotated100`, `staggered100`, and own-state continuation. Native and own-state worker declarations were prepared but never executed. There are no cold or warm stability claims.

## Audit and totals

**20 unique training gates: 17 PASS, 3 FAIL, 0 training ERROR; 24,580 actual CUDA Adam updates.** Two focused CUDA mechanism checks pass (the analytic penalty/response check has 17 cases); both perform zero optimizer updates. The ledger also retains one receipt-audit ERROR and its PASS repair. Thus regression entries are 3 PASS / 1 ERROR, including that correction. There are 49 SKIPPED entries across the three proposals and their stability prerequisites.

The receipt checker initially assumed one regularizer call per step on unipolar. Its frozen host evaluates two scales: 800 penalty calls with 800 Adam calls over 400 steps. The saved receipt was re-audited without training. The original audit error remains in `candidates/direct_particle_response/unipolar/audit-before-count-correction.json`; the corrected audit is beside it. Mid-scale’s four-scale call count is also checked from its frozen source constants.

The final read-only audit verifies all 1,614 CUDA source files, exact parent/candidate code hashes, each declared spec rebuilt from the candidate recipe (including image regularizer aliases), original frozen verdicts, pinned initialization, CUDA FP32 parameters/gradients/state, and exact update counts. Seven retained fixtures were used first. Missing fixtures were captured from pinned CPU constructors, stopping after the second Adam constructor; CPU autograd was explicitly forbidden and update counts are zero. Their hashes and capture commands are retained.

## Artifacts and exact replay

Attempt directory: `/ml2/hypergan/gan-attempts/formulations-20260925T005324Z/adversarial_mobility/20260925T005324Z-2154605`
Research artifacts: `/ml2/hypergan/gan-attempts/formulations-20260925T005324Z/adversarial_mobility/20260925T005324Z-2154605/repo/reports/toy100/adversarial-mobility-attempt`

- `../tests.jsonl`: append-only gate ledger with metrics, timings, artifacts and skipped/error records.
- `final-audit.json`: audited leaderboard and every measured gate; `artifact-hashes.json`: retained file checksums.
- `candidates/<proposal>/declaration.json`: immutable pre-execution formulation and hashes.
- `candidates/<proposal>/{config.json,mechanism.py,probe.py,response.py}`: candidate snapshot (`response.py` applies to proposals 2 and 3). **Config alone is not this formulation.**
- `candidates/<proposal>/<gate>/result.json`: raw metrics, observations, actual displacement/gradients, random stream hashes, regularizer and CUDA state/update receipts.
- `prepared/prepared-sources.json` and `prepared/repos/cuda/`: checksum-verified archived host sources, prepared once.
- `commands.jsonl`, `initial-command.json`, `fixture-captures.jsonl`: exact commands and runtime environment.
- `progress.log`, `<proposal>-<gate>.log`, `latest.log`: concise logs; `tail -F reports/toy100/adversarial-mobility-attempt/latest.log`.

The requested `../current-research-base.json` and `../dimension-rms-base/README.md` were absent. The committed equivalents under `reports/toy100/` were read and verified; no earlier attempt or original CPU recipe was used as the starting candidate.

From the repository root, replay any measured candidate/gate into a new local directory:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/adversarial-mobility-attempt/replay.py \
  direct_particle_response two_pole \
  --output reports/toy100/adversarial-mobility-attempt/replays/direct-two-pole
```

The replay helper pins the assigned GPU UUID, one CPU thread, deterministic FP32, TF32 off and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It verifies source/config/fixture hashes before running, then requires all non-timing results, gradient/displacement records, random draws, initial parameters and update receipts to equal the saved run. This helper was syntax checked but not executed here: completed gates were not rerun. Use `prior_recent_rms two_pole`, `coherent_particle_response mode_hold`, or `direct_particle_response vector_unequal_width` with distinct new output directories to replay the failures.

## Recommendations

Keep `direct_particle_response` as a provisional mobility improvement, not a release-qualified default. It fixes the assigned two-pole blocker while preserving all six selected-base regressions.

For a subsequent authorized attempt, prioritize unequal-width component covariance fidelity on the unchanged dimension-RMS latent path. Inspect the first component’s generated covariance and critic gradients, then choose one target-independent critic or latent-update change. Retain the current gate order and add this measured failure to qualification. Increasing direct-particle motion again cannot address a host where that mechanism is inactive. Full22 must clear before the declared own-state continuation is run.
