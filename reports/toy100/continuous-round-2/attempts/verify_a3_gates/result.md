Unchanged A3 verification is **complete: 4 PASS / 18 FAIL / 0 ERROR / 0 NOT_RUN across 22 frozen GPU toys**. A3 is not a verified replacement. K3P remains selected; `current-research-base.json` was not changed. No new formulation was proposed and no baseline, hold, shift, seed repeat, or restart experiment was run.

| Formulation | Hold | Extension | Raw shift | Toy gates |
|---|---|---|---|---|
| K3P selected parent (preserved) | PASS 1200/1200 | PASS 300/300 | FAIL 28/81; delay 1130 | PASS 22/22 |
| A3 unchanged | PASS 1200/1200 (prior) | PASS 300/300 (prior) | FAIL 71/81; delay 810; pre-hold 89/120 (prior) | 4 PASS / 18 FAIL / 0 ERROR / 0 NOT_RUN |

Both formulations fail the required hold + extension + timely-recovery conjunction. K3P retains its measured 22/22; A3's better partial recovery never authorizes promotion. Parent and prior A3 protocol rows are preserved evidence, not new measurements in this attempt.

The four sensitive gates ran first. `mode_hold` fails at the original 1200-update budget: 8 modes but final HQ .818115, only 2/24 passing observations, no stable confirmation. This is consistent with prior A3 hold convergence only at update 1795. `vector_unequal_mass` fails HQ .810303 < .85 and component covariance error 3.255216 > .85; 0/24 observations pass. `vector_unequal_width` passes its final metric cells but only 1/24 observations and no sustained confirmation. `img_stripes2` passes. The assigned verification exception required completing all remaining gates even after these failures.

| Native (7000 updates, seed 1234) | Coverage | Accuracy | Final modes | Final HQ | Worst terminal HQ | Accuracy checks |
|---|---|---|---|---|---|---|
| grid100 | FAIL | FAIL | 100 | 0.91385 | 0.91385 | 0/5 |
| rotated100 | FAIL | FAIL | 49 | 0.518 | 0.5055 | 0/5 |
| staggered100 | FAIL | FAIL | 94 | 0.84575 | 0.81445 | 0/5 |

All native terminal accuracy windows retain the original checks at 6000, 6250, 6500, 6750 and 7000, with independent 100000-sample holdouts. Grid's holdout center RMS is 1.209386 sigma (limit .2) and radial KS .239676 (limit .04); rotated's are 1.873611 and .436217. These are substantial precision/acquisition regressions, beyond the ring's missed recovery deadline. `vector_two_broad` also collapses entirely into one component (counts 0/4096, mass TV .5), despite HQ .996582; high HQ alone is not retention. No EMA score substitutes for the required live score, no early checkpoint was selected, and no quality threshold or budget was changed.

| Gate | A3 | K3P preserved | Evidence |
|---|---|---|---|
| mode_hold | FAIL | PASS | hq 0.81812 (requires >= 0.9); no sustained confirmation; 2/24 checks pass |
| vector_unequal_mass | FAIL | PASS | hq 0.8103 (requires >= 0.85); component_covariance_error 3.2552 (requires <= 0.85); no sustained confirmation; 0/24 checks pass |
| vector_unequal_width | FAIL | PASS | no sustained confirmation; 1/24 checks pass |
| img_stripes2 | PASS | PASS | confirmed at 450; 11/24 checks pass |
| grid100 | FAIL | PASS | coverage FAIL; accuracy FAIL; terminal accuracy 0/5; final HQ 0.91385 |
| rotated100 | FAIL | PASS | coverage FAIL; accuracy FAIL; terminal accuracy 0/5; final HQ 0.518 |
| staggered100 | FAIL | PASS | coverage FAIL; accuracy FAIL; terminal accuracy 0/5; final HQ 0.84575 |
| ae_gan_hold | FAIL | PASS | recon_mse 0.090029 (requires <= 0.05); no sustained confirmation; 0/24 checks pass |
| cover_leftover | FAIL | PASS | u_kept 0.31451 (requires >= 0.85); content_kept 0.47243 (requires >= 0.75); pole_rel_err_plus 0.49056 (requires <= 0.2); pole_rel_err_minus 0.47287 (requires <= 0.2); no sustained confirmation; 0/24 checks pass |
| img_bars4 | FAIL | PASS | no sustained confirmation; 1/24 checks pass |
| img_blobs4 | FAIL | PASS | modes 2 (requires >= 4); hq 0.6875 (requires >= 0.9); no sustained confirmation; 0/24 checks pass |
| img_intensity2 | FAIL | PASS | hq 0.84375 (requires >= 0.9); no sustained confirmation; 0/24 checks pass |
| mid_scale_identity | FAIL | PASS | concept_cos_minus 0.84026 (requires >= 0.85); concept_mag_plus 0.51115 (requires >= 0.75); concept_mag_minus 0.42121 (requires >= 0.75); identity_at_0 0.50819 (requires >= 0.85); identity_at_mid 0.84439 (requires >= 0.85); no sustained confirmation; 0/24 checks pass |
| residual_student | FAIL | PASS | success_rate 0.91667 (requires >= 1); wrong_pad_rate 0.083333 (requires <= 0); no sustained confirmation; 0/24 checks pass |
| trajectory | FAIL | PASS | identity_mse 0.27271 (requires <= 0.02); no sustained confirmation; 0/24 checks pass |
| two_pole | FAIL | PASS | mean_abs 0.1189 (requires >= 0.3); no sustained confirmation; 0/24 checks pass |
| unipolar | FAIL | PASS | cover 0.62902 (requires >= 0.85); neu_hold 0.81685 (requires >= 0.85); no sustained confirmation; 0/24 checks pass |
| unused_token_hold | FAIL | PASS | concept_move 0.13769 (requires >= 0.85); no sustained confirmation; 0/24 checks pass |
| vector_anisotropic | PASS | PASS | confirmed at 1050; 8/24 checks pass |
| vector_overlap | PASS | PASS | confirmed at 350; 22/24 checks pass |
| vector_spiral | PASS | PASS | confirmed at 667; 19/24 checks pass |
| vector_two_broad | FAIL | PASS | sw1_normalized 0.63357 (requires <= 0.18); mass_tv 0.5 (requires <= 0.15); component_min_eigen_ratio 0 (requires >= 0.15); no sustained confirmation; 0/24 checks pass |

Failed gates: mode_hold, vector_unequal_mass, vector_unequal_width, grid100, rotated100, staggered100, ae_gan_hold, cover_leftover, img_bars4, img_blobs4, img_intensity2, mid_scale_identity, residual_student, trajectory, two_pole, unipolar, unused_token_hold, vector_two_broad. Missing canonical toy gates: none.

The frozen A3 rules are unchanged: Adam base G/D LR .000425 and learned-prior multiplier 2; stationary schedule multipliers 1; input noise 0 and output noise .029 without warmup; critic mixing `s=0` from initialization, explicitly decoupled from LR. The penalty is coefficient .5 times real/fake L2 gradient caps at 1 plus `w * mean(||grad D - grad D_EMA||²)/dimension`. Critic EMA decay is .999, with next-call `w=1+relu(-cos(g_current,g_previous))` in [1,2] from raw total critic gradients. The original critic-only 5x bias-corrected Adam RMS guard starts after 200 prior steps. Generator EMA .995, direct-response rule, bounded sparse-latent rule, GAN objectives, learned prior, architecture, data, auxiliary AE/token losses and fixture seeds are retained. The unused mechanism `FLOOR=.01` receipt field is not an applied schedule.

`stationary_policy.py` was observed executing in transfer and native adapters. Every recorded multiplier is 1, output sigma .029 and mixing weight 0. Native input-noise calls are absent because its configured input noise is disabled; transfer calls return 0. Native driver events record actual post-Adam G/D rates .000425 and prior .00085. Transfer traces use the frozen driver's post-Adam rates; any inherited direct-particle response remains separately visible in receipts. The external observer only records function return values and optimizer metadata; it does not change candidate source, tensors, rates, RNG or verdicts. All 206 sampled mobility rows in the new 1200-update mode_hold exactly match the preserved A3 shift prefix.

The prior A3 raw shift remains **FAIL**: stationary 2/5, continued pre-hold 89/120, deadline 71/81, worst deadline HQ .812988 and sustained delay 810 updates (deadline 400). Its hold 1200/1200 and extension 300/300 remain PASS, with worst HQ .930420 and .985596. Post-shift saved evidence contains 1200 additional G and D updates, constant .000425/.000425/.00085 rates and positive sampled displacement (minimum RMS .0000529237). A final quality PASS stamp does not override the failed protocol. Under the supervisor's clarification, a successful live conjunction would initially be UNCONFIRMED pending matched frozen control; A3 does not meet that conjunction.

Horizon scope: preserved prior ring audit PASS compares 90 state tensor hashes after 1000 updates for horizons 1200 versus 7000 and host budgets 7500 versus 9000, including optimizer/controller/EMA/RNG state. That audit is not rerun or counted as new evidence. The four A3 policy primitives ignore step/budget arguments; current transfer and full native runs verify their installation and actual values. Remaining budget reads belong to host termination, accounting/evaluation and the native adapter's cap-versus-total branch; the latter selects equal constant-rate paths here. Config retains cap1600, anneal metadata and total_steps7000. A paired full-state native horizon audit remains **NOT_RUN**; no whole-native horizon-equivalence, restart equivalence or indefinite-learning claim is made.

Matched frozen recovery control, delayed-change stress and repeated uninterrupted change are **NOT_RUN**: the existing shift prerequisites and deadline already fail, and this lane is unchanged toy verification only. No partial score is promoted.

Compute: 35530 fixed-budget GAN updates across the matrix; 38282 measured extra EMA-critic forward/input-gradient evaluations; no extra optimizer updates. Each penalty evaluation after anchor initialization adds an EMA evaluation. Some hosts apply multiple penalties per optimizer update: `mid_scale_identity` has 800 updates, 3200 penalty calls and 3195 EMA evaluations (anchor initializes on call 5); `unipolar` also evaluates multiple conditions. Per-gate receipts retain exact counts. Driver seconds total 1160.25 on the shared GPU, descriptive rather than a controlled speed comparison. K3P's historical zero extra-forward counters are unincremented and must not be interpreted as zero cost. The prior hold took A3 3295 updates versus K3P 2900 (+395); prior equal-3600-update shifts took A3 3598 versus K3P 2636 inferred EMA evaluations (+962). Those are preserved protocol costs, separate from this matrix.

Audit results: source_runtime_fixture_integrity: PASS 650/650 assertions; full22_frozen_protocol_completeness: PASS 113/113 assertions; stationary_policy_cuda_fp32_and_activity: PASS 311/311 assertions. Executed ledger currently has 22 candidate gates and 3 saved-evidence regression gates. The one serial benchmark worker used the explicit assigned GPU UUID, `/tmp/pr38-default-env/bin/python`, one CPU thread, deterministic CUDA algorithms, FP32 floating state and TF32 off. Frozen runtime (614 files), all 19 fixtures, candidate bundle and selected-base hashes are checked. Sources and runtime were never edited. Training began at 16:55:54 UTC, within five minutes of launch.

| Unchanged candidate file | SHA256 |
|---|---|
| config.json | `f5271982a4fb05caa2ffcde9c07d7334c0be1a869db98977b4182a63f119d33d` |
| mechanism.py | `6c2569ed70e17a992e787bdcbbf80f4390b542500f27ebacca71751c7352eca0` |
| stationary_policy.py | `99d8eeddb8f130a66c8d5f10b1e03a1038cff6ef6d2a6731049d72813e7ab95e` |
| response.py | `7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d` |
| latent.py | `197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139` |
| probe.py | `e8653d7e450268310d9c7fc529262279f202a2cb10b3bf1470efc7763d4452bc` |
| native100.py | `d550b6ed701273381450a3c66f6b264767550921e89ba9690ceafb3bc1c0e997` |

Exact source bundle: [/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/sources/a3_reversal_strength](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/sources/a3_reversal_strength). Full source/runtime/fixture provenance: [setup.json](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/setup.json), [provenance audit](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/provenance-audit.json), [qualification audit](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/qualification-audit.json). Canonical numerical evidence: [summary.json](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/summary.json), [matrix.md](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/matrix.md), per-gate `runs/GATE/result.json`, `policy-audit.json` and `mechanism-receipt.json`. Preserved prior evidence with original paths and exact hashes: [prior-evidence.json](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/prior-evidence.json).

Logs and exact invocation environment: `/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/logs/GATE.log` and `GATE.command.json`. Ledger: [tests.jsonl](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/tests.jsonl). CSVs: [actual rates](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/traces/applied-rates.csv), [mixing and strength](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/traces/mixing-and-strength.csv), [rate/noise primitives](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/traces/rate-noise-primitives.csv). [Replay commands](/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/replay_commands.sh) cover every gate with a fresh `-replay` output and the full explicit environment; these commands were not executed and do not inherit an expired launcher deadline.

```sh
tail -f /ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/progress.jsonl
/tmp/pr38-default-env/bin/python /ml2/hypergan/gan-attempts/formulations-20260925T165310Z/verify_a3_gates/20260925T165310Z-3675706/repo/reports/toy100/verify-a3-attempt/summarize.py
```

Recommendation: keep K3P selected and retain A3 as a negative diagnostic reference. Ring hold after slow acquisition did not preserve fixed-budget acquisition or native precision. Any future candidate must address these measured regressions together with the full shift conjunction; higher mobility or partial recovery alone is insufficient. This lane makes no new mechanism proposal and does not search from A3.
