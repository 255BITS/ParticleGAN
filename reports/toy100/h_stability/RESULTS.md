# eps_net_1m: diagnostic audit complete; no promotion

**19 older hosts: 10 PASS / 9 FAIL. Own-state1200: FAIL at update1692, HQ .86474609375 with8 modes, after491 passing dense checks.**

Executed the authorized17 remaining hosts individually plus one own-state hold, using at most3 CPU workers and one thread per worker. First batch started **2026-09-24 19:00:13 UTC**, within5 minutes; finished19:01:18 UTC. Live audit gates: **18 executed, 9 PASS, 9 FAIL, 0 ERROR**. Ring/two_pole were independently measured earlier and only their retained episodes were regraded here (1 PASS /1 FAIL). Every audit/regrade runtime ledger row is `candidate=regression`; these are diagnostic results, not candidate toy passes.

| Recipe / ledger candidate | Older19 | Original10 | Own short200 | Own long1200 | Promotion |
|---|---|---|---|---|---|
| eps_net_1m / regression | 10 PASS,9 FAIL | 7 PASS,3 FAIL | First200 reproduced; min HQ .918945313 | FAIL:491/492 checks pass; stopped1692 | Unqualified |

**New proposals: 0.** The audit exposed regressions in image quality and unequal-mass coverage, alongside conditional retention and the late HQ excursion. These measurements do not isolate a new permitted fix outside the existing movement/combination work. No coefficient sweep or auxiliary-host invariant study was started; unused candidate/time budgets are intentional.

| Frozen host | Updates | Status | Decisive measured values |
|---|---:|---|---|
| two_pole † | 80 | FAIL | mean_abs 0.0298888311 (requires >= 0.3) |
| mode_hold † | 1200 | PASS | modes 8; hq 0.996826172; suffix14 |
| unipolar | 400 | FAIL | neu_hold 0.810458109 (requires >= 0.85) |
| mid_scale_identity | 800 | FAIL | identity_at_0 0.718450265 (requires >= 0.85) |
| cover_leftover | 800 | FAIL | u_kept 0.441448624 (requires >= 0.85); content_kept 0.444301657 (requires >= 0.75); pole_rel_err_plus 0.419609755 (requires <= 0.2); pole_rel_err_minus 0.460157573 (requires <= 0.2) |
| trajectory | 400 | PASS | identity_mse 0.00175289263 |
| residual_student | 400 | PASS | identity_mse 0.0022454001; success_rate 1 |
| img_stripes2 | 600 | PASS | modes 2; hq 1; distribution_tv 0.03125 |
| img_bars4 | 600 | PASS | modes 4; hq 0.96875; distribution_tv 0.03125 |
| vector_overlap | 1200 | PASS | sw1_normalized 0.0366718283; mean_error 0.0295113381; covariance_error 0.0383533239 |
| img_blobs4 | 600 | FAIL | modes 3 (requires >= 4); hq 0.84375 (requires >= 0.9) |
| img_intensity2 | 600 | FAIL | hq 0.875 (requires >= 0.9) |
| vector_unequal_mass | 1200 | FAIL | component_covariance_error 1.60670958 (requires <= 0.85); component_min_eigen_ratio 0.0183518231 (requires >= 0.15); min_mass_ratio 0.131460339 (requires >= 0.25) |
| vector_unequal_width | 1200 | PASS | component_covariance_error 0.540445052; component_min_eigen_ratio 0.386122525; min_mass_ratio 0.78515625 |
| vector_two_broad | 1200 | PASS | sw1_normalized 0.0218545327; component_covariance_error 0.129680716 |
| vector_anisotropic | 1200 | PASS | sw1_normalized 0.0421745975; component_min_eigen_ratio 0.250084281 |
| vector_spiral | 1600 | PASS | sw1_normalized 0.0292750988; covariance_error 0.0419114493 |
| ae_gan_hold | 250 | FAIL | recon_mse 3.05687213 (requires <= 0.05) |
| unused_token_hold | 200 | FAIL | unused_hold 0.788209453 (requires >= 0.85); concept_move 0.423579831 (requires >= 0.85) |

† Retained independent cold evidence, regraded without training. All other17 rows were executed once in separate directories with the complete eps_net_1m policy. All budgets completed and saved-episode/receipt checks succeeded; partial endpoints are not credited.

The hold restored its own cold checkpoint `112b407dfb58baf9c3b1ce015f3024051524c642163cb5ca07ad71aa86de6b42`, including Adam moments and RNG. Initial samples matched bitwise. Updates1201–1691 passed (minimum HQ .905517578125); update1692 failed. Update1691 had HQ1.0. Executed492/1200 updates, with492 G and492 D commits; final Adam step1692. **No later hold checks ran.** The cold checkpoint has HQ .996826172; the earlier handoff’s .999755859 was the own200 endpoint, now corrected.

The recipe remains a GAN: G and particles use only the inherited logistic relativistic discriminator objective. D retains R1+R2 .6, mixup consistency .01, and fixed input noise .05. Actual constant G/D/prior rates are .001125/.0015/.00225, Adam betas(0,.999), epsilon .001/.001/1e-8. **No new adversarial signal or preconditioner was added.** Frozen host architecture, data, seeds, training budgets and scoring are unchanged. Existing pure-adversarial auxiliary overrides remain in force; no direct reconstruction, label, center or metric-fitting objective was introduced.

**Code:** `adam_response_cold.py` now retains final model/optimizer/tensor/RNG state for residual_student, ae_gan_hold and unused_token_hold through the existing capture hook; its training policy is unchanged. `baseline-qualification/run_audit.py` invokes the existing runners once per assigned host and publishes each completed result. `START.md` and `current-base.json` now contain the measured blockers; the selected recipe/checkpoint/options/rates are unchanged.

**Tests: 32 passed, 0 failed, 0 errors, 0 skipped; 32 distinct unit cases in one run.** Existing suites: Adam response, continuous candidates, legacy noise adapters and remaining-host noise. XML and log retain actual totals. Full unit suite SKIPPED because no viable final candidate exists. Production own1200 and native100 SKIPPED; the authorized diagnostic does not enable promotion.

**Next experiment priority:** require a new policy to repair two_pole80 then reacquire cold ring1200. Retention gates remain unipolar → mid_scale_identity → cover_leftover. The original-ten regressions in img_blobs4, img_intensity2 and vector_unequal_mass must also be resolved before another full own-state hold or native100. Keep the documented AE/unused-token limitations as blockers; do not spend another baseline audit or restore fitting objectives.

**Artifacts and logs**

- Audit table, raw metrics and episode/checkpoint paths: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/audit-results.json`.
- All17 host checkpoints: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/<host>/eps_net_1m/<host>/final-state.pt`; per-host config, options, protocol, source archive and `signal-policy.json.gz` are adjacent.
- Hold metrics, losses, declaration, final model/Adam/RNG state and receipt: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/own_state_diagnostic_1200`.
- Batch declaration and original selection: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/declaration.json`, `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/current-base.json`.
- Applied source: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/source.tar.gz` and `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/source-manifest.json` (131 source files plus orchestration/declarations; inherited AVX2 and one-thread environment recorded).
- Leaderboard/test totals: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/leaderboard.json`, `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/unit-test-totals.json`, `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/regression.xml`.
- Exact executed subprocess commands: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo/reports/toy100/h_stability/baseline-qualification/audit/commands.json`. Runtime ledger: `/ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/tests.jsonl`.

Exact replay commands, from this checkout, using fresh output paths:

```bash
cd /ml2/hypergan/gan-attempts/eps-net-20260924T185715Z/baseline_qualification/20260924T185715Z-1646701/repo
/tmp/pr38-default-env/bin/python -u reports/toy100/h_stability/baseline-qualification/run_audit.py --output reports/toy100/h_stability/baseline-qualification/replay
# Individual host replay; substitute any of the17 host names in the table:
/tmp/pr38-default-env/bin/python -u reports/toy100/h_stability/adam_response_cold.py --declaration reports/toy100/h_stability/baseline-qualification/audit/declaration.json --output reports/toy100/h_stability/baseline-qualification/replay-unipolar --ledger reports/toy100/h_stability/baseline-qualification/replay-unipolar.jsonl --workers 1 --tasks unipolar
# Own checkpoint diagnostic, always stop at the first failed dense check:
/tmp/pr38-default-env/bin/python -u reports/toy100/h_stability/selected_base_probe.py --output reports/toy100/h_stability/baseline-qualification/replay-own1200 --steps 1200
/tmp/pr38-default-env/bin/python -m pytest -q tests/test_adam_response.py tests/test_continuous_candidates.py tests/test_legacy_noise_adapters.py tests/test_legacy_noise_remaining.py
tail -F reports/toy100/h_stability/baseline-qualification/progress.log ../tests.jsonl
```
