# Selected-family and non-seed research run-attempt coverage

Every record identifies an attempt to run an unchanged selected or archived research recipe and task. PASS/FAIL diagnostics remain separate from ordinary qualification. BLOCKED means an actual start was refused by its constructor or required checkpoint; it is not a numerical failure.

| Family | Attempted / distinct tasks | Pending | Outcome counts |
| --- | ---: | ---: | --- |
| atlas | 35/35 | 0 | 35 BLOCKED |
| bcap-pure | 35/35 | 0 | 4 BLOCKED, 14 FAIL, 17 PASS |
| e22 | 35/35 | 0 | 35 BLOCKED |
| k3p | 35/35 | 0 | 4 BLOCKED, 15 FAIL, 16 PASS |
| ka2 | 35/35 | 0 | 4 BLOCKED, 16 FAIL, 15 PASS |
| r1r2 | 35/35 | 0 | 4 BLOCKED, 17 FAIL, 14 PASS |
| release07-gan-v3 | 35/35 | 0 | 4 BLOCKED, 17 FAIL, 14 PASS |
| signed-confidence-lr-gaussian-v1 | 8/29 | 21 | 3 FAIL, 3 PASS, 21 PENDING, 2 RUNNING |
| G-only coherent mean (original seed0) | 2/29 | 27 | 1 FAIL, 1 PASS, 27 PENDING |
| kinetic-local-dualnorm-g-history-u-v1 | 3/29 | 26 | 2 FAIL, 1 PASS, 26 PENDING |
| kinetic_transport_local_v2 | 1/29 | 28 | 1 PASS, 28 PENDING |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | 3/29 | 26 | 1 FAIL, 2 PASS, 26 PENDING |
| cell-quota-dualnorm-cross-v1 | 1/29 | 28 | 1 FAIL, 28 PENDING |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | 3/29 | 26 | 2 FAIL, 1 PASS, 26 PENDING |
| kinetic-local-dualnorm-g-history-reference8-current-v1 | 2/29 | 27 | 2 FAIL, 27 PENDING |
| Fresh matched protected NS/transport control | 2/29 | 27 | 1 FAIL, 1 PASS, 27 PENDING |
| generator-idle-ema-bcap-direction-v1 | 6/29 | 23 | 2 INCOMPLETE, 4 PASS, 23 PENDING |
| kinetic-local-dualnorm-g-history-reference8-v1 | 2/29 | 27 | 1 FAIL, 1 PASS, 27 PENDING |
| generator-confidence-lr-bcap-direction-v1 | 8/29 | 21 | 1 FAIL, 7 PASS, 21 PENDING |
| bcap-default-baseline-direction-v1 | 6/29 | 23 | 1 INCOMPLETE, 5 PASS, 23 PENDING |
| positive-evidence-lr-gaussian-v1 | 2/29 | 27 | 1 FAIL, 1 PASS, 27 PENDING |
| confidence-lr-critic-orientation-gaussian-v1 | 2/29 | 27 | 1 FAIL, 1 PASS, 27 PENDING |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | 2/29 | 27 | 1 FAIL, 1 PASS, 27 PENDING |
| confidence-lr-finite-cap-gaussian-v1 | 1/29 | 28 | 1 FAIL, 28 PENDING |
| kinetic-local-dualnorm-g-history-v1 | 3/29 | 26 | 1 FAIL, 2 PASS, 26 PENDING |
| kinetic-local-nsgda-cross-v1 | 1/29 | 28 | 1 FAIL, 28 PENDING |

[Exact per-task attempts](attempt-coverage.json) · [Main technique leaderboard](technique-inventory.md)

The runner reuses matched source/recipe/task receipts, resumes pending work, and never replays a completed attempt automatically. It uses at most four workers, two per GPU. Each child retains the task's original timeout. Raw logs and checkpoints are stored outside Git in the configured state directory.

```sh
python -m experiments.forge.attempt_coverage plan --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY
python -m experiments.forge.attempt_coverage run --source-root FROZEN_CHECKOUT --state-root RUN_DIRECTORY --refresh
python -m experiments.forge.attempt_coverage status --state-root RUN_DIRECTORY
```

For a durable user service, adapt [the systemd template](../../ops/forge-attempt-coverage.service.example), then run `systemctl --user enable --now particlegan-attempt-fill.service`. Follow its starts and completions with `journalctl --user -u particlegan-attempt-fill.service -f` or `tail -f RUN_DIRECTORY/events.jsonl`. Each event names the family, task and attempt directory ID.

`--watch` rescans the local selected rows every 30 seconds after draining. `--refresh` regenerates the local main leaderboard during work and when the queue drains. An optional `--completion-hook ops/publish_attempt_coverage.py` publishes the completed batch through a generated-report-only commit and a checked GitHub PR on the dedicated branch. A changed source or recipe requires its matching frozen checkout; a mismatched source stops preparation. Interrupted attempts are retained for review, not replayed. The research intake includes the 19 seed-zero variants already shown in the leaderboard; historical seed-only repetitions are excluded. Original pinned requests supply each recipe and snapshot, and original source task declarations supply the missing questions. A mismatch with a frozen evaluator source blocks the attempt. Completed new same-source measurements fill previously empty research cells; archived verdicts and selected-family qualification never change. Checkpoint/data dependencies still require passing certified artifacts; gate-only failures are bypassed solely for this authorized diagnostic lane. No new qualification is granted.

## Attempted blockers and incomplete runs

These are recorded start attempts, not missing runs. A source or formulation change is not applied to make them pass.

| Recipe | Task | Outcome | Reason |
| --- | --- | --- | --- |
| atlas | ae_gan_hold | BLOCKED | ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | clockfree_audit | BLOCKED | clockfree_audit: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | clockfree_audit_measurement_v1 | BLOCKED | clockfree_audit_measurement_v1: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | cover_leftover | BLOCKED | cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| atlas | five_word_joint_smoke | BLOCKED | five_word_joint_smoke: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | gaussian1d_smoke | BLOCKED | gaussian1d_smoke: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: BLOCKED |
| atlas | grid100 | BLOCKED | grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | grid100_14k | BLOCKED | checkpoint parent grid100: BLOCKED |
| atlas | img_bars4 | BLOCKED | img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | img_blobs4 | BLOCKED | img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | img_intensity2 | BLOCKED | img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | img_stripes2 | BLOCKED | img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | mid_scale_identity | BLOCKED | mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | mode_hold | BLOCKED | mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | residual_student | BLOCKED | residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | ring16_acquisition | BLOCKED | ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | ring_extension | BLOCKED | checkpoint parent ring_hold: BLOCKED |
| atlas | ring_hold | BLOCKED | ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | rotated100 | BLOCKED | rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | rotated100_14k | BLOCKED | checkpoint parent rotated100: BLOCKED |
| atlas | staggered100 | BLOCKED | staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | staggered100_14k | BLOCKED | checkpoint parent staggered100: BLOCKED |
| atlas | target_shift_recovery | BLOCKED | target_shift_recovery: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | trajectory | BLOCKED | trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | two_pole | BLOCKED | two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | unipolar | BLOCKED | unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | unused_token_hold | BLOCKED | unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | vector_anisotropic | BLOCKED | vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | vector_overlap | BLOCKED | vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | vector_spiral | BLOCKED | vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | vector_two_broad | BLOCKED | vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | vector_unequal_mass | BLOCKED | vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| atlas | vector_unequal_width | BLOCKED | vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| bcap-pure | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| e22 | ae_gan_hold | BLOCKED | ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | clockfree_audit | BLOCKED | clockfree_audit: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | clockfree_audit_measurement_v1 | BLOCKED | clockfree_audit_measurement_v1: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | cover_leftover | BLOCKED | cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | five_word_joint_hold | BLOCKED | checkpoint parent five_word_joint_smoke: BLOCKED |
| e22 | five_word_joint_smoke | BLOCKED | five_word_joint_smoke: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | gaussian1d_smoke | BLOCKED | gaussian1d_smoke: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | gaussian1d_stability | BLOCKED | checkpoint parent gaussian1d_smoke: BLOCKED |
| e22 | grid100 | BLOCKED | grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | grid100_14k | BLOCKED | checkpoint parent grid100: BLOCKED |
| e22 | img_bars4 | BLOCKED | img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | img_blobs4 | BLOCKED | img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | img_intensity2 | BLOCKED | img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | img_stripes2 | BLOCKED | img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | mid_scale_identity | BLOCKED | mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | mode_hold | BLOCKED | mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | residual_student | BLOCKED | residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | ring16_acquisition | BLOCKED | ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | ring_extension | BLOCKED | checkpoint parent ring_hold: BLOCKED |
| e22 | ring_hold | BLOCKED | ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | rotated100 | BLOCKED | rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | rotated100_14k | BLOCKED | checkpoint parent rotated100: BLOCKED |
| e22 | staggered100 | BLOCKED | staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | staggered100_14k | BLOCKED | checkpoint parent staggered100: BLOCKED |
| e22 | target_shift_recovery | BLOCKED | target_shift_recovery: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | trajectory | BLOCKED | trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | two_pole | BLOCKED | two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | unipolar | BLOCKED | unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | unused_token_hold | BLOCKED | unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | vector_anisotropic | BLOCKED | vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | vector_overlap | BLOCKED | vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | vector_spiral | BLOCKED | vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | vector_two_broad | BLOCKED | vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | vector_unequal_mass | BLOCKED | vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| e22 | vector_unequal_width | BLOCKED | vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation |
| k3p | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| ka2 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| r1r2 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
| release07-gan-v3 | ring_extension | BLOCKED | checkpoint parent ring_hold: FAIL |
