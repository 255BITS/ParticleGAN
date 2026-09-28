# Handoff: PR #155 fresh-reservoir row EM

Updated 2026-09-28 while the full PR #155 `all22` replay is running.

## Objective and current status

Continue the native100 work on PR #155. The fresh-reservoir sampling package
already passed the frozen native100 gate on all three tasks: grid100,
rotated100, and staggered100 each passed all five final full-accuracy checks
and the independent 100,000-sample holdout. The exact receipts and audit are
in this directory's `README.md` and `audit.json`.

The user then requested the full PR #155 leaderboard and all tests. The
frozen `all22` preset is 11 quick gates + 8 custom hosts + 3 native100 tasks.
The candidate is registered as `row-em-renew-all22`, config hash
`5c4e70b71d184f3c`, package SHA-256
`1fc12d29e2c5964f91c9523006e569cba743eba2e3e9851a68afa063c08cd324`.

Current completed results:

| Group | Result | Detail |
|---|---:|---|
| Quick gates | 7/11 PASS | mode_hold FAIL; intensity2 FAIL; blobs4 FAIL; stripes2 PASS; bars4 FAIL; all six vector tasks PASS |
| Custom hosts | 0/8 PASS | all eight returned `ERROR parity` before training because the custom engine refuses the package's `_sigma_intrinsic_scale` hook |
| Native100 | In progress | reruns are in `lrfree-20260926/runs/row-em-renew-all22/{grid100,rotated100,staggered100}` |

The user explicitly said not to debug the custom-host parity refusals. Keep
them recorded as errors; do not modify the custom engine or candidate to
force these tests through.

## How the sampler works

The training loop adds real batches to a FIFO reservoir. Once the non-sigma G
stationarity scale reaches 1/64 or lower and at least 16,384 fresh real
examples have arrived, calibration records five jittered generator outputs
for every particle row. It treats each output as a Gaussian component using
the current training output width. An expectation step assigns each real
reservoir point fractional responsibility across nearby components. Summed
responsibilities become the row sampling weights; residual distances estimate
a shared sampling width. A spatial tree prunes negligible Gaussian tails.

Every later fit starts from uniform row mass and the current training width,
so old weights do not follow rows moved by birth/death indefinitely. The
fitted weights and width affect public sampling only. G, D, and birth/death
continue using uniform rows and the original training width; fitting uses a
private RNG stream. This preserves the critic-floor learner updates and
training RNG streams exactly.

## Exact locations and protocol

- Candidate package: `/ml2/hypergan/gan-attempts/row-em-renew-20260928/candidate/package`
- Recipe overrides: `/ml2/hypergan/gan-attempts/row-em-renew-20260928/candidate/overrides.json`
- Native receipts from the successful frozen 3/3 run: `/ml2/hypergan/gan-attempts/row-em-renew-20260928/runs/`
- Full-suite receipts as they finish: `/ml2/hypergan/lrfree-20260926/runs/row-em-renew-all22/`
- Generated screening leaderboard: `/ml2/hypergan/lrfree-20260926/LEADERBOARD.md`
- Queue ledger and pool log: `/ml2/hypergan/lrfree-20260926/ledger.jsonl` and `pool.log`
- Full-suite monitor: `python harness/wait.py row-em-renew-all22 --timeout 7200 --poll 20`
- PR branch before the full-suite update: `codex/k3p-continuous-search`, commit `a41ba114`

The full-suite run was submitted using the frozen `all22` preset with noisy
evaluation, strict stream checks, and the candidate package and overrides
above. The screening pool writes each task to the ledger and rebuilds
`LEADERBOARD.md` after completion. Native formal verdicts require all five
full-accuracy checks at 6,000–7,000 and the independent holdout; do not use the
coverage-only `ok` field.

## Remaining work

1. Wait for the three all22 native reruns and their holdouts to finish.
2. Record the final 22-cell matrix. At the time of this handoff, the native
   runs have not reached their verdicts.
3. Update this handoff and `README.md` with the complete matrix, including the
   8 parity errors, and archive compact result receipts.
4. Update the PR #155 body to replace “full 22-check suite unscored” with the
   measured all22 result; retain the native-only 3/3 result and explain the
   sampling-only method and runtime cost.
5. Commit and push the updated report, then verify the PR head and body.

The all22 package is a research candidate; the full suite result does not
automatically change the project default. The prior native-only result costs
about 4.4–4.8 times the critic-floor wall time due to 486–615 EM fits per
task.
