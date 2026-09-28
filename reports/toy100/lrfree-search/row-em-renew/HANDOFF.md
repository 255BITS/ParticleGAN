# Handoff: PR #155 fresh-reservoir row EM

Updated 2026-09-28 after the full PR #155 `all22` replay completed.

## Outcome

The frozen candidate `row-em-renew-all22` completed all 22 tasks:

| Group | PASS | FAIL | ERROR | Notes |
|---|---:|---:|---:|---|
| Quick gates | 7/11 | 4 | 0 | All six vector gates and `img_stripes2` pass. `mode_hold`, `img_intensity2`, `img_blobs4`, and `img_bars4` fail. |
| Custom hosts | 0/8 | 0 | 8 | Each returned `ERROR parity` before training; the custom engine refused `_sigma_intrinsic_scale`. These are errors, not trained-task failures. |
| Native100 | 3/3 | 0 | 0 | `grid100`, `rotated100`, and `staggered100` pass every terminal full-accuracy check and independent 100,000-sample holdout. |
| **Total** | **10/22** | **4/22** | **8/22** | |

Native receipts for the all22 replay are under
`/ml2/hypergan/lrfree-20260926/runs/row-em-renew-all22/`; the frozen native-only
receipts and detailed state audit are under
`/ml2/hypergan/gan-attempts/row-em-renew-20260928/runs/`. The committed
`all22-summary.json` and `all22-leaderboard.md` provide compact result records.

The custom errors were deliberately left as reported, per the user's request
not to debug them. The quick-gate failures are ordinary scoring failures. No
default promotion is implied by the native100 result.

## Candidate and reproducibility

- Candidate: `row-em-renew-all22`
- Config hash: `5c4e70b71d184f3c`
- Frozen package SHA-256: `1fc12d29e2c5964f91c9523006e569cba743eba2e3e9851a68afa063c08cd324`
- Package: `/ml2/hypergan/gan-attempts/row-em-renew-20260928/candidate/package`
- Overrides: `/ml2/hypergan/gan-attempts/row-em-renew-20260928/candidate/overrides.json`
- Run used the frozen `all22` preset, the package and overrides above, and candidate options `strict_streams=true`, `diagnostics=true`, and `eval_output_noise=true`.
- Pool monitor: `python /ml2/hypergan/lrfree-20260926/harness/wait.py row-em-renew-all22 --timeout 7200 --poll 20`

The leaderboard is generated at `/ml2/hypergan/lrfree-20260926/LEADERBOARD.md`.
Its row is candidate ID 26 and now shows 10/14 scored hosted/native fields,
with all eight custom slots explicitly marked `ERROR parity`.

## How calibration works

Training batches fill a FIFO reservoir of 16,384 real examples. Once the
non-sigma generator stationarity scale is at most 1/64 and the reservoir has
been renewed, calibration records five jittered generator outputs per prior
row. It models each output as a Gaussian component at the current training
width. An expectation step assigns reservoir examples fractional responsibility
to nearby components; summed responsibility estimates each row's sampling
mass, while residual distances estimate a shared sampling width. A cKDTree
prunes negligible tails.

Every renewed-reservoir fit starts from uniform row mass and current training
width, so the sampler can adapt as birth/death and training move support. The
fitted weights and width affect public sampling only. G, D, and birth/death
continue to train against uniform rows at the original width, and EM uses a
private RNG. The detailed audit confirms exact learner state, rates, and
training streams versus the critic-floor baseline, with zero stream
deviations. Repeated fits are costly: 486–615 fits and roughly 4.4–4.8x the
baseline wall time on the native-only replay.

## Useful files

- [Full native100 and implementation report](README.md)
- [Compact all22 results](all22-summary.json)
- [All22 leaderboard snapshot](all22-leaderboard.md)
- [Critic-gap diagnostic and interpretation](critic-gap-investigation.md)
- [Frozen-checkpoint diagnostic script](critic_gap.py)
- [Candidate implementation](../../../../../../gan-attempts/row-em-renew-20260928/candidate/package/particlegan/row_em.py)
