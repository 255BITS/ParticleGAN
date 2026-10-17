# Round 6: D-LR floors — max(prior-scale, churn-rate) leads 13/16 honest (September 28)

Follow-up to [sigdata-bdpair](../sigdata-bdpair/README.md) (cf1 base: grid+staggered
PASS, rotated FAIL trace +.124) and [dtrack2](../dtrack2/README.md) (D×2 fixes rotated
shape, precision binds at .957). Round 6 replaces the blunt D-LR multiplier with
state-driven floors on D's settle scale. All runs 7,000 updates, QR
`batch_feature_zero`, noisy scoring, `total_steps=null`, no metric/task conditioning.

## Candidates (one declared mechanism each, all on frozen cf1)

- **codex α-.75** (`cf1-bdpair-dtracks-075-prior`): D scale floor
  `max(own, 0.75 × prior_scale)`. Executed package + runs:
  `/ml2/hypergan/gan-attempts/formulations-20260928T040504Z/custom_follow/20260928T040504Z-4138988/`
  (`candidates/cf1-bdpair-dtracks-075-prior/`, `runs/cf1-bdpair-dtracks-075-prior-{grid100,rotated100,staggered100}/`).
- **codex α-1.0** (`cf1-bdpair-dtracks-prior`): same floor at full prior scale.
- **mine2 H2 bdmoves**: D scale floor `max(own, 0.75 × churn)` with churn from
  cumulative BD backlog counters. Session:
  `/ml2/hypergan/gan-attempts/rotshape-20260928/mine2/20260928T180736Z-1530980/`
  (`repo/runs/h2-bdmoves-{staggered100,rotated100,grid100}/`).
- **mine3 both2** (`both2-rate`): D scale floor
  `max(own, 0.75 × prior_scale, 0.75 × churn_rate)` with churn as the per-step
  BD-backlog increment smoothed by a 0.99 EMA (~100-step memory, releases when
  the table quiets). Package:
  `/ml2/hypergan/gan-attempts/rotshape-20260928/mine3/pkg-both2/` (diff vs
  pkg-both is `training.py` only + 2 checkpoint floats).
- Rejected along the way: v1 cumulative churn (ratchet — rotated plateaus .965),
  H1 gated floor (ties α-.75 on staggered), H3 siggroup floor (ties α-.75),
  EMA-anchor damping (killed at 6/34).

## Frozen native results

| Candidate | Grid | Rotated | Staggered | Natives |
|---|---|---|---|---|
| cf1 base | PASS 20/34, .985/.196 | FAIL 0/34, .952/.266 | PASS 18/34, .976/.176 | 2/3 |
| codex α-.75 | PASS 23/34, .983/.183 | PASS 8/34, .973/.135 | FAIL 18/34, .974/.199 | 2/3 |
| codex α-1.0 | PASS 19/34, .982/.186 | PASS 10/34, .972/.159 | FAIL 18/34, .975/.207 | 2/3 |
| h2-bdmoves | PASS 22/34, .983/.198 | FAIL 0/34, .967/.155 | PASS 21/34, .976/.177 | 2/3 |
| **both2-rate** | PASS 23/34, .983/.183 | PASS 8/34, .973/.135 | FAIL 18/34, .974/.199 | 2/3 |

(prec / centre-σ; trace and KS pass on all non-FAIL cells. both2 grid rerun clean
`both2-grid100-clean`: 23/34, holdout .983.)

## Full-toy matrix (13 quick + ring/stationary + 3 natives = 16 scored)

Pool sweep 2026-09-28 (`lrfree-20260926` ledger cands `cf1-bdpair`, `cf1-075prior`,
`cf1-prior10`, `h2-bdmoves`, `both2-rate`; clean eval):

| Task | cf1 | α-.75 | α-1.0 | h2 | both2 |
|---|---|---|---|---|---|
| mode_hold | FAIL | PASS | FAIL | PASS | PASS |
| img_intensity2 | FAIL | FAIL | FAIL | FAIL | FAIL |
| img_blobs4 | PASS | PASS | FAIL | PASS | PASS |
| img_stripes2 | PASS | PASS | PASS | PASS | PASS |
| img_bars4 | FAIL | FAIL | FAIL | FAIL | FAIL |
| vec_two_broad | PASS | PASS | PASS | PASS | PASS |
| vec_unequal_mass | PASS | PASS | FAIL | PASS | PASS |
| vec_unequal_width | PASS | PASS | PASS | PASS | PASS |
| vec_anisotropic | PASS | PASS | PASS | PASS | PASS |
| vec_overlap | FAIL | PASS | PASS | FAIL | PASS |
| vec_spiral | PASS | PASS | PASS | PASS | PASS |
| ring_shift | PASS | PASS | PASS | PASS | PASS |
| stationary | PASS | PASS | PASS | PASS | PASS |
| grid100 | PASS | PASS | PASS | PASS | PASS |
| rotated100 | FAIL | PASS | PASS | FAIL | PASS |
| staggered100 | PASS | FAIL | FAIL | PASS | FAIL |
| **Total** | **10/16** | **12/16** | **10/16** | **11/16** | **13/16** |

## Read

- The D floor strictly helps the quick suite (base 8/13 → both2 11/13): it fixes
  mode_hold and vec_overlap with zero quick-gate regressions.
- α-1.0 is too strong: breaks blobs4 and unequal_mass that .75 keeps.
- Natives trade rotated against staggered across every variant: prior-scale floor
  takes grid+rot, churn floor takes grid+stag, the max takes grid+rot with the
  best rotated margin (streak 8, prec .973). Nothing honest takes all three.
- img_intensity2 + img_bars4 fail lineage-wide (all five) — a base image weakness,
  not a floor effect. (Noisy-vs-clean note: pool eval column reads `clean` for
  these rows; the native receipts above are noisy-scored per the frozen protocol.)
- Staggered's centre miss (.199 across all D variants incl. both2) is not
  D-pressure: both floor terms read near-zero there late. Suspect BD-side
  (76k deaths vs 11k births); D-rate shaping is exhausted.

## Raw receipts

Native `result.json` per run live in the local dirs above; the pool ledger
(`/ml2/hypergan/lrfree-20260926/ledger.jsonl`) holds all 65 quick-gate rows.
both2 native finals:
`/ml2/hypergan/gan-attempts/rotshape-20260928/mine3/logs/both2-{rotated100,staggered100,grid100-clean}/result.json`.
