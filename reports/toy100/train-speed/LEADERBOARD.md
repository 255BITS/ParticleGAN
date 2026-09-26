# Toy train-speed leaderboard (live — updates with winners)

Goal: fastest wall-clock full-gate training of the frozen simpler22 toy suite
with **identical** numerics gates. Every promotion is a full 22/22 regrade
PASS at frozen seeds/thresholds that is faster than the previous best.
Ranked by `train_seconds` on the pinned CPU profile (1 thread, AVX2,
seed 1234). Timing-only runs without a full-gate PASS are screening
evidence, not winners. Append rows; never rewrite history.

| Rank | Candidate (config / change) | 22/22 | Train seconds ↓ | vs baseline | Evidence | Status |
| --- | --- | :---: | ---: | --- | --- | --- |
| 1 | **Baseline**: `constraints_simple_regularization.json` (exact `b_cap` every step, foreach off) | PASS | 374.8 (toy100 ×3 incl. eval+GIF; per-problem 98.8 / 99.9 / 176.1) | — | `reports/toy100/simpler22/`, `reports/toy100/README.md` | **Current winner** |
| — | *Your candidate here* | — | — | — | `artifacts/toy-suite/<candidate>`, `<candidate>.live.log` | — |

## Update contract

- Each winner adds one row above plus its evidence paths (run dir, live log,
  regrade output). Keep the old rows.
- A winner must attach: exact config (or diff vs baseline), full
  `benchmarks.toy_suite run` + `regrade` commands, `train_seconds` with the
  train/eval/IO split, and confirmation that seeds, thresholds, budgets and
  the pinned CPU profile are unchanged.
- Screening rows (partial gates, single-problem timings) go in attempt
  `result.md` files, not in the table, until they earn a full-gate PASS.
