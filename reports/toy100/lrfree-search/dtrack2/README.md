# Round 5: critic tracking (d_lr_mult 2.0) — shape fixed, precision now binds

Follow-up to [sigdata-bdpair](../sigdata-bdpair/README.md) (cf1-bdpair: grid+staggered
PASS, rotated FAIL trace +.124 / KS .063). One declared mechanism: `d_lr_mult`
1.0 → 2.0 on the frozen cf1 package — no `.py` change, no sampler change, no
metric/task conditioning. Dose check at 1.5x included.

## Frozen results (7k, QR, noisy)

| Candidate | Gate | Verdict | Checks | Prec | TV | Centre/σ | Trace | KS |
|---|---|---|---|---:|---:|---:|---:|---:|
| cf1 base | rotated100 | FAIL | 0/34 | .952 | .034 | .266 | +.124 | .063 |
| rs1-dtrack (D×2) | rotated100 | FAIL | 0/34 | .957 | .039 | .193 | −.022 | .006 |
| rs1b (D×1.5) | rotated100 | FAIL | 0/34 | .953 | .034 | .236 | +.082 | .045 |
| rs1-dtrack (D×2) | grid100 | PASS | 22/34, arr 1500, streak 11 | .983 | .030 | .188 | .025 | .019 |
| rs1-dtrack (D×2) | staggered100 | FAIL | 9/34, arr 4250, streak 7 | .971 | .036 | .200 | .007 | .013 |

Dose-response on rotated shape is monotone (.124 → .082 → −.022), confirming the
under-tracking diagnosis. But precision is not (.952 → .953 → .957): every gate's
residual failure is now outlier mass, not per-mode width. BD counters unchanged
by the D-rate move (~11k moves, 96k waited deaths on rotated), so the shape fix
came from the critic, not from fewer teleports.

Raw results, fixtures, headers, 34 scored rows, 7,000 rate rows and diagnostics
per run are committed here. Local runs + executed package (byte copy of cf1):
`/ml2/hypergan/gan-attempts/rotshape-20260928/20260928T160049Z-1357513/repo/`
(`runs/rs1-{dtrack-{rotated100,grid100,staggered100},...}`, `candidates/rs1-dtrack/`).
Full agent ledger: `tests.jsonl` + `result.md` in the attempt dir
`/ml2/hypergan/gan-attempts/rotshape-20260928/20260928T160049Z-1357513/`.

## Next: outlier mass at fixed shape (round 6)

Rotated BD still proposes 87k deaths / waits 76–96k / stale-resets 77k for ~11k
moves; prior rms 5.27σ/interval with 50 mode-switch rows (grid: 2.8σ, 4 rows).
H2 BD-evidence tightening and H3 sigma-descent were never tested (round 5 used
2/3 slots). Do NOT raise D LR further (shape at −.022, overshoot risk).
