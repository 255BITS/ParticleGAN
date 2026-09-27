# No-R1 screen: 20 arms x 7 tasks

Every arm is `gs2_c03_lr2_d05` with R1 on reals removed, plus one spike-control term and one settling mechanism (see
README.md). The run was 140 tasks with `screen.sh`, one `run_nr.py` launcher per GPU and 6 jobs each, in about 62 min
of wall time. There were 0 crashes and 0 receipt failures, and every cell's receipt shows R1 evals = 0, stock penalty
calls = 0, and the arm's spike, settle and guard buffer engaged. The reference rows are read-only runs from
k3p-constant-fix.
Nothing was rerun.

## Verdict

**Removing R1 does not fix the within-mode covariance, and it breaks the ring.** None of the 20 replacements recovers
either.

- **Covariance, the hypothesis under test:** the R1-removed control `nr_none_none` has a worst per-mode covariance
  ratio of 0.15. gs2 has 0.14 and k3p_simple 0.14. Every arm still fails the native gates (0/30 checks, the same as
  gs2). Across all 20 arms the eigenvalue range stays about 0.1-2.3, which is the same "too thin on one axis, too wide
  on the other" shape that gs2 has. R1's slope flattening is not what fails the toy100 covariance gate.
- **Rings:** gs2 has 0 fails outside transit on both ring tasks. Every nr arm has 38-656 fails (shift + multishift),
  and many never complete arrivals ("x"). On the rings, R1 is load-bearing.
- **rotated100 HQ** falls from .969 (gs2) to .85-.94 in most arms without an anchor.
- **Settle factor means** (ring fails / img_bars4+two_pole cells): anchor 124 / 15.8, oadam 355 / 9.0, none 399 / 7.2,
  hinge 404 / 4.0. The anchor is the only mechanism that clearly helps the rings and the toys. It also collapses the
  within-mode width, with a worst cov ratio of 0.01-0.02 on every native task. The hinge is the worst on the toys, and
  both hinge arms without a value term fail two_pole.
- **Spike factor means** (ring fails / toy cells): none 244, pathcap 242, symcap 311, pairsec 369, dvalcap 436. The toy
  cells are about 8.8 for every spike term except dvalcap (10.0). No spike term beats no term on the rings. On the ring
  the real-side slope cap is active on 86-100% of calls (symcap / pathcap / pairsec spike term > 0), so the caps do
  engage. Clipping the slope at RMS 1 just does not do R1's job there.
- **img_bars4:** only `nr_dvalcap_anchor` passes it (9/24; gs2 has 19/24). As the README caveat says, every cap term
  (symcap, pathcap, pairsec and the fake cap) is 0 on img_bars4, so those arms train it with no gradient penalty at all.

Real-side grad norm and max|D(real)| are not logged by the hosts. The closest measurement is the per-call spike term
mean in `nr_receipt.json` (`term_stats`). For symcap that term is mean relu(||grad D(r)||/sqrt d - 1)^2: .0007-.024 on
native, and .0015-.09 on ring8-shift, where it is positive on nearly every call.

## Top (no arm keeps 0 ring fails, so these are ranked by native passes, then toy cells, then worst cov)

1. `nr_dvalcap_anchor`: 0/3 native, img_bars4 9/24 + two_pole 14/24, worst cov .016, ring fails 228
2. `nr_none_anchor`: 0/3 native, 0 + 14/24, worst cov .0125, ring fails 38
3. `nr_pathcap_anchor`: 0/3 native, 0 + 14/24, worst cov .0113, ring fails 52

These are not recommended as R1 replacements. All three are worse than gs2 on every axis except two_pole.

## Recommendations

- Keep R1 on reals, or a real-side term that pins the slope toward 0 near the data the way R1 does. The ring results
  show it is doing necessary local-convergence work that neither a slope cap, a value cap, OAdam, the hinge nor the
  anchor replaces.
- Look for the covariance failure somewhere other than R1. It is present with and without R1 at the same magnitude, and
  k3p_stock passes it at 0.57-1.21. The candidates are what stock has and simple lacks (instance noise, the annealed LR
  and the EMA anchor at annealed LR, direct-particle response), ablated one at a time on top of gs2 with R1 kept.
- Do not run the full 26-task suite for any nr arm. The screen already rules them out on the rings.

## Leaderboard (summarize_nr.py --tasks screen)

Native cell: P/F cov+acc checks | final HQ | per-mode cov eig ratio min-max (gate .4-1.7) | worst center RMS/sigma.
Ring cell: fails outside transit f, first arrival(s) (+steps, x = never), departures d.

| # | arm | passes | ring fails | native checks /30 | cov err | grid100 | rotated100 | staggered100 | img_bars4 | two_pole | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3p_stock (ref) | 6/7 | 2 | 30 | 0.50 | P 5/5 0.987 0.61-1.19 c0.18 | P 5/5 0.982 0.65-1.20 c0.16 | P 5/5 0.987 0.57-1.21 c0.16 | P 10/24 | P 9/24 | P f0 +1220 d1 | F f2 +1990/x/+1640 d4 |
| 2 | gs2_c03_lr2_d05 (ref) | 4/7 | 0 | 0 | 1.76 | F 0/0 0.980 0.14-2.62 c0.33 | F 0/0 0.969 0.15-2.02 c0.42 | F 0/0 0.966 0.23-1.96 c0.39 | P 19/24 | P 13/24 | P f0 +940 d0 | P f0 +940/+310/+110 d0 |
| 3 | k3p_simple (ref) | 2/7 | 0 | 0 | 1.82 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 3/24 | F 0/24 | P f0 +430 d0 | P f0 +430/+190/+20 d0 |
| 4 | nr_dvalcap_anchor | 2/7 | 228 | 0 | 4.00 | F 0/0 0.980 0.02-2.68 c0.41 | F 0/0 0.960 0.02-1.75 c0.86 | F 0/0 0.983 0.02-2.21 c0.43 | P 9/24 | P 14/24 | F f99 +40 d2 | F f129 +40/+360/+70 d20 |
| 5 | nr_none_anchor | 1/7 | 38 | 0 | 3.97 | F 0/0 0.990 0.04-3.10 c0.43 | F 0/0 0.958 0.01-2.10 c0.74 | F 0/0 0.982 0.01-1.85 c0.42 | F 0/24 | P 14/24 | F f17 +770 d6 | F f21 +770/+300/+270 d8 |
| 6 | nr_pathcap_anchor | 1/7 | 52 | 0 | 3.75 | F 0/0 0.982 0.05-2.36 c0.47 | F 0/0 0.946 0.01-1.67 c0.87 | F 0/0 0.981 0.02-1.92 c0.38 | F 0/24 | P 14/24 | F f26 +780 d1 | F f26 +780/+160/+180 d1 |
| 7 | nr_pairsec_anchor | 1/7 | 112 | 0 | 4.44 | F 0/0 0.998 0.01-2.48 c0.58 | F 0/0 0.960 0.01-2.01 c0.81 | F 0/0 0.980 0.01-2.71 c0.41 | F 0/24 | P 14/24 | F f53 +240 d11 | F f59 +240/+70/+170 d12 |
| 8 | nr_symcap_anchor | 1/7 | 188 | 0 | 4.09 | F 0/0 0.991 0.01-2.95 c0.56 | F 0/0 0.967 0.01-1.92 c0.60 | F 0/0 0.979 0.05-2.15 c0.40 | F 0/24 | P 14/24 | F f92 +120 d8 | F f96 +120/+60/+40 d11 |
| 9 | nr_none_oadam | 1/7 | 234 | 0 | 2.11 | F 0/0 0.984 0.20-2.02 c0.33 | F 0/0 0.871 0.06-1.65 c0.99 | F 0/0 0.984 0.15-1.72 c0.37 | F 3/24 | P 6/24 | F f117 x d3 | F f117 x/x/x d3 |
| 10 | nr_pathcap_none | 1/7 | 280 | 0 | 1.96 | F 0/0 0.977 0.14-2.10 c0.38 | F 0/0 0.885 0.12-2.06 c1.03 | F 0/0 0.981 0.17-2.27 c0.50 | F 1/24 | P 6/24 | F f120 x d0 | F f160 x/x/+460 d19 |
| 11 | nr_symcap_none | 1/7 | 326 | 0 | 2.27 | F 0/0 0.980 0.20-1.92 c0.43 | F 0/0 0.904 0.06-1.69 c0.93 | F 0/0 0.982 0.09-2.01 c0.46 | F 1/24 | P 7/24 | F f139 +1170 d10 | F f187 +1170/+700/+210 d25 |
| 12 | nr_pathcap_oadam | 1/7 | 339 | 0 | 2.68 | F 0/0 0.984 0.11-2.42 c0.35 | F 0/0 0.851 0.03-1.50 c1.01 | F 0/0 0.987 0.10-1.62 c0.38 | F 3/24 | P 7/24 | F f116 x d3 | F f223 x/+1690/+990 d35 |
| 13 | nr_none_hinge | 1/7 | 342 | 0 | 1.90 | F 0/0 0.991 0.27-2.11 c0.43 | F 0/0 0.926 0.10-1.99 c0.92 | F 0/0 0.979 0.13-1.95 c0.53 | F 0/24 | P 5/24 | F f149 +480 d10 | F f193 +480/+1050/+250 d25 |
| 14 | nr_symcap_oadam | 1/7 | 357 | 0 | 1.95 | F 0/0 0.983 0.24-2.18 c0.36 | F 0/0 0.940 0.07-1.88 c0.64 | F 0/0 0.985 0.16-1.88 c0.37 | F 3/24 | P 7/24 | F f113 x d5 | F f244 x/+1780/+230 d32 |
| 15 | nr_pairsec_oadam | 1/7 | 358 | 0 | 3.72 | F 0/0 0.988 0.23-2.33 c0.33 | F 0/0 0.902 0.00-1.76 c0.78 | F 0/0 0.983 0.11-1.77 c0.38 | F 3/24 | P 7/24 | F f112 x d7 | F f246 x/+910/+940 d28 |
| 16 | nr_none_none | 1/7 | 364 | 0 | 1.69 | F 0/0 0.986 0.15-2.17 c0.72 | F 0/0 0.886 0.18-1.74 c0.87 | F 0/0 0.985 0.23-1.77 c0.52 | F 1/24 | P 6/24 | F f182 +1420 d8 | F f182 +1420/x/x d8 |
| 17 | nr_pairsec_none | 1/7 | 370 | 0 | 1.74 | F 0/0 0.987 0.21-2.56 c0.46 | F 0/0 0.912 0.09-1.79 c0.83 | F 0/0 0.983 0.29-1.88 c0.38 | F 1/24 | P 7/24 | F f125 +2140 d1 | F f245 +2140/+1280/+420 d18 |
| 18 | nr_dvalcap_hinge | 1/7 | 371 | 0 | 1.87 | F 0/0 0.991 0.29-2.14 c0.33 | F 0/0 0.931 0.06-1.45 c0.93 | F 0/0 0.983 0.23-1.86 c0.44 | F 0/24 | P 5/24 | F f147 +1050 d9 | F f224 +1050/+170/+700 d39 |
| 19 | nr_dvalcap_oadam | 1/7 | 489 | 0 | 8.32 | F 0/0 0.986 0.21-2.12 c0.32 | F 0/0 0.926 0.07-1.77 c0.83 | F 0/0 0.983 0.00-2.91 c0.45 | F 0/24 | P 6/24 | F f122 +2140 d5 | F f367 +2140/+90/+330 d30 |
| 20 | nr_dvalcap_none | 1/7 | 656 | 0 | 2.32 | F 0/0 0.991 0.13-1.78 c0.39 | F 0/0 0.935 0.04-1.70 c0.93 | F 0/0 0.985 0.19-1.78 c0.40 | F 0/24 | P 6/24 | F f279 +390 d14 | F f377 +390/+1980/+350 d35 |
| 21 | nr_pathcap_hinge | 0/7 | 298 | 0 | 2.08 | F 0/0 0.988 0.13-1.86 c0.43 | F 0/0 0.895 0.07-1.74 c0.92 | F 0/0 0.964 0.22-1.72 c0.52 | F 0/24 | F 4/24 | F f149 +570 d6 | F f149 +570/x/+1830 d6 |
| 22 | nr_symcap_hinge | 0/7 | 372 | 0 | 1.61 | F 0/0 0.982 0.34-2.01 c0.40 | F 0/0 0.932 0.12-1.69 c0.90 | F 0/0 0.983 0.19-1.75 c0.47 | F 0/24 | F 3/24 | F f164 +1030 d25 | F f208 +1030/x/+770 d39 |
| 23 | nr_pairsec_hinge | 0/7 | 637 | 0 | 2.03 | F 0/0 0.991 0.18-1.96 c0.34 | F 0/0 0.909 0.09-1.73 c0.76 | F 0/0 0.980 0.14-2.11 c0.44 | F 0/24 | F 3/24 | F f235 +300 d14 | F f402 +300/+300/+550 d81 |
