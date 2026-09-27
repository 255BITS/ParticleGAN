# K3P no-R1: full 26-task suite for the screen top 3, plus a reg_coeff 1.0 check

**Verdict: none of the three arms should replace gs2_c03_lr2_d05.** They score 13, 13 and 11 out of 26, below
k3p_simple (14), gs2_c03_lr2_d05 (17) and k3p_stock (18). No arm passes a task that gs2 fails. Every arm fails both
rings, which gs2 passes. Every arm still fails all 3 native covariance/accuracy gates, and its within-mode covariance
is worse than with R1. Raising reg_coeff from .3 to 1.0 makes all three arms worse again.

Launcher: `full.sh`. It runs `run_nr.py --tasks all` for the three arms on cuda:1 (7 jobs) and `--tasks screen` for the
`_c1` arms on cuda:0 (5 jobs). `run_nr.py` skipped each arm's 7 screen cells because `result.json` already existed. The
`arms.json` entries for these arms did not change (checked), so those cells have identical config. That left 57 new
full-suite cells and 21 coefficient-check cells, all rc=0, in about 21 minutes. All 78 new receipts pass
`summarize_nr.receipt_ok`. Each shows 0 R1 evaluations, 0 stock penalty calls, only `rp_softplus` d_loss calls,
Adam steps on the critic, guard reads of `max_exp_avg_sq` only, an active anchor, noise 0 and constant LR. The K3P
kernel `coeff` is .3 in every cell of the base arms and 1.0 in every `_c1` cell. The `_c1` arms come from `gen_arms.py`
and are identical to their parent arms except for `config.reg_coeff`.

Follow the logs with `tail -f logs/*.log` (one line per task) or `tail -f logs/<arm>/<task>.log` (one line per eval).

## Full suite: passes out of 26, by group

| arm | passes /26 | native /3 | transfer /19 | hold /1 | shift /1 | ring /2 | ring fails (shift+multi) | native cov err |
|---|---|---|---|---|---|---|---|---|
| k3p_stock (ref) | **18** | 3 | 14 | 0 | 0 | 1 | 2 | 0.50 |
| gs2_c03_lr2_d05 (ref) | 17 | 0 | 15 | 0 | 0 | 2 | 0 | 1.76 |
| k3p_simple (ref) | 14 | 0 | 12 | 0 | 0 | 2 | 0 | 1.82 |
| nr_pathcap_anchor | 13 | 0 | 13 | 0 | 0 | 0 | 52 (26+26) | 3.75 |
| nr_dvalcap_anchor | 13 | 0 | 13 | 0 | 0 | 0 | 228 (99+129) | 4.00 |
| nr_none_anchor | 11 | 0 | 11 | 0 | 0 | 0 | 38 (17+21) | 3.97 |

The native cov err is the mean over the 3 native tasks of max(|ln min eig ratio|, |ln max eig ratio|), where 0 means
the within-mode shape is exact. Without R1 the worst per-mode eigen ratio falls to .01-.05, against .14-.23 for gs2.
Removing R1 flattens modes even further instead of freeing them. So R1 is not the cause of the covariance failure.

## Per task: where each arm differs from gs2_c03_lr2_d05

A `-` means gs2 passes the task and the arm fails it. No arm has a `+`.

| arm | lost vs gs2 | gained |
|---|---|---|
| nr_pathcap_anchor | ring8-shift (f26), ring8-multishift (f26), toy-img_bars4 (3/4 modes), toy-img_blobs4 (hq .938) | none |
| nr_dvalcap_anchor | ring8-shift (f99), ring8-multishift (f129), toy-vector_overlap (sw1 .133 vs .091), toy-img_stripes2 (never confirmed) | none |
| nr_none_anchor | ring8-shift (f17), ring8-multishift (f21), toy-trajectory, toy-residual_student (never confirmed), toy-img_bars4, toy-img_blobs4 | none |

These failures pass no gate, but some are closer to passing than gs2:
- nr_pathcap_anchor is the only run in this set, including the references, whose hold-mode_hold converges (at step
  5497, then it fails its hold at 5502 with min HQ .82). On shift-mode_hold it reaches 19/81 deadline windows with
  delay 1020, where every other arm gets 0/81.
- On toy-vector_overlap and toy-img_stripes2, nr_none_anchor and nr_pathcap_anchor give identical metrics. The
  path cap never fires on those toys, which is the same inertness the screen reported for img_bars4.
- The ring failures follow one pattern: without R1 the rings arrive much earlier (dvalcap +40 against +940 for
  gs2), but they do not hold. They show 17-129 fails outside transit and 1-20 departures.

## Coefficient check: reg_coeff 1.0 on the 7 screen tasks

| arm | passes /7 (c .3 -> 1.0) | ring fails | native cov err | worst eig ratio min | rotated100 final HQ | two_pole suffix |
|---|---|---|---|---|---|---|
| nr_pathcap_anchor | 1 -> 1 | 52 -> 90 | 3.75 -> 11.40 | .01 -> .00 | .946 -> .866 | 14 -> 18 |
| nr_dvalcap_anchor | 2 -> 1 (loses img_bars4) | 228 -> 356 | 4.00 -> 11.45 | .02 -> .00 | .960 -> .862 | 14 -> 18 |
| nr_none_anchor | 1 -> 1 | 38 -> 251 | 3.97 -> 11.47 | .01 -> .00 | .958 -> .812 | 14 -> 18 |

At c = 1.0 the modes collapse to lines: the minimum eigen ratio is about 0, and on grid100 nr_none_anchor_c1 has max
ratio .04. rotated100 loses modes (the center RMS is inf because a mode is missing), and the rings get worse. The
only gain is the two_pole suffix, and two_pole passes at .3 anyway. The base's .3 was tuned with R1 present, but the
low coefficient is not what holds these arms back. With the anchor on, a larger c scales the EMA-anchor prox term as
well. That term pulls D toward its lagged copy. A stronger pull is consistent with the worse covariance and ring holding, but this check does not separate the prox term from the spike terms.

## Recommendations

1. **Keep R1 in gs2_c03_lr2_d05 and stop this no-R1 line.** Across 20 screen arms and 3 full-suite arms, no
   replacement for R1 recovers native covariance. All of them lose the rings. The best no-R1 arm scores 13/26
   against gs2's 17/26.
2. **Look for the covariance fix in the components where k3p_stock differs from gs2.** k3p_stock is the only arm
   that passes the native covariance gates, with a worst eig ratio of .57-.65 and 30/30 checks. The candidates are
   its instance noise and its annealed LR. The annealed LR is also what activates its EMA anchor. The next test
   should add one stock component at a time to gs2 (noise only, then anneal only) on the 3 natives plus both rings.
   A forced anchor at constant LR is already ruled out by this study.
3. The partial hold/shift progress of nr_pathcap_anchor is not worth pursuing on its own. It costs both rings and 2
   image toys.

## Full-suite leaderboard (`summarize_nr.py --tasks all`, these arms and the references)

Native cell format: P/F, coverage/accuracy checks, final HQ, per-mode covariance eigen-ratio min-max (the gate is
.4-1.7), and the worst center RMS/sigma. Ring cell format: P/F, fails outside transit, arrivals, departures.

| # | arm | passes | ring fails | native checks /30 | cov err | grid100 | rotated100 | staggered100 | hold-mode_hold | ring8-multishift | shift-mode_hold | ring8-shift | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold | vector_two_broad | vector_unequal_mass | vector_unequal_width | vector_anisotropic | vector_overlap | vector_spiral | img_stripes2 | img_bars4 | img_blobs4 | img_intensity2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | k3p_stock (ref) | 18/26 | 2 | 30 | 0.50 | P 5/5 0.987 0.61-1.19 c0.18 | P 5/5 0.982 0.65-1.20 c0.16 | P 5/5 0.987 0.57-1.21 c0.16 | F 0/1200+0/0 | F f2 +1990/x/+1640 d4 | F 0/81 | P f0 +1220 d1 | P 9/24 | P 23/24 | P 20/24 | P 19/24 | P 22/24 | P 13/24 | P 11/24 | P 17/24 | F 0/24 | P 23/24 | F 0/24 | F 0/24 | P 6/24 | P 11/24 | P 21/24 | P 22/24 | P 10/24 | F 0/24 | F 0/24 |
| 2 | gs2_c03_lr2_d05 (ref) | 17/26 | 0 | 0 | 1.76 | F 0/0 0.980 0.14-2.62 c0.33 | F 0/0 0.969 0.15-2.02 c0.42 | F 0/0 0.966 0.23-1.96 c0.39 | F 0/1200+0/0 | P f0 +940/+310/+110 d0 | F 0/81 | P f0 +940 d0 | P 13/24 | P 22/24 | P 18/24 | P 22/24 | P 24/24 | P 17/24 | P 20/24 | P 21/24 | F 0/24 | P 21/24 | F 0/24 | F 0/24 | F 0/24 | P 6/24 | P 22/24 | P 8/24 | P 19/24 | P 19/24 | P 9/24 |
| 3 | k3p_simple (ref) | 14/26 | 0 | 0 | 1.82 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 0/1200+0/0 | P f0 +430/+190/+20 d0 | F 0/81 | P f0 +430 d0 | F 0/24 | P 22/24 | P 8/24 | P 18/24 | P 22/24 | P 11/24 | P 10/24 | P 16/24 | F 0/24 | P 21/24 | F 0/24 | F 0/24 | P 5/24 | P 9/24 | P 24/24 | P 5/24 | F 3/24 | F 0/24 | F 0/24 |
| 4 | nr_pathcap_anchor | 13/26 | 52 | 0 | 3.75 | F 0/0 0.982 0.05-2.36 c0.47 | F 0/0 0.946 0.01-1.67 c0.87 | F 0/0 0.981 0.02-1.92 c0.38 | F 5/1200+0/0 | F f26 +780/+160/+180 d1 | F 19/81 | F f26 +780 d1 | P 14/24 | P 21/24 | P 12/24 | P 22/24 | P 17/24 | P 18/24 | P 19/24 | P 21/24 | F 0/24 | P 19/24 | F 0/24 | F 0/24 | F 0/24 | P 11/24 | P 23/24 | P 10/24 | F 0/24 | F 1/24 | P 10/24 |
| 5 | nr_dvalcap_anchor | 13/26 | 228 | 0 | 4.00 | F 0/0 0.980 0.02-2.68 c0.41 | F 0/0 0.960 0.02-1.75 c0.86 | F 0/0 0.983 0.02-2.21 c0.43 | F 0/1200+0/0 | F f129 +40/+360/+70 d20 | F 0/81 | F f99 +40 d2 | P 14/24 | P 19/24 | P 15/24 | P 22/24 | P 17/24 | P 17/24 | P 19/24 | P 21/24 | F 0/24 | P 22/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | P 24/24 | F 4/24 | P 9/24 | P 11/24 | P 9/24 |
| 6 | nr_none_anchor | 11/26 | 38 | 0 | 3.97 | F 0/0 0.990 0.04-3.10 c0.43 | F 0/0 0.958 0.01-2.10 c0.74 | F 0/0 0.982 0.01-1.85 c0.42 | F 0/1200+0/0 | F f21 +770/+300/+270 d8 | F 0/81 | F f17 +770 d6 | P 14/24 | F 3/24 | F 4/24 | P 22/24 | P 17/24 | P 17/24 | P 19/24 | P 21/24 | F 0/24 | P 22/24 | F 0/24 | F 0/24 | F 0/24 | P 11/24 | P 24/24 | P 10/24 | F 0/24 | F 1/24 | P 10/24 |
