# Regularizer study -- promotion leaderboard

Champion = promotion candidate for the repo's default recipe. Sharp board requires: all seeds pass the repo bar (100 modes & hq>=0.9), mean hq>=0.95, zero collapses. If per_mode_std_ratio < 0.8, the row is flagged 'variance-compressed' and cannot hold CHAMPION.

290 runs / 58 groups from the main tree; 60 audit runs annotated 12 group(s) with `per_mode_std_ratio` (marked ᴬ).


## 1. Sharp board -- the promotion track

All seeds clear the repo bar (100 modes & hq >= 0.9 by step 7000), mean hq >= 0.95, zero collapses. Ranked by mean final exact W1, lowest first.

| # | group | arm | coeff | n | W1 (exact) | hq | mode recall | per-mode σ ratio | collapse | steps→bar (n reached) | bar pass | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `b_cap_c1p0_lr2p0` | b_cap | 1.0 | 5 | 0.2901 | 0.986 | 0.998 | 2.727 ± 0.392 ᴬ | 0.00 | 5500 (5/5) | 1.00 | **CHAMPION** |
| 2 | `b_cap_c1p0_nl1_lr2p0` | b_cap | 1.0 | 5 | 0.3064 | 0.992 | 0.992 | 2.143 ± 0.245 | 0.00 | 5620 (5/5) | 1.00 |  |
| 3 | `b_cap_c1p0_nlinf_lr2p0` | b_cap | 1.0 | 5 | 0.3252 | 0.990 | 0.988 | 2.419 ± 0.341 | 0.00 | 5580 (5/5) | 1.00 |  |
| 4 | `b_cap_c1p0_nl1` | b_cap | 1.0 | 5 | 0.3312 | 0.978 | 0.992 | 2.579 ± 0.220 | 0.00 | 4860 (5/5) | 1.00 |  |
| 5 | `d_asym_c1p0_lr2p0` | d_asym | 1.0 | 5 | 0.3377 | 0.986 | 0.996 | 2.810 ± 0.154 ᴬ | 0.00 | 5500 (5/5) | 1.00 |  |
| 6 | `b_cap_c1p0_anndel` | b_cap | 1.0 | 5 | 0.3671 | 0.955 | 0.980 | 2.852 ± 0.459 | 0.00 | 3860 (5/5) | 1.00 |  |
| 7 | `b_cap_c1p0` | b_cap | 1.0 | 5 | 0.3675 | 0.967 | 0.976 | 2.873 ± 0.451 ᴬ | 0.00 | 3860 (5/5) | 1.00 |  |
| 8 | `b_cap_c1p0_nlinf` | b_cap | 1.0 | 5 | 0.3888 | 0.967 | 0.968 | 3.086 ± 0.181 | 0.00 | 4280 (5/5) | 1.00 |  |
| 9 | `d_asym_c1p0` | d_asym | 1.0 | 5 | 0.4139 | 0.962 | 0.966 |  | 0.00 | 4500 (5/5) | 1.00 |  |
| 10 | `c_eikonal_c0p1_lr2p0` | c_eikonal | 0.1 | 5 | 0.4143 | 0.953 | 0.984 | 4.143 ± 0.381 ᴬ | 0.00 | 5380 (5/5) | 1.00 |  |
| 11 | `b_cap_c1p0_lazy16` | b_cap | 1.0 | 5 | 0.4425 | 0.960 | 0.948 |  | 0.00 | 4920 (5/5) | 1.00 |  |
| 12 | `c_eikonal_c0p1_lazy16` | c_eikonal | 0.1 | 5 | 0.4459 | 0.960 | 0.938 |  | 0.00 | 4920 (5/5) | 1.00 |  |
| 13 | `c_eikonal_c0p1` | c_eikonal | 0.1 | 5 | 0.4522 | 0.962 | 0.952 | 4.804 ± 0.510 ᴬ | 0.00 | 3840 (5/5) | 1.00 |  |
| 14 | `e_interp_c1p0` | e_interp | 1.0 | 5 | 0.4564 | 0.954 | 0.960 | 5.427 ± 0.574 ᴬ | 0.00 | 4700 (5/5) | 1.00 |  |
| 15 | `c_eikonal_c1p0` | c_eikonal | 1.0 | 5 | 0.4570 | 0.958 | 0.948 |  | 0.00 | 4740 (5/5) | 1.00 |  |
| 16 | `d_asym_c0p1` | d_asym | 0.1 | 5 | 0.4581 | 0.955 | 0.964 |  | 0.00 | 4160 (5/5) | 1.00 |  |

## 2. Transport board -- coverage of the target

Mean mode_recall >= 0.95, zero collapses. Ranked by mean final exact W1, lowest first. A group can top this board while being too blurry for the sharp board; CHAMPION is never awarded here.

| # | group | arm | coeff | n | W1 (exact) | hq | mode recall | per-mode σ ratio | collapse | steps→bar (n reached) | bar pass | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a_r1r2_c1p0` | a_r1r2 | 1.0 | 5 | 0.1834 | 0.598 | 1.000 | 3.594 ± 0.076 ᴬ | 0.00 | 7000 (0/5) | 0.00 |  |
| 2 | `a_r1r2_c0p3_lr2p0` | a_r1r2 | 0.3 | 5 | 0.1899 | 0.841 | 1.000 |  | 0.00 | 7000 (0/5) | 0.00 |  |
| 3 | `a_r1r2_c1p0_lr2p0` | a_r1r2 | 1.0 | 5 | 0.1991 | 0.538 | 1.000 |  | 0.00 | 7000 (0/5) | 0.00 |  |
| 4 | `a_r1r2_c1p0_oadam` | a_r1r2 | 1.0 | 5 | 0.2031 | 0.609 | 1.000 | 4.282 ± 0.133 | 0.00 | 7000 (0/5) | 0.00 |  |
| 5 | `a_r1r2_c1p0_lr0p5` | a_r1r2 | 1.0 | 5 | 0.2137 | 0.562 | 1.000 |  | 0.00 | 7000 (0/5) | 0.00 |  |
| 6 | `a_r1r2_c0p3` | a_r1r2 | 0.3 | 5 | 0.2294 | 0.868 | 1.000 | 2.422 ± 0.122 ᴬ | 0.00 | 7000 (0/5) | 0.00 |  |
| 7 | `a_r1r2_c0p1_lr2p0` | a_r1r2 | 0.1 | 5 | 0.2608 | 0.943 | 1.000 | 2.666 ± 0.172 ᴬ | 0.00 | 5160 (5/5) | 1.00 |  |
| 8 | `b_cap_c1p0_lr2p0` | b_cap | 1.0 | 5 | 0.2901 | 0.986 | 0.998 | 2.727 ± 0.392 ᴬ | 0.00 | 5500 (5/5) | 1.00 |  |
| 9 | `a_r1r2_c0p1_oadam` | a_r1r2 | 0.1 | 5 | 0.2971 | 0.889 | 0.996 | 4.276 ± 0.650 | 0.00 | 6200 (4/5) | 0.80 |  |
| 10 | `b_cap_c1p0_nl1_lr2p0` | b_cap | 1.0 | 5 | 0.3064 | 0.992 | 0.992 | 2.143 ± 0.245 | 0.00 | 5620 (5/5) | 1.00 |  |
| 11 | `b_cap_c1p0_nlinf_lr2p0` | b_cap | 1.0 | 5 | 0.3252 | 0.990 | 0.988 | 2.419 ± 0.341 | 0.00 | 5580 (5/5) | 1.00 |  |
| 12 | `b_cap_c1p0_nl1` | b_cap | 1.0 | 5 | 0.3312 | 0.978 | 0.992 | 2.579 ± 0.220 | 0.00 | 4860 (5/5) | 1.00 |  |
| 13 | `d_asym_c1p0_lr2p0` | d_asym | 1.0 | 5 | 0.3377 | 0.986 | 0.996 | 2.810 ± 0.154 ᴬ | 0.00 | 5500 (5/5) | 1.00 |  |
| 14 | `a_r1r2_c0p1` | a_r1r2 | 0.1 | 5 | 0.3381 | 0.943 | 0.984 | 2.533 ± 0.256 ᴬ | 0.00 | 4500 (5/5) | 1.00 |  |
| 15 | `b_cap_c1p0_annlin` | b_cap | 1.0 | 5 | 0.3453 | 0.929 | 0.990 | 2.472 ± 0.232 | 0.00 | 3700 (5/5) | 1.00 |  |
| 16 | `b_cap_c1p0_oadam` | b_cap | 1.0 | 5 | 0.3665 | 0.935 | 0.986 | 5.298 ± 0.868 | 0.00 | 5880 (5/5) | 1.00 |  |
| 17 | `b_cap_c1p0_anndel` | b_cap | 1.0 | 5 | 0.3671 | 0.955 | 0.980 | 2.852 ± 0.459 | 0.00 | 3860 (5/5) | 1.00 |  |
| 18 | `b_cap_c1p0` | b_cap | 1.0 | 5 | 0.3675 | 0.967 | 0.976 | 2.873 ± 0.451 ᴬ | 0.00 | 3860 (5/5) | 1.00 |  |
| 19 | `b_cap_c1p0_nlinf` | b_cap | 1.0 | 5 | 0.3888 | 0.967 | 0.968 | 3.086 ± 0.181 | 0.00 | 4280 (5/5) | 1.00 |  |
| 20 | `b_cap_c0p1` | b_cap | 0.1 | 5 | 0.3944 | 0.950 | 0.954 |  | 0.00 | 4140 (5/5) | 1.00 |  |
| 21 | `b_cap_c1p0_lr0p5` | b_cap | 1.0 | 5 | 0.4034 | 0.931 | 0.970 |  | 0.00 | 4280 (5/5) | 1.00 |  |
| 22 | `d_asym_c1p0` | d_asym | 1.0 | 5 | 0.4139 | 0.962 | 0.966 |  | 0.00 | 4500 (5/5) | 1.00 |  |
| 23 | `c_eikonal_c0p1_lr2p0` | c_eikonal | 0.1 | 5 | 0.4143 | 0.953 | 0.984 | 4.143 ± 0.381 ᴬ | 0.00 | 5380 (5/5) | 1.00 |  |
| 24 | `c_eikonal_c0p1_anndel` | c_eikonal | 0.1 | 5 | 0.4488 | 0.942 | 0.952 | 4.916 ± 0.460 | 0.00 | 3840 (5/5) | 1.00 |  |
| 25 | `c_eikonal_c0p1` | c_eikonal | 0.1 | 5 | 0.4522 | 0.962 | 0.952 | 4.804 ± 0.510 ᴬ | 0.00 | 3840 (5/5) | 1.00 |  |
| 26 | `e_interp_c1p0` | e_interp | 1.0 | 5 | 0.4564 | 0.954 | 0.960 | 5.427 ± 0.574 ᴬ | 0.00 | 4700 (5/5) | 1.00 |  |
| 27 | `d_asym_c0p1` | d_asym | 0.1 | 5 | 0.4581 | 0.955 | 0.964 |  | 0.00 | 4160 (5/5) | 1.00 |  |
| 28 | `e_interp_c0p1` | e_interp | 0.1 | 5 | 0.4652 | 0.934 | 0.952 |  | 0.00 | 5320 (5/5) | 1.00 |  |
| 29 | `e_interp_c0p02` | e_interp | 0.02 | 5 | 0.5010 | 0.877 | 0.954 |  | 0.00 | 7000 (0/5) | 0.00 |  |

## 3. Wasserstein runs -- informational, NOT promotable

`loss_type: wasserstein` optimizes a different objective, so these W1 numbers are not commensurable with the logistic-objective boards above. They are listed for reference only and are never pooled with, ranked against, or promoted over the logistic groups.

| # | group | arm | coeff | n | W1 (exact) | hq | mode recall | per-mode σ ratio | collapse | steps→bar (n reached) | bar pass | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `wgan_a_r1r2_c1p0` | a_r1r2 | 1.0 | 5 | 0.1540 | 0.876 | 1.000 | 2.374 ± 0.083 | 0.00 | 7000 (0/5) | 0.00 |  |
| 2 | `wgan_b_cap_c1p0` | b_cap | 1.0 | 5 | 0.1635 | 0.951 | 0.998 | 1.735 ± 0.242 | 0.00 | 5080 (5/5) | 1.00 |  |
| 3 | `wgan_b_cap_c0p1` | b_cap | 0.1 | 5 | 0.1643 | 0.944 | 0.998 | 1.493 ± 0.165 | 0.00 | 5180 (5/5) | 1.00 |  |
| 4 | `wgan_a_r1r2_c0p1` | a_r1r2 | 0.1 | 5 | 0.1653 | 0.898 | 1.000 | 2.163 ± 0.225 | 0.00 | 6040 (5/5) | 1.00 |  |
| 5 | `wgan_c_eikonal_c0p1` | c_eikonal | 0.1 | 5 | 0.1771 | 0.938 | 1.000 | 2.050 ± 0.483 | 0.00 | 5640 (5/5) | 1.00 |  |
| 6 | `wgan_e_interp_c0p1` | e_interp | 0.1 | 5 | 0.1925 | 0.895 | 0.994 | 3.158 ± 0.477 | 0.00 | 5880 (4/5) | 0.80 |  |
| 7 | `wgan_c_eikonal_c1p0` | c_eikonal | 1.0 | 5 | 0.1955 | 0.946 | 0.998 | 2.344 ± 0.264 | 0.00 | 5200 (5/5) | 1.00 |  |
| 8 | `wgan_e_interp_c1p0` | e_interp | 1.0 | 5 | 0.1991 | 0.978 | 0.994 | 1.682 ± 0.334 | 0.00 | 5060 (5/5) | 1.00 |  |
| 9 | `wgan_f_none_c0p0` | f_none | 0.0 | 5 | 2.7324 | 0.247 | 0.380 | 11.389 ± 3.091 | 0.80 | 7000 (0/5) | 0.00 |  |

## Notes

1. **steps→bar** is the first eval step in `metrics.jsonl` where `modes >= 100` and `hq >= 0.9` in the same row. Seeds that never get there (or that have no usable timeseries) count as 7000, so a group with `n reached` below `n` reports a lower bound on speed. **bar pass** is the fraction of seeds that did reach it.

2. **per-mode σ ratio** is `final.per_mode_std_ratio` -- generated per-mode spread over the real per-mode spread. Blank means the metric predates the group's summaries, which is *not* treated as a failure: an unmeasured group is never flagged. Cells marked ᴬ come from the deterministic re-run in the audit tree; every other column on that row still comes from the original run.

3. A row flagged **variance-compressed** (`per_mode_std_ratio` < 0.8) keeps its rank but cannot hold CHAMPION: tight, under-dispersed clusters flatter W1 and hq precisely by failing to reproduce the target's spread, so the title passes to the next unflagged row.

4. `collapse` is the fraction of seeds with `collapse_events > 0`. Both promotion boards require it to be exactly 0.

