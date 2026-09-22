# Behavioral search notes

The rank is the [live leaderboard](../behavioral_baseline/README.md). This folder does not edit it.

A result counts only when `behavior-v1` trained all 9 toys and scored all 29 live bounds. Overall **PASS** also requires the 10 shared checks. The order is passed toys, then passed bounds, then live ring modes, HQ, and effective modes. EMA, a 3-toy screen, a late-checkpoint note, and a missing host are not wins. `MISSING` is not a pass.

## Leaderboard

Codex recorded these full runs. Shared checks passed. Five configs are overall PASS. The other seven full runs fail.

| Rank | Config | Toys | Bounds | Live modes | Live HQ | Effective modes | Overall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `bcap_k1p25_c3p0_lr0p85` | 9/9 | 29/29 | 8/8 | 100% | 7.54 | **PASS** |
| 2 | `b_cap_k1_25_c2_lr0_85` | 9/9 | 29/29 | 8/8 | 100% | 7.51 | **PASS** |
| 3 | `b_cap_k1_25_c2_0_no_l2` | 9/9 | 29/29 | 8/8 | 91.75% | 7.21 | **PASS** |
| 4 | `bcap_k1p25_c2p0_lr0p8` | 9/9 | 29/29 | 8/8 | 91.67% | 7.48 | **PASS** |
| 5 | `r1_r2_0_1` | 9/9 | 29/29 | 7/8 | 100% | 6.37 | **PASS** |
| 6 | `r1_r2_0_1_no_l2` | 8/9 | 28/29 | 8/8 | 100% | 7.61 | **FAIL** trajectory |
| 7 | `no_particle_l2` | 8/9 | 28/29 | 8/8 | 74.05% | 7.72 | **FAIL** mode_hold |
| 8 | `b_cap_no_l2_coeff_5` | 8/9 | 27/29 | 4/8 | 48.12% | 3.80 | **FAIL** mode_hold |
| 9 | `b_cap_no_l2_coeff_2` | 7/9 | 26/29 | 6/8 | 65.84% | 5.73 | **FAIL** trajectory, mode_hold |
| 10 | `locked_shared` | 7/9 | 26/29 | 5/8 | 82.30% | 4.73 | **FAIL** trajectory, mode_hold |
| 11 | `b_cap_no_l2_lr_half` | 6/9 | 26/29 | 7/8 | 92.33% | 6.71 | **FAIL** two_pole, trajectory, cover_leftover |
| 12 | `b_cap_no_l2_lr_quarter` | 2/9 | 17/29 | 5/8 | 75.29% | 4.58 | **FAIL** two_pole, trajectory, unipolar, cover_leftover, unused_token_hold, mid_scale_identity, mode_hold |

`r1_r2_0_1_no_l2` is not a win. The live ring is 8/8 at 100% HQ and the trajectory bound still fails, so the row is FAIL. The [62-attempt ledger](../behavioral_baseline/search/README.md) keeps 3-toy screens. Those rows did not train the other six hosts, so they are not on this rank.

## Full runs in this folder

Same 9-toy / 29-bound scorer. Shared checks passed. These rows are not on the leaderboard. This CPU does not reproduce the leaderboard’s `r1_r2_0_1` or `bcap_k1p25_c3p0_lr0p85` rows, so a PASS here is not a rank above those rows.

| Config | Toys | Bounds | Live modes | Live HQ | Overall | Failed live toys |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `r1_r2_0_1_l2_004` | 9/9 | 29/29 | 8/8 | 100% | **PASS** | None |
| `r1_r2_0_1_l2_005` | 9/9 | 29/29 | 8/8 | 100% | **PASS** | None |
| `r1_r2_0_1_l2_007` | 9/9 | 29/29 | 8/8 | 92.1% | **PASS** | None |
| `r1_r2_0_15` | 8/9 | 28/29 | 8/8 | 100% | **FAIL** | trajectory |
| `r1_r2_0_1_l2_006` | 8/9 | 28/29 | 6/8 | 92.2% | **FAIL** | mode_hold |
| `r1_r2_0_05` | 8/9 | 27/29 | 5/8 | 73.9% | **FAIL** | mode_hold |
| `music_cover` | 8/9 | 27/29 | 4/8 | 49.2% | **FAIL** | mode_hold |
| `r1_r2_0_1_l2_003` | 8/9 | 27/29 | 3/8 | 31.9% | **FAIL** | mode_hold |
| `r1_r2_0_1` (control) | 8/9 | 27/29 | 1/8 | 8.3% | **FAIL** | mode_hold |
| `r1_r2_0_1_cover_1` | 8/9 | 27/29 | 1/8 | 8.3% | **FAIL** | mode_hold |
| `no_vicreg` | 7/9 | 27/29 | 7/8 | 82.7% | **FAIL** | trajectory, mode_hold |
| `r1_r2_0_1_l2_001` | 7/9 | 27/29 | 6/8 | 100% | **FAIL** | trajectory, mode_hold |
| `r1_r2_0_08` | 7/9 | 27/29 | 5/8 | 100% | **FAIL** | trajectory, mode_hold |
| `no_l2_no_vicreg` | 7/9 | 26/29 | 6/8 | 82.4% | **FAIL** | trajectory, mode_hold |
| `bcap_k1p25_c3p0_lr0p85` (repro) | 7/9 | 26/29 | 5/8 | 66.7% | **FAIL** | trajectory, mode_hold |
| `r1_r2_0_12` | 7/9 | 26/29 | 3/8 | 59.1% | **FAIL** | trajectory, mode_hold |
| `r1_r2_0_1_no_vicreg` | 7/9 | 26/29 | 2/8 | 33.3% | **FAIL** | trajectory, mode_hold |
| `r1_r2_0_1_no_l2` | 7/9 | 26/29 | 1/8 | 7.9% | **FAIL** | trajectory, mode_hold |
| `base_regularization` | 7/9 | 24/29 | 6/8 | 58.3% | **FAIL** | cover_leftover, mode_hold |
| `no_cover` | 7/9 | 23/29 | 4/8 | 49.2% | **FAIL** | cover_leftover, mode_hold |
| `r1_r2_0_1_cover_0` | 7/9 | 23/29 | 1/8 | 8.3% | **FAIL** | cover_leftover, mode_hold |
| `r1_r2_0_2` | 6/9 | 23/29 | 3/8 | 40.6% | **FAIL** | trajectory, residual_student, mode_hold |

`r1_r2_0_1_l2_005` was trained twice and both runs passed. The leaderboard’s `r1_r2_0_1` fails mode hold on this CPU (1/8, 8.3% HQ). The leaderboard’s `bcap_k1p25_c3p0_lr0p85` fails trajectory and mode hold here (7/9, 26/29). Two-pole for that repro still matches the leaderboard value to about 1e-7. The leaderboard rows are unchanged.

`r1_r2_0_1_no_l2` fails in both places. On the leaderboard the failure is trajectory only. Here it is trajectory and mode hold.
