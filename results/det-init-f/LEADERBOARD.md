# Family F particle-prior screen

Torch `2.14.0+cpu` git `08187d9e0fba026dc8217405802ab5381dc88d90` file `/home/ubuntu/.local/lib/python3.12/site-packages/torch/__init__.py`.

Determinism: 128 pairs checked, 0 mismatches.

## Calibration (random K3P init on this CPU build)

| init | ring | unequal | ring seeds | unequal seeds |
| --- | --- | --- | --- | --- |
| k3p | 5/8 | 2/8 | P8 F7 F5 P8 F8 P8 P8 P8 | P F F F F F F P |

## Ranked screen

Rank is ring passes, then unequal-mass passes, then name. `P`/`F` plus ring mode count. `.` is not finished.

| rank | init | ring | unequal | ring seeds | unequal seeds |
| --- | --- | --- | --- | --- | --- |
| 1 | qr_pb_pq_sobol_b | 7/8 | 6/8 | P8 P8 F8 P8 P8 P8 P8 P8 | P P F P P P P F |
| 2 | qr_pb_pq_lhs_b | 7/8 | 5/8 | P8 P8 F6 P8 P8 P8 P8 P8 | P F P F P P F P |
| 3 | hid_q_r2_s | 7/8 | 1/8 | P8 P8 P8 F8 P8 P8 P8 P8 | F F F F P F F F |
| 4 | qr_pb_pq_sobol_x | 6/8 | 6/8 | F7 P8 P8 P8 P8 F7 P8 P8 | P P F P P P P F |
| 5 | qr_pb_pq_strat | 6/8 | 4/8 | F8 P8 F6 P8 P8 P8 P8 P8 | F P F F P F P P |
| 6 | hid_q_halton_b | 6/8 | 1/8 | P8 F7 P8 P8 P8 P8 P8 F8 | F F F F F F F P |
| 7 | hid_q_lhs | 6/8 | 0/8 | P8 P8 P8 F8 F7 P8 P8 P8 | F F F F F F F F |
| 8 | hid_q_lhs_b | 6/8 | 0/8 | P8 F7 P8 P8 P8 P8 F7 P8 | F F F F F F F F |
| 9 | qr_pb_pq_strat_s | 5/8 | 6/8 | P8 F8 F7 P8 P8 F7 P8 P8 | P P F P P P F P |
| 10 | qr_pb_pq_halton_s | 5/8 | 5/8 | F7 P8 F3 F6 P8 P8 P8 P8 | P F P F P P F P |
| 11 | qr_pb_pq_weyl_s | 5/8 | 5/8 | P8 F7 P8 P8 P8 P8 F4 F7 | P F P P F P F P |
| 12 | hid_q | 5/8 | 4/8 | P8 F7 F6 P8 P8 F6 P8 P8 | F P P P F P F F |
| 13 | qr_pb_pq_weyl | 5/8 | 4/8 | P8 P8 P8 P8 F8 F6 F6 P8 | F P F P F F P P |
| 14 | qr_pb_pq | 5/8 | 3/8 | P8 F7 P8 P8 P8 P8 F1 F8 | F F F P P F F P |
| 15 | qr_pb_pq_r2_s | 5/8 | 3/8 | F8 F7 P8 F7 P8 P8 P8 P8 | P P P F F F F F |
| 16 | hid_q_weyl_b | 5/8 | 2/8 | F6 P8 F8 P8 P8 F0 P8 P8 | F F P F F F P F |
| 17 | hid_q_sobol_b | 5/8 | 1/8 | F3 F8 P8 P8 P8 F7 P8 P8 | F F F F F F P F |
| 18 | hid_q_halton_s | 5/8 | 0/8 | F7 P8 P8 P8 P8 F8 P8 F8 | F F F F F F F F |
| 19 | qr_pb_pq_lhs_x | 4/8 | 8/8 | F8 P8 P8 F8 P8 P8 F7 F7 | P P P P P P P P |
| 20 | qr_pb_pq_halton_b | 4/8 | 7/8 | F7 P8 F7 P8 F8 F7 P8 P8 | P P P F P P P P |
| 21 | qr_pb_pq_strat_x | 4/8 | 7/8 | F8 F7 P8 P8 F8 F5 P8 P8 | P P P P P P P F |
| 22 | hid_q_r2_x | 4/8 | 5/8 | F2 P8 F7 F6 P8 P8 F7 P8 | P P P P F P F F |
| 23 | qr_pb_pq_r2_x | 4/8 | 5/8 | P8 P8 F7 P8 F0 F8 P8 F8 | P P P P P F F F |
| 24 | qr_pb_pq_strat_b | 4/8 | 5/8 | F8 F7 P8 P8 F8 F5 P8 P8 | P F F P P P F P |
| 25 | qr_pb_pq_lhs_s | 4/8 | 4/8 | F8 P8 P8 F7 F7 F8 P8 P8 | F F P P F F P P |
| 26 | hid_q_r1_s | 4/8 | 2/8 | F7 F7 F7 P8 P8 P8 F7 P8 | P F F F F F F P |
| 27 | hid_q_sobol | 4/8 | 2/8 | F7 P8 P8 P8 P8 F7 F6 F4 | F P F F P F F F |
| 28 | hid_q_strat_x | 4/8 | 2/8 | P8 F8 P8 F1 F7 P8 P8 F7 | F F F P P F F F |
| 29 | hid_q_weyl_s | 4/8 | 2/8 | P8 F6 P8 P8 F0 F7 F2 P8 | P F F F F F F P |
| 30 | qr_pb_pq_weyl_x | 4/8 | 2/8 | F6 F7 P8 P8 P8 P8 F8 F7 | P F F F P F F F |
| 31 | hid_q_fib_b | 4/8 | 1/8 | F5 P8 P8 P8 F3 P8 F4 F7 | F P F F F F F F |
| 32 | hid_q_r1_b | 4/8 | 1/8 | F0 P8 F6 P8 F7 F5 P8 P8 | F F F F F P F F |
| 33 | hid_q_r2 | 4/8 | 1/8 | F7 P8 P8 P8 P8 F7 F6 F1 | F F F F P F F F |
| 34 | hid_q_r2_b | 4/8 | 1/8 | F7 P8 P8 P8 F8 F8 F6 P8 | F P F F F F F F |
| 35 | hid_q_sobol_s | 4/8 | 1/8 | P8 F7 F6 F7 P8 P8 P8 F7 | F F F F P F F F |
| 36 | hid_q_sobol_x | 4/8 | 1/8 | P8 P8 F0 F7 F7 F8 P8 P8 | F F F F F P F F |
| 37 | hid_q_r1_x | 4/8 | 0/8 | F7 P8 P8 F7 P8 P8 F7 F5 | F F F F F F F F |
| 38 | hid_q_strat_b | 4/8 | 0/8 | P8 F8 P8 F1 F7 P8 P8 F7 | F F F F F F F F |
| 39 | qr_pb_pq_sobol | 3/8 | 5/8 | P8 F0 F7 P8 F7 P8 F7 F8 | P P F P F P P F |
| 40 | qr_pb_pq_sobol_s | 3/8 | 5/8 | F7 F7 F7 F5 P8 P8 P8 F4 | P F P P F P F P |
| 41 | qr_pb_pq_fib_b | 3/8 | 4/8 | P8 F5 P8 F0 F5 F0 P8 F1 | F F P F P P P F |
| 42 | qr_pb_pq_fib_x | 3/8 | 4/8 | P8 F3 F7 P8 F1 F6 P8 F0 | F P F F F P P P |
| 43 | qr_pb_pq_r1_s | 3/8 | 4/8 | P8 P8 F6 F6 P8 F6 F6 F4 | P F F F P P P F |
| 44 | qr_pb_pq_weyl_b | 3/8 | 3/8 | F7 P8 F7 F7 F7 P8 P8 F3 | P F F F P F F P |
| 45 | hid_q_lhs_x | 3/8 | 2/8 | F7 F7 P8 F6 P8 P8 F7 F8 | F F P F P F F F |
| 46 | hid_q_fib_s | 3/8 | 1/8 | P8 F8 F6 F7 P8 F8 P8 F8 | P F F F F F F F |
| 47 | hid_q_lhs_s | 3/8 | 1/8 | P8 F7 F7 P8 F0 P8 F7 F7 | F F F F P F F F |
| 48 | hid_q_strat | 3/8 | 1/8 | F1 P8 P8 P8 F7 F8 F7 F2 | F F F P F F F F |
| 49 | qr_pb_pq_fib_s | 3/8 | 1/8 | P8 F7 F6 P8 P8 F7 F7 F8 | F F F F P F F F |
| 50 | qr_pb_pq_halton | 2/8 | 7/8 | F7 P8 F7 F6 F7 P8 F7 F0 | F P P P P P P P |
| 51 | qr_pb_pq_r1_b | 2/8 | 7/8 | F6 F8 F7 F6 P8 F7 P8 F7 | P P F P P P P P |
| 52 | qr_pb_pq_halton_x | 2/8 | 6/8 | F5 F7 F7 P8 F5 F8 F6 P8 | P P P P P F F P |
| 53 | qr_pb_pq_fib | 2/8 | 5/8 | F6 F7 P8 F2 F8 F6 P8 F6 | P P P P F F P F |
| 54 | qr_pb_pq_lhs | 2/8 | 5/8 | F5 F5 F7 P8 F6 F0 P8 F7 | P P F F P P P F |
| 55 | qr_pb_pq_r1 | 2/8 | 4/8 | F7 F7 F7 F7 P8 F6 F5 P8 | P P F F P F P F |
| 56 | hid_q_halton_x | 2/8 | 3/8 | P8 F6 P8 F8 F6 F7 F7 F7 | F F P P F P F F |
| 57 | qr_pb_pq_r1_x | 2/8 | 3/8 | F2 F7 F4 F7 F7 F7 P8 P8 | F P F F P P F F |
| 58 | hid_q_fib | 2/8 | 1/8 | F7 P8 F7 F7 F8 F8 F7 P8 | F F F F P F F F |
| 59 | hid_q_halton | 2/8 | 1/8 | F5 F8 F8 F6 F8 F6 P8 P8 | F F F F P F F F |
| 60 | hid_q_r1 | 2/8 | 1/8 | P8 F6 F6 F7 F7 P8 F0 F0 | F F F F P F F F |
| 61 | hid_q_weyl_x | 2/8 | 1/8 | P8 F7 F7 F7 F5 F6 P8 F7 | F P F F F F F F |
| 62 | hid_q_strat_s | 2/8 | 0/8 | F4 P8 F7 F7 P8 F8 F7 F6 | F F F F F F F F |
| 63 | qr_pb_pq_r2_b | 1/8 | 5/8 | F1 F7 F8 F6 P8 F6 F8 F8 | F F P P P P F P |
| 64 | hid_q_fib_x | 1/8 | 4/8 | F7 F7 F7 F7 F7 P8 F2 F7 | F F F P F P P P |

## Priority gates (repo seed)

| init | ring | unequal | stripes | blobs | hold | shift | grid100 | rotated100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qr_pb_pq_sobol_b | P | P | P | P | post-fail | F | F | F |
| qr_pb_pq_lhs_b | P | P | P | F | post-fail | F | F | F |

## Priority gates (repo seed, detail)

`post-fail` is `POST_CONVERGENCE_FAIL`. Both finalists pass ring, unequal mass, and stripes at offset 0. `qr_pb_pq_sobol_b` also passes blobs (4 modes, hq 0.94, suffix 15). `qr_pb_pq_lhs_b` misses blobs on hq 0.875 against 0.90.

Hold reaches a converged ring and then loses it before the hold budget ends. Sobol-box converges at step 1797 and fails the hold at step 1836 (39 hold checks, 325 settling failures). LHS-box converges at step 1400, holds 1043 checks, and fails at step 2443.

Shift's stationary block passes 5/5 for both. The shift-recovery block then scores 0/120 with 0 modes, so the gate is FAIL. LHS continued-hold is 120/120; that does not clear recovery.

Grid100 and rotated100 both reach 100 modes by step 1000 and never record a passing terminal check (0/5). Grid finishes near hq 0.93–0.94 with precision under 0.97, covariance eigenvalue ratio above 1.7, and radial median ratio above 1.4. Rotated finishes near precision 0.95, short of 0.97. These are absolute CPU gate results. Random K3P was not re-run on hold, shift, stripes, blobs, grid, or rotated on this build.

## Reading

Rank is ring passes, then unequal-mass passes, then name. Calibration on this build is random K3P at ring 5/8 and unequal 2/8. The published weight inits, with their published priors, land on that same ring rate and a better unequal rate: `hid_q` 5/8 and 4/8, `qr_pb_pq` 5/8 and 3/8.

The two finalists keep the `qr_pb_pq` weights and replace only the particle table with a low-discrepancy cube mapped into a variance-matched box (half-width = declared std × √3), then exact per-coordinate mean 0 and population std 0.5. `qr_pb_pq_sobol_b` is unscrambled Sobol. `qr_pb_pq_lhs_b` is a centered Latin hypercube. Ring 7/8 and unequal 6/8 and 5/8 both beat calibration on both gates. The shared ring miss is offset 202: Sobol has 8 modes but suffix 3, LHS has 6 modes. Offset 202 is also a random-K3P ring miss (5 modes).

Inverse-normal (`g`) does not win. The published R2 Gaussian prior `qr_pb_pq` is rank 14. Exact marginals without the box (`x`) can perfect unequal mass (`qr_pb_pq_lhs_x` is 8/8) while leaving ring at 4/8. The equal-norm sphere lifts the Householder arm's ring (`hid_q_r2_s` is 7/8) and drops unequal to 1/8. Bounded support plus exact scale is the combination that moves both screen gates.

Send `qr_pb_pq_sobol_b` and `qr_pb_pq_lhs_b` to an A6000. CPU remains a screen: grid and rotated quality, and the hold/shift failures, need that machine before either init is treated as a recipe change. Patches are `reports/toy100/det-init-f/variants/<name>.patch` against develop. Pass the filename stem as `--init`.
