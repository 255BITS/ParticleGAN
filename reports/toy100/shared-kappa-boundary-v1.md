# Narrow κ boundary screen

Four predeclared values, κ=1.16, 1.17, 1.18, and 1.19, changed only `reg_kappa` and `name` from the exact earlier 18/19 shared recipe (base config SHA-256 `52f2971a…`, source `1c1a086`). Fixed host seeds, budgets, resources, and gates were retained. The staged screen used residual-student, trajectory, mode-hold, then seven vector/image bottlenecks; full 19 would run fresh only after 10/10.

| κ | Residual-student | Trajectory | Mode-hold | Remaining seven | Result |
| ---: | --- | --- | --- | --- | --- |
| 1.16 | PASS | FAIL, MSE .33625 | skipped | skipped | no promotion |
| 1.17 | PASS | FAIL, MSE .24595 | skipped | skipped | no promotion |
| 1.18 | PASS | PASS, MSE .000912 | PASS, 8/8 with five-check suffix | 5/7 | 8/10 only |
| 1.19 | PASS | FAIL, MSE .03868 | skipped | skipped | no promotion |

At κ=1.18, unequal-mass, unequal-width, overlap, blobs, and intensity passed. Stripes failed final HQ .84375, and bars reached only 3/4 quality modes (the fourth had fraction .09375 versus .125 required). This is a real overlap of the three earlier conflicting hosts and all tested vectors, but **not** a common-19 or common-22 result. No full-19 run was attempted.

Complete evidence (`artifacts/toy100-accuracy/compatibility/shared-kappa-boundary-v1`) includes all four exact configs and hashes, predeclared stage plan, compressed episodes, source archives, live curves, independent regrades, and skip receipts. All 94 retained files matched RAM originals by SHA-256, and relocated evidence independently regraded with the same 4/4 residual, 1/4 trajectory, 1/1 mode-hold, and 0/1 remaining-seven strict results.
