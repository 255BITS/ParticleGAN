# Staged 64-row shared-recipe search

This frozen-source search varied four global fields around the earlier 18/19 recipe: κ in [.99, 1.10], D/G LR multiplier in [.97, 1.03], prior LR multiplier in [1.90, 2.10], and output noise σ in [.0285, .0295]. Halton indices 1–64 with bases 2, 3, 5, and 7 fixed every candidate before training. All other fields, host resources, seeds, budgets, and gates stayed fixed. The executable source was commit `1c1a086` and the base config SHA-256 was `52f2971a…`; historical global output-noise RNG remained in use.

The staged result was **no winner**. Residual-student passed 33/64. Of those 33, trajectory passed 2/33. Both survivors failed mode-hold, so the remaining seven bottlenecks and a fresh full-19 replay were correctly skipped. These subset results cannot be called a full-19 failure or pass for any row.

| Row | κ | D/G LR | Prior LR | Output σ | Residual-student | Trajectory MSE | Mode-hold |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 29 | 1.06906 | 1.01074 | 2.0616 | .0287245 | PASS | .000878, PASS | 5/8 modes, HQ .589, FAIL |
| 57 | 1.05703 | .97815 | 1.9912 | .0286662 | PASS | .000875, PASS | 5/8 modes, HQ 1.0, FAIL |

Both mode-hold failures persisted through the final five checks. Row 29 consistently missed modes 1, 3, and 7; row 57 missed 3, 5, and 7. Row 57's perfect final HQ shows its problem was absent support, rather than noisy samples around the modes it retained. Among the other 31 residual-student passing rows, the closest trajectory failure had a worst final-five MSE of .02147 versus the .02 limit (row 55); none held a passing final suffix.

The narrowest joint boundary in saved evidence is the earlier exact-base κ bracket: κ=1.15 passed residual-student and mode-hold 8/8 but failed trajectory at MSE .0353; κ=1.20 passed residual-student and trajectory at MSE .000895 but ended mode-hold at 7/8 with HQ 1.0. A four-value κ-only bracket at **1.16, 1.17, 1.18, and 1.19**, keeping the exact 18/19 base otherwise fixed, would test whether these three gates overlap between the two observations. It should use the same staged order and run the remaining seven/full 19 only after a strict three-host pass. This is a proposed diagnostic, not a promotion: unequal-mass and overlap also failed at κ=1.20, so a three-host pass alone would not establish a common recipe.

Complete evidence (`artifacts/toy100-accuracy/compatibility/shared-halton64-v1`) includes the predeclared 64 config bytes and hashes, Halton coordinates, persistent driver hash, frozen source archive, every attempted compressed episode, independent stage grades, and explicit skip receipts. All 831 retained files matched RAM originals by SHA-256. Every attempted stage was independently regraded from the relocated archive, including the source and optimizer/noise receipts.
