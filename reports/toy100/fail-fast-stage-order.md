# Cost audit for staged older-host screening

We are failing fast **between complete frozen-budget host runs**: a candidate advances only after the preceding strict gate passes. No attempted host is stopped early within its training budget. These are episode `index.json` training/evaluation seconds; process startup, archiving, regrading, and shared-disk waits are excluded.

| Evidence cohort / host | Runs | Strict failures | Median seconds per run |
| --- | ---: | ---: | ---: |
| Halton64: residual-student | 64 | 31 (48%) | 1.20 |
| Halton64: trajectory, conditional on residual pass | 33 | 31 (94%) | 1.17 |
| Halton64: mode-hold, conditional on both passes | 2 | 2 (100%) | 6.67 |
| Six complete ten-host screens: trajectory | 6 | 2 | 2.00 |
| Six complete ten-host screens: residual-student | 6 | 0 | 1.21 |
| Six complete ten-host screens: stripes | 6 | 4 | 5.97 |
| Six complete ten-host screens: mode-hold | 6 | 3 | 8.07 |
| Six complete ten-host screens: bars | 6 | 2 | 6.30 |
| Six complete ten-host screens: overlap | 6 | 3 | 7.93 |
| Six complete ten-host screens: blobs | 6 | 1 | 5.81 |
| Six complete ten-host screens: unequal mass | 6 | 2 | 20.95 |
| Six complete ten-host screens: unequal width | 6 | 0 | 9.29 |

The complete-screen cohort is the four predeclared κ rows at source `a3be165` plus κ=1.180 and 1.176 at frozen source `1c1a086`. The latter two reached all ten through staged selection; these counts are descriptive, **not unconditional failure probabilities**. Halton64's trajectory and mode-hold rates are also conditional on previous passes. We have no trajectory outcomes for its 31 residual failures. Mixing source epochs and nearby recipe families further limits prediction. All counts come from archived `index.json` records and strict stage receipts in `shared-halton64-v1` (`artifacts/toy100-accuracy/compatibility/shared-halton64-v1`), `network-floor-kappa-bracket-v1` (`artifacts/toy100-accuracy/compatibility/network-floor-kappa-bracket-v1`), `shared-kappa-boundary-v1` (`artifacts/toy100-accuracy/compatibility/shared-kappa-boundary-v1`), and `shared-kappa-fine-v1` (`artifacts/toy100-accuracy/compatibility/shared-kappa-fine-v1`).

For the **next predeclared ten-host screen**, use this fixed order: trajectory → residual-student → stripes → mode-hold → bars → overlap → blobs → intensity → unequal mass → unequal width. Trajectory and residual are both about 1–2 seconds; trajectory rejected 31/33 residual-passing Halton rows and 8/12 in the fine κ bracket, so it should go first. Stripes cost less than mode-hold and rejected four of the six complete-screen rows; bars uniquely rejected nearby κ=1.177 after the first four passed. The expensive unequal-mass and low-rejection unequal-width checks can wait until late. This order is a screening heuristic, not a change to any host's budget, threshold, seed, or optimizer. Every candidate that passes ten must still run a fresh full 19 and then the combined 22 gate.

For the six complete screens, applying that order retrospectively would have spent about 121 total measured case-seconds before first failures, versus 128 for the previous residual → trajectory → mode-hold → stripes order and its remaining hosts. This small difference is evidence about those six rows only. If future evidence changes the rejection pattern, declare a new order before testing. A genuine within-host early stop would need to prove that the remaining scheduled checkpoints cannot form the required terminal passing streak; any truncated case would be labeled screen-only and could not support a full-suite verdict. No such within-host truncation was used here.
