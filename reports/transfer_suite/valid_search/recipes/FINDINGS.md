# Shared optimizer search: six valid data toys

**No shared recipe passed all six data toys.** The best two fully checked recipes each passed four, with different failures. This search does establish a rare-mass GAN witness at the original 256-particle capacity and original 1,200-step budget, without changing the loss or regularizer. It does not justify promoting a new default.

| Shared optimizer recipe | Broad | Rare mass | Unequal width | Anisotropic | Overlap | Spiral | Sustained total |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| G .00075, D .0015, prior .0225; Adam(0,.999) | PASS | PASS | FAIL | FAIL | PASS | PASS | 4/6 |
| G .00075, D .0015, prior .015; Adam(0,.995) | PASS | FAIL | PASS | PASS | Final only | PASS | 4/6 |
| G .00075, D .00225, prior .00375; Adam(0,.999) | PASS | FAIL | FAIL | FAIL | PASS | PASS | 3/6 |

All three use one G and one D update per outer step. Their exact names are `b999_lr075_d2_p30`, `b995_lr075_d2_p20`, and `b999_lr075_d3_p5`. Every recipe retains Rp logistic, b_cap coefficient 3 / kappa 1.25, prior regularization .05, no particle L2, the original G/D architecture, batch size 128 and cosine schedule. Each target uses its original budget: 1,200 outer steps, except spiral at 1,600. EMA never contributes to the live verdict.

## What improved

The first recipe passes rare mass for the final **6/24** observations, confirming at step 1,150. Final live HQ is **97.754%**, mass TV **.03400**, component covariance error **.65821**, minimum eigenvalue ratio **.54950**, and minimum mass ratio **.90739**. Its rare 2% component receives 2.173% of evaluated samples. EMA also passes separately. The same recipe passes overlap for the final 7/24 checks, broad modes for 16/24, and spiral for all 24.

Relative to the original vector host, this keeps D's absolute LR at .0015, lowers G's LR by 25%, raises the prior LR from .01 to .0225, and changes beta2 from .99 to .999. The result supports coordinated optimization as one way to address the rare-component failure; it is not an isolated one-knob causal experiment.

The second recipe passes unequal width for the final **7/24** checks: covariance error **.42185** and minimum eigenvalue ratio **.89768**, with HQ **98.022%**. It also retains broad, anisotropic and spiral success. Its overlap ends with all numerical metrics passing but only a three-check passing suffix, so it remains a failure under the frozen stability rule.

## What remains wrong

The rare-mass recipe misses unequal-width covariance (**1.76637**, bound .85), despite passing every other final bound. One component has covariance error 6.5522, while the other three errors are .1121, .0914 and .3098. It also misses anisotropic covariance (**1.02653**, bound .85). High HQ and correct occupancy do not guarantee correct within-component spread.

The width recipe badly regresses rare mass: covariance error **6.90173** and minimum eigenvalue ratio **.000156**. The two wins cannot be combined by choosing a different recipe for each task when ranking a shared recipe.

Reduced-G-cadence candidates made overlap stable, but did not solve rare mass or unequal width within the unchanged outer budget. Momentum and LR combinations also produced several final-only passes; these remain failures. The [complete matrix](MATRIX.md) preserves all 18 cards and every untested cell.

## Bounded protocol and provenance

Eighteen coordinated cards were frozen before the screen. The screen evaluated each on rare mass, unequal width and overlap: **54 episodes**. The three finalists were chosen by hard-screen sustained pass count, then mean normalized final shortfall, and each received the remaining three data toys: **9 additional episodes**. Total: **63 GAN episodes**, **1,512 fixed live observations**, **434.235 seconds** summed recorded wall time on a shared CPU host. No longer-training or extra-capacity runs were performed.

Every run uses seed 0, complete 24-observation curves, the original behavioral thresholds, and at least five final passing observations. No diagnostic dynamics, bad-architecture cases, reserved families, required regressions or image tasks were evaluated in this sub-study. Those are separate parent evaluations. The declared cards were checked against 90 prior executed episodes from the existing screen, combinations, geometry and regression studies; none duplicated an effective configuration.

Source checkout: `a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff`. The source archive contains all 57 fingerprinted source files; numerical source stayed unchanged throughout. Full episode artifacts retain original and effective specs, exact shared candidate, live/EMA curves, actions, actual role-update counts, runtime and independently recomputed verdicts. All failures are retained.
