# Solvability audit: target and representation controls

Source HEAD `afe615264221eda47c5d1b7fd2cf4082552e30e9`; frozen metrics unchanged. Seed0 support construction, CPU thread1; evaluation seed990, target991, projection992. These are **oracle controls, not GAN training successes**. Thirteen ranking/seen-cadence specifications cover seven distinct target distributions. No new reserved data or architecture was inspected.

| Task | Direct target | 256 empirical particles | 512 | 1024 |
|---|---:|---:|---:|---:|
| vector_two_broad | PASS | PASS | PASS | PASS |
| vector_unequal_mass | PASS | FAIL | PASS | PASS |
| vector_unequal_width | PASS | PASS | PASS | PASS |
| vector_anisotropic | PASS | PASS | PASS | PASS |
| vector_overlap | PASS | PASS | PASS | PASS |
| vector_spiral | PASS | PASS | PASS | PASS |
| stress_fast_critic | PASS | PASS | PASS | PASS |
| stress_slow_critic | PASS | PASS | PASS | PASS |
| stress_small_batch | PASS | PASS | PASS | PASS |
| stress_large_critic | PASS | PASS | PASS | PASS |
| stress_long_horizon | PASS | PASS | PASS | PASS |
| stress_r1_r2 | PASS | PASS | PASS | PASS |
| reserved_alternating_critic_updates | PASS | PASS | PASS | PASS |

All 52 controls have complete24-point observation schedules; 51 pass every bound at every observation. These are stationary deterministic witnesses, so their confirmation steps do **not** measure training convergence.

The sole empirical-support failure is unequal_mass with256 particles: support counts[142,79,28,7], minimum covariance eigen ratio0.04877 below0.15. HQ0.97583, mass TV0.02375 and mean component covariance error0.34984 pass. The512/1024 supports pass, with minimum eigen ratios0.70441/0.57789. This is one fixed-seed construction per size, not a sampling-frequency or confidence estimate.

A separate deterministic moment-balanced256-particle construction allocates[141,77,33,5] atoms and places each component on a regular polygon at Mahalanobis radius sqrt(2). It passes all bounds: SW1 .03083, mass TV .00908, HQ1.0, covariance error .04009, minimum eigen ratio .90293, minimum target-mass ratio .95215. The exact existing G64x2,z4 MLP is set algebraically to identity over the first two prior coordinates. Therefore existing architecture and particle count can represent a passing output distribution. The polygon has matching moments but is not an exact Gaussian density; coarse metric passing is not density identity.

Tail sensitivity is strong but mathematically consistent: replacing one of4096 exact-target outputs by[-8,-8] on unequal_width changes covariance error from.04160 to3.00835, while HQ changes only.98828 to.98804. Covariance uses untrimmed Euclidean-assigned component samples, so small distant contamination is heavily penalized in narrow components. HQ and covariance are not contradictory; they test different distribution properties.

The2% target mode corresponds to5.12 of256 uniform particles. Mass granularity1/256 is compatible with the bounds (five atoms approximate2% closely); nonzero2D spread additionally needs at least three distinct non-collinear atoms. A batch128 has2.56 expected rare real samples and7.53% probability of none; batch64 has1.28 expected and27.45% probability of none. This is a legitimate imbalance challenge, not defective data. Counts among4096 sampled outputs do not identify distinct learned particles. Archived training JSON has metrics rather than output coordinates, so no claim about unique learned outputs is made here.

Practical conclusion: healthy vector/ring tasks have jointly attainable metrics at existing capacity. Adversarial optimization and stability remain to be demonstrated by actual training; these controls cannot explain every failure causally. Larger support reduces the observed rare-mode covariance problem in this one comparison, but no universal particle-count threshold or success probability is established.

Files: [protocol](protocol.json.gz), [full control summary](summary.json.gz), [balanced256 witness](vector_unequal_mass__balanced_support256.json.gz), [tail sensitivity](tail_sensitivity.json.gz), [source archive](source.tar.gz), [control script](audit.py), [balanced construction](balanced_control.py), [log](progress.log). All per-control files retain full24-point curves and support coordinates.
