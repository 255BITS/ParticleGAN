# Fresh-bank sample-anchor free-output stability

The sampled MST group-anchor objective passes this **free-output** time test. From the archived warm 1324 cloud, all 1,200 fixed 4,096-draw grades retain 8 modes and HQ ≥ 0.9. From the archived cold step-1 cloud, grades fail at updates 1 and 2, first pass at update 3, then pass through update 1,200. The terminal five cold checks pass. Both streams inferred 8 groups from every fresh native real128 minibatch; this number was inferred by the MST gap, never configured.

| Saved start | Passing checks | First pass | Subsequent failures | Final | Accepted MM targets |
|---|---:|---:|---:|---|---:|
| Warm 1324 pre-step | 1200/1200 | 1 | 0 | 8 modes, HQ 1.0000 | 1200/1200 |
| Cold 1 pre-step | 1198/1200 | 3 | 0 | 8 modes, HQ 1.0000 | 1200/1200 |

Each update draws exactly one real128 batch from the saved data RNG, forms current-bank groups with `mst_groups`, minimizes one active quadratic piece of the distinct-anchor plus nearest-group objective with `output_mm_step`, then adopts the exact free-point target only if the *whole current-bank objective* strictly decreases. There is no critic, generator, prior, Adam, or host training in this test. Quality uses the same fixed late-noise 4,096 draws at diagnostic clock `240+t` as the prior split-output calculation. The initial support, initial grade, and data RNG after 1,200 batches match that prior calculation exactly for both starts, so the bank streams and evaluation clock are aligned. All 2,400 proposed targets strictly reduced their current-bank objective after float32 conversion.

Unlike a fixed-bank rest proof, fresh empirical centroids keep the free supports moving: the per-update maximum particle displacement has median 0.0571 in the warm run (95th percentile 0.0831, maximum 0.1311). The cold run's first maximum is 2.919, then its median over all steps is 0.0568. Thus the measured stability is **quality and coverage stability under continual stochastic updates**, not parameter stillness or a fixed objective equilibrium. The same-stream split C+Q free-output comparator passed 1198/1200 warm checks; this anchor diagnostic passed all 1200, but the two methods optimize different objectives and no neural-method ranking follows.

This is a necessary geometry filter only. The MST assumes well-separated sample groups, and its count happened to be 8 on these two streams; it could fail or vary on other data. A joint G/prior pullback may fail to realize these output targets, or its accepted moves may perturb optimizer state, critic feedback, or other tasks. The mathematical exact-MM argument assumes fixed centers and freely movable outputs; this experiment adds finite-batch center motion and checks quality empirically. It does not establish indefinite training stability or production-host success.

The source, declarations, every update's group count/objective/grade/support, first cold failure state, logs, and SHA-256 manifest are archived at [round6-sample-anchor-free](continuous-evidence/round6-sample-anchor-free/manifest.json). Run source: [sample_anchor_free1200.py](sample_anchor_free1200.py).
