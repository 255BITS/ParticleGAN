# Expanding trainable centers improves this scout

The 4096-center continuation beats the matched 1024-center control at both FID50k evaluations and improves on the common 10k parent. Final FID is 18.5207 versus 19.8932: a 1.3724-point gain with 1.71% extra measured training time. This is evidence that changing the particle prior's flexibility/learning dynamics helps this plateau. It does not establish a unique root cause or a hard 1024-mode ceiling, and the <13 target remains unmet.

| Arm | Initial FID50k | FID50k 15k | FID50k 20k | Test MSE 20k | Train minutes | Wall minutes |
|---|---:|---:|---:|---:|---:|---:|
| 4096 cloned centers | 19.44812 | 18.93574 | 18.52071 | 0.14468 | 7.37 | 10.94 |
| 1024 control | 19.44808 | 19.86990 | 19.89316 | 0.14707 | 7.25 | 10.82 |

The expanded arm's advantage grows from 0.9342 at 15k to 1.3724 at 20k. Its FID also decreases between both evaluations. A single matched trajectory does not estimate run-to-run uncertainty; no seed replicates were run. Contemporary control scores differ modestly from earlier controls because production CUDA execution is not bitwise deterministic. The unchanged deterministic full-state replay passed.

## What changed and what stayed fixed

Four initially coincident descendants replace each original center. G/D/E architecture and state, sigma, d0, reconstruction routing, rates, Adam step counts and EMA remain unchanged. Original data/particle-parent/noise RNG streams are paired; a separate saved stream selects children. Prior per-row Adam moments are copied to descendants without LR scaling. The config retains num_particles=1024 as the reference initialization count; expansion_factor=4 means 4096 live rows.

Reference-count corrections preserve initial standardized centers and the particle variance/covariance regularizer, to floating-point tolerance. Initial FID differs by only 0.0000458. Thus the observed gain develops during joint training rather than being supplied by a better initialization image distribution. Expansion necessarily changes per-row exposure, gradient allocation and optimizer dynamics; the experiment cannot isolate abstract capacity from those effects.

Three CUDA tests passed: exact original control continuation; actual-parent state/moment mapping and coupled output identity; exact full-state expanded resume and E-only reconstruction gradient recipients. Both actual-parent eight-step smokes and both full scouts passed pipeline certification. Backbone, sigma and original parent hashes remain unchanged. All original data/noise streams finish with matching hashes. Historical/shared certified sources were not edited. See ../particle_expansion/PREFLIGHT.md and TESTS.txt.

## The extra centers actually differentiate

At 15k/20k, per-coordinate RMS displacement around the sibling mean is 0.09140/0.11283, or 0.430/0.531 times saved sigma (0.21262). In the fixed 32-parent panel, between-sibling pixel variation under identical noise rises from 0.001020 to 0.001517, versus within-child noise variation 0.002325/0.002453. The corresponding Inception mean-square values are 0.01854/0.02035 between siblings and 0.02660/0.02494 within a child. These measures show different outputs, not semantic coverage or recall.

Every expanded center was sampled 250–397 times across D/G draws over 10k updates (mean 312.5), versus mean 1250 for control. Centers are not unused. Direct selection counts do not enumerate all gradients: standardization and regularization couple rows.

Initial, 15k and 20k sibling grids were inspected. Siblings still mostly preserve object/pose/layout while differing in appearance and shape. Some local variety has increased, but the grid does not show four new semantic modes per parent. No claim of a fourfold increase in semantic diversity is justified.

Historical original-center movement from 10k to 20k is 0.311 per coordinate (1.46 sigma), reaching 0.569 by 50k (2.68 sigma). Thus the old centers were not simply frozen; the new intervention adds independently adaptable descendants. These latent movement measurements alone do not establish image coverage.

## Decision and follow-up

Continue both respective 20k checkpoints to 40k on the two GPUs, FID50k every 5k, unchanged rates and one D update. The observed gain is worthwhile for its ~2% cost, unlike the earlier expensive double-D intervention. The question is whether improvement persists or whether expansion only delays the plateau. No automatic 200k promotion.

Skip the small symmetry-breaking jitter test for now: the unperturbed descendants already differentiate and improve FID. Consider 8192 only after the 4096 advantage persists. Discriminator feedback quality/robustness remains a competing or additional limitation; if the advantage fades, return to that hypothesis. Dense bcap has not yet been tested in matched joint FID.

Current continuation log: `tail -F runs/cifar_particle_ae/particle_expansion_40k/PIPELINE.log`. Its pipeline will write a certified leaderboard and findings when complete; nothing beyond 40k is queued.
