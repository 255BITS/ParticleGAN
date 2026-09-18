# Particle support and sampling sensitivity

All three read-only probes completed and passed certification. Broader inference noise worsened FID at every checkpoint. The generator does use the original noise: replacing it with particle centers alone substantially worsens FID. This does not settle whether the trained mixture offers enough distinct image configurations.

The grouped sample grid is informative. In the inspected 16 control particles, different noise draws largely retain object, pose and scene layout while altering local appearance. Across the balanced 128-particle, 16-draw measurement, within-particle variation contributes about 32–33% of total Inception-feature sum of squares but only about 5% in pixel space. Neither fraction is a semantic coverage score, and this observation is not proof of a 1024-mode ceiling.

Center-only sampling repeatedly emits a finite set of 1024 outputs. Its high FID includes the consequences of that discrete support and cannot by itself diagnose poor center quality or collapsed training. The noise-x2 measurement changes inference only: failure does not rule out retraining with a broader prior.

The global latent table has effective covariance rank essentially 64/64 at both 10k and 50k; there is no evidence of latent dimension collapse. A generator can still map a full-rank latent distribution into restricted image variation. Frozen G/prior state and original checkpoint hashes are verified unchanged by the sampling probe. Standard sampling reproduces the existing benchmark scores within 0.001 in all three observed baselines. The 50k FID reference/protocol is unchanged.

## Next controlled test

A useful next intervention is to expand the particle table from 1024 to 4096 starting from the same checkpoint, with a matched 1024 control and unchanged G/D architecture. This directly tests additional trainable centers. Keep the saved sigma fixed, preserve G/D/E/EMA/optimizer state, and handle expanded prior Adam/EMA explicitly. Before joint training, measure whether the expansion changed initial generated distribution/FID. Duplicating rows interacts with the unbiased standard deviation in `prior.means()`, so simply repeating raw rows is not exactly distribution-preserving; do not claim exact identity without accounting for that.

The competing hypothesis remains discriminator feedback quality. The lower-G-LR scout increased measured G gradient norm ~4x without improving FID, joining the earlier negative D-warmup and weaker-bcap results. These tests weaken a simple feedback-strength explanation but do not establish that pretrained features are robust. If expanded support fails, prioritize a controlled discriminator feedback/robustness intervention. Dense bcap was tested D-only, not yet in matched joint FID; it remains an open option.

No long training promotion is supported by these results.


| Checkpoint | Centers only FID50k | Normal FID50k | Double noise FID50k |
|---|---:|---:|---:|
| parent_10k | 41.8580 | 19.4481 | 21.7837 |
| control_20k | 42.4706 | 20.3039 | 21.7199 |
| half_g_20k | 43.4404 | 20.7005 | 23.5012 |

Sampling pipeline completed in about 7 minutes. Baseline reproduction errors: parent_10k -0.000117, control_20k -0.000552, half_g_20k +0.000847. Both GPUs are idle; no follow-up training queued.
