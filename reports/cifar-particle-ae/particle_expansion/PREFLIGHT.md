# Expansion preflight

Three CUDA tests passed (11.90 seconds): exact unchanged full-state replay against the historical balance trainer; actual 10k parent cloning/moment mapping/coupled sampling and image identity; exact expanded full-state save/resume plus E-only reconstruction gradient recipients. Full log: TESTS.txt. Both real-parent eight-update pipeline smokes passed source/config certification, frozen-feature/sigma checks, original data/noise RNG pairing, and Adam counters 10008.

Live and EMA center maximum absolute error after cloning: 9.5367e-7. Coupled initial generated image maximum error: 2.4438e-6 on [-1,1]. Live regularizer 6.7501038e-5 versus 6.7501031e-5; EMA regularizer 2.8957867e-5 versus 2.8957893e-5. Maximum difference between original gradient and summed descendant gradients: 2.13e-11. This is distribution preservation to floating-point tolerance, not bitwise identity after changing reduction size.

Saved sigma 0.212616428732872 and d0 8.504656791687012 retained; no recalibration after cloning. Prior Adam first/second moments repeated per row, step 10000 preserved, no LR compensation. Parent checkpoint and all historical/shared source files unchanged. The config's num_particles remains the reference initialization count (1024); expansion_factor=4 means 4096 live rows. Expanded checkpoint extra state records reference count, child RNG and exposure. Evaluation resets/restores a separate child RNG, preserving training streams. EMA parameters update normally; its evaluation child stream does not alter the live training stream.

Eight updates select 1024 centers across D and G draws in either arm: mean exposure 1.0 per control center, 0.25 per expanded center. Direct exposure is not the same as receiving a gradient: global standardization and particle regularization couple rows. Expanded exposure is cumulative since expansion; control exposure records the current continuation.

Initial small smoke FIDs are approximately 143.6 because only 128 images were requested. They are not benchmark scores. The full scout requests initial FID50k and matched 15k/20k FID50k.

An initial unit test caught a diagnostic assuming at least 32 parents; the small replay fixture has 16. Corrected panel size to min(32,N), then reran successfully before pipeline training. No benchmark training result was affected. The harmless scalar-conversion warning in logs comes from preflight measurement of a differentiable tensor; it does not change gradients or values.

Full initial FID50k: control 19.4480765065; expanded 19.4481222578. Difference +0.0000457513. Both reproduce parent 19.4482 within 0.001.
