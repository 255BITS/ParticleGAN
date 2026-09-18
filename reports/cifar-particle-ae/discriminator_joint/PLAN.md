# Matched joint continuations after discriminator diagnosis

Three arms from identical CNN E-only 10k parent, continuing to20k with FID50k/test reconstruction at15k and20k. Control unchanged; weaker changes bcap coefficient1→0.1 retaining every8; warmstart installs only D/Adam from current-regularization 2048-step D-only adaptation, then restores original joint recipe. Parent G/E/prior/EMA/G optimizer and original joint RNG preserved in all arms. No architecture changes, no seed replicates, one D update per joint update.

Selection: D-only test AUC current0.9194, weaker0.9552, every-step0.9471. Test/train ranking agrees. Both pixel and pretrained feature heads learn, so no fresh-head/backbone intervention is justified by this diagnostic. Every-step costs159.5s vs48.6s forcurrent and47.7s forweaker over2048 D-only updates; retain lazy8 for the cheaper first FID test. This does not rule out every-step FID benefits.

Warmstart tests whether a one-time head start persists during joint training; weaker tests sustained regularization relief without a warmup confound. FID, not AUC or gradient magnitude, decides value. Do not automatically promote a small isolated gain.

Preflight: exact old/new unchanged full-state continuation under deterministic CUDA passed (all models, EMA, both Adam states and RNG). Three actual-parent eight-step pipeline smokes passed, including warmstart D optimizer counter12056 vs G10008 and unchanged frozen backbone. Historical/shared sources untouched.

Tail: `tail -F runs/cifar_particle_ae/discriminator_joint/PIPELINE.log`
