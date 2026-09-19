# SAGAN duration test: 40k to 200k

Continue the existing SAGAN-style G/D attention checkpoint on GPU 1 to 200,000 total steps (160,000 additional). Starting FID50k19.6425 at40k; previous30k/35k22.2370/20.4561. Test whether the late improvement persists.

Unchanged historical trainer and recipe: 16,384 independently initialized particles, fixed sigma0.212616428732872, wide GroupNorm deconv with active attention in G and trainable pixel D, frozen pretrained ResNet18 features, E-only reconstruction, one D update, bcap every8 steps multiplied by8. Constant G/E LR0.0003, prior0.003, D0.00045. Full optimizer, EMA, RNG and particle state restored. No seed/control repeat.

FID50k and reconstruction10k every5,000 steps; retain each checkpoint and report best sampled separately from final. Reuse existing wide-deconv/CNN references with unequal-duration and prior-history caveats. No automatic stopping for FID regression or training beyond200k. Existing nonfinite/time checks remain; six-hour active-training cap gives headroom over the roughly2.5-hour wall-time estimate based on the scout.

Source/config and checkpoint preflight passed. Controller writes certified results, leaderboard, checkpoint hashes and findings on completion. Review samples and trajectory when complete; FID alone does not identify capacity, discriminator or coverage as the cause.

Tail: `tail -f runs/cifar_particle_ae/sagan_gd_16k_200k/PIPELINE.log`
Controller errors: `runs/cifar_particle_ae/sagan_gd_16k_200k/launcher.log`
