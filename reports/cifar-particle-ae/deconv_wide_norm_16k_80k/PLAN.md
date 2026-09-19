# Continue wider GroupNorm deconv to 80k on GPU 0

User explicitly requested continuing the wide deconv to80k while SAGAN G/D trains on the other card. Resume the completed40k checkpointFID50k18.22851292930784; SHA129f905658fa71a0c69c7deeeec3860bf62ee52c07a30abdcdbeb1cceac96bd2. Exact full-state resume ofG/D/E/prior, EMA, optimizers and RNG with unchanged recipe and constant learning rates. Keep16,384 particles and fixedsigma0.212616428732872. WiderG has925,763 parameters, GroupNorm, no attention.

Train40k->80k onGPU0, FID50k every5k, full checkpoints and sample/reconstruction grids retained. Estimated30–35 minutes. Generous3-hour training cap; no automatic promotion. GPU1 SAGAN40k continues independently.

Parent config/source certificate and checkpointSHA verified; strict trainer load validates empty interventions. Existing11-test suite covers exact CUDA full-state split resume. Trainer source is unchanged; verify actual restore and first updates rather than rerunning the suite. Pipeline certifies final config/sources and unchanged frozen features/sigma, expected FID sample counts and rates. Best sampled checkpoint across the original scout and this continuation is retained separately from final.

Follow: `tail -F runs/cifar_particle_ae/deconv_wide_norm_16k_80k/PIPELINE.log`
