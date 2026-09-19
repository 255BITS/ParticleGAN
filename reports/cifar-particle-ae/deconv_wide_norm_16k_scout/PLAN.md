# Wider deconvolution with normalization

User selected wider WITH normalization for GPU 0 while the original small deconv continues on GPU 1. Use GroupNorm to match the normalization family in the historical residual CNN.

Generator: z64 -> Linear4096 -> reshape256x4x4 -> GroupNorm(8,256) -> ReLU -> ConvTranspose2d(256,128,4,2,1) -> GroupNorm(8,128) -> ReLU -> ConvTranspose2d(128,64,4,2,1) -> GroupNorm(8,64) -> ReLU -> ConvTranspose2d(64,3,4,2,1) -> tanh. No output normalization, residual blocks, or intermediate latent conditioning.

Train from scratch0->40k with16,384 particles. Copy the completed small-deconv full config, changing only generator architecture, output path and wall-time cap. Keep identical prior initialization seed, D/E initialization RNG, fixedsigma0.212616428732872, E-only reconstruction, frozen pretrained D features, one D update, every8-step bcap multiplied by8, rates and EMA. FID50k every5k with full checkpoints/grids. Reuse the existing small-deconv curve; no repeat-seed baseline.

This is a combined width+normalization upgrade, not an isolated width or normalization test. Compare at matched5k intervals against the existing scratch small generator. Historical residual-CNN comparisons additionally differ in prior initialization/training history.

Subagent owns the new standalone trainer and tests; preserve all historical source certificates. Validate normalization placement, gradients, E-only isolation, unchanged D/E RNG and exact resume onGPU0. Then production-path16-update smoke with16k particles, lazy double backprop, small FID/reconstruction and full-state outputs. Follow with40k run onGPU0, automated certified reports and checkpoint hashes. No automatic promotion.

Follow: `tail -F runs/cifar_particle_ae/deconv_wide_norm_16k_scout/PIPELINE.log`
