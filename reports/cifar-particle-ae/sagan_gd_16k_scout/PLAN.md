# SAGAN-style attention in G and D, active from initialization

User requested stopping the200k small-deconv continuation early and using that card for a40k experiment with attention in both G and D. Explicitly no phase-in. Subagent implements new standalone trainer/tests; historical trainers and shared sources remain immutable.

GPU1: scratch0->40k,16,384 particles. Start from the wider GroupNorm deconv architecture256/128/64; insert one spatial self-attention block at16x16 after the second hidden deconv/GN/ReLU and before the final RGB deconv. D gets one attention block in its trainable pixel branch at16x16 after the first residual block/pool. Its pretrained ResNet18 feature branch stays frozen. Both use `h + attention(h)` with fixed coefficient1 from initialization, standard nonzero weights, no zero-initialized gate/output, no warm-up. Manual bmm/softmax supports the existing double-backprop bcap. Query/key widthC/8, value widthC/2, output projectionC.

Keep same particle initialization, base G/D/E weights, noise scale0.212616428732872, loss, E-only reconstruction, optimizers/rates, EMA, batch64, oneD update and lazy bcap every8 multiplied by8. Additional attention construction uses isolated RNG so base initialization remains comparable. No new spectral normalization, class conditioning, or SAGAN-specific learning-rate changes: this tests attention within our existing recipe and is not a full paper reproduction.

FID50k every5k, saved sample/reconstruction grids and full checkpoints. Compare against the ongoing/completed wider GroupNorm run at matching steps; no baseline retraining. No automatic promotion. Joint G+D intervention cannot identify which side contributes; a stable single run cannot prove all instability mechanisms are fixed.

Before launch validate nonidentity attention fromstep0, gradients toattention parameters, E-only isolation, frozen pretrained features, nonzero bcap double backward through attention, exact full-state resume and unchanged existing weights. Then actual16k-particle16-update GPU1 smoke including penalty at8/16, FID128, reconstruction/checkpoint paths. Smoke scores are not benchmark results.

Stopped previous run at logged86,600; latest preserved complete checkpoint80k FID50k22.00219. See `deconv_16k_200k/STOPPED.json`. Do not treat200k as completed or86.6k as a saved checkpoint. GPU0 wider no-attention run is allowed to finish.

Follow: `tail -F runs/cifar_particle_ae/sagan_gd_16k_scout/PIPELINE.log`
