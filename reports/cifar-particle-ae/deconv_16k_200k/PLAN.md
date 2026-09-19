# Extend plain deconv to 200k

User requested much longer training to find where the plain generator caps. Continue the certified 40k checkpoint on GPU 1 to200k total steps. Exact full-state resume: same architecture, 16,384 particle centers, no normalization, fixed sigma0.212616428732872, E-only reconstruction, frozen pretrained D features, one D update and bcap every8 steps multiplied by8. Preserve all optimizer/EMA/RNG state and constant learning rates. No intervention except duration/evaluation spacing.

FID50k every10k steps, retained full checkpoints and sample/reconstruction grids; best checkpoint across the original scout and continuation is recorded separately from the endpoint. Estimated wall time110–125 minutes based on scout throughput; generous5-hour training cap. No automatic promotion beyond200k.

Original standalone trainer is unchanged. Its10-test suite already covered exact full-state resume; its actual16k-particle production smoke passed. Validate parent certificate/SHA and actual restoration/startup instead of repeating those tests. Pipeline will certify final sources/config, unchanged sigma/frozen features, evaluation sample counts, learning rates and empty resume interventions.

User selected a wider deconv WITH normalization on the other card. New GPU0 scout uses hidden channels256/128/64 with GroupNorm, otherwise the same16k scratch prior and recipe as the original40k deconv scout. This tests the combined upgrade; width and normalization effects are not isolated. See `deconv_wide_norm_16k_scout/PLAN.md`.

Follow: `tail -F runs/cifar_particle_ae/deconv_16k_200k/PIPELINE.log`
