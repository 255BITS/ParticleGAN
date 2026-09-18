# Scratch generator architecture and reconstruction routing

User requested a transformer generator, implemented by a subagent, and accepted reusing the historical full-reconstruction CNN benchmark. User then explicitly added a fresh E-only CNN because the earlier E-only measurement was a checkpoint intervention. Three new runs; no CNN full-reconstruction retraining and no seed experiments.

| New arm | Generator | Reconstruction updates | Steps |
|---|---|---|---:|
| transgan_all | TransGAN-style | E, G and particles | 0→50,000 |
| transgan_e_only | TransGAN-style | E only | 0→50,000 |
| cnn_e_only | Existing CNN | E only | 0→50,000 |

Historical CNN with full reconstruction: FID50k18.9012 at50k, test MSE0.04022, 645,123 generator parameters. This trajectory began from scratch, then continued at30k with full saved state and the same recipe. Its training certificates and previously audited early FID50k checkpoint hashes were checked before reuse. `HISTORICAL_CNN.json` records the curve. The earlier E-only CNN intervention at50k→60k remains background information, not a scratch control.

## Hypotheses and interpretation

1. The CNN family or its capacity limits useful generation: compare transformer E-only with CNN E-only. This is a direct comparison of new scratch runs with common training settings. The transformer is larger, so it tests architecture plus capacity, not attention in isolation.
2. Reconstruction updates steer the generative model toward a poor unconditional solution: compare full versus E-only reconstruction within the transformer, with matched initialization and training random streams. Compare CNN E-only against the historical CNN full-reconstruction curve as supporting evidence. E-only blocks both G and prior reconstruction updates; it does not distinguish which of those two was responsible.
3. A shared mechanism limits all architectures: if the new arms plateau similarly, examine pretrained critic feedback, particle sampling/support, and optimization next. A failed transformer run alone does not prove a shared bottleneck; the inherited optimizer/loss recipe may suit architectures differently.

E-only means detached particle centers during encoding and temporarily frozen G parameters during the reconstruction forward. The image gradient still travels through G into E. GAN loss continues to update G and the particles; particle regularization remains unchanged. Reconstruction metrics therefore measure how well E can encode into a generator trained adversarially, not whether G was optimized for MSE.

## Architecture and shared recipe

Transformer: latent64 projected to8×8 tokens, width512→128→32 via pixel shuffle, transformer depths5/4/2, four attention heads, MLP ratio4, pre-LayerNorm, learned absolute and relative spatial positions, and tanh RGB output in[-1,1]. A final token LayerNorm and RGB Xavier gain0.1 adapt the unbounded upstream output head to our bounded image interface. This is a TransGAN-style generator inside our AE-GAN, not a reproduction of the full published TransGAN setup.

Primary architecture references: [official generator](https://github.com/VITA-Group/TransGAN/blob/master/models_search/ViT_custom_rp.py), [official CIFAR configuration](https://github.com/VITA-Group/TransGAN/blob/master/exps/cifar_train.py). Official configuration uses generator width1024 and a different latent size, discriminator, losses, augmentation, optimizer settings and budget. We preserve our latent64 and shared AE-GAN recipe to interpret the comparison.

Common configuration: seed24002, batch64, scratch width32 encoder, original frozen ImageNet ResNet18 discriminator features and trainable heads/pixel branch, latent64, 1,024 particles, sigma_rel0.025, temperature0.125, reconstruction weight1, G/E LR0.0003, prior LR0.003, D LR0.00045, original Adam betas, constant LR, one D update, EMA0.995, lazy double-backprop bcap every8 with coefficient multiplied by8. Same data flips and training RNG streams. Alternative G construction preserves historical D/E initialization.

## Execution and checks

Standalone trainer preserves historical source certificates. Subagent owns trainer and tests; primary agent owns experiment pipeline, configs, benchmark, validation, reports and launch. Ten tests passed, covering reconstruction routing, finite output/latent gradients, shared initialization, the official Linear initialization convention, resistance to100× residual activation growth, and full-state checkpoint continuation for both transformer routes. Run three full-width512-step smokes through the real pipeline, including the original bcap path. Record actual throughput and memory before launching.

An initial128-step preflight caught a porting mismatch before any long launch: blanket Xavier initialization of Linear matrices drove live-G tanh saturation to100%. Official TransGAN applies Xavier to Conv2d weights only, leaving Linear initialization at framework defaults. Corrected that mismatch, retained upstream RGB bias initialization, and added a regression test. Initial saturation was lower, but both corrected preflights again showed the same severe loss pattern by128steps. The original failures and read-only probe are retained in `PREFLIGHT_INITIALIZATION_FAILURE.json`, `INITIALIZATION_PROBE.json` and `PREFLIGHT_DEFAULT_INIT_FAILURE.json`; superseded runs/source archives remain in pipeline history. This motivated the explicit final-normalization/smaller-RGB-gain adaptation above. G/E learning rates and all shared losses remain unchanged. Failed preflights are implementation checks, not ranked FID experiments.

Scouts use both GPUs with one worker per card and the third run queued. Evaluate unconditional FID50k and test10k reconstruction every10k, keeping full checkpoints for continuation. Ranking uses final50k endpoints; intermediate minima and recent trends are shown separately. Report FID versus steps and training time, parameter counts, wall-clock cost and recommendations. No automatic long promotion.

Launch: `bash experiments/cifar_ae_transgan_pipeline.sh transgan_scout 0,1`.

Tail: `tail -F runs/cifar_particle_ae/transgan_scout/PIPELINE.log`.

Completion report: `reports/cifar-particle-ae/transgan_scout/LEADERBOARD.md`.
