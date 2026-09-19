# Plain deconvolution scout on GPU 1

User authorized a simple linear -> deconvolution -> tanh generator with 16k particles and no normalization. The existing GPU 0 continuation remains active. Work stays on feat/cifar-ae-gan-pretrained-encoder.

Architecture: 64D z -> Linear(2048), ReLU, reshape 128x4x4 -> ConvTranspose2d(128,64,4,2,1), ReLU -> ConvTranspose2d(64,32,4,2,1), ReLU -> ConvTranspose2d(32,3,4,2,1), tanh. No normalization, residual connections, attention, or intermediate latent conditioning. Standard PyTorch initialization.

Train G, E, D head, and 16,384 independently initialized particle centers from scratch for 40,000 updates. D retains the same frozen ImageNet ResNet18 features. Keep seed 24002, batch 64, latent dimension 64, E-only reconstruction, fixed sigma 0.212616428732872, G/E learning rate 0.0003, prior learning rate 0.003, D learning rate 0.00045, EMA 0.995, one D update, and bcap every 8 steps multiplied by 8. Override the noise scale explicitly after calibration to avoid making particle count change the noise level. Calibration distance and actual sigma are recorded separately.

FID50k every 5,000 steps, test reconstruction on 10,000 images, retained full checkpoints and sample grids. Budget cap 7,200 training seconds. No seed-repeat experiments and no automatic promotion. New standalone trainer preserves historical source certificates.

Historical residual CNN references: 16k at 40k FID50k 17.0982, best at 80k 15.7527. Those models expanded a trained 1,024-particle checkpoint at 10k. This scratch run tests the simpler architecture's viability; architecture and prior initialization/training history both differ, so it is not a matched causal architecture ablation.

Before launch: architecture, gradient-routing, D/E initialization and resume tests; actual GPU 1 pipeline smoke with 16k particles, 16 updates, two lazy penalties, 128-sample FID and reconstruction/checkpoint paths. Smoke FID is not a benchmark. Full run uses GPU 1 only. Automatic reports include curve, best and final checkpoint hashes, historical references and a trajectory-based recommendation requiring review.

Follow: `tail -F runs/cifar_particle_ae/deconv_16k_scout/PIPELINE.log`

Launcher: `.venv/bin/python -u experiments/cifar_ae_deconv_scout.py`
