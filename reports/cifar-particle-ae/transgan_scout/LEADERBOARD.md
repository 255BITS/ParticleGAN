# CIFAR AE-GAN: scratch transformer and reconstruction routing

0/3 certified scouts complete.

All new runs start from scratch for 50k updates. Same frozen pretrained ResNet18 D, scratch E, particle prior, optimizer rates, one D update, EMA and lazy bcap every8 with coefficient×8. E-only reconstruction detaches particle centers and freezes G parameters for the reconstruction forward while preserving the gradient through G into E. Adversarial gradients still train G and the prior.

| Rank | New run | G parameters | Final FID50k ↓ | Test MSE ↓ | Train min | Wall min | Updates/s |
|---:|---|---:|---:|---:|---:|---:|---:|

Historical CNN with full reconstruction at50k: **FID 18.9012**, test MSE 0.04022; 645,123 G parameters. Its original scratch trajectory resumed at30k with full optimizer/RNG state and unchanged recipe. It is reused at user request and is not a new concurrent control.

FID uses 50k prior samples, EMA G/prior and the existing TF-compatible Inception CIFAR train50k reference. Reconstruction uses test10k. All arms use the same seed; no seed sweeps. Transformer size differs from CNN size, so this tests architecture plus capacity. It is a TransGAN-style generator inside AE-GAN, with a normalized low-gain tanh RGB head to prevent preflight saturation, not a reproduction of the full TransGAN recipe.

| Run | Step | FID50k ↓ | Test MSE ↓ | Training min |
|---|---:|---:|---:|---:|
| cnn_all_historical | 5,000 | 22.5396 | 0.08629 | 3.77 |
| cnn_all_historical | 10,000 | 19.6110 | 0.07755 | 7.51 |
| cnn_all_historical | 15,000 | 19.9602 | 0.06760 | 11.24 |
| cnn_all_historical | 20,000 | 20.1046 | 0.05816 | 14.96 |
| cnn_all_historical | 25,000 | 19.6419 | 0.05111 | 18.69 |
| cnn_all_historical | 30,000 | 19.4391 | 0.04819 | 22.41 |
| cnn_all_historical | 40,000 | 19.4241 | 0.04220 | 29.89 |
| cnn_all_historical | 50,000 | 18.9012 | 0.04022 | 37.33 |

## Interpretation and recommendation

Pending or uncertified: transgan_all, transgan_e_only, cnn_e_only. No winner selected.

![Learning curves and cost](curves.png)
