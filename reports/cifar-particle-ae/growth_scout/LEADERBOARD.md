# CIFAR AE-GAN checkpoint capacity scouts

0/4 certified runs complete.

All arms resume the same one-D 50k checkpoint (FID50k 18.9012), run one D update per G update, and retain the original reconstruction routing. G growth adds one identity-initialized residual refinement at each resolution. D growth adds identity-initialized refinements to the three trainable pretrained-feature heads; the ResNet18 backbone stays frozen.

| Rank | Run | Final FID50k ↓ | Δ vs control ↓ | Test MSE ↓ | Train min | Wall min | G updates/s |
|---:|---|---:|---:|---:|---:|---:|---:|

FID uses 50,000 prior samples, EMA G/prior, and the existing TF-compatible Inception CIFAR train50k reference. Reconstruction uses the 10k test split. Same seed and parent state throughout; these are architecture interventions, not seed experiments. Final endpoints determine ranking. Intermediate minima are reported separately.

| Run | Global step | FID50k ↓ | Test MSE ↓ | Train min since 50k |
|---|---:|---:|---:|---:|

## Interpretation and recommendation

Pending or uncertified: control, grow_g, grow_d, grow_both. Do not select a winner yet.
