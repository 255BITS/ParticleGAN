# CIFAR AE-GAN feature and selective-gradient scouts

0/4 certified runs complete.

All arms resume the original one-D 50k checkpoint (FID50k18.9012). ResNet34 replaces only the frozen discriminator backbone while retaining trainable heads and Adam state. This changes D scores immediately and tests replacement plus adaptation, not capacity alone. G growth adds identity-initialized refinements; reconstruction gradients are blocked only on those new parameters, retaining reconstruction updates on old G, E and prior. No extra D warmup, head reset, or change in the one-D update ratio.

| Rank | Run | Final FID50k ↓ | Δ vs control ↓ | Test MSE ↓ | Train min | Wall min | G updates/s |
|---:|---|---:|---:|---:|---:|---:|---:|

FID uses 50,000 prior samples, EMA G/prior, and the existing TF-compatible Inception CIFAR train50k reference. Reconstruction uses the 10k test split. Same seed and parent state throughout; these are architecture interventions, not seed experiments. Final endpoints determine ranking. Intermediate minima are reported separately.

| Run | Global step | FID50k ↓ | Test MSE ↓ | Train min since 50k |
|---|---:|---:|---:|---:|

## Interpretation and recommendation

The earlier all-gradient G-growth scout ended at FID22.8981; its contemporaneous control was19.2033. Use that as historical context for selective routing, not a same-run paired estimate. Reconstruction is never an unconditional-generation benchmark.
Pending or uncertified: control, resnet34, grow_g_adv, both. Do not select a winner yet.
