# Anima transplant experiment

Read [results and recommendation](READOUT.md) and [design/reproduction](PLAN.md).

The frozen pretrained donor beats its matched random control at 10k, FID50k
30.263 versus32.550, but does not beat the faster attention U-Net29.327.
All six runs completed. No default promotion; no active or queued work.

- [1k scouts](scouts/TABLE.md)
- [10k validation](promotions/TABLE.md)
- [Training throughput](promotion_speed/TABLE.md)

Configs live in `configs/cifar_ddgan/anima_{profile,1k,10k}`. The branch is
`experiment/anima-transplant`; constant learning rates and the ParticleGAN
DDGAN/UCD formulation are preserved.
