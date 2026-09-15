# Trainable Anima transplant

Read [results](READOUT.md) and [design](PLAN.md).

At10k, trainable pretrained Anima reaches FID50k 36.558 in 24.89 training
minutes; the random control collapses to yellow images, FID 496.351.
Frozen pretrained was 30.263 in 19.98 minutes. No default promotion.

- [1k scouts](scouts/TABLE.md)
- [10k validation](promotions/TABLE.md)
- [Throughput](promotion_speed/TABLE.md)
- [Final output audit](output_probe_10k.json)

Configs: `configs/cifar_ddgan/anima_trainable_{profile,1k,10k}`.
All six processes completed; no active or queued training. Keep constant
learning rates and the existing ParticleGAN DDGAN formulation.
