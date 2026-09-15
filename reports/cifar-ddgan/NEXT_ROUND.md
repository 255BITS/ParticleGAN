# Latest result: frozen Turbo did not help

The user requested testing the frozen pretrained Anima transplant with Turbo.
The matched 10k comparison completed on both GPUs: Base31.306 FID50k in20.29
training minutes versus Turbo v1.1 37.330 in20.88 minutes. Both kept the entire
existing ParticleGAN DDGAN formulation and constant LR. Full source/checkpoint
verification passed. See [readout](anima_turbo/READOUT.md).

No defaults changed. Recommend parking this donor setting. No next experiment
is selected or queued; both GPUs are free. Work remains on the Anima feature
branch. The user requested committing/pushing it and returning to master for
future experiments, without merging. Preserve unrelated .claude/ and sparse-ucd.log.

The earlier suggestions below remain untested ideas, not an automatic queue.

# Next round after unfreezing Anima

The [trainable Anima round](anima_trainable/READOUT.md) completed six processes
on both GPUs. At 10k, pretrained FID50k was 36.558 in 24.89 training minutes;
random collapsed to solid-yellow outputs (496.351). Frozen pretrained was
30.263 in 19.98 minutes. Attention remains better/faster at 10k (29.327,
10.73 minutes). No default promotion; no active or queued runs.

The user requested unfreezing the donor. All donor attention/MLP/time/AdaLN
parameters now train when anima_trainable:true. G, donor, and particle/D
rates retain the existing constant recipe; no added donor LR multiplier,
clipping, schedule, or loss. FP32 master parameters/Adam/EMA and live timestep
modulation are necessary parts of the implementation. Frozen mode still works.

The inherited donor rate was 0.0006. Pretrained donor relative L2 movement
was 1.435 at 1k and 2.380 at 10k, versus 0.400/0.665 for random. This motivates
one smaller constant donor rate if we continue, but it does not prove that
pretraining was erased or that the smaller rate will help. Do not launch a
broad hyperparameter search or another long run automatically. The one-block
transplant idea also remains untested. Do not reintroduce LR decay.

Preserve ParticleGAN learned latent particles, four-step DDGAN, Gaussian step
noise, joint UCD, exact lazy-4 bcap, Rp logistic and VICReg. Plain U-Net stays
the no-argument baseline; its best certified longer result remains FID 25.397
at 50k in 45.73 training minutes. The old attention 30k checkpoint is gone.

Current branch: experiment/anima-transplant, continuing frozen commit 6e53731.
See the runbook and configs/cifar_ddgan/anima_trainable_{profile,1k,10k}.
Use fresh out_dirs, --workers_per_gpu 1, both GPUs for useful independent
comparisons, and matched sample exposure. No seed repeats. The user wants
experiment check-ins only after completion.

Historical tail: tail -F results/cifar_ddgan/anima_trainable.live.log
