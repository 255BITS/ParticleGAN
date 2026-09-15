# Next session: return to the baseline after Anima experiments

Start with [ANIMA_HANDOFF.md](ANIMA_HANDOFF.md). The Anima feature branch is
committed through 5102d8c, and work has returned to master without merging its
experimental code. All twelve processes across the frozen/trainable rounds
completed; both GPUs are free. No next experiment is queued or selected.

At 10k updates, the from-scratch attention U-Net remains better and faster:
FID50k 29.327 in 10.73 training minutes, versus frozen pretrained Anima 30.263
in 19.98 minutes and trainable pretrained 36.558 in 24.89 minutes. The trainable
random control collapsed. These results do not justify promoting Anima.

The user rejected LR decay because of its tuning burden. Keep constant rates;
the previous gentler-tail proposal is withdrawn. Preserve the same four-step
ParticleGAN DDGAN, joint UCD, Gaussian step noise, exact lazy-4 bcap, VICReg
and optimizer recipe. No seed-only repeats.

Plain U-Net remains the no-argument baseline and has the best certified longer
result, FID 25.397 at 50k in 45.73 training minutes. Optional attention remains
available. See [attention results](attention_duration/READOUT.md). The old
attention 30k checkpoint is gone; intermediate-checkpoint retention is still
unimplemented. Do not infer FID50k from diagnostic FID5k.

Use config files, fresh output directories, both GPUs when useful, and
`--workers_per_gpu 1`. Preserve sample exposure if changing batch size. Make
logs easy to tail and report results only after experiments complete.

The feature branch retains the complete transplant code/configs/reports.
Revisiting a smaller donor or a smaller constant donor LR are optional ideas,
not instructions to restart that work automatically.
