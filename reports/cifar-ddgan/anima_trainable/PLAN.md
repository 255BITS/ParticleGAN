# Unfreeze the Anima transplant

The user requested trainable Anima weights after the frozen two-block
transplant reached FID50k 30.263 at 10k. This round changes
`anima_trainable: true` in otherwise matched configs. Constant learning rates
remain: the donor uses the ordinary G learning rate, 0.0006, with the same
Adam settings. No donor-specific rate, decay, clipping, or auxiliary loss.

Both GPUs compare pretrained and random initialization, with the same U-Net,
two donor blocks, trainable adapters, particles, batch64, seed and sample
budget. These are not seed repeats. Historical frozen results provide context;
the simultaneous trainable pair tests whether initialization helps once the
donor can adapt. Both arms start fresh, rather than resuming frozen checkpoints.

## Implementation

- Every donated parameter trains: self/cross-attention, MLP, Q/K norms,
  timestep embedding/norm and all block AdaLN modulation weights.
- Learned timestep modulation is recomputed from current weights. Caching
  it would silently freeze those parameters or give EMA stale conditioning.
  Only the deterministic sinusoidal timestep inputs are reused.
- Trainable donor weights, gradients, Adam moments and EMA weights are
  float32. Large donor matrix operations use bfloat16 autocast. Q/K RMS
  normalization uses float32 with trainable master weights, then casts Q/K
  to V's dtype for attention. This avoids a mixed-dtype fused-norm warning.
- The original frozen mode retains bfloat16 weight storage and cached
  timestep modulation. Trainable/frozen comparisons consequently include the
  precision/storage changes needed for reliable trainable updates.
- EMA computes conditioning from its own current donor parameters even in
  eval/no-grad mode. The `anima_trainable` configuration, not the transient
  `requires_grad` setting, determines whether conditioning is cached.
- The zero-start output adapter remains. Donor gradients are zero on the
  first step; once the output adapter moves, gradients reach all donor weights.

The core four-step ParticleGAN DDGAN, joint UCD, Gaussian step noise, exact
lazy-4 bcap, Rp logistic, VICReg, particle count, optimizer ratios and EMA .995
are unchanged. No-argument training remains the plain U-Net.

## Execution

1. Paired 128-update profiles, including the full optimizer/backward path.
   Their ten-sample FID is only a smoke check, not a quality comparison.
2. Paired 1k scouts with FID5k, after correctness/runtime checks pass.
3. Promote practical promising arms to 10k with FID50k, judging quality and
   training time against the frozen experiment and established baselines.

Full configs: `configs/cifar_ddgan/anima_trainable_{profile,1k,10k}/*.yaml`.
Pinned donor bundle/revision/hash are the same as the frozen experiment.
Use one worker per GPU. Do not inspect progress until each run completes.

```
tail -F results/cifar_ddgan/anima_trainable.live.log
```

Validation before profiles: 56 tests and 13 subtests passed; the new dtype
warning was then fixed and all 18 Anima tests passed without that warning.
The new tests require finite nonzero gradients and actual updates for **every**
donor parameter, float32 optimizer state, live time modulation, EMA updates,
and exact output/modulation agreement after restoring EMA state. Frozen-mode
tests still pass. Sources remain fixed while runs are active.

## Completed screen and validation decision

Both full-size GPU checks passed before scouts: exact initial output identity,
finite nonzero gradients and actual updates for every donated parameter after
two optimizer steps, float32 Adam state, and active particle/image gradients.
The profiles completed with about 5.60 GiB training allocation and about 30%
more steady step time than the frozen transplant. The ten-sample profile FID
produced singular-covariance warnings; those tiny-sample scores are not used
to judge quality. No such warning arose in the 5k-sample scout evaluations.

The paired 1k scouts completed: trainable pretrained FID5k 82.771 in 2.52 min;
trainable random 74.709 in 2.57 min. Frozen pretrained was 67.514 and frozen
random 74.227. Unfreezing therefore lost the early pretrained advantage.
There were no nonfinite failures, and every donated parameter was verified
trainable. The cost is practical, so both arms receive a10k validation to test
whether this is a slower learning curve or a lasting regression. This is a
deliberate extension of the initial promotion criterion, not a claim that the
screen improved quality. The user was informed after the scouts completed.
