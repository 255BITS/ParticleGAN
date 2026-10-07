# Ring16: uninterrupted serialized autograd

Prospective GPU diagnostic based on develop `2859975707eb3ea4d31958ad1b03e9e69e148a53`.
The [frozen protocol](protocol.json) asks whether fresh live training with
serialized autograd reaches the full Ring16 distribution gate without a restart.
A separate boundary-only arm asks whether the third trajectory isolated in
[PR334](https://github.com/255BITS/ParticleGAN/pull/334) converges.

The shared [intervention runner](../../../benchmarks/toy_audit/ring16_interventions.py)
comes from [PR332](https://github.com/255BITS/ParticleGAN/pull/332), extended with
execution-context hooks rather than a separate training loop. The continuous
arm uses public `GANTrainer.serial_backward=True` through a declared Forge
extension. Its mode is checkpointed and incompatible modes cannot be restored.
The boundary arm keeps the baseline trainer flag False, serializes all autograd
work only at401, and continues normally afterward. Neither learner loads a
checkpoint. The boundary400 and401 states must match archived controls after
only the declared unused stream registrations/external allowance are projected.

Both arms retain seed0, public deterministic initialization, z4, G4→64→64→2,
Fourier D10→64→64→1, learned256-location MoG sigma.1, batch128, constant
G/D/prior rates .012/.018/.03, full polar normalization, recipe horizon400,
clean live serving, and96 scheduled4096-sample evaluations through1600.
All original bounds remain:16 modes, massTV≤.15, HQ≥.85, covariance error≤.85,
minimum component eigenvalue ratio≥.15. Smoke needs a full scheduled pass and
one independent unchanged-state confirmation; failed confirmation is never
retried. Both arms finish1600 regardless of acquisition. Subsequent failures,
terminal streak and five-terminal-pass verdict are separate diagnostics.

Freeze limits:2 attempts/3200 training updates/600 subprocess seconds/194
scored draws, plus one separate CUDA API software check of at most3 tiny-model
updates/30 seconds. No scientific retries, seed experiments, annealing or
automatic retention work. Actual-training GIFs will be rendered from saved
CUDA-generated observations without new model calls. Frozen metric math on
saved outputs uses CPU, matching the prior scorer cohort; models and optimizers
execute on CUDA. Raw evidence stays outside Git.

Run from this checkout with the project Python environment. One worker uses
physicalGPU0, exposed as logicalcuda:0. Easy-tail log:
`runs/reports/ring16-serialized-live/execution.log`.

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.ring16_interventions plan \
  --protocol reports/forge/ring16-serialized-live/protocol.json \
  --output runs/api/ring16-serialized-live-v1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m benchmarks.toy_audit.ring16_interventions run \
  --protocol reports/forge/ring16-serialized-live/protocol.json \
  --output runs/api/ring16-serialized-live-v1
```

Plan is read-only. Run creates an exclusive campaign directory and refuses
reuse. This Ring-only diagnostic supplies no ordinary qualification or default
change; compare with existing truncation/noise evidence under original source
identities after completion.
