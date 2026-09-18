# Plateau investigation

Goal: diagnose the AE-GAN FID plateau, test targeted interventions, then launch a continuation ending at global step 200,000. Feature branch: `feat/cifar-ae-gan-pretrained-encoder`.

1. Audit EMA checkpoints at 50k (best observed) and 100k (latest). Regenerate FID50k with the existing protocol. Compare uniform prior against encoder-frequency sampling, a full-covariance Gaussian fitted to train offsets (uniform/frequency particle selection), and a shrunk conditional diagonal Gaussian. Train-code replay and 10k held-out reconstructions are explicitly diagnostic, not unconditional generation benchmarks.
2. On live training weights, measure adversarial/reconstruction generator-gradient magnitudes and cosine similarity across 16 fixed 64-image batches, critic real/fake score separation, and input-gradient norms relative to the bcap threshold. Snapshot diagnostics cannot alone establish a training cause.
3. Resume the common 50k checkpoint for 10k updates: reconstruction weight 0.1; all learning rates ×0.25; both; two discriminator updates per generator update. Use historical unchanged 60k as endpoint control. Compare FID50k at 55k and 60k, not FID5k. No seed sweeps. The 2D variant spends additional compute and consumes additional data/RNG draws.
4. Select using completed final measurements and trend; launch the chosen continuation to 200k with periodic FID50k/checkpoints. A gradual learning-rate decay for the long run is a prospective intervention, not a measured scout result.

The new standalone trainer preserves historical source certificates and records all allowed continuation interventions. Model/EMA/Adam/RNG state is restored; architecture, base learning rates, data, and original model initialization remain checked. The scheduled scale overrides restored Adam learning rates each update. N=8 exact double-backprop bcap remains coefficient-scaled by eight and is indexed by discriminator updates when using two D updates.

Logs:

```
tail -F runs/cifar_particle_ae/plateau_diagnostics/gpu{0,1}.log
tail -F runs/cifar_particle_ae/plateau_scout/PIPELINE.log
tail -F runs/cifar_particle_ae/plateau_200k/PIPELINE.log
```

## Follow-up: reconstruction should not move particles

The user identified the reconstruction-to-prior gradient as an unwanted coupling. Existing routing uses `means[ids]` with gradients; differentiable mean/std normalization couples selected-center gradients to every raw particle row. The follow-up trainer `train_cifar_ae_routing.py` detaches means only for reconstruction. Adversarial and spread losses continue to train the prior.

Two additional 50k→60k scouts isolate gradient recipients at the original learning rates and reconstruction weight: `no_recon_prior` updates G/E from reconstruction, and `encoder_only_recon` updates only E from reconstruction. The latter temporarily freezes G parameters for the reconstruction forward while preserving input gradients and the separately built adversarial G graph. Three tests verify identical forward values, the exact intended gradient recipients, restored parameter flags, and unchanged adversarial G/prior gradients. Both real-checkpoint 16-update smokes passed. Historical checkpoints record both routing flags as true in the intervention journal.

```
tail -F runs/cifar_particle_ae/routing_scout/PIPELINE.log
```
