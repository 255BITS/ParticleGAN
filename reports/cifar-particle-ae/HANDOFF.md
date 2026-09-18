# AE-GAN investigation handoff — 2026-09-17

All capacity/duration pipelines completed successfully. Both GPUs are idle.
User requested compaction after completion, then investigation of AE-GAN's
generation/reconstruction disconnect. No additional training is queued.

## Objective and constraints

Target CIFAR-10 FID50k below 13. Work on branch
`feat/cifar-ae-gan-pretrained-encoder`. Use the experiment pipeline, keep logs
easy to tail, summarize completed experiments with leaderboards and recommendations.
No seed sweeps. Do not launch further training before investigating the current
failure mode. User accepted investigating encoder/prior mismatch after compaction.

## Completed results

| Model | Updates | FID50k | Test reconstruction MSE |
|---|---:|---:|---:|
| Original width32 | 20k | 20.1046 | 0.05816 |
| Original width32 | 30k | 19.4391 | 0.04819 |
| Original width32, intermediate checkpoint | 50k | **18.9012** | 0.04022 |
| Original width32, final | 100k | 20.8058 | **0.03371** |
| Width64 | 20k | 35.4405 | 0.04964 |
| Width64, depth2 | 20k | 25.7782 | 0.04244 |

50k is the best observed checkpoint of the duration run, not its final result.
30k→100k reduced reconstruction MSE by about 30%, but worsened final FID by 1.367.
Continuation took 52.17 training minutes / 60.21 wall minutes. It restored model,
EMA, optimizer and RNG state from 30k; it was not a restart. Six tests passed,
including deterministic continuation replay; all three real configuration smokes
passed. Source/configuration certificates passed on completion.

Duration FID50k at 40/50/60/70/80/90/100k:
19.424 / 18.901 / 19.977 / 20.355 / 20.340 / 21.720 / 20.806.
Capacity width64 FID10k→20k: 21.020→35.440; depth2: 37.000→25.778.
Capacity improved reconstruction but did not beat width32 at the same budget.

## Investigation hypothesis, not an established cause

Reconstruction uses `G(mu[encoder_selected_id] + sigma * bounded_offset(x))`.
Generation uses `G(mu[uniform_id] + sigma * normal_noise)`.
Offsets are bounded coordinatewise via `3*tanh(offset/3)`; bounding does not
match their distribution to Gaussian noise. Training has adversarial generation,
pixel reconstruction and particle spread losses, without an explicit aggregate
encoder-distribution matching term.

Final 100k diagnostics on 10k held-out test images:

- Reconstruction MSE 0.03371; retaining encoded offsets but shuffling selected
  particle centers gives 0.03870. Image information appears concentrated in offsets.
- Encoder selects 405/1024 particles; entropy-effective count 359.9; usage TV
  distance from uniform 0.6147. Generation samples all particles uniformly.
- Offset RMS 1.1956 versus approximately 1 for standard Gaussian noise;
  conditional offset-mean RMS 0.7749, saturation 1.60%.
- Zero/random-offset reconstruction MSE 0.41183/0.41580. These destroy the
  input-specific encoding, so high paired MSE alone does NOT prove poor sample
  quality or explain FID.
- Feature variance ratio 1.0837 checks only total feature variance, not feature
  means/full covariance. GAN losses near 0.693 are not evidence of good FID.

Proposed next work: frozen-checkpoint diagnostics before another training sweep.
Compare ordinary prior sampling, sampling particles by encoder frequencies, and
sampling offsets fitted from TRAIN encodings. Separate frequency and offset
changes to identify which matters. Avoid fitting distributions on held-out test
images. Any FID of reconstructions/empirically replayed encodings must be labeled
as a diagnostic, not an unconditional generation benchmark. Inspect sample grids,
verify evaluation protocol, and distinguish distribution mismatch from insufficient
generator/discriminator quality. Do not assume lower reconstruction weight is
the right fix until these diagnostics inform that choice.

## Files and reproducibility

- Trainer: `experiments/train_cifar_ae_capacity.py`; isolated copy preserves old
  checkpoint source hashes. Original trainer and lib files remain intact.
- Prior: `particlegan/particle_prior.py:MoGParticlePrior`.
- Encoder routing/G: `lib/image_particle_autoencoder.py`.
- Configs: `configs/cifar_particle_ae/{duration_100k,capacity_scout}/`.
- Pipeline: `experiments/cifar_ae_capacity_pipeline.sh`.
- Reports: [duration](duration_100k/LEADERBOARD.md),
  [capacity](capacity_scout/LEADERBOARD.md), [prior curve](lazy-long/README.md).
- Runs: `runs/cifar_particle_ae/duration_100k/n08/` and
  `runs/cifar_particle_ae/capacity_scout/{g64,g64_deep}/`.
- Final100k `checkpoint.pt` SHA256:
  `c1e29cf45f69528d09fcfd2849cbbe863ecb76fb7c71e50db53129b00f7f9478`.
- Best observed50k `checkpoint_050000.pt` SHA256:
  `10fe8bbc22afb29ff6838ad1ede86e142320e5d7bce43c745b97de24e23ee8d6`.
- Parent30k: `runs/cifar_particle_ae/lazy_long/n08/checkpoint.pt`, SHA256
  `b7e5e974be8053514cb04fcff1d9baa11e43a3e2d6c6dcf3649289b9c3a007af`.
- Numbered checkpoints, sample/reconstruction PNGs and `metrics.jsonl` retained.
- Historical audit script `experiments/audit_cifar_particle_ae.py` builds the
  original G only; capacity diagnostics need the new trainer's model builder.

Shared settings: seed24002, batch64, latent64, 1024 particles, D/E width32,
scratch encoder, frozen pretrained ResNet18 discriminator features, recon_weight1,
G/E lr0.0003, prior lr0.003, D lr0.00045, EMA0.995, sigma_rel0.025,
temperature0.125. N=8 exact double-backprop bcap, applied coefficient multiplied
by8. G widths32/64; depth2 adds one residual block at each output resolution.
All new evaluations use FID50k against CIFAR train50k with TF-compatible Inception.

Earlier scouts: pretrained E lost to scratch; N8 selected over N4/N16.
GPU1 performance bundle was ~7.7% faster but worsened 5k FID (23.203 vs22.799),
so it was not adopted. BigGAN comparison is not exact protocol parity: ours is
unconditional with pretrained D; user target remains below13.

Environment: `.venv/bin/python`, two A6000 GPUs, data and weights cached.
Do not alter unrelated untracked `.claude/`, results/hopfield*, results/motion,
`sparse-ucd.log` or untracked run artifacts. New diagnostic files can preserve
existing source certificates; changing lib/trainer files invalidates live-source
checks for historical artifacts. No need to repeat successful training tests.
