# CIFAR-10 AE-GAN: pretrained encoder scouts

Branch: `feat/cifar-ae-gan-pretrained-encoder`.

Four 5,000-update experiments use the existing direct particle AE-GAN trainer,
grid runner, and tagged log follower. One shared seed (24002), no seed sweep.
Labels are unused. Each job uses one GPU; the default queue runs on GPUs 0 and 1.

| Config | Encoder | Reconstruction weight | G/E LR | Prior LR | D LR |
|---|---|---:|---:|---:|---:|
| 01_scratch_control | Existing trainable CNN | 1.0 | .0003 | .003 | .00045 |
| 02_pretrained | Frozen ResNet18 + learned heads | 1.0 | .0003 | .003 | .00045 |
| 03_pretrained_recon03 | Frozen ResNet18 + learned heads | 0.3 | .0003 | .003 | .00045 |
| 04_pretrained_half_lr | Frozen ResNet18 + learned heads | 1.0 | .00015 | .0015 | .000225 |

The pretrained encoder uses cached torchvision ImageNet1K V1 weights, resizes
32px images to 64px with bilinear interpolation, applies ImageNet normalization,
and retains layer3 spatial features (256×4×4). Learned query and bounded-offset
heads route into the same 1,024 particles with 64-dimensional codes. Backbone
weights and batch normalization statistics remain frozen, including in EMA.
Unconditional generation uses only the generator and prior.

All arms retain the existing pretrained discriminator and width32 generator.
The baseline rates are half the historical trainer defaults, following the
previous report's stability recommendation. The two pretrained variants test
weaker reconstruction pressure and a more conservative optimizer separately.
This first round tests frozen feature transfer; it does not test backbone tuning.

Rank by **final EMA FID5k** at step 5,000, with a same-count measurement at 2,500.
Both use the cached CIFAR train50k Inception reference. Reconstruction and
offset/routing diagnostics use the first 1,000 held-out test images.
FID5k is provisional and should not be compared directly with historical FID50k.
Review sample grids and late regression before selecting a longer run.

```sh
# Launch (the pipeline locks itself against duplicate invocations).
bash experiments/cifar_ae_pretrained_pipeline.sh

# All tagged training progress and the final report:
tail -F runs/cifar_particle_ae/pretrained_scout/PIPELINE.log

# One run:
tail -F runs/cifar_particle_ae/pretrained_scout/02_pretrained/log.txt
```

The pipeline writes [LEADERBOARD.md](LEADERBOARD.md), `leaderboard.json`, and
`winner_long.yaml` after completion. Promotion requires every requested result
to have a matching source/config/summary certificate. Failed or incomplete
grids produce a provisional report without a winner config.

The suggested next experiment runs the winner for 30k updates from the same
initial seed, with final FID50k and all 10k test reconstructions. It is prepared
but not launched. The trainer saves optimizer/RNG states and numbered evaluation
checkpoints; it does not currently implement checkpoint continuation.

Preflight: unit tests cover frozen encoder state, trainable head gradients,
batch independence, checkpoint loading, final-score ranking and incomplete-grid
promotion blocking. Separate 20-step GPU smoke configs exercise scratch and
pretrained training/evaluation/checkpoint paths; these are not ranked scouts.

Reanalyze without training:

```sh
.venv/bin/python experiments/analyze_cifar_ae_scout.py \
  --config_manifest configs/cifar_particle_ae/pretrained_scout/manifest.json \
  --report reports/cifar-particle-ae/pretrained-scout
```
