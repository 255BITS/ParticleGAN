# Changelog

## Unreleased

- Add K3P as package components: `GradRegularizer(arm="k3p")` with
  `blend_weight()`, `after_critic_step()` and `state_dict()`, plus
  `CriticAnchor`, `CriticSpikeGuard`, `LatentRowDamping` and
  `DirectParticleResponse` in `particlegan.k3p`. One instance per critic, with
  no optimizer hooks or module globals. The caller allocates the EMA critic
  and the history buffers. With one critic, the result matches the frozen K3P
  mechanism bit for bit (`tests/test_k3p.py`). The recipe default does not
  change yet.

## 0.7.0 — 2026-09-24

- Add the strict 100-mode toy gate (`python -m benchmarks.toy100 run`); its
  default is the simpler shared recipe that passes all 22 toys. The README
  animation now shows that default converging on the 10×10 grid.

- Keep training control flow separate from recipes: construct
  `GANTrainer(recipe, G, D, ...)` explicitly. Remove `Recipe.make_trainer`;
  recipes retain hyperparameters and small component factories. The current
  winning hyperparameters remain unchanged.

- Restore named model-family selection with `get_recipe(name="gan", **overrides)`:
  GAN, MoG, DDGAN, AE-GAN, VAE-GAN and AE-DDGAN configurations share current
  optimizer, cap and spread defaults. Names select components without training
  control flow or legacy hyperparameters. Full saved recipe dictionaries still restore.
- Preserve historical leaderboard comparisons as benchmark inputs outside the
  installable package. The quickstart uses the winning batch-distance D directly.

- Add `particlegan.locked_shared.LOCKED_SHARED`, the demo RpGAN + `b_cap`
  stamp (cover 1.5, FM off, 12 particles, host critic). Builders are
  `make_gan_loss` and `make_b_cap`. This stamp is separate from the current
  recipe hyperparameters.

## 0.6.0 — 2026-09-24

- Require explicit keyword `sigma` in `MoGParticlePrior`; construction no longer
  searches nearest neighbors. The shared isotropic noise remains fixed in training.
- Add optional `calibrate_mog_sigma(centers, sigma_rel)` returning `(sigma, d0)`;
  retain exact even-count median and historical dtype rounding. Recipes explicitly
  calibrate their initialized centers unless `make_prior(sigma=...)` overrides them.
- Preserve legacy checkpoint centers, sigma, d0, read settings, samples and RNG
  behavior. Load with matching dimensions and `sigma=0`, then `load_state_dict`.
- Migrate fixed-noise integrations to `MoGParticlePrior(..., sigma=fixed_sigma)`
  and remove post-construction sigma overwrites. Replace `calibrate()` with the
  standalone helper only when spacing-based calibration is intended.

## 0.5.0 — 2026-09-17

- Add public `particle_ae`, `particle_vae` and `ParticleEncoding`, plus
  `get_recipe("ae_gan")`, `get_recipe("vae_gan")` and `get_recipe("ae_ddgan")`.
  Caller-owned encoders select learned MoG particles for reconstruction.
- Default VAE uses one selected particle with prior-matching fixed-sigma noise:
  its joint KL is constant, so no KL penalty is needed in training.
  Reconstruction helpers never add KL. Gaussian negative ELBO reporting and
  the soft categorical posterior are explicit opt-ins; hard routing has a
  biased straight-through gradient. The genuine VAE evidence is toy-only.
- Add `recipe.encode(...)` and optional `encoder=E` to `make_optimizers` with
  shared-parameter deduplication. Existing recipes retain their defaults;
  networks, loops, loss composition, EMA and optimizers remain caller-owned.
- Document AE-DDGAN one-step reconstruction, inference and numerical variation
  audits. Matched CIFAR32 at 10k updates: FID50k 19.483 direct GAN, 20.054
  AE-GAN, 43.233 AE-DDGAN and 49.475 DDGAN. These single-trajectory results
  do not establish universal superiority or image VAE performance.
- Include queued toy/image experiments, configs, portable leaderboards,
  provenance and tests. Document narrower-than-real modes and late instability.

See the [particle autoencoder guide](docs/particle-autoencoders.md) for objectives,
examples, public contracts and evidence. No additional core dependencies.

## 0.4.0 — 2026-09-17

- Add `get_recipe("ddgan_mog")`: DDGAN with class-only UCD, 400 MoG components,
  z_dim=4, sigma_rel=1/40, standardized reads, 100,000 updates, a constant learning
  rate, prior LR multiplier 100 (0.06), and prior Adam betas (0.5, 0.999).
  Existing GAN, MoG and DDGAN presets retain their defaults; keyword overrides
  remain supported.
- Support MoG latent priors in `train_denoising` and checkpoint probes, including
  raw-mean regularization and separate prior optimizer settings. Add
  `generator_hidden` to vary generator width independently of the discriminator.
- Include matched 14k/100k GAN/DDGAN × atoms/MoG studies, frozen-noise probes,
  configs, metrics and readouts. At 100k, DDGAN+MoG reaches 79.57% joint HQ and
  100 modes with core width ratio .889; the one-shot models retain higher HQ
  but cover 77 modes. These are single-seed findings with a small generator,
  not a universal quality guarantee or an isolated capacity result.

The new recipe supplies package hyperparameters from the 100k study. Networks,
data, training/sampling loops and EMA remain caller-owned; the study used G
width 32 and D width 128. See the
[100k readout](reports/denoising-toy/mog_capacity_100k/READOUT.md).

## 0.3.0

- Add public `MoGParticlePrior`: a uniform mixture with learned means and a
  shared, fixed Gaussian sigma calibrated from initial nearest-neighbor spacing.
  Defaults to **400 components, z_dim=4, sigma_rel=1/40, standardized reads**.
- Add `get_recipe("mog")`, the selected **400-component, 28k-step** recipe:
  prior LR 0.06, prior Adam betas (0.5, 0.999), and the existing GAN defaults.
  Recipe factories support selecting the prior and setting prior betas separately.
- Add `configs/mog/default.toml` to run that experiment with the existing
  100-Gaussian trainer. Existing examples and the `ParticlePrior` atoms API retain
  their behavior.
- MoG supports explicit sampling generators, fixed epsilon snapshots, noisy
  module forward calls for DDP, raw-center regularization, EMA, and state-dict
  restoration of read configuration and calibrated noise. Legacy experimental
  checkpoints remain loadable with their original standardization setting.
- Keep the core dependency on PyTorch alone. The optional `mog` extra installs
  SciPy for faster calibration of large low-dimensional tables; a memory-bounded
  exact Torch fallback is available without it.

The selected compact model passed the frozen C0 envelope on 100 Gaussians at
28k steps: HQ/real 0.99953, width/real 0.92636, KL 0.02888. It uses 50 times fewer
components and four times the training steps of the original baseline. This is
a single-seed configuration result, not a claim of universal superiority.
See [the full experiment report](results/mog/COMPONENT_SCALE.md).
