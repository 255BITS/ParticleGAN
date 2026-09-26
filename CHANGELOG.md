# Changelog

## Unreleased

- **The G/D LR horizon scales with the training budget.** The new
  `Recipe.network_lr_horizon_fraction` (default `1600/7000`) sets the
  generator/critic horizon to `round(fraction × total_steps)` updates, and
  `network_lr_horizon_cap` now defaults to `None`. At the qualified 7,000
  updates the horizon is still exactly 1,600, so the toy results are
  unchanged; at 200,000 it is 45,714 instead of 1,600, so long runs no longer
  sit at the 1% floor (and finish K3P's R1-to-anchor handover) in their first
  1% of training. `recipe.network_lr_horizon` reports the resolved value.
  - Short budgets anneal sooner too: below 7,000 updates the horizon is now
    under 1,600 (e.g. 46 at 200), where budgets up to 1,600 used to anneal
    over the full run. The short qualified gates (`lib/yue2_particle_toy.py`,
    `lib/safe_fast_landing.py`) pin `network_lr_horizon_cap=1600`, and
    `benchmarks/legacy` keeps the recorded fixed horizon.
  - An integer `network_lr_horizon_cap` still fixes the horizon in updates and
    overrides the fraction. Pass `network_lr_horizon_cap=1600` to keep the
    former fixed horizon at other budgets.
  - **Changed meaning:** `network_lr_horizon_cap=None` used to mean the full
    budget; it now means "use the fraction". Use
    `network_lr_horizon_fraction=1.0` for the full budget.
  - `GANTrainer` checkpoints compare the recipe by the horizon it resolves to,
    so a checkpoint recorded with `network_lr_horizon_cap: 1600` at 7,000
    updates resumes under the new default. Recipes saved before this field
    load with fraction 1.0, so their `None` cap keeps meaning the full budget.

## 0.8.0 — 2026-09-25

- **K3P is the default and only formulation.** The critic penalty blends R1 +
  fake-gradient cap into one-sided gradient caps plus an EMA-critic gradient
  anchor as the critic LR decays; the recipe optimizers add the critic spike
  guard, sparse latent-row damping and direct-particle response. It replaces
  GAN v3, which lost quality over longer training runs.
- **Plain PyTorch loop.** Build everything through the recipe:
  `recipe.make_optimizers(G, D, prior, ema_critic=...)`,
  `recipe.make_critic_optimizer(D2, ema_critic=...)` for extra critics, and
  `penalty = recipe.make_critic_penalty(opt_d)`; then
  `d_loss = adv + penalty(D, real, fake, *cond)` and the usual
  `zero_grad()/backward()/step()`. All regularization state lives in the
  optimizers' `state_dict()`s, so standard checkpoints resume exactly.

- **One formulation, no technique menu.** Removed from `particlegan`: the
  `Recipe` fields `loss_type`, `gan_mode`, `reg_arm` and `reg_method`; the
  non-K3P penalty arms, norms, center annealing and finite differences of the
  critic gradient penalty; the hinge/Wasserstein/LSGAN/vanilla/RaGAN modes of
  `GANLoss` (now RpGAN logistic only); `recipe.make_gradient_penalty`; the
  unused `generator=` of `make_critic_penalty`; the `particlegan.locked_shared`
  stamp; and the `GradientPenalty` top-level export. Training with the default
  recipe is bit-identical. `GANTrainer` still loads checkpoints whose recipe
  recorded the removed fields at their only supported values.
- Benchmarks that replay archived GAN v3 / locked_shared / arm-study
  configurations now resolve them through frozen copies in `benchmarks/legacy/`
  (`LegacyRecipe`, the multi-arm penalty, the multi-mode loss, the stamp).
- Removed the arm-study drivers and configs (`experiments/train_arm.py`,
  `gen_configs.py`, `compare_priors.py`, `make_video.py`, `provenance.py`,
  `probe_cifar_fd.py`, top-level `configs/*.yaml`, `configs/audit/`), the
  `lib/gan_loss.py` / `lib/grad_regularizers.py` shims, the docs for GAN v3,
  locked_shared, prior controls and the develop API review, and tests that
  only asserted removed behaviour.
- Added ablation switches `Recipe.reg_anchor_weight` (1.0; 0 removes the
  critic penalty's EMA-anchor term) and `Recipe.direct_particle_gain` (True;
  False keeps the direct-particle LR unscaled).
- Rewrote the README for newcomers around a runnable plain-PyTorch loop
  (executed by `tests/test_readme.py`).

- **Every repository trainer uses the default formulation.** Examples,
  experiments and `lib/` trainers build their optimizers and critic penalties
  only through `recipe.make_optimizers` / `make_critic_optimizer` /
  `make_generator_optimizer` / `make_critic_penalty`, on the recipe's LR
  schedule (`scale_learning_rates`). Removed the pinned
  `GradientPenalty(arm="b_cap", lazy_k=4)` critics (MoG AE/VAE, CIFAR particle
  AE/DDGAN, gym slider/particle fine-tunes, YuE2 and safe-fast 2D gates), the
  gym fine-tune `adv_posture` (`yue2` / `locked_shared`) and its
  `locked_shared.yaml`, the `--reg_arm`/`--r1_gamma`/`--loss_type`/`--gan_mode`
  menu of `examples/100gaussians.py`, the `reg_arm` key of the config-driven
  trainers (CIFAR DDGAN, denoising, 100-Gaussian runner, sparse; stripped from
  their launch configs; the finite-difference speed configs are deleted), and
  the hand-built Adam optimizers for G/E/prior/critics. Image-scale trainers
  keep a lazy penalty (`reg_every=4`; laziness changes only frequency and
  coefficient) and a full-budget network LR horizon as recipe fields. Frozen
  bundles under `reports/` and the `benchmarks/` research harnesses are
  unchanged.
- **Familiar GAN loop: the recipe's optimizers do the step-time work.** The
  loop is plain PyTorch — `d_loss = adv + penalty(D, real, fake)`, then
  `opt_d.zero_grad(); d_loss.backward(); opt_d.step()` and the same for
  `opt_g`. `recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))`
  returns `torch.optim.Adam` subclasses whose `step()` runs the spike guard,
  EMA-critic update and LR record (critic) and A2 latent damping /
  direct-particle response (generator). New role-named factories
  `recipe.make_critic_optimizer(D, ema_critic=...)` (additional critics) and
  `recipe.make_generator_optimizer(params, latent_table=None,
  direct_particles=None)`. `recipe.make_critic_penalty(opt_d, output=None,
  generator=None, collect_stats=False)` returns a callable penalty paired with
  that optimizer: `penalty(D, real, fake, *cond, **cond_kw)` returns a scalar,
  forwards conditioning to the critic and its EMA (roles of a shared module
  and wrappers such as the new `InputNoise` resolve to the matching EMA),
  takes the logits from a tuple's first element unless `output=` says
  otherwise, reads its step from the optimizer (no `step` argument) and
  exposes `last_stats`/`diagnostics()`. All regularization state (EMA critic,
  LR record, counters, histories) is in `optimizer.state_dict()` under
  `"regularizer"`, so the usual `torch.save({... "opt_g": opt_g.state_dict(),
  "opt_d": opt_d.state_dict()})` resumes exactly. `GANTrainer` checkpoints
  are schema 3 (schema 2 is upgraded on load); `trainer.penalty` is the recipe
  penalty and `trainer.critic` / `trainer.generator_regularizer` are gone.
  Removed: `make_critic_regularizer`, `make_generator_regularizer` and their
  `step`/`before_step`/`after_step`/`around` surface. Behavior note:
  `make_optimizers` now applies the recipe's spike guard (`d_guard_ratio`) and,
  for a learnable `ParticlePrior`, A2 damping (`latent_damping_max_rate`) even
  without a penalty; set both to 0 for plain Adam steps. GANTrainer,
  `examples/pytorch_loop.py` and the frozen-K3P parity tests are bit-identical.
- **Regularizers come from the recipe.** `recipe.make_critic_regularizer(D,
  opt_d, **penalty_kwargs)` (one call per critic optimizer) and
  `recipe.make_generator_regularizer(opt_g, latent_table=prior.z,
  direct_particles=None)` build whatever the current best formulation needs
  (K3P today) behind a formulation-agnostic interface: `penalty`,
  `ema_critic`, `before_step`/`after_step`/`step`, `diagnostics`,
  `state_dict`/`load_state_dict`. They replace `K3PCritic(recipe, ...)` and the
  `make_critic_anchor`, `make_critic_guard`, `make_latent_damping` and
  `make_direct_response` factories (removed). `K3PCritic`, the new
  `K3PGeneratorRegularizer` and the K3P primitives move to / stay in
  `particlegan.k3p` and are no longer exported from `particlegan`.
  `GANTrainer`, examples, experiments, `lib/` and docs use the factories;
  `trainer.generator_regularizer` is new. Math and checkpoints are unchanged.
- **K3P is the default.** `Recipe()`/`get_recipe()` now resolve to the qualified
  K3P config: `reg_arm="k3p"`, coefficient 1, κ 1, betas (0, .999), no particle
  spread, batch 2048, z_dim 2, plus new fields `network_lr_floor` (.01),
  `network_lr_horizon_cap` (1600), `reg_anchor_decay`, `d_guard_ratio`,
  `d_guard_min_steps`, `latent_damping_max_rate`, `direct_particle_betas` and
  the input/output noise schedules. New `learning_rate_scales(step, recipe)`.
- A per-critic-optimizer K3P bundle (now `recipe.make_critic_regularizer`)
  (trainer-allocated EMA critic with buffer averaging and side-effect-free
  forwards, penalty, spike guard, `state_dict`). Several critics use several
  bundles; one module with several roles passes a per-role `ema_critic`.
- `GANTrainer` trains K3P end to end (role-wise LR schedule, critic input and
  generator output noise from its own stream, spike guard, A2 latent damping)
  and checkpoints it (schema 2). Schema-1 GAN v3 checkpoints are rejected.
  `trainer.ema_D` is now a property of `trainer.critic`.
- Custom loops in `experiments/` and `lib/` use the critic bundle; hosts whose
  protocol requires `b_cap` pin it explicitly. Historical benchmarks resolve
  archived recipes through `benchmarks.gan_v3` (`GAN_V3_FIELDS`,
  `legacy_recipe`, `legacy_dict`), so their receipts are unchanged.
- `GradientPenalty()` now defaults to `arm="k3p"`; legacy direct callers pin
  `arm="b_cap"`. New `scale_learning_rates(step, recipe, optimizers,
  base_rates, prior)` sets G/D groups to the network schedule and prior groups
  to the prior one, so a custom loop's critic LR reaches the same floor K3P's
  blend weight uses. The custom loops, `examples/five_modes.py` and the
  README/API examples use it.
- Docs: new `docs/k3p.md`; GAN v3 docs marked superseded. The shipped
  `configs/100gaussians` and `configs/denoising` defaults follow the recipe.
- Add K3P as package components: `GradRegularizer(arm="k3p")` with
  `blend_weight()`, `after_critic_step()` and `state_dict()`, plus
  `CriticAnchor`, `CriticSpikeGuard`, `LatentRowDamping` and
  `DirectParticleResponse` in `particlegan.k3p`. One instance per critic, with
  no optimizer hooks or module globals. The caller allocates the EMA critic
  and the history buffers. With one critic, the result matches the frozen K3P
  mechanism bit for bit (`tests/test_k3p.py`). The recipe default does not
  change yet.
- `GANTrainer` and `examples/100gaussians.py` support `reg_arm="k3p"`: they
  call `after_critic_step` after each critic step and allocate the EMA critic.
  The trainer also averages the critic's buffers and saves both `ema_D` and
  the penalty state in its checkpoint.
- A k3p regularizer now raises if it is used on a second critic without an
  explicit `ema_critic=`, and `after_critic_step` accepts a tensor LR.

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
