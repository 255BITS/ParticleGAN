# custom22: candidate-policy engine and bindings for the 8 custom hosts

Status: IMPLEMENTED 2026-09-27; see §9 for what was built, how it was verified, smoke verdicts and the
deviations from the design text below (§0-§8 are the design as written). Goal: every candidate gets the full 22-check suite (19 frozen
older hosts + 3 native100). The 14 already runnable are mode_hold, 4 images, 6 vectors and 3 native100.
This document covers the missing 8: two_pole, trajectory, residual_student, unipolar, ae_gan_hold,
cover_leftover, unused_token_hold and mid_scale_identity.

Rule used throughout: **the learner is the candidate package's `GANTrainer` policy.** It is re-expressed as
components that fire in the same order as `GANTrainer._step`. **The task is the frozen host.** Its data,
architecture, conditioning, auxiliary losses, scorers and thresholds stay verbatim. Nothing from the old
`compare_defaults` learner route survives into learner policy: no `optimizer_defaults`, no `FixedControl`,
no `schedule_optimizer`, no legacy loss or penalty, and no host LR schedule. Where a host does not fix a
choice, the choice is derived from GANTrainer semantics and listed in §7. Items that would change the
candidate's policy are in §8.

## 0. Pinned sources (read-only PR #155 checkout; last commit touching them c720645e)

| file | sha256 (prefix) | note |
|---|---|---|
| benchmarks/locked_shared/two_pole.py | 65237880beaf | = prior audit hash |
| benchmarks/locked_shared/trajectory.py | f6f7b5578f4b | = audit |
| benchmarks/locked_shared/hosts/residual_student.py | 42c4dd16dae4 | = audit |
| benchmarks/locked_shared/hosts/unipolar.py | aeed81a291dc | = audit |
| benchmarks/locked_shared/hosts/ae_gan_hold.py | 7cf5e66dc637 | = audit |
| benchmarks/locked_shared/hosts/cover_leftover.py | d6556a851ee6 | = audit |
| benchmarks/locked_shared/hosts/unused_token_hold.py | 8ac5536238f7 | = audit |
| benchmarks/locked_shared/hosts/mid_scale_identity.py | ca36f2badc34 | = audit |
| benchmarks/locked_shared/observation.py | 04067872ce22 | `sustained` |
| benchmarks/transfer_suite/protocol.py | 7626dd8236c0 | `test_verdict`, `requirements` |
| benchmarks/transfer_suite/plans/default_comparison.json | 2e62a935dd70 | 19 frozen specs, 8 used here |
| benchmarks/transfer_suite/legacy_noise_adapters.py | 2e4ebc45fc92 | EVAL_SCOPES and noise application sites |
| benchmarks/legacy/{locked_shared,gan_loss,grad_regularizers}.py | 9bac7def8e5c / ee26f6d8ef56 / cfe252f52a6d | import-time deps of the hosts only |

The candidate packages are `candidates/dv12-ams/package` (dv12-ams-rc3, overrides in
`runs/dv12-ams-rc3/candidate.json`) and `candidates/dv12-st/package` (st-10, overrides in
`runs/st-10/candidate.json`). dv12-st adds the following over dv12-ams: `output_noise_mode`
(fixed/learnable/mobility), `lr_control` (mobility/stationarity: `StationarityLR`/`SettleTest`),
`critic_payoff_damping`, `critic_r1_real` (internal to the penalty), `particle_birth_death` and
`output_sigma()`/`last_output_sigma`.

## 1. Files (all additive; no existing task or config hash changes)

| path | content |
|---|---|
| `harness/components.py` | engine: `build()`, the `Update` phase object, evaluation scope, state/receipts, refusals |
| `harness/custom22.py` | one `run_<host>(ctx, task)` per host, plus the verdict, observation and logging glue (reuses `screen.Context`) |
| `harness/hosts/custom/` | byte-for-byte copies of the 13 pinned files above, with sha256 checked at import. There are also two **import-only shims**, `benchmarks/gan_v3.py` and `benchmarks/legacy/recipe.py`, which raise if called. `ae_gan_hold` imports them at module level, but the ports never use the legacy recipe. |
| `harness/tasks/custom22_specs.json` | the 8 frozen specs copied from `default_comparison.json` (hash-checked) |
| `harness/tests/test_components_parity.py` | the parity levels in §3 |
| `harness/lrlib.py` (edit) | `CUSTOM_TASKS` (8), `ALL_TASKS += CUSTOM_TASKS`, presets `custom` and `all22` (= gates + native + custom = 22), and `SHORT` names. `gates`/`all` are unchanged. |
| `harness/screen.py` (edit) | `--task` choices follow `ALL_TASKS`; one `elif task in CUSTOM_TASKS: custom22.run(ctx, task)`; for custom tasks the header skips CUDA device queries, so no CUDA context is created |

The pool is unchanged: `task_kind()` returns the task name, and the default timeout is 3600 s, which is ample.
The config hash is package + overrides + behaviour options; it does not include the task, so it is unchanged.
Custom jobs are CPU jobs. They take a pool slot but never touch the GPU.

## 2. Engine contract (`harness/components.py`)

### 2.1 Build: `eng = build(package, overrides, spec, device, seed=0, optimizer_options={'foreach': False, 'fused': False}, serial_backward='auto')`

`spec` is a role registry supplied by the host binding. No learning-rate numbers enter through it.

| spec field | meaning | GANTrainer analogue |
|---|---|---|
| `generator` | G-side network module or `None`; init key 0 | `G` |
| `encoder` | optional G-side module; init key 2 | `make_optimizers(encoder=)` |
| `tables` | list of package `ParticlePrior`/`MoGParticlePrior`; `tables[0]` is "the prior" | `prior` |
| `critic` | host critic module; init key 1 | `D` |
| `resources` | recipe fields forced to host values (§4a), like screen.py's host resources | `get_recipe(**overrides, num_particles=..)` |
| `init_fresh` | False when the host pins deterministic critic weights (two_pole only): the recipe is built with `initialization=None` | package docstring: "use `initialization=None` for custom weights" |

The build order copies `GANTrainer.__init__` exactly:

1. `recipe = get_recipe(**overrides, **resources)`.
2. `opt_g, opt_d = recipe.make_optimizers(generator, critic, tables[0] or None, encoder=encoder, ema_critic=deepcopy(critic), **optimizer_options)`. This gives role groups `[g(+E), prior]`, the prior group at `lr*prior_lr_mult` with `prior_betas or betas`, `amsgrad` on every group, A2 when `tables[0]` is a plain `ParticlePrior`, `KA2CriticAdam` with spike guard and EMA critic, and `batch_feature_zero` on fresh modules.
3. For each `tables[k>=1]`: `opt_g.add_param_group({'params': [t.z], 'lr': lr*prior_lr_mult, 'betas': prior_betas or betas})`, plus its own `k3p.LatentRowDamping(t.z, zeros_like(t.z), recipe.latent_damping_max_rate)`. The engine wraps `opt_g.step()` with that damping's `around()` (see §5.4).
4. Learnable sigma, when `output_noise_mode == 'learnable'`: `log_output_sigma = Parameter(log(output_noise_std))` and `opt_g.add_param_group({'params': [it], 'lr': recipe.lr})`.
5. Then, as in GANTrainer: `initial_lrs`, `roles` (prior by parameter identity with the tables), `loss = recipe.make_loss()`, `prior_regularizer = recipe.make_prior_regularizer(weight=1.0)`, EMA deep copies of every G-side module and every table (`requires_grad_(False)`, eval), and streams `latent/penalty/eval/noise = Generator(device).manual_seed(seed + 2/3/4/5)`.
6. Next: `_noisy_D = InputNoise(critic, 0, noise)`, `penalty = recipe.make_critic_penalty(opt_d)`, `controller = DataDriftController(policy)`, then `controller.observe_prior(table_view)` when a table exists. For dv2..dv12 the penalty is linked back: `penalty.regularizer.continuous_controller = controller`.
7. `lr_settle = StationarityLR((opt_g, opt_d), prior_param=tables[0].z)` when `lr_control == 'stationarity'`.

Refusals (ERROR, never a silent drop): `particle_birth_death=True`; `model != 'gan'`; `conditioning != 'scalar'`; dv11 on a host without a latent-to-sample map (`observe_support` needs `G(latent)`); and any recipe field outside the engine's known set whose value differs from the `Recipe` default, i.e. a new policy knob the engine has not been parity-checked for. The engine also records the set of `GANTrainer` method names, and a set that differs from the parity-checked one is refused as well.

The engine exposes the attributes `screen.Context` reads from a trainer: `opt_g`, `opt_d`, `recipe`, `completed_steps`, `controller`, `penalty`, `initial_lrs`, `log_output_sigma`, `last_output_sigma`, `output_sigma()`, `lr_settle`. The existing `rates.jsonl` and `diag` code therefore works unchanged. The `real_grad_norm` probe is skipped (`probe_real=None`) because host critics are conditional.

### 2.2 Per-update phases: `with eng.update(real) as u:`

The whole update runs under `set_multithreading_enabled(False)` when `serial_backward`, like `GANTrainer.step`. A state machine asserts the order below. Each phase runs once per update, except `sample`, `generate` and `noise`, which run once per role or site.

| # | engine phase | exactly as `GANTrainer._step` lines (dv12-st numbering) |
|---|---|---|
| P0 | enter | budget check (`total_steps`); `real = _batch(real)` |
| P1 | enter | `controller.observe_prior(table_view)`; `controller.observe_game(penalty.regularizer.record)` |
| P2 | enter | LR: without `lr_settle`, `(net, pri) = controller.observe_real(real)` (or `learning_rate_scales(completed, recipe)` when there is no controller) and `lr = initial*(pri if prior else net)`. With `lr_settle`: `observe_real`; `reopen = data_score > 3`; testers `restart(reopen=True)`/`begin`; `lr = initial*tester.s` |
| P3 | enter | D groups `*= controller.critic_scale()` if `getattr(recipe, 'critic_payoff_damping', True)` |
| P4 | enter | `sigma_in = input_noise_std(recipe, completed)`; `sigma_out = _output_sigma(output_noise_std(...), detach=False)` (package function when present); `last_output_sigma` |
| P5 | `u.d_phase()` | `D.train()`; G-side modules `.eval()` |
| P6 | `u.sample(latent_map, latent, table)` | `no_grad`: `observe_support(...)`, then `perturb_latent(latent, noise, table, record=True)`, then map, then `+ sigma*randn(noise)`. Sigma is detached. |
| P7 | `u.observe_pair(real_all, fake_all)` | once, on the role-union batches |
| P8 | `u.penalty(view, x_real, x_fake, *cond)` | `penalty.collect_stats = obs_step`; **exactly one call per update** (§5.5) |
| P9 | `u.d_step(loss_d_adv, penalty)` | `opt_d.zero_grad(); (loss_d_adv + penalty).backward(); opt_d.step()`; `_settle_observe(1)` |
| P10 | `u.g_phase()` | `D.eval()`; G-side `.train()`; `D.requires_grad_(False)` (restored at exit) |
| P11 | `u.generate(latent_map, latent, table)` | perturb, map, `+ sigma*randn(noise)`. Sigma is **attached** (learnable). |
| P11' | `u.noise(x)` | output noise at a host site with no latent (e.g. the ae reconstruction path): `+ sigma*randn(noise)`, attached in the G phase and detached in the D phase |
| P11'' | `u.input(x)` | critic input noise at the host's data coordinate. It is the identity with no RNG draw while `sigma_in == 0` (always the case for continuous policies). |
| P12 | `u.g_step(loss_g, loss_gan, loss_d_adv)` | `opt_g.zero_grad(); loss_g.backward(); controller.observe_generator(G_side, loss_gan, loss_d_adv)`; then extra-table A2 `around(opt_g.step())`; `_settle_observe(0)` |
| P13 | exit | restore D flags; EMA `mul_(d).add_(x, 1-d)` on params and `copy_` on buffers for every G-side module and table; `completed += 1`; rate row |

`loss_gan` and `loss_d_adv` are the **adversarial terms only**, with the host's role weights. They play the
roles of GANTrainer's `loss_gan` and `loss_d - penalty`. Auxiliary terms enter only `loss_g`.

`table_view` is an object with `.z`. For one plain table it is the table itself, which is bitwise identical
to GANTrainer. For special cases see §5.3 (two tables) and §8 (MoG).

### 2.3 Evaluation: `with eng.evaluate(step, ema=False) as e:`

The block runs under `torch.random.fork_rng`. Global CPU RNG is seeded to `seed + 402 + step`, the frozen
`NoisePolicy.evaluation` rule. That rule fixes the host's own eval data draws. The block has a private eval
noise stream `Generator().manual_seed(seed + 402 + step + 1901)` (the frozen `OUTPUT_NOISE_SEED_OFFSET`)
and sets modules to eval. `e.generate(...)` applies `perturb_latent(record=False)` and then output noise at
`sigma = output_sigma()`. The candidate's sampling law therefore includes the latent perturbation, as in
screen.py. Each observation is scored twice from identically seeded streams: **noisy** (the record) and
clean (sigma 0; the perturbation is kept, as screen.py does). Training state and RNG are unchanged; a hash
receipt of the training streams is taken before and after each observation.

## 3. Parity plan (a gate before any custom run)

| level | what | pass criterion |
|---|---|---|
| L1 scalar step | `ScalarBinding(eng).step(real, generator_real=cb, collect_stats=...)` re-expresses `GANTrainer._step` through the phases. It runs next to a real `package.GANTrainer` built from twin-constructed G/D/prior (reseeded identically). Configs: C1 is the mode_hold resources (12 particles, z 4, batch 128, `mode_hold_host` MLPs, fresh stream0 data with a `generator_real` callback). C2 is a sparse table (512 particles, batch 64, z 2) so the A2 sparse path runs. There are **1000 updates**, which cover KA2 pure-A to blend (call 800), `sur_base`, spike guard (>200) and SettleTest windows. The matrix is dv12-ams-rc3 and st-10 × C1/C2 × CPU and CUDA (deterministic flags as in screen.py): 8 runs. | **Bitwise after every update**: returned losses (`loss_d`, `loss_g`, `loss_gan`, `prior_regularization`, `penalty`, `step`), every param and buffer of G/D/prior/ema_G/ema_prior/`opt_d.ema_critic`, both optimizer `state_dict()`s (Adam and AMSGrad moments, KA2 record, guard, A2 history), every group `lr`, `controller.state_dict()` (tensors with `torch.equal`), `lr_settle.state_dict()`, `log_output_sigma`, `last_output_sigma`, all 4 stream states, global CPU and CUDA RNG, `penalty_stats` at collect steps. At the end, `trainer.state_dict()` equals the engine's GANTrainer-schema export. |
| L2 gate tasks | A test-only shim makes `screen.run_mode_hold`, `run_image('img_intensity2')` and `run_vector('vector_two_broad')` drive an engine-backed trainer object for both candidates (GPU, direct run, no pool). | `harness/compare.py` against `runs/dv12-ams-rc3-ce/*` and `runs/st-10/*`: `rates.jsonl` byte-identical and every observation row equal except `seconds`. The perturb_latent speedup is documented as bitwise-neutral. |
| L3 component units (CPU, seconds) | (a) Two-table A2: engine `opt_g` + extra `LatentRowDamping` equals two separate `K3PGeneratorAdam`s, one per table, bitwise over 200 sparse steps. (b) Role-union penalty (§5.5) equals `Σ w_r·KA2_r` computed from per-role `_k3p_penalty` term functions on a frozen record, to about 1 ulp, and its EMA-anchor view evaluates `ema_critic.score` (checked against a manual computation). (c) Construction consumes no global RNG: after build, the global state equals the verbatim host construction's. (d) LR ownership: group LRs change only in P2/P3 (asserted every update in the ports). | exact (a, c, d); (b) documented tolerance, value only |
| L4 host smoke (CPU) | each port for 30 updates, run twice | identical `metrics.jsonl`/`rates.jsonl`; phase-order asserts; one penalty call per update |

Gate: `custom22.run` requires `runs/_parity/<package_sha256>.json` with L1 (CPU) = PASS. The first custom job
of a package runs L1-CPU under an atomic lock file; failure makes every custom task ERROR `engine parity`.
L1-CUDA, L2 and L3 are run once per engine revision and recorded in this report.

## 4. Per-host bindings

### 4a. Roles, resources and removed policy (line numbers are the pinned host file)

| host (budget) | G-side (role g, init key 0/2) | tables (role prior) | critic (role d, key 1) | recipe resources forced | removed host policy |
|---|---|---|---|---|---|
| two_pole (80) | none (identity map) | `particles` (12×1, zeros) becomes a package `ParticlePrior` built with a private generator and set to zeros, so no global RNG is used | `HostCritic`, **pinned weights** → `init_fresh=False` | num_particles 12, z_dim 1, batch 12 | L110-121 (Adam×2, legacy loss/cap), L133, L145 `schedule_optimizer` |
| trajectory (400) | `_Generator(slow,z)` | host `ParticlePrior(12,4,init_std=.1, gen seed 0)` (preserved) | `_Critic(slow,fast)` | 12 / 4 / 12 | L157-158, L160-166 (L159 `spread` kept, weight mapped), L181, L197 |
| residual_student (400) | `ResidualHead` | same as trajectory | `_Critic` | 12 / 4 / 12 | L197-202, L204-209 (L203 `spread` kept), L249, L270 |
| unipolar (400) | `FreeOriginResidual` (odd, even, origin) | none | `ScaleCritic` (input_scale buffer) | none | L255-274 (legacy GAN/penalty, Adam, `initial_lr`), L283-284 `_apply_lr` delayed cosine, L301, L316 |
| ae_gan_hold (250) | decoder `MLP(2,2)` key 0 + encoder `MLP(2,4)` key 2 | `recipe.make_prior()` → `MoGParticlePrior` (candidate init + calibration) | `MLP(2,1)` | num_particles 12, z_dim 2, batch 64, prior_kind mog, sigma_rel .025, encoder_mode ae | L162 legacy `make_recipe` (its d_lr_mult 1.5, prior_lr_mult 10, prior_betas, total_steps 6000), L169 legacy `make_optimizers`, L172-173, L208, L236 |
| cover_leftover (800) | `_Residual` (w_odd, w_even) | `prior_p`, `prior_m` (host `ParticlePrior(12,4,.05)`, global RNG, preserved) | `_FourierCritic` (bank buffer seed+17) | 12 / 4 / 32 | L398-406, L408-409, L415-423 (L407 `spread` kept; one shared prior group → two groups), L426 host `_EMA` (engine EMA), L457-461 delayed cosine, L484, L502 |
| unused_token_hold (200) | `SharedSlotStudent` (shared, slot; neu buffer) | none | `SlotCritic` | none | L229-237, L255, L276 |
| mid_scale_identity (800, CPU) | `MidScaleResidual` (odd, even, origin, mid) | none | `ScaleCritic` | none | L444-463, L472-473 `_apply_lr`, L491, L509 |

For every host, the host's `noise_policy` hooks (`set_step`, `discriminator()`, `output`, `evaluation`) are
replaced by P4, P6, P11, P11' and §2.3 at the same call sites. Host prints and `_emit`/`_log` previews are
dropped (they are logs only; `evaluate` restores the RNG), and the harness emits its own JSON lines (§6).

### 4b. Controller and penalty hook meaning

| host | `real` (P2, P7) | latent / table for P6 and P11 | fake (P7) | penalty call (P8): view, coordinates | role weights (adv D, adv G) |
|---|---|---|---|---|---|
| two_pole | `real_batch(12)`, fixed | latent = the whole table in index order (host uses every particle); map = identity | perturbed particles + noise | raw `critic`; x = particles (dim 1) | 1 / 1 |
| trajectory | `paired` fast arcs (12×16), fixed; slow is conditioning and not observed | latent = the whole `prior.z` in row order (identity association with slow row i); map `z ↦ G(slow, z)` | `G(slow,·)` + noise | `_FastView` with `.slow = slow` (the EMA view copies it); x = fast (dim 16) | 1 / 1 |
| residual_student | as trajectory | map `z ↦ head(slow, z)` | as trajectory | as trajectory | 1 / 1 |
| unipolar | `cat(real[0], real[1])` in raw units (16×4) | no latent; P6/P11 are replaced by `u.noise(delta(s) rows)` | `cat` of noisy `delta(s)` rows | one call: `RoleView(critic, scales=(0,1), rows=8)` whose forward is the row-block `critic.score(z_b, s_b)`; x = `cat(real_s/input_scale)`, `cat(fake_s/input_scale)` (§5.5) | 0.5 each / 0.5 each |
| ae_gan_hold | `data = sample_data(64)` (host global RNG) | latent = `prior.sample(64)` codes (host global RNG); table = MoG (§8 B1); map = decoder | `decoder(perturbed codes)` + noise | raw `critic`; x = data (dim 2) | 1 / `adversarial_weight` 1 |
| cover_leftover | host `real = cat(real_p, real_m)` (32×4) | per branch: `prior_x.sample(16)` (host RNG), then P6/P11 perturb with `table = prior_x`, then **host jitter** `+.01·randn_like` (host RNG), then map `neu + delta(±1) + z`; `observe_prior` on the union view (§5.3) | `cat(fake_p, fake_m)` + noise (one site, as the adapter does) | raw `critic`; x dim 4 | 1 / 1 |
| unused_token_hold | `CONCEPT_DIR` × 8, fixed | no latent; `u.noise(embeds(1)[CONCEPT] rows)` | noisy concept rows | raw `critic`, called before the adversarial term (host order); x dim 2 | 1 / 1 |
| mid_scale_identity | `cat(reals[s] for s in (-1,0,.5,1))` (32×4) | no latent; `u.noise(state(s) rows)` per scale | `cat` of noisy state rows | one call: `RoleView(critic, scales=(-1,0,.5,1), rows=8)`, normalized coordinates | 1/4 each / 1/4 each |

`observe_generator(G_side, ...)` receives the g-group network modules in optimizer order. That means
decoder + encoder for ae; the empty module for two_pole, where only the payoff update runs because there are
no G gradients. Tables and sigma are excluded, as with GANTrainer's `self.G`.

### 4c. Output noise, evaluation scope, and what is scored

The application sites are the frozen `legacy_noise_adapters` wrap points (EVAL_SCOPES). Sigma comes from
the package. Noise draws use the engine `noise` stream (GANTrainer seed+5) in host call order, never the
global RNG.

| host | EVAL_SCOPE | training sites (D: detached, G: attached) | scored object at the 24 observations | noisy ≠ clean? |
|---|---|---|---|---|
| two_pole | learned_particles_and_critic_gradient | D fake, G generated | `mean_abs(particles)`, `_grad_median(critic, real, particles)` (unwrapped critic, no draws) | no |
| trajectory | generated_samples | D fake, G fake (cover sees the noisy fake) | `identity_mse(e.generate(G(slow,·), prior.z), fast)` | **yes** |
| residual_student | generated_samples | D fake, G fake (cover and residual see it) | identity_mse, `landing_stats` of `e.generate(head)` | **yes** |
| unipolar | learned_residual_parameters | per-scale D and G fakes | `score_residual(student)` | no |
| ae_gan_hold | generated_and_reconstructed_samples | D fake, G reconstruction `decoder(encoded)` (P11'), G generated | `evaluate()` re-expressed: `recon = decoder(encode(data)) + noise`, `hold` on `e.generate(decoder, prior.sample(1024))` | **yes** |
| cover_leftover | learned_residual_parameters | D and G concatenated fakes | `score_geometry(residual, ...)` (the live residual; final `live` is taken before the EMA copy, as the host does) | no |
| unused_token_hold | learned_embedding_parameters | D fake, G fake_g (the hold loss uses clean embeds, as the host does) | `score_student(student)` | no |
| mid_scale_identity | learned_residual_parameters | per-scale D and G fakes (cover MSE uses clean `state(s)`, as the host does) | `score_hold(student, EVAL_SCALES, 'matched')` | no |

EMA (diagnostic only) scores the same scorer on EMA copies: EMA table + live critic for two_pole, and
ema G/encoder/tables for the rest.

### 4d. Auxiliary terms (verbatim host expressions; coefficients as the frozen compare route resolved them)

| host | loss_g = adversarial + ... | coefficient source |
|---|---|---|
| two_pole | `particle_l2·particles²` | particle_l2 = 0 (`compare_defaults.candidate`) |
| trajectory | `cover_weight·_cover(fake, fast) + particle_l2·z² + spread(z)` | cover 1.5 (`baseline.Candidate`), particle_l2 0, spread weight = `recipe.prior_reg` (the frozen VICReg mapping) |
| residual_student | as trajectory + `RESIDUAL_WEIGHT·mse(fake[mask], fast[mask])` | residual 1.0 (host) |
| unipolar | none | `cover_weight` 0 (unused by the host) |
| ae_gan_hold | `reconstruction_weight·recon + particle_l2·z² + cover_weight·cover` (+fm if >0) + GANTrainer's `prior_reg·prior_regularizer(z)` | recon 1.0 (HoldConfig), particle_l2 0, cover 1.5, fm 0; prior_reg from the candidate (0) |
| cover_leftover | `spread(cat z) + particle_l2·cat z² (if >0) + cover_weight·pole MSE` | spread weight = prior_reg (target_std .05 host), particle_l2 0, cover 1.5 |
| unused_token_hold | `hold_weight·unused_hold_loss` (+fm if >0) | hold 1.0, fm 0 |
| mid_scale_identity | `cover_weight·Σ_s mse(state(s), target_s)/4` | cover 1.5 (FORMULATION) |
| two_pole (table, no VICReg site) | GANTrainer's `prior_reg·prior_regularizer(particles)` | candidate prior_reg (0) |

The D objective is the host's adversarial expression through the package `GANLoss` (RpGAN logistic,
row-paired) with host role weights, plus the one KA2 penalty. The KA2 settings come from the candidate
recipe (reg_coeff 3, kappa 1, reg_every 1, anchor weight 1). The legacy b_cap and `GradRegularizer` are gone.

### 4e. Frozen verdict (spec from `default_comparison.json`; verdict = verbatim `protocol.test_verdict`)

There are 24 observations at `ceil(i·budget/24)`. PASS requires a complete curve, a passing suffix of at
least 5 (`sustained`) **and** every final `live` cell passing. EMA never counts.

| host | thresholds (live) |
|---|---|
| two_pole | mean_abs ≥ .30, grad_med ≤ 1.0 |
| trajectory | identity_mse ≤ .02 |
| residual_student | identity_mse ≤ .02, success_rate ≥ 1.0, wrong_pad_rate ≤ 0 |
| unipolar | cover ≥ .85, off_caption ≤ .05, neu_hold ≥ .85 |
| ae_gan_hold | recon_mse ≤ .05, hold ≤ .35 |
| cover_leftover | u_kept ≥ .85, content_kept ≥ .75, leak_ratio ≤ .20, pole_rel_err_± ≤ .20, same_dir ≤ .25 |
| unused_token_hold | unused_hold ≥ .85, concept_move ≥ .85 |
| mid_scale_identity | concept_cos_± ≥ .85, .75 ≤ concept_mag_± ≤ 1.25, identity_at_0 ≥ .85, identity_at_mid ≥ .85 |

`final` is the host's post-loop block, verbatim: the same seeded evaluation as step = budget, and the live
residual for cover_leftover.

## 5. Structural resolutions (derived from GANTrainer, no new constants)

1. **Hosts without a sampled prior.**
   - two_pole: the direct particles are a prior-role table (the frozen route already says "direct particles remain prior-owned") behind an identity G. That gives `lr·prior_lr_mult`, `prior_betas`, A2 (normally inactive: every row gets a gradient every step) and prior scale `(.05+.95m)·gt`. P6/P11 perturb the full-table latent, because the perturbation is a function of the table geometry and sits where GANTrainer puts it: after the latent draw, before the map. `DirectParticleResponse` is not used, since GANTrainer never builds it.
   - unipolar, unused_token_hold, mid_scale_identity: there is no table. `observe_prior` and `perturb_latent` never fire and `latent_bandwidth` stays None; the dv12 kernel is inert by construction. `prior_scale` has no group, so every LR is network or critic.
2. **Latent selection belongs to the host.** Hosts that use the full table in index order (two_pole, trajectory, residual) keep it. Hosts that sample (ae, cover) keep their global-RNG draws. The engine `latent` stream is allocated but unused, much as screen.py hands the host stream to GANTrainer.
3. **Two tables (cover_leftover).** Each table gets its own prior group and its own A2 (A2 is defined per sparse table and needs the table alone in its group). `perturb_latent` uses the table the sample came from as `prior`, so the exclusion radius comes from the law the sample was drawn from. `observe_prior` runs once per update on a union view `z = cat(p.z, m.z)`: one bandwidth state for the one G that consumes both tables. The perturbation goes between `prior.sample` and the host jitter, i.e. directly on the sample, as GANTrainer does.
4. **A2 composition.** `make_optimizers` handles `tables[0]`. The other tables use the package's own `LatentRowDamping.around(opt_g)`, which touches only its own group, so nesting is exact (checked in L3a). Splitting the host's single shared prior group into per-table groups is numerically neutral for elementwise Adam/AMSGrad. It only changes SettleTest granularity (one tester per group, which is package semantics).
5. **Multi-role scale-conditioned critics (unipolar R=2, mid R=4).** The controller advances once per update: one `observe_game`, one `observe_real` on the role-union real batch, one `critic_scale`, one `observe_pair`, one `observe_generator(loss_gan = Σ w_r g_r, loss_d_adv = Σ w_r d_r)`. **KA2 is called once per update on the role-union batch** through a row-block `RoleView` module (the paired EMA view works because the critic is a child). Both hosts use uniform weights `w_r = 1/R` and equal role sizes (8 rows), so the union KA2 value equals the host's `Σ w_r·cap_r` term by term (sample means, relu terms and prox mean), up to summation order. The code asserts uniform weights and equal sizes. This keeps GANTrainer's clock (one penalty call, one `advance_blend`, one surprise sample per critic step). Per-role calls would push R copies of the same completed-step surprise into `sur_hist`, apply `alpha *= game_trust` R times, and reach the 800-call warmup at update 400/R. Adversarial terms stay per-role and verbatim.
6. **Conditional slow/fast pairing (trajectory, residual).** The controller `real` is the critic's data coordinate (fast), not the conditioning (slow). This is the coordinate the frozen adapter perturbs (`data_index=1`) and that `_FastView` penalizes. RpGAN pairs row i with row i (same identity). The latent is `prior.z` row i (identity association), and P6/P11 perturb it before `G(slow_i, ·)`.
7. **MoG prior + encoder (ae_gan_hold).** The prior and encoder come from the package's own components: `recipe.make_prior()` with the mog/ae host resources, `recipe.encode(..., offset=)`, `reconstruction_loss`, and `make_optimizers(decoder, critic, prior, encoder=)`, which already assigns the MoG means to the prior role without A2. The perturbation applies only to prior-sampled codes, never to encoder-routed codes, because GANTrainer's `_generate` is the prior-sampling path. Output noise applies at every decoder output (the frozen site set). The host's post-construction RNG reset is kept verbatim. How the controller reads MoG geometry is open (§8 B1).
8. **Controller `real` aggregation.** `real` is the critic's real data-coordinate batch in raw units, concatenated over roles in host order, with conditioning excluded. `observe_real` standardizes by its own first-batch location and scale, so penalty-normalized versus raw coordinates is a global rescale and immaterial.
9. **Learnable sigma (st-10).** Sigma is a G-role parameter trained by the host's full G objective. It is attached at the frozen G-phase sites, so auxiliary terms computed on noisy fakes (cover, residual, ae reconstruction) also send it gradient. It is detached in the D phase and at evaluation, as GANTrainer detaches in the D step and in `sample`. Its group is appended after the tables at `recipe.lr` and scaled like G.
10. **Initialization.** The candidate's `batch_feature_zero` applies through `make_optimizers` to every host module whose weights come from a random draw (PyTorch default Linear init). This is the harness precedent: native100 Xavier weights are replaced too. Deterministic host-assigned weights count as custom weights; only two_pole's `HostCritic` has them, so two_pole uses `initialization=None`. Its particles (zeros) and every host-supplied table are preserved anyway, except ae_gan_hold's MoG table, which
    the candidate's `recipe.make_prior()` builds (initialization + component-width calibration) instead of the host's
    `make_recipe(cfg).make_prior()` draw (§9.5 B2).
11. **Budget.** The candidate's own `total_steps` applies (None for both). A scheduled candidate would follow its own horizon, as in GANTrainer, not the host budget as `FixedControl` did.
12. **EMA.** The engine keeps the GANTrainer EMA (`recipe.ema_decay`) for G-side modules and tables. It replaces cover_leftover's host `_EMA` (same .995) and is reported only.
13. **Device.** All 8 hosts run on CPU with 1 thread and deterministic algorithms. The frozen baseline protocol is "device cpu, threads 1", mid_scale_identity refuses CUDA, and the tensors are ≤ 64 rows. Engine parity is still proven on CPU and CUDA.

## 6. Outputs and logs (per job, `runs/<cand>/<task>/`)

- `log.txt`: one compact JSON line per observation, e.g. `{"step":..,"ok":0|1,<threshold metrics>,"noisy":{..} or "clean":{..},"ema":{..},"lr":[g,prior,d],"sigma":..,"pe":..,"m":..,"gt":..}`, then the `RESULT` line.
- `metrics.jsonl`: full rows plus `diag` (controller, penalty, `lr_settle`, KA2 record counters).
- `rates.jsonl`: per update, the lr of every group, `out_noise` (= `last_output_sigma`) and `in_noise`.
- `result.json`: the screen.py schema (`status`, `passing_checks`, `observations`, `first_arrival`, `final_streak`, `final`, `ema_final`, `thresholds`), `noisy_*`/`clean_*`, the protocol verdict dict, source/package/spec hashes, a construction RNG receipt, role/group ownership, resolved recipe and aux coefficients, KA2 call count (= updates), and the parity file id.
- Primary scoring follows the candidate's `eval_output_noise` exactly as on the other 14 tasks, so hashes and the leaderboard `eval` column stay consistent. The 22-check leaderboard of record reads the **noisy** verdict, which is always computed. For parameter-scored hosts, noisy equals clean.

## 7. Decisions (mirrors the structured summary)

The numbered rules are in §5. The remaining ones: host modules are ported as verbatim copies, and only the
loop is re-expressed with engine calls (with the source line references in §4a); import-only shims cover the
legacy recipe modules; fixed `real` batches are observed every update; host call order is kept for the
penalty relative to the adversarial term; `collect_stats` runs only at observation steps; the parity gate is
keyed by package sha; unknown recipe fields are refused.

## 8. Blockers (need the user's call: they would change or extend the candidate's policy)

| id | item | why it is policy, not binding | recommendation |
|---|---|---|---|
| B1 | ae_gan_hold MoG prior and the dv12 latent kernel | GANTrainer rejects MoG priors (`type(prior) is ParticlePrior`), so dv12's `observe_prior`/`perturb_latent` are undefined there. The code reads raw `prior.z`, but MoG samples live in `means()` space (standardized). Raw z gives a bandwidth and exclusion radius in the wrong coordinates. | Pass a view with `.z = prior.means().detach()` (the actual component centres). Alternatively the user may choose "no latent kernel on MoG hosts". Either choice extends the policy. |
| B2 | ae_gan_hold as a whole is outside the package's declared GANTrainer scope (`encoder_mode != 'none'`, `prior_kind='mog'` raise) | Running it means applying the dv12/KA2 policy to a model family the candidate never defined | Accept as an extension via the package's own components (§5.7); the verdict is labelled "extended scope" |
| B3 | Role-union KA2 call (§5.5) versus the package docstring, which allows one call per role ("multiple roles may consume the same completed step's surprise") | The two choices give different KA2 clocks, surprise histories and alpha decay on unipolar and mid_scale | Union call (the GANTrainer clock). Needs sign-off because the package text permits the other reading. |
| B4 | Learnable sigma gradient from host auxiliary losses (§5.9, st-10 only) | In GANTrainer, sigma learns from the adversarial loss only; here cover, residual and recon terms push it too | Keep the frozen adapter sites (sigma is a G parameter of the host's objective). The alternative is detaching sigma in the auxiliary terms (same noise sample). |
| B5 | Two-table single bandwidth (§5.3) | The controller has one `latent_bandwidth`; per-table bandwidths would need a package change | Union view (no package change) |
| B6 | `particle_birth_death` on custom hosts | BD's R/F density pools are scored at q = G(z) and moved rows copy G/EMA/optimizer/A2 state. No custom host exposes a one-arg G(z) map over a plain table: trajectory/residual_student take (slow, z); two_pole/unused_token_hold/mid_scale_identity bind no generator; unipolar (`FreeOriginResidual.delta(scale)`) and cover_leftover (`_Residual`, no forward) expose delta-only modules; ae_gan_hold is MoG. Any binding needs a host-specified scoring map. | Refuse (keeps native100 as the only BD evidence). Revisit only with a host-specified G(z) map, which is a policy extension. |

## 9. Implementation (2026-09-27)

All additive; the pool, `pool-config.json`, `queue/` and the `gates`/`all`/`native` presets are untouched. Config
hashes of dv12-ams-rc3, st-10, API-RP12 and st5-scaleaware-qr and the existing presets were recomputed before and
after the `lrlib.py`/`screen.py` edits: identical.

### 9.1 Files

| path | what |
|---|---|
| `harness/components.py` | engine: `Engine.build` (GANTrainer.__init__ order), `Update` phases P0-P13 with asserted order, `Evaluation`, `RoleView`, refusals, `scalar_step` (GANTrainer._step through the phases). Reuses the package's own `_output_sigma`, `output_sigma`, `_settle_observe`, `make_optimizers`, `make_critic_penalty`, `LatentRowDamping`, `DataDriftController`, `StationarityLR` |
| `harness/components_parity.py` | L1 gate CLI (`--package-root --overrides --steps --device --configs --out`) |
| `harness/custom22.py` | the 8 host loops (each line tagged `# Lnnn` with its original host line), `Observer` (24 noisy+clean, live+EMA observations, logs), verdict, parity gate (lock + cache), `run(ctx, task)` for screen.py |
| `harness/hosts/custom/` | 17 byte-identical copies (8 hosts, observation, two_pole/trajectory deps, legacy loss/penalty/locked_shared, toy100/device, package `__init__`s) with a 3-line provenance header; 3 import-only shims (`benchmarks/gan_v3.py`, `benchmarks/legacy/recipe.py`, `benchmarks/toy100/__init__.py`); `MANIFEST.json` (source commit fe7ed255, per-file sha256, shim hashes, referenced-file hashes); `frozen_verdict.py` (verbatim `score_metrics`, `requirements`, `test_verdict` function sources, hash-pinned) |
| `harness/tasks/custom22_specs.json` | the 8 frozen specs from `default_comparison.json` (source sha256 2e62a935…) + EVAL_SCOPES |
| `harness/tests/test_components_parity.py`, `harness/tests/custom22_host_policy.py`, `harness/tests/custom22_host_noise.py` | tests (§9.3); the host-policy driver runs a loop with the host's own learner or the original PR #155 host; the host-noise driver does the same with one output-noise policy on both sides |
| `harness/screen.py` (edit) | `CUSTOM_TASKS`; `--task` choices `ALL_TASKS + CUSTOM_TASKS`; one dispatch branch; no `get_device_name` for custom tasks |
| `harness/lrlib.py` (edit) | `CUSTOM_TASKS`, `ALL_TASKS += CUSTOM_TASKS`, presets `custom` and `all22` (= gates + custom + native = 22), `SHORT` names, cell labels `ERROR parity` / `ERROR refused` |
| `README.md` (edit) | layout rows + section "Custom22 hosts" |

### 9.2 Deviations from §0-§8 (all structural; no new constants)

1. **Three import-only shims, not two**: `benchmarks/toy100/__init__.py` imports the toy100 problem registry; the
   hosts only need `toy100/device.py` (copied verbatim) through `observation.py`.
2. **Copies carry a header**; the recorded sha256 is of the bytes below it and equals the original file's.
3. **Noisy is the primary score for custom tasks regardless of `eval_output_noise`** (task instruction and the
   user rule "scoring of record is noisy"). `result.json` says `primary_scoring: noisy`; clean is under `clean` /
   `clean_*`. The LEADERBOARD `eval` column is derived from the candidate's options, so for a clean-eval candidate
   such as dv12-ams-rc3 the custom cells are still noisy (only trajectory, residual_student and ae_gan_hold differ).
4. **`observe_generator`'s D argument is GANTrainer's own expression** `(loss_d - penalty)` with
   `loss_d = adversarial + penalty` (bitwise GANTrainer), not the adversarial terms alone (these differ in the last
   bits). `loss_gan` is the adversarial G term with the host role weights.
5. **Refusals are broader**: dv11 is refused on every custom host; critic input noise > 0 is refused (0 for every
   continuous policy); a GANTrainer defining any method outside the dv12-ams/dv12-st set is refused (st6's
   `_extragradient` is); `particle_birth_death`, non-gan/scalar recipes and non-default unknown recipe fields as
   designed.
6. **Parity cache key** = package sha + overrides hash + `components.py` sha + `components_parity.py` sha
   (`runs/_parity/<16>-<12>-<12>-<12>.json`), not the package alone: `lr_control`, `output_noise_mode` etc. are
   overrides, and an engine or gate edit re-verifies. The gate exercises only c1 (scalar mode_hold) and c2 (sparse
   table); host-specific engine paths (initialization=None on two_pole, second table + own A2 on cover_leftover,
   `RoleView` on unipolar/mid_scale_identity, encoder + MoG on ae_gan_hold) are covered by L3a/L3b and HostSmokeTest,
   not by the per-package gate. Every result says so in `custom22.parity.scope`.
7. **L2 (gate tasks through the engine) was not run.** L1 checks the same scalar path per update for 1000 updates.
8. The host scorers' own legacy `pass` flag (cover_leftover, mid_scale_identity) is kept as `host_pass`; `pass` in
   a harness row is the frozen-threshold pass. ae_gan_hold's pre-training `measure(0)` is kept as
   `custom22.initial`.
9. **mid_scale_identity critic input round-trip.** The frozen noisy route wires `critic.noise_policy`, so
   `ScaleCritic.score` computes `policy.input(z*input_scale)/input_scale` (original L243-247, L442-443). With
   input_std=0 that is the identity up to float rounding (input_scale is not a power of two; unipolar's 0.5 is, so
   its round-trip is exact). The binding does not wire the critic policy, because input noise > 0 is refused, so it
   matches the frozen noisy route in behaviour but not bitwise (first difference ~4e-7 relative, 12-update check).
   The noise-site test removes the round-trip on the original side for this host only.
10. **Initialization** (§5.10): the candidate's `batch_feature_zero` replaces the PyTorch-default init of every host
   `nn.Linear` on 7 hosts (two_pole keeps `initialization=None`); each result records it as
   `custom22.initialization`. Using the host init everywhere would be `initialization=None` in every `RESOURCES`
   entry (a structural choice left to the user).

### 9.3 Verification

| check | result |
|---|---|
| L1 CPU, 1000 updates, c1 (mode_hold resources, callable `generator_real`) + c2 (512-row sparse table) | **PASS, bitwise after every update**, dv12-ams-rc3 and st-10 (losses, penalty stats, every G/D/prior/EMA tensor, both optimizer state_dicts incl. KA2 record, guard, EMA critic, A2 history, all group LRs, controller and SettleTest state, sigma, 4 streams, global RNG). Coverage: 1000 KA2 calls (blend from call 800, anchor started), spike guard clipped (rc3 c1 4, st-10 c2 5), A2 sparse path active on c2 (row rate .118), SettleTest decisions (drift, stationary, reversal; st-10), learnable sigma .029 -> .011-.012 |
| L1 CUDA (cuda:0, shared GPU) | PASS, 100 updates c1 + c2, both candidates (and 50 in the test suite) |
| comparator mutation | a one-ulp change of the critic LR at update 4 is reported as the first mismatch (`step4…`) |
| L3a two-table A2 | engine opt_g + `LatentRowDamping.around(opt_g)` == two separate `K3PGeneratorAdam`s, bitwise, 200 sparse steps with the rho-damping path active every step (row rate .225) |
| L3b role-union KA2 | union call vs `sum_r cap_r / R` (4 roles): pure-A relative difference 9.4e-8 (summation order), blend phase (anchor term .1295 active) exactly equal |
| L3c construction RNG | engine build never consumes the global RNG (asserted at every build; `construction_global_rng_untouched`) |
| L3d LR ownership | group LRs are compared with the P2/P3 values before every optimizer step (asserted) |
| (c) engine disabled | the re-expressed loops on the copies, driven by the host's own learner (`HostPolicy`: host Adam, legacy loss/penalty, host default coefficients) reproduce the **original** PR #155 hosts for all 8 tasks, 12 updates: all 24 per-step sha256 of (LRs, params, grads) before each `optimizer.step()`, the observation curves and the final metrics are equal |
| (c') noise sites | as (c) with output noise ON: one duck-typed noise policy (isolated stream, sigma .05, per-evaluation reseed) goes through the bindings' `noise/sample/generate` sites and, as `noise_policy=`, into the original hosts. 12 updates, all 8 hosts: per-step digests, curves and final metrics are equal, and the sequence of training noise calls (step, shape, input hash) is identical. The copies' eval-scope calls are a subsequence of the originals'; the extra calls come from logging-only evaluations (residual_student `_emit`, ae_gan_hold `_log`). Test-only on the original side: host LR schedule disabled, mid_scale round-trip removed (§9.2.9) |
| (b) + L4 | every host x both candidates through `screen.py`, 30 updates, run twice: `metrics.jsonl` (minus seconds) and `rates.jsonl` identical; 24 observations, 30 KA2 calls, noisy == clean on the 5 parameter-scored hosts |
| sources | copies byte-identical below their headers; verdict functions equal to the checkout's source text; specs equal to `default_comparison.json` |

Command: `$PY -m unittest -v harness/tests/test_components_parity.py` (11 tests, all pass; ~7.5 min, CUDA part
skipped with `LRFREE_TEST_CUDA=0`).

Review fixes (2026-09-27, second pass; no binding logic changed): added the (c') noise-site regression test;
parity cache key now includes the `components_parity.py` sha (new PASS records
`runs/_parity/{ca2feb43…,7effbb0f…}-2b45e1b86997-7200a410c7ec.json`, c1+c2 1000 updates each, rc3 and st-10;
the older 3-part records are orphaned, not deleted); results gain `custom22.parity.scope` and
`custom22.initialization`; the B1/B2 `blockers_applied` labels now state the inactive MoG kernel and the
candidate-built MoG table. The full suite passed after the edits (L1 CPU 60 + CUDA 50 both candidates, L1 1000-update
gate both candidates via the smoke test, (b)/(c)/(c'), L3, sources). The 8 dv12-ams-rc3 full-budget screens were
rerun into `scratch-custom22/`: `metrics.jsonl` (minus seconds) and final blocks are identical to the previous
runs (§9.4 unchanged). Config hashes of the 4 registered candidates and the existing presets: unchanged.

### 9.4 Smoke verdicts at the full frozen budgets (informational; CPU, noisy record)

Direct `screen.py` runs, no pool: `scratch-custom22/<task>/` (dv12-ams-rc3) and `scratch-custom22/st-10/<task>/`.
Each job trains in 1.4-8 s; the first job of a package also runs the L1 gate (~40 s). Clean verdicts are
identical to the noisy ones in every cell.

dv12-ams-rc3 (4/8 PASS):

| task | status | passing | first | suffix | final live (noisy) | failing |
|---|---|---:|---:|---:|---|---|
| two_pole | FAIL | 0/24 | - | 0 | mean_abs .0247, grad_med .0021 | mean_abs (>= .30) |
| trajectory | PASS | 11/24 | 234 | 11 | identity_mse .0011 | - |
| residual_student | PASS | 22/24 | 50 | 22 | identity_mse .00095, success 1, wrong_pad 0 | - |
| unipolar | PASS | 11/24 | 234 | 11 | cover .955, off_caption .000001, neu_hold .902 | - |
| ae_gan_hold (extended scope) | PASS | 23/24 | 21 | 23 | recon_mse .0037, hold .0079 | - |
| cover_leftover | FAIL | 0/24 | - | 0 | u_kept .680, content .931, leak .0002, pole err .212/.224, same_dir .011 | u_kept, pole_rel_err_± |
| unused_token_hold | FAIL | 0/24 | - | 0 | unused_hold .999, concept_move .709 | concept_move (>= .85) |
| mid_scale_identity | FAIL | 0/24 | - | 0 | cos .987/.989, mag .883/.913, id_0 .790, id_mid .974 | identity_at_0 (>= .85) |

st-10 (5/8 PASS):

| task | status | passing | first | suffix | final live (noisy) | failing |
|---|---|---:|---:|---:|---|---|
| two_pole | FAIL | 0/24 | - | 0 | mean_abs .0376, grad_med .0077 | mean_abs |
| trajectory | FAIL | 0/24 | - | 0 | identity_mse .263 | identity_mse (<= .02) |
| residual_student | PASS | 21/24 | 50 | 19 | identity_mse .00018, success 1, wrong_pad 0 | - |
| unipolar | PASS | 15/24 | 167 | 15 | cover .995, off .00002, neu_hold .987 | - |
| ae_gan_hold (extended scope) | PASS | 23/24 | 21 | 23 | recon_mse .010, hold .0059 | - |
| cover_leftover | PASS | 13/24 | 400 | 13 | u_kept .987, content 1.000, pole err .008/.008 | - |
| unused_token_hold | FAIL | 2/24 | 192 | 2 | unused_hold .998, concept_move .870 | final passes; suffix 2 < 5 |
| mid_scale_identity | PASS | 12/24 | 434 | 12 | cos .9998/.9998, mag 1.002/1.000, id_0 .971, id_mid .999 | - |

Reading (no tuning implied): with rc3 the controller's mobility decays (final m .026 on cover_leftover, .045 on
mid_scale_identity), so the LRs fall (cover_leftover ends at G 1.5e-4 / prior 6.3e-4) while the geometry is still
approaching the bound; st-10's stationarity test keeps every group at its base LR there and passes cover/mid. two_pole moves its 12 zero-initialized particles only to
mean |x| .02-.04 in 80 updates under both. KA2 never leaves pure-A on the <=400-update hosts (800-call warmup);
the two 800-update hosts reach the blend on their last update only.

### 9.5 Open issues (need the user's call; each run records the ones it applied in `custom22.blockers_applied`)

- **B1** ae_gan_hold MoG kernel: implemented (kept) with the view `z = prior.means().detach()`, and labelled in
  `blockers_applied` as effectively inactive on this host. Observed consequence: MoG
  codes are `means[k] + sigma*eps`, so their nearest centre is their own component (not excluded, distance != 0) and
  the dv12 exclusion radius becomes half the MoG draw offset (rc3: radius mean .012, clipped fraction ~1, bandwidth
  .276): the latent kernel is almost fully clipped on this host. The alternative (no latent kernel on MoG hosts)
  is one line in `Engine.perturb_prior`.
- **B2** ae_gan_hold runs outside GANTrainer's declared scope (encoder_mode=ae, prior_kind=mog); `extended_scope:
  true` in its result. Its MoG table is the candidate recipe's (`batch_feature_zero` + width calibration from that
  table's geometry), not the host's `make_recipe(cfg).make_prior()` draw, so the latent sampling width differs too
  (seed 0, rc3: host sigma .0140 vs candidate .0183). The data stream is unchanged (host RNG reset L178-182 kept).
  The alternative (host table: `initialization=None` for the MoG prior only, keeping the candidate's optimizer and
  controller) needs the user's call; `blockers_applied` records the choice in use.
- **B3** unipolar / mid_scale_identity use one role-union KA2 call per update (GANTrainer clock); the per-role
  reading would reach the 800-call warmup at update 400/R and change the surprise history.
- **B4** st-10's learnable sigma also receives gradient from cover / residual / reconstruction terms computed on
  noisy samples (frozen noise sites).
- **B5** cover_leftover's two tables share one latent bandwidth (union view).
- Leaderboard `eval` label vs noisy-primary custom cells (§9.2.3).
- L2 not run (§9.2.7); dv11, particle birth-death and critic input noise are refused on custom hosts.
- Packages whose `_step` differs from the dv12-ams/dv12-st protocol (e.g. new LR rules in st* variants) are
  accepted only if they pass the L1 gate; otherwise every custom cell is `ERROR parity`.

### 9.6 two_pole investigation (2026-09-27): port defect fixed; dv12 failure is genuine

**Symptom.** two_pole FAILed 0/24 for every policy (final mean_abs .025 rc3, .038 st-10, .101 for the develop-K3P
control `k3p-noin-c22-h80`), while the frozen K3P route passes it (`reports/toy100/gap-fill-20260925/results/
k3p-toy-two_pole.json.gz`: PASS 10/24, mean_abs .645, grad_med .147).

**Frozen route rerun on CPU** (scratch copies of `sources/k3p/*`, only the CUDA device assert removed; same repo,
fixture and config): PASS 9/24, mean_abs .626 with the declared input noise .5; PASS 10/24 (first 50), .664 with
input noise 0. Input noise is not the cause.

**Per-update comparison (K3P control, input noise 0).** Scheduled LRs are identical (particles .0085 -> .000444 on
the prior schedule, critic .00425 -> 5.3e-5), update 1 is identical (grad .0052, step .0085), penalty/loss/sigma
agree (KA2 pure-A until call 67 in both; output-noise warmup to .029 by update 16). The difference is the
particle group's optimizer: the frozen route treats two_pole's opt_p group as **direct sample particles**
(`response.py` scope "direct sample particles only; registered ParticlePrior parameters excluded"): Adam betas
(0, .9) during the step and LR gain `1 + relu(cos(center(g_t), center(g_prev)))` (effective LR .0170 at update 2,
x1.9-2.0 from update ~40, final .00089), no A2. The binding instead wrapped the particles in a `ParticlePrior`
(latent-table class): betas `prior_betas or betas` = (0, .999), no gain, A2 attached (never started: 960/960 rows
observed). Displacement at update 2: .0076 vs .0153; mean_abs at update 50: .067 vs .320; final .101 FAIL.
The package defines this case: `recipe.make_generator_optimizer(params, direct_particles=[...])` (docs/k3p.md,
recipe fields `direct_particle_betas` (0, .9) / `direct_particle_gain` True, declared by every candidate).
Engine decomposition (K3P control): betas .9 only .230 FAIL; gain only .466 PASS 6/24; both .604 PASS 9/24.
**Verdict: port defect** (§5.1's "prior-role ParticlePrior, DirectParticleResponse not used" contradicts the package
API and the frozen route; "prior-owned" there means the LR role only).

**Fix (additive).** `components.Engine.build(direct_particles=...)` + `Engine._direct_optimizers`: the group is built
exactly as `make_optimizers` builds its prior group (`lr * prior_lr_mult`, `prior_betas or betas`, prior role/LR
scale), passed to the package's `make_generator_optimizer(groups, direct_particles=...)`; the critic optimizer is
`make_critic_optimizer` (make_optimizers' own call); requires `initialization=None` (two_pole already forces it);
tensor EMA of the particles (reported only); `state_dict` gains `direct`/`ema_direct` only when present.
`custom22.run_two_pole`: `particles = nn.Parameter(zeros(12, 1))` (host L107 verbatim), output noise at the two
frozen sites via `u.noise`, no latent table, hence no A2, no dv12 latent kernel and no GANTrainer prior
regularizer (both inert before: kernel perturbation rms <= 2e-4, prior_reg = 0 for every candidate). Observation
diag gains `direct` (last gain, betas). The default `make_optimizers` path and every other host are unchanged.

**Full-budget reruns** (`scratch-custom22/two_pole-fix/<cand>/`, screen.py direct, CPU, new parity records PASS):

| cand | status | passing | first | suffix | final mean_abs / grad_med | before |
|---|---|---:|---:|---:|---|---|
| k3p-noin-c22-h80 | **PASS** | 9/24 | 54 | 9 | .6035 / .1709 | FAIL .1009 |
| dv12-ams-rc3-c22 | FAIL | 0/24 | - | 0 | .0113 / .0088 | FAIL .0247 |
| st-10-c22 | FAIL | 0/24 | - | 0 | .0314 / .0087 | FAIL .0376 |

**dv12 failure is the candidates' own policy.** The particle gradient falls from .0048 (update 1) to <= .0003 by
update 10 and stays there, with grad_mean ~ -grad_rms (the 12 particles move together and never split toward the
poles); K3P's stays .001-.007 and splits (grad_mean ~ 0 at updates 40-50). Diagnostic arms on the fixed binding
(not candidates): rc3 with amsgrad off .478 (4/24, suffix 4 < 5), + reg_coeff 1 .584 PASS, reg_coeff 1 alone .104
FAIL; K3P + amsgrad .143 FAIL, K3P + reg_coeff 3 .447 PASS. Main driver: AMSGrad keeps the particle group's
second-moment maximum from update 1 while the gradient shrinks ~20x, so the step collapses (the package applies
`amsgrad` to every group, direct particles included); rc3's stronger penalty compounds it. The dv12 prior scale
(m .98 -> .67, prior LR .0085 -> .0058) is secondary. Tests: the full `test_components_parity.py` suite passes
after the change (CUDA part skipped: shared GPUs).
