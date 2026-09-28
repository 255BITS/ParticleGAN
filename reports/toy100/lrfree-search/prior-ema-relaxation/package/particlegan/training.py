"""A checkpointable training loop using the current recipe formulation."""
from copy import deepcopy
import math

import torch
from torch import nn

from .particle_prior import ParticlePrior
from .recipes import Recipe, learning_rate_scales


def input_noise_std(recipe, completed_steps):
    """Critic input-noise std for the next update (peak, linear to 0)."""
    if recipe.continuous_policy is not None:
        return 0.0
    end = recipe.input_noise_anneal_end * recipe.total_steps
    return float(recipe.input_noise_std * max(0.0, 1.0 - completed_steps / end))


def output_noise_std(recipe, completed_steps):
    """Generator output-noise std after ``completed_steps`` (linear warmup)."""
    if recipe.continuous_policy is not None:
        return float(recipe.output_noise_std)
    if recipe.output_noise_warmup == 0:
        return float(recipe.output_noise_std)
    return float(recipe.output_noise_std
                 * min(1.0, completed_steps / (recipe.output_noise_warmup * recipe.total_steps)))


class InputNoise(nn.Module):
    """``critic(x + std * eps, ...)`` with fresh ``eps`` per call from ``generator``.

    Plain ``critic(x, ...)`` (no draw) while ``std == 0``; set ``std`` as the
    schedule moves. Pass the wrapper to the recipe's critic penalty like the
    critic itself: its EMA critic is evaluated through the same wrapper (same
    ``std`` and noise stream).
    """

    def __init__(self, critic, std=0.0, generator=None):
        super().__init__()
        self.critic, self.std, self.generator = critic, float(std), generator

    def forward(self, x, *args, **kwargs):
        if self.std == 0:
            return self.critic(x, *args, **kwargs)
        noise = torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        return self.critic(x + self.std * noise, *args, **kwargs)


# Fields added to Recipe after checkpoints were written, with the value that
# reproduces the older behaviour; a saved recipe without them is upgraded.
_ADDED_RECIPE_FIELDS = {"continuous_policy": None, "amsgrad": False, "critic_r1_real": True,
                        "critic_payoff_damping": True, "output_noise_mode": "fixed",
                        "lr_control": "mobility", "particle_birth_death": False}


class GANTrainer:
    """Own the recipe's update mechanics; callers supply networks and real batches.

    Supports scalar, unconditional GAN recipes with a particle prior. Fresh real
    batches for the generator can be passed as ``generator_real`` tensors or
    zero-argument callables. Data-loader position is caller-owned and must be
    saved separately when checkpointing. Checkpoints restore global PyTorch RNG
    state as well as this trainer's sampling streams for exact continuation on
    the same device. Sampling never advances training RNG streams.

    Per update: role-wise LR schedule (``learning_rate_scales``), critic step
    with input noise and the recipe's penalty (``recipe.make_critic_penalty``;
    the critic optimizer's ``step()`` runs the spike guard and anchor EMA),
    then a generator/prior step with output noise (the generator optimizer's
    ``step()`` applies A2 latent damping). The trainer allocates the EMA
    critic ``ema_D`` (a frozen deep copy). Noise comes from a trainer stream;
    fresh network weights follow ``recipe.initialization``. ``sample`` includes
    the output noise (``output_sigma()``; see ``recipe.output_noise_mode``).

    ``serial_backward=True`` executes the whole update with autograd
    multithreading disabled. Use it for exact CUDA checkpoint continuation
    with higher-order critic penalties. This changes gradient summation order
    from the historical runtime, so it is explicit and checkpointed; loading
    across execution modes is rejected. The caller's autograd mode is restored.
    False inherits the caller's autograd setting; that ambient setting is not
    captured by a legacy checkpoint. True enforces the serialized constraint.
    """

    def __init__(self, recipe, generator, discriminator, *, prior=None, seed=0,
                 latent_generator=None, penalty_generator=None,
                 optimizer_options=None, penalty_options=None, serial_backward=False):
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
        if type(serial_backward) is not bool:
            raise TypeError("serial_backward must be a boolean")
        self.serial_backward = serial_backward
        if (recipe.model != "gan" or recipe.conditioning != "scalar"
                or recipe.encoder_mode != "none" or recipe.prior_kind != "particles"):
            raise ValueError("GANTrainer supports unconditional scalar GANs with particle priors and no encoder")
        self.recipe, self.G, self.D = recipe, generator, discriminator
        parameters = list(generator.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("generator must have trainable parameters")
        self.device, self.dtype = parameters[0].device, parameters[0].dtype
        self.prior = (recipe.make_prior().to(device=self.device, dtype=self.dtype)
                      if prior is None else prior)
        if type(self.prior) is not ParticlePrior:
            raise ValueError("prior must be a ParticlePrior")
        if self.prior.z.shape != (recipe.num_particles, recipe.z_dim):
            raise ValueError("prior dimensions must match the recipe")
        if not any(p.requires_grad for p in discriminator.parameters()):
            raise ValueError("discriminator must have trainable parameters")
        seen = set()
        for module in (self.G, self.D, self.prior):
            for value in (*module.parameters(), *module.buffers()):
                if value.device != self.device or (value.is_floating_point() and value.dtype != self.dtype):
                    raise ValueError("generator, discriminator and prior must share one device and floating dtype")
            for parameter in module.parameters():
                if id(parameter) in seen:
                    raise ValueError("generator, discriminator and prior must not share parameters")
                seen.add(id(parameter))
        self.optimizer_options = dict(optimizer_options or {})
        self.penalty_options = dict(penalty_options or {})
        # The recipe picks the regularization formulation; its
        # step-time work runs inside these optimizers' step().
        self.opt_g, self.opt_d = recipe.make_optimizers(
            self.G, self.D, self.prior, ema_critic=deepcopy(self.D), **self.optimizer_options)
        # output_noise_mode="learnable": trainer-owned log-sigma, its own generator
        # optimizer group at G's base LR (scaled like G by the LR schedule/controller).
        self.log_output_sigma = None
        self.last_output_sigma = None  # std applied in the last update (host float)
        if recipe.output_noise_mode == "learnable":
            self.log_output_sigma = nn.Parameter(torch.full(
                (), math.log(recipe.output_noise_std), device=self.device, dtype=self.dtype))
            self.opt_g.add_param_group({"params": [self.log_output_sigma], "lr": recipe.lr})
        self.initial_lrs = [[group["lr"] for group in opt.param_groups]
                            for opt in (self.opt_g, self.opt_d)]
        prior_ids = {id(p) for p in self.prior.parameters()}
        self.roles = [["prior" if any(id(p) in prior_ids for p in group["params"]) else "generator"
                       for group in self.opt_g.param_groups], ["critic"] * len(self.opt_d.param_groups)]
        self.loss = recipe.make_loss()
        self.prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
        self.ema_G, self.ema_prior = deepcopy(self.G).eval(), deepcopy(self.prior).eval()
        for module in (self.ema_G, self.ema_prior):
            module.requires_grad_(False)
        self.latent_generator = self._stream(latent_generator, seed + 2)
        # Reserved stream (the penalty draws no randomness); kept so the
        # checkpoint schema and the other streams' seeds stay unchanged.
        self.penalty_generator = self._stream(penalty_generator, seed + 3)
        self.eval_generator = self._stream(None, seed + 4)
        self.noise_generator = self._stream(None, seed + 5)
        self._noisy_D = InputNoise(self.D, 0.0, self.noise_generator)
        self.penalty = recipe.make_critic_penalty(self.opt_d, **self.penalty_options)
        self.completed_steps = 0
        self.prior_averaging_events = []
        from .continuous import DataDriftController
        self.controller = (None if recipe.continuous_policy is None else
                           DataDriftController(recipe.continuous_policy))
        if self.controller is not None:
            self.controller.observe_prior(self.prior)
        if recipe.continuous_policy in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            self.penalty.regularizer.continuous_controller = self.controller
        # lr_control="stationarity": per-group settling tests own the LR scale.
        self.lr_settle = None
        if recipe.lr_control == "stationarity":
            from .continuous import StationarityLR
            self.lr_settle = StationarityLR((self.opt_g, self.opt_d), prior_param=self.prior.z)
        # particle_birth_death: Fisher-Rao moves on the table (private stream seed + 6).
        self.birth_death = None
        if recipe.particle_birth_death:
            from .birth_death import ParticleBirthDeath
            self.birth_death = ParticleBirthDeath(self, seed + 6)

    @property
    def latent_damping(self):
        return self.opt_g.latent_damping

    @property
    def latent_history(self):
        return self.opt_g.latent_history

    @property
    def ema_D(self):
        """The trainer-owned EMA critic used by the anchor penalty."""
        return self.opt_d.ema_critic

    def _stream(self, generator, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed) if generator is None else generator
        device = generator.device
        # CUDA generators may expose the unindexed current-device spelling,
        # while parameter tensors always expose an explicit index.
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != self.device:
            raise ValueError("random generators must use the model device")
        return generator

    def _batch(self, batch, name):
        if (not isinstance(batch, torch.Tensor) or batch.ndim < 2 or not len(batch)
                or batch.device != self.device or batch.dtype != self.dtype):
            raise ValueError(f"{name} must be a nonempty batch on the model device and dtype")
        return batch.detach()

    def _output_sigma(self, base, detach=True):
        """Output-noise std from the fixed schedule value ``base`` and the recipe's mode.

        fixed: ``base``; mobility: ``base * controller.mobility`` (the mobility
        behind the current LR scales); learnable: ``max(exp(log_output_sigma),
        base * settle)`` where ``settle`` is the settledness of the generator
        and prior LR scales owned by the stationarity test: 1 while any G/prior
        scale is still open (s>1/64, i.e. not yet cut to a settled floor) and
        relaxing toward the controller mobility only once every G/prior group
        has settled. The state-driven floor keeps per-mode width near the data
        scale while centres are still moving; it relaxes automatically once the
        state test declares the table settled, so the learned width can refine
        late. No metric, accuracy, or coverage signal enters. A zero ``base``
        always means no noise.
        """
        mode = self.recipe.output_noise_mode
        if mode == "fixed":
            return base
        if mode == "mobility":
            return base * self.controller.mobility
        if not base:
            return 0.
        sigma = self.log_output_sigma.exp()
        settle = self.controller.mobility
        lr_settle = getattr(self, "lr_settle", None)
        testers = getattr(lr_settle, "testers", None)
        if testers:
            try:
                scales = [t.s for t in testers[0]]
                if scales and all(s is not None for s in scales):
                    settle = 1.0 if any(s > 1. / 64. for s in scales) else self.controller.mobility
            except Exception:
                pass
        floor = base * settle
        sigma = torch.maximum(sigma, torch.as_tensor(floor, device=sigma.device, dtype=sigma.dtype))
        return float(sigma.detach()) if detach else sigma

    def output_sigma(self):
        """The output-noise std that ``sample`` currently adds (a float)."""
        return float(self._output_sigma(output_noise_std(self.recipe, self.completed_steps)))

    def _generate(self, model, latent, sigma, stream):
        """``model(latent) + sigma * eps``; no draw when sigma == 0."""
        if self.controller is not None:
            prior = self.ema_prior if model is self.ema_G else self.prior
            latent = self.controller.perturb_latent(latent, stream, prior, record=stream is self.noise_generator)
        y = model(latent)
        if sigma == 0:
            return y
        return y + sigma * torch.randn(y.shape, generator=stream, device=y.device, dtype=y.dtype)

    def step(self, real, *, generator_real=None, collect_stats=False):
        """Perform one D update and one G/prior update; return detached losses.

        ``step`` in the result is the completed update count. The generator
        loss pairs fakes with ``generator_real`` (a tensor or a callable;
        default: ``real``). ``collect_stats`` additionally
        returns the gradient penalty's synchronized diagnostic dictionary
        (including the penalty's blend weight ``s``).
        """
        if self.serial_backward:
            # Higher-order critic gradients otherwise mix nodes created on
            # different autograd threads. Their thread-local sequence numbers
            # can reorder floating-point accumulation after a process restart.
            # Scope this to the complete update and restore the caller's mode.
            with torch.autograd.set_multithreading_enabled(False):
                return self._step(real, generator_real=generator_real, collect_stats=collect_stats)
        return self._step(real, generator_real=generator_real, collect_stats=collect_stats)

    def _step(self, real, *, generator_real=None, collect_stats=False):
        recipe = self.recipe
        if recipe.total_steps is not None and self.completed_steps >= recipe.total_steps:
            raise RuntimeError("recipe training budget exhausted")
        real = self._batch(real, "real")
        if generator_real is not None and not callable(generator_real):
            generator_real = self._batch(generator_real, "generator_real")
            if generator_real.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if len(generator_real) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
        if self.controller is not None:
            self.controller.observe_prior(self.prior)
            self.controller.observe_game(self.penalty.regularizer.record)
        prior_tester = None
        prior_scale_before = None
        prior_group_index = None
        if self.lr_settle is not None:
            for group_index, (tester, role) in enumerate(zip(self.lr_settle.testers[0], self.roles[0])):
                if tester is not None and role == "prior":
                    prior_tester = tester
                    prior_scale_before = tester.s
                    prior_group_index = group_index
                    break
        if self.birth_death is not None:
            self.birth_death.observe_real(real)
        if self.lr_settle is None:
            network, prior_scale = (learning_rate_scales(self.completed_steps, recipe)
                                    if self.controller is None else self.controller.observe_real(real))
            for optimizer, rates, roles in zip((self.opt_g, self.opt_d), self.initial_lrs, self.roles):
                for group, rate, role in zip(optimizer.param_groups, rates, roles):
                    group["lr"] = rate * (prior_scale if role == "prior" else network)
        else:
            # Mobility/data_score still update (output_noise_mode="mobility", reopen);
            # the LR is base * s per group (D also gets the payoff damping below).
            self.controller.observe_real(real)
            reopen = self.controller.data_score > 3.
            for group, tester in self.lr_settle.pairs((self.opt_g, self.opt_d)):
                if reopen:
                    tester.restart(group["params"], reopen=True)
                else:
                    tester.begin(group["params"])
            prior_scales = [tester.s for group, tester, role in
                            zip(self.opt_g.param_groups, self.lr_settle.testers[0], self.roles[0])
                            if role == "prior" and tester is not None and tester.s is not None]
            prior_scale = max(prior_scales) if prior_scales else None
            for index, (optimizer, rates, testers) in enumerate(
                    zip((self.opt_g, self.opt_d), self.initial_lrs, self.lr_settle.testers)):
                for group, rate, tester in zip(optimizer.param_groups, rates, testers):
                    own_scale = 1. if tester is None else tester.s
                    if index == 1 and prior_scale is not None:
                        own_scale = max(own_scale, prior_scale)
                    group["lr"] = rate * own_scale
        if self.controller is not None and recipe.critic_payoff_damping:
            for group in self.opt_d.param_groups:
                group["lr"] *= self.controller.critic_scale()
        sigma_in = input_noise_std(recipe, self.completed_steps)
        sigma_out = self._output_sigma(output_noise_std(recipe, self.completed_steps), detach=False)
        self.last_output_sigma = float(sigma_out.detach() if torch.is_tensor(sigma_out) else sigma_out)
        noise = self.noise_generator
        critic = self._noisy_D
        critic.std = sigma_in
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, _ = self.prior.sample(len(real), generator=self.latent_generator)
            if self.controller is not None:
                self.controller.observe_support(self.G, self.D, latent, sigma_out, noise)
            fake = self._generate(self.G, latent, sigma_out, noise)
        if self.controller is not None:
            self.controller.observe_pair(real, fake)
        loss_d = self.loss.d_loss(critic(real), critic(fake))
        self.penalty.collect_stats = collect_stats
        penalty = self.penalty(critic, real, fake)
        penalty_stats = self.penalty.last_stats
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        if self.lr_settle is not None:
            self._settle_observe(1)

        # Experimental critic-tracking candidate: one additional critic-only
        # update per generator/prior update. The host callback must supply a new
        # real batch on each invocation; this added draw is part of the candidate.
        if generator_real is not None:
            if not callable(generator_real):
                raise ValueError("two-D-update candidate requires a fresh real-batch callback")
            real_d2 = self._batch(generator_real(), "critic_real_extra")
            latent_d2, _ = self.prior.sample(len(real_d2), generator=self.latent_generator)
            with torch.no_grad():
                if self.controller is not None:
                    latent_d2 = self.controller.perturb_latent(
                        latent_d2, noise, self.prior, record=False)
                fake_d2 = self.G(latent_d2)
                if sigma_out:
                    fake_d2 = fake_d2 + sigma_out * torch.randn(
                        fake_d2.shape, generator=noise, device=fake_d2.device, dtype=fake_d2.dtype)
            loss_d = self.loss.d_loss(critic(real_d2), critic(fake_d2))
            penalty = self.penalty(critic, real_d2, fake_d2)
            penalty_stats = self.penalty.last_stats
            loss_d = loss_d + penalty
            self.opt_d.zero_grad()
            loss_d.backward()
            self.opt_d.step()
            if self.lr_settle is not None:
                self._settle_observe(1)

        self.D.eval()
        self.G.train()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, indices = self.prior.sample(len(real), generator=self.latent_generator)
            fake_logits = critic(self._generate(self.G, latent, sigma_out, noise))
            real_g = generator_real() if callable(generator_real) else generator_real
            real_g = real if real_g is None else self._batch(real_g, "generator_real")
            if real_g.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if len(real_g) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
            real_logits = critic(real_g)
            loss_gan = self.loss.g_loss(fake_logits, real_logits)
            prior_reg = loss_gan.new_zeros(())
            if self.prior.z.requires_grad:
                raw = self.prior.z if recipe.num_particles <= 1024 else self.prior.z[torch.unique(indices)]
                prior_reg = self.prior_regularizer(raw)
            loss_g = loss_gan + recipe.prior_reg * prior_reg
            self.opt_g.zero_grad()
            loss_g.backward()
            if self.controller is not None:
                self.controller.observe_generator(self.G, loss_gan.detach(), (loss_d - penalty).detach())
            self.opt_g.step()
            if self.lr_settle is not None:
                self._settle_observe(0)
            prior_scale_reduced = (prior_tester is not None and prior_scale_before is not None
                                   and prior_tester.s < prior_scale_before)
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
        with torch.no_grad():
            for target, source in ((self.ema_G, self.G), (self.ema_prior, self.prior)):
                for averaged, current in zip(target.parameters(), source.parameters()):
                    averaged.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
                for averaged, current in zip(target.buffers(), source.buffers()):
                    averaged.copy_(current)
        prior_stationary_relaxation = (prior_tester is not None and not prior_scale_reduced
                                       and prior_tester.last_decisive == -1
                                       and prior_tester.counts.get("reopens", 0) == 0)
        if prior_scale_reduced or prior_stationary_relaxation:
            # A stationarity decision retains the original full EMA handoff.
            # Subsequent ordinary updates relax toward that same EMA on the
            # tester's intrinsic block timescale. Translate the positional
            # anchor by the applied non-gradient displacement without clearing
            # accumulated gradient blocks or correlation evidence.
            with torch.no_grad():
                table = self.prior.z
                old = table.detach().clone()
                stream_states = {name: getattr(self, name).get_state().clone()
                                 for name in self._STREAMS}
                if prior_scale_reduced:
                    table.copy_(self.ema_prior.z)
                    relaxation_alpha = 1.0
                else:
                    group = self.opt_g.param_groups[prior_group_index]
                    intrinsic_delta = group["lr"] / self.initial_lrs[0][prior_group_index]
                    relaxation_alpha = -math.expm1(-intrinsic_delta / prior_tester.b)
                    table.lerp_(self.ema_prior.z, relaxation_alpha)
                displacement = table - old
                if prior_scale_reduced:
                    rows = torch.arange(len(table), device=table.device)
                    prior_tester.rebase([table], rows)
                else:
                    prior_tester.anchor.add_(displacement.reshape(-1))
                if self.birth_death is not None and self.birth_death.anchor is not None:
                    self.birth_death.anchor.copy_(self.G(table).flatten(1))
                if any(not torch.equal(state, getattr(self, name).get_state())
                       for name, state in stream_states.items()):
                    raise RuntimeError("prior EMA handoff unexpectedly consumed a training RNG stream")
                self.prior_averaging_events.append(dict(
                    step=self.completed_steps + 1,
                    reason=("stationarity_scale_reduction" if prior_scale_reduced else "stationary_intrinsic_relaxation"),
                    s_before=prior_scale_before, s_after=prior_tester.s,
                    intrinsic_delta=(None if prior_scale_reduced else intrinsic_delta),
                    relaxation_alpha=relaxation_alpha,
                    decision=dict(prior_tester.last), displacement_rms=float(displacement.square().mean().sqrt()),
                    displacement_max=float(displacement.norm(dim=1).max()),
                    optimizer_block_rms_valid_rows_zeroed=(
                        float(torch.nan_to_num(prior_tester.last_block).square().mean().sqrt())
                        if prior_scale_reduced or prior_tester.tau == 0 else None),
                    optimizer_block_valid_fraction=(
                        float(torch.isfinite(prior_tester.last_block).float().mean())
                        if prior_scale_reduced or prior_tester.tau == 0 else None),
                    prior_rms=float(table.square().mean().sqrt()),
                    latent_history_rms=(None if self.opt_g.latent_history is None else
                                        float(self.opt_g.latent_history.square().mean().sqrt())),
                    birth_death_anchor_rebased=bool(self.birth_death is not None and
                                                    self.birth_death.anchor is not None),
                    tester_evidence_preserved=not prior_scale_reduced))
        if self.birth_death is not None:
            event = self.birth_death.maybe_apply(self, self.last_output_sigma)
            if event and event.get("moves") and self.lr_settle is not None:
                # A teleport is not gradient movement: re-anchor the moved rows of the prior
                # tester's current block (restart() if the group is not the bare table).
                for group, tester, role in zip(self.opt_g.param_groups, self.lr_settle.testers[0], self.roles[0]):
                    if role == "prior" and tester is not None:
                        tester.rebase(group["params"], self.birth_death.moved_rows)
        self.completed_steps += 1
        if __import__("os").environ.get("PARTICLEGAN_ADAM_DIAG_DIR"):
            from .optvar_diag import capture as _capture_optvar
            _capture_optvar(self)
        result = {key: value.detach() for key, value in
                  dict(loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
                       prior_regularization=prior_reg, penalty=penalty).items()}
        result["step"] = self.completed_steps
        if collect_stats:
            result["penalty_stats"] = penalty_stats
        return result

    def _settle_observe(self, index):
        optimizer = (self.opt_g, self.opt_d)[index]
        for group, rate, tester in zip(optimizer.param_groups, self.initial_lrs[index], self.lr_settle.testers[index]):
            if tester is not None:
                tester.observe(group["params"], group["lr"] / rate, step=self.completed_steps + 1)

    @torch.no_grad()
    def sample(self, n, *, ema=False, generator=None):
        """Draw live or EMA samples (with the current output noise) without
        changing modes or training RNGs."""
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        stream = self.eval_generator if generator is None else self._stream(generator, 0)
        if stream in (self.latent_generator, self.penalty_generator, self.noise_generator):
            raise ValueError("sampling requires a stream separate from training")
        model, prior = (self.ema_G, self.ema_prior) if ema else (self.G, self.prior)
        modes = [(module, module.training) for root in (model, prior) for module in root.modules()]
        devices = [self.device.index] if self.device.type == "cuda" else []
        try:
            model.eval()
            prior.eval()
            with torch.random.fork_rng(devices=devices):
                latent, _ = prior.sample(n, generator=stream)
                return self._generate(model, latent,
                                      self._output_sigma(output_noise_std(self.recipe, self.completed_steps)), stream)
        finally:
            for module, flag in modes:
                module.training = flag

    _STREAMS = ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")

    def state_dict(self):
        """Return an independent checkpoint; save the caller's data cursor too."""
        names = ("G", "D", "prior", "ema_G", "ema_prior")
        return deepcopy({
            **({"serial_backward": True} if self.serial_backward else {}),
            **({"controller": self.controller.state_dict()} if self.controller is not None else {}),
            "schema": 4, "recipe": self.recipe.to_dict(),
            "optimizer_options": self.optimizer_options, "penalty_options": self.penalty_options,
            "device": str(self.device), "dtype": str(self.dtype),
            "models": {name: getattr(self, name).state_dict() for name in names},
            "requires_grad": {name: {key: p.requires_grad for key, p in getattr(self, name).named_parameters()}
                              for name in names},
            "optimizers": [self.opt_g.state_dict(), self.opt_d.state_dict()],
            "initial_lrs": self.initial_lrs, "completed_steps": self.completed_steps,
            "streams": {name: getattr(self, name).get_state() for name in self._STREAMS},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            **({"output_noise": {"log_sigma": self.log_output_sigma.detach()}}
               if self.log_output_sigma is not None else {}),
            **({"lr_settle": self.lr_settle.state_dict()} if self.lr_settle is not None else {}),
            **({"birth_death": self.birth_death.state_dict()} if self.birth_death is not None else {}),
        })

    def load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state.

        Recreate the same parameter freezing before loading. Validation of both
        optimizers (which carry the KA2 state) and all RNG states precedes any
        mutation of the live trainer. Earlier formulations cannot be resumed
        under KA2; use the release that wrote those checkpoints.
        """
        if isinstance(state, dict) and state.get("schema") in (1, 2, 3):
            raise ValueError(
                f"schema-{state['schema']} GANTrainer checkpoints come from an older formulation "
                "and cannot resume under KA2; pin the release that wrote the checkpoint "
                "(0.8.0 for K3P), or start a new run")
        expected = self.state_dict()
        if isinstance(state, dict) and (
                type(state.get("serial_backward", False)) is not bool
                or state.get("serial_backward", False) != self.serial_backward):
            raise ValueError("checkpoint serial_backward execution mode does not match trainer")
        if not isinstance(state, dict) or state.keys() != expected.keys() or state.get("schema") != 4:
            raise ValueError("invalid GANTrainer checkpoint schema")
        saved_recipe = state["recipe"]
        if isinstance(saved_recipe, dict):
            # Preserve pre-continuous / pre-amsgrad checkpoint compatibility.
            saved_recipe = {**_ADDED_RECIPE_FIELDS, **saved_recipe}
        if (not isinstance(saved_recipe, dict)
                or saved_recipe.get("initialization") not in (None, "batch_feature_zero")
                or {k: v for k, v in saved_recipe.items() if k != "initialization"}
                != {k: v for k, v in expected["recipe"].items() if k != "initialization"}):
            raise ValueError("checkpoint recipe does not match trainer")
        # Saved weights replace construction-time initialization completely.
        for key in ("optimizer_options", "penalty_options", "device", "dtype", "requires_grad"):
            if state[key] != expected[key]:
                raise ValueError(f"checkpoint {key} does not match trainer")
        steps = state["completed_steps"]
        if type(steps) is not int or steps < 0 or (self.recipe.total_steps is not None and steps > self.recipe.total_steps):
            raise ValueError("invalid checkpoint step count")
        rates = state["initial_lrs"]
        if (not isinstance(rates, list) or len(rates) != 2
                or any(not isinstance(a, list) or len(a) != len(b) for a, b in zip(rates, self.initial_lrs))
                or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for row in rates for v in row)):
            raise ValueError("invalid checkpoint initial learning rates")
        if (not isinstance(state["models"], dict) or not isinstance(state["streams"], dict)
                or state["models"].keys() != expected["models"].keys()
                or state["streams"].keys() != expected["streams"].keys()):
            raise ValueError("invalid checkpoint model or RNG schema")
        for name, tensors in state["models"].items():
            current = expected["models"][name]
            if not isinstance(tensors, dict) or tensors.keys() != current.keys() or any(
                    not isinstance(tensors[k], torch.Tensor) or tensors[k].shape != v.shape
                    or tensors[k].dtype != v.dtype for k, v in current.items()):
                raise ValueError(f"checkpoint model {name} has incompatible tensors")
        if not isinstance(state["optimizers"], list) or len(state["optimizers"]) != 2:
            raise ValueError("invalid checkpoint optimizer schema")
        try:
            for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
                deepcopy(optimizer).load_state_dict(deepcopy(values))
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid checkpoint optimizer state") from error
        try:
            for value in state["streams"].values():
                torch.Generator(device=self.device).set_state(value.cpu())
            torch.Generator(device="cpu").set_state(state["cpu_rng"].cpu())
            if self.device.type == "cuda":
                torch.Generator(device=self.device).set_state(state["cuda_rng"].cpu())
            elif state["cuda_rng"] is not None:
                raise ValueError("CPU trainer cannot load CUDA RNG state")
        except (TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid checkpoint RNG state") from error
        if self.log_output_sigma is not None:
            value = state["output_noise"].get("log_sigma") if isinstance(state["output_noise"], dict) else None
            if (not isinstance(value, torch.Tensor) or value.shape != self.log_output_sigma.shape
                    or value.dtype != self.log_output_sigma.dtype or not torch.isfinite(value).all()):
                raise ValueError("invalid checkpoint learnable output noise")
        if self.controller is not None:
            deepcopy(self.controller).load_state_dict(state["controller"])
        if self.lr_settle is not None:
            deepcopy(self.lr_settle).load_state_dict(state["lr_settle"], (self.opt_g, self.opt_d))
        if self.birth_death is not None:
            self.birth_death.check_state(state["birth_death"])
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        if self.log_output_sigma is not None:
            with torch.no_grad():
                self.log_output_sigma.copy_(state["output_noise"]["log_sigma"])
        for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
            optimizer.load_state_dict(deepcopy(values))
        if self.controller is not None:
            self.controller.load_state_dict(state["controller"])
        if self.lr_settle is not None:
            self.lr_settle.load_state_dict(state["lr_settle"], (self.opt_g, self.opt_d))
        if self.birth_death is not None:
            self.birth_death.load_state_dict(state["birth_death"])
        self.initial_lrs, self.completed_steps = deepcopy(rates), steps
        self.recipe = self.recipe.replace(initialization=saved_recipe.get("initialization"))
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
