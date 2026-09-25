"""A small, checkpointable K3P training loop for unconditional particle GANs."""
from copy import copy, deepcopy
import math

import torch

from .k3p import CriticAnchor
from .particle_prior import ParticlePrior
from .recipes import Recipe, learning_rate_scales


def _buffer_pairs(ema, live):
    live_b, ema_b = dict(live.named_buffers()), dict(ema.named_buffers())
    if live_b.keys() != ema_b.keys():
        raise ValueError("ema_critic buffer names differ from critic's")
    for name, b in live_b.items():
        if ema_b[name].shape != b.shape or ema_b[name].dtype != b.dtype:
            raise ValueError(f"ema_critic buffer {name} differs in shape or dtype")
    return [(ema_b[name], b) for name, b in live_b.items()]


class RobustCriticAnchor(CriticAnchor):
    """``CriticAnchor`` that also averages buffers and never mutates state on forward.

    Floating-point buffers (e.g. BatchNorm running statistics) are averaged
    with the parameters; integer buffers are copied. Every EMA forward runs
    in the live critic's per-module train/eval mode and restores any EMA
    buffer it changed (BatchNorm statistics, spectral-norm ``u``/``v``), so
    evaluating the anchor never changes the EMA or the live critic. No
    ``.data`` swapping is involved.
    """

    def __init__(self, critic, ema_critic, decay=0.999):
        super().__init__(critic, ema_critic, decay)
        self._buffer_pairs = _buffer_pairs(ema_critic, critic)
        ema_modules, live_modules = list(ema_critic.modules()), list(critic.modules())
        if len(ema_modules) != len(live_modules):
            raise ValueError("ema_critic module structure differs from critic's")
        self._module_pairs = list(zip(ema_modules, live_modules))
        # (owning module, buffer name) for every EMA buffer.
        self._owned = [(module, name) for module in ema_modules
                       for name, buffer in module._buffers.items() if buffer is not None]

    @torch.no_grad()
    def start_(self):
        super().start_()
        for e, b in self._buffer_pairs:
            e.copy_(b)

    @torch.no_grad()
    def update_(self):
        super().update_()
        for e, b in self._buffer_pairs:
            if e.is_floating_point():
                e.mul_(self.decay).add_(b, alpha=1.0 - self.decay)
            else:
                e.copy_(b)

    def forward(self, fn, x):
        """Evaluate ``fn(ema_critic, x)`` without side effects on any module state."""
        modes = [(e, e.training) for e, _ in self._module_pairs]
        for e, live in self._module_pairs:
            e.training = live.training
        # The forward runs on private copies of every EMA buffer: a train-mode
        # BatchNorm or spectral norm updates (and may save for backward) the
        # copies, and the EMA's own buffers are put back untouched afterwards.
        # No in-place restore, so the caller's later input-gradient is valid.
        originals = [(module, name, module._buffers[name]) for module, name in self._owned]
        try:
            for module, name, buffer in originals:
                module._buffers[name] = buffer.clone()
            return fn(self.ema_critic, x)
        finally:
            for module, name, buffer in originals:
                module._buffers[name] = buffer
            for e, flag in modes:
                e.training = flag

    def __call__(self, x):
        return self.forward(lambda module, inputs: module(inputs), x)


class K3PCritic:
    """Best-practice K3P bundle for one critic optimizer.

    Allocates the EMA critic (``deepcopy(critic)``, frozen), a
    ``RobustCriticAnchor`` over it, the recipe's gradient penalty and critic
    spike guard. Use one per critic optimizer; one module used in several
    roles shares one bundle and passes an ``ema_critic`` per role::

        k3p = K3PCritic(recipe, D, opt_d)
        loss = adv + k3p.penalty(D, real, fake, step)
        # shared module: k3p.penalty(lambda x: D.role(x), xr, xf, step,
        #                            ema_critic=k3p.ema_critic(lambda m, x: m.role(x)))
        opt_d.zero_grad(); loss.backward(); k3p.step()

    ``step()`` applies the guard, steps the optimizer and records the step
    for K3P. ``state_dict()`` holds the penalty scalars, EMA critic and guard.
    ``optimizer`` may be None for penalty-only use (``step()`` then raises).
    """

    def __init__(self, recipe, critic, optimizer, **penalty_overrides):
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
        self.critic, self.optimizer = critic, optimizer
        arm = penalty_overrides.get("arm", recipe.reg_arm)
        self.ema = self.anchor = None
        if arm == "k3p":
            self.ema = deepcopy(critic).requires_grad_(False)
            self.anchor = RobustCriticAnchor(critic, self.ema, decay=recipe.reg_anchor_decay)
        self.regularizer = recipe.make_gradient_penalty(anchor=self.anchor, **penalty_overrides)
        self.guard = recipe.make_critic_guard()

    def ema_critic(self, fn=None):
        """Callable ``x -> fn(ema_module, x)`` (default ``ema_module(x)``), side-effect free."""
        if self.anchor is None:
            return None
        if fn is None:
            return self.anchor
        return lambda x: self.anchor.forward(fn, x)

    def penalty(self, D, x_real, x_fake, step, *, generator=None, collect_stats=False, ema_critic=None):
        """``(penalty, stats)`` for one critic (or one role of it)."""
        options = {} if ema_critic is None else {"ema_critic": ema_critic}
        return self.regularizer.penalty(D, x_real, x_fake, step, generator, collect_stats, **options)

    def step(self):
        """Guard, ``optimizer.step()``, then ``after_critic_step`` (anchor EMA + LR record)."""
        if self.optimizer is None:
            raise RuntimeError("K3PCritic was built without an optimizer")
        if self.guard is not None:
            self.guard.apply_(self.optimizer)
        self.optimizer.step()
        self.regularizer.after_critic_step(self.optimizer)

    def state_dict(self):
        return {"penalty": self.regularizer.state_dict(),
                "ema": None if self.ema is None else self.ema.state_dict(),
                "guard": None if self.guard is None else self.guard.state_dict()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"penalty", "ema", "guard"}:
            raise ValueError("invalid K3PCritic state")
        if (state["ema"] is None) != (self.ema is None) or (state["guard"] is None) != (self.guard is None):
            raise ValueError("K3PCritic state does not match this recipe")
        self.regularizer.load_state_dict(state["penalty"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        if self.guard is not None:
            self.guard.load_state_dict(state["guard"])


def input_noise_std(recipe, completed_steps):
    """Critic input-noise std for the next update (peak, linear to 0)."""
    end = recipe.input_noise_anneal_end * recipe.total_steps
    return float(recipe.input_noise_std * max(0.0, 1.0 - completed_steps / end))


def output_noise_std(recipe, completed_steps):
    """Generator output-noise std after ``completed_steps`` (linear warmup)."""
    if recipe.output_noise_warmup == 0:
        return float(recipe.output_noise_std)
    return float(recipe.output_noise_std
                 * min(1.0, completed_steps / (recipe.output_noise_warmup * recipe.total_steps)))


class GANTrainer:
    """Own the K3P update mechanics; callers supply networks and real batches.

    Supports scalar, unconditional GAN recipes with a particle prior. Fresh real
    batches for the generator can be passed as ``generator_real`` tensors or
    zero-argument callables. Data-loader position is caller-owned and must be
    saved separately when checkpointing. Checkpoints restore global PyTorch RNG
    state as well as this trainer's sampling streams for exact continuation on
    the same device. Sampling never advances training RNG streams.

    Per update: role-wise LR schedule (``learning_rate_scales``), critic step
    with input noise, K3P penalty, spike guard and anchor EMA
    (``K3PCritic``), then a generator/prior step with output noise and A2
    latent damping. Noise is applied functionally from a trainer stream; the
    caller's modules are never wrapped. ``sample`` includes the output noise.
    """

    def __init__(self, recipe, generator, discriminator, *, prior=None, seed=0,
                 latent_generator=None, penalty_generator=None,
                 optimizer_options=None, penalty_options=None):
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
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
        self.opt_g, self.opt_d = recipe.make_optimizers(
            self.G, self.D, self.prior, **self.optimizer_options)
        self.initial_lrs = [[group["lr"] for group in opt.param_groups]
                            for opt in (self.opt_g, self.opt_d)]
        prior_ids = {id(p) for p in self.prior.parameters()}
        self.roles = [["prior" if any(id(p) in prior_ids for p in group["params"]) else "generator"
                       for group in self.opt_g.param_groups], ["critic"] * len(self.opt_d.param_groups)]
        self.loss = recipe.make_loss()
        self.critic = K3PCritic(recipe, self.D, self.opt_d, **self.penalty_options)
        self.penalty = self.critic.regularizer
        self.prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
        self.latent_history = self.latent_damping = None
        if self.prior.z.requires_grad and recipe.latent_damping_max_rate > 0:
            group = self.opt_g.param_groups[self.roles[0].index("prior")]
            if len(group["params"]) != 1 or group["betas"][0] != 0.0:
                raise ValueError("A2 latent damping needs the prior table alone with beta1 == 0; "
                                 "set latent_damping_max_rate=0 to train without it")
            self.latent_history = torch.zeros_like(self.prior.z, requires_grad=False)
            self.latent_damping = recipe.make_latent_damping(self.prior.z, self.latent_history)
        self.ema_G, self.ema_prior = deepcopy(self.G).eval(), deepcopy(self.prior).eval()
        for module in (self.ema_G, self.ema_prior):
            module.requires_grad_(False)
        self.latent_generator = self._stream(latent_generator, seed + 2)
        self.penalty_generator = self._stream(penalty_generator, seed + 3)
        self.eval_generator = self._stream(None, seed + 4)
        self.noise_generator = self._stream(None, seed + 5)
        self.completed_steps = 0

    @property
    def ema_D(self):
        """The trainer-owned EMA critic (K3P anchor), or None for other arms."""
        return self.critic.ema

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

    @staticmethod
    def _noisy(fn, sigma, stream):
        """``x -> fn(x + sigma * eps)`` with fresh eps per call; identity when sigma == 0."""
        if sigma == 0:
            return fn
        return lambda x: fn(x + sigma * torch.randn(x.shape, generator=stream, device=x.device, dtype=x.dtype))

    @staticmethod
    def _generate(model, latent, sigma, stream):
        """``model(latent) + sigma * eps``; no draw when sigma == 0."""
        y = model(latent)
        if sigma == 0:
            return y
        return y + sigma * torch.randn(y.shape, generator=stream, device=y.device, dtype=y.dtype)

    def step(self, real, *, generator_real=None, collect_stats=False):
        """Perform one D update and one G/prior update; return detached losses.

        ``step`` in the result is the completed update count. Only ``rp`` and
        ``ra`` invoke a generator-real callback. ``collect_stats`` additionally
        returns the gradient penalty's synchronized diagnostic dictionary
        (including K3P's blend weight ``s``).
        """
        recipe = self.recipe
        if self.completed_steps >= recipe.total_steps:
            raise RuntimeError("recipe training budget exhausted")
        real = self._batch(real, "real")
        if recipe.gan_mode in ("rp", "ra") and generator_real is not None and not callable(generator_real):
            generator_real = self._batch(generator_real, "generator_real")
            if generator_real.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if recipe.gan_mode == "rp" and len(generator_real) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
        network, prior_scale = learning_rate_scales(self.completed_steps, recipe)
        for optimizer, rates, roles in zip((self.opt_g, self.opt_d), self.initial_lrs, self.roles):
            for group, rate, role in zip(optimizer.param_groups, rates, roles):
                group["lr"] = rate * (prior_scale if role == "prior" else network)
        sigma_in = input_noise_std(recipe, self.completed_steps)
        sigma_out = output_noise_std(recipe, self.completed_steps)
        noise = self.noise_generator
        critic = self._noisy(self.D, sigma_in, noise)
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, _ = self.prior.sample(len(real), generator=self.latent_generator)
            fake = self._generate(self.G, latent, sigma_out, noise)
        loss_d = self.loss.d_loss(critic(real), critic(fake))
        ema = self.critic.ema_critic()
        penalty, penalty_stats = self.critic.penalty(
            critic, real, fake, self.completed_steps + 1, generator=self.penalty_generator,
            collect_stats=collect_stats, ema_critic=None if ema is None else self._noisy(ema, sigma_in, noise))
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.critic.step()

        self.D.eval()
        self.G.train()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, indices = self.prior.sample(len(real), generator=self.latent_generator)
            fake_logits = critic(self._generate(self.G, latent, sigma_out, noise))
            real_logits = None
            if recipe.gan_mode in ("rp", "ra"):
                real_g = generator_real() if callable(generator_real) else generator_real
                real_g = real if real_g is None else self._batch(real_g, "generator_real")
                if real_g.shape[1:] != real.shape[1:]:
                    raise ValueError("generator_real must match the real sample shape")
                if recipe.gan_mode == "rp" and len(real_g) != len(real):
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
            if self.latent_damping is None:
                self.opt_g.step()
            else:
                with self.latent_damping.around(self.opt_g):
                    self.opt_g.step()
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
        with torch.no_grad():
            for target, source in ((self.ema_G, self.G), (self.ema_prior, self.prior)):
                for averaged, current in zip(target.parameters(), source.parameters()):
                    averaged.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
                for averaged, current in zip(target.buffers(), source.buffers()):
                    averaged.copy_(current)
        self.completed_steps += 1
        result = {key: value.detach() for key, value in
                  dict(loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
                       prior_regularization=prior_reg, penalty=penalty).items()}
        result["step"] = self.completed_steps
        if collect_stats:
            result["penalty_stats"] = penalty_stats
        return result

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
                return self._generate(model, latent, output_noise_std(self.recipe, self.completed_steps), stream)
        finally:
            for module, flag in modes:
                module.training = flag

    _STREAMS = ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")

    def _k3p_state(self):
        latent = None
        if self.latent_damping is not None:
            latent = {"state": self.latent_damping.state_dict(), "history": self.latent_history}
        return {"critic": self.critic.state_dict(), "latent": latent}

    def state_dict(self):
        """Return an independent checkpoint; save the caller's data cursor too."""
        names = ("G", "D", "prior", "ema_G", "ema_prior")
        return deepcopy({
            "schema": 2, "recipe": self.recipe.to_dict(),
            "optimizer_options": self.optimizer_options, "penalty_options": self.penalty_options,
            "device": str(self.device), "dtype": str(self.dtype),
            "models": {name: getattr(self, name).state_dict() for name in names},
            "requires_grad": {name: {key: p.requires_grad for key, p in getattr(self, name).named_parameters()}
                              for name in names},
            "k3p": self._k3p_state(),
            "optimizers": [self.opt_g.state_dict(), self.opt_d.state_dict()],
            "initial_lrs": self.initial_lrs, "completed_steps": self.completed_steps,
            "streams": {name: getattr(self, name).get_state() for name in self._STREAMS},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        })

    def load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state.

        Recreate the same parameter freezing before loading. Validation of both
        optimizers, K3P state and all RNG states precedes any mutation of the
        live trainer. Schema-1 (GAN v3) checkpoints are rejected.
        """
        if isinstance(state, dict) and state.get("schema") == 1:
            raise ValueError("schema-1 GANTrainer checkpoints use the GAN v3 formulation and cannot "
                             "resume under K3P; retrain, or pin the old release to continue them")
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys() or state.get("schema") != 2:
            raise ValueError("invalid GANTrainer checkpoint schema")
        for key in ("recipe", "optimizer_options", "penalty_options", "device", "dtype", "requires_grad"):
            if state[key] != expected[key]:
                raise ValueError(f"checkpoint {key} does not match trainer")
        steps = state["completed_steps"]
        if type(steps) is not int or not 0 <= steps <= self.recipe.total_steps:
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
        k3p = state["k3p"]
        try:
            if not isinstance(k3p, dict) or set(k3p) != {"critic", "latent"}:
                raise ValueError("k3p schema")
            probe = copy(self.critic)
            probe.regularizer, probe.guard = copy(self.critic.regularizer), copy(self.critic.guard)
            probe.ema = None if self.critic.ema is None else deepcopy(self.critic.ema)
            probe.load_state_dict(deepcopy(k3p["critic"]))
            if (k3p["latent"] is None) != (self.latent_damping is None):
                raise ValueError("latent damping presence")
            if self.latent_damping is not None:
                history = k3p["latent"]["history"]
                if (not isinstance(history, torch.Tensor) or history.shape != self.latent_history.shape
                        or history.dtype != self.latent_history.dtype):
                    raise ValueError("latent history")
                copy(self.latent_damping).load_state_dict(dict(k3p["latent"]["state"]))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("invalid checkpoint K3P state") from error
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
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
            optimizer.load_state_dict(deepcopy(values))
        self.critic.load_state_dict(deepcopy(k3p["critic"]))
        if self.latent_damping is not None:
            self.latent_damping.load_state_dict(dict(k3p["latent"]["state"]))
            with torch.no_grad():
                self.latent_history.copy_(k3p["latent"]["history"])
        self.initial_lrs, self.completed_steps = deepcopy(rates), steps
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
