"""A small, checkpointable K3P training loop for unconditional particle GANs."""
from copy import deepcopy
import math

import torch
from torch import nn

from .particle_prior import MoGParticlePrior, ParticlePrior
from .recipes import Recipe, learning_rate_scales


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


class GANTrainer:
    """Own the shared update mechanics; callers supply networks and real batches.

    Supports scalar, unconditional GAN recipes with a particle or MoG prior. Fresh real
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
    critic ``ema_D`` (a frozen deep copy). Noise comes from a trainer stream.
    Networks train from the weights they arrive with; initialize them first
    (e.g. ``particlegan.init.deterministic_orthogonal_``). ``sample`` omits
    training output noise unless explicitly requested.
    """

    def __init__(self, recipe, generator, discriminator, *, prior=None, seed=0,
                 latent_generator=None, penalty_generator=None,
                 noise_generator=None, input_noise_generator=None,
                 prior_noise_generator=None, eval_generator=None, model_generator=None,
                 require_latent_damping=None, max_steps=None,
                 optimizer_options=None, penalty_options=None):
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
        if (recipe.model != "gan" or recipe.conditioning != "scalar"
                or recipe.encoder_mode != "none"):
            raise ValueError("GANTrainer supports unconditional scalar GANs with particle priors and no encoder")
        self.recipe, self.G, self.D = recipe, generator, discriminator
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        # The execution budget is independent of every recipe schedule horizon.
        # Continuing a run never stretches or restarts its original schedule.
        self.max_steps = recipe.total_steps if max_steps is None else max_steps
        parameters = list(generator.parameters())
        if not parameters or not any(p.requires_grad for p in parameters):
            raise ValueError("generator must have trainable parameters")
        self.device, self.dtype = parameters[0].device, parameters[0].dtype
        self.prior = (recipe.make_prior().to(device=self.device, dtype=self.dtype)
                      if prior is None else prior)
        expected_prior = MoGParticlePrior if recipe.prior_kind == "mog" else ParticlePrior
        if type(self.prior) is not expected_prior:
            raise ValueError("prior must match the recipe's ParticlePrior or MoGParticlePrior kind")
        if type(self.prior) is MoGParticlePrior and self.prior.standardize != recipe.standardize:
            raise ValueError("prior standardize must match the recipe")
        if prior_noise_generator is not None and type(self.prior) is not MoGParticlePrior:
            raise ValueError("prior_noise_generator requires a MoG prior")
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
        if require_latent_damping is None:
            require_latent_damping = self.prior.z.requires_grad and recipe.latent_damping_max_rate > 0
        if type(require_latent_damping) is not bool:
            raise TypeError("require_latent_damping must be a boolean or None")
        # The recipe picks the regularization formulation (currently K3P); its
        # step-time work runs inside these optimizers' step().
        self.opt_g, self.opt_d = recipe.make_optimizers(
            self.G, self.D, self.prior, ema_critic=deepcopy(self.D),
            require_latent_damping=require_latent_damping, **self.optimizer_options)
        self.prior_mechanisms = self.opt_g.prior_mechanisms
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
        self.eval_generator = self._stream(eval_generator, seed + 4)
        self.noise_generator = self._stream(noise_generator, seed + 5)
        # Keep the old atom default streams/checkpoint keys byte-compatible.
        # Extra streams exist only for new optional bindings and MoG priors.
        self._STREAMS = type(self)._STREAMS
        if input_noise_generator is not None:
            self.input_noise_generator = self._stream(input_noise_generator, seed + 7)
            self._STREAMS += ("input_noise_generator",)
        else:
            self.input_noise_generator = self.noise_generator
        if type(self.prior) is MoGParticlePrior:
            self.prior_noise_generator = self._stream(prior_noise_generator, seed + 6)
            self._STREAMS += ("prior_noise_generator",)
        else:
            self.prior_noise_generator = None
        self.model_generator = None
        if model_generator is not None:
            self.model_generator = self._stream(model_generator, seed + 8)
            self._STREAMS += ("model_generator",)
        training_streams = [getattr(self, name) for name in self._STREAMS if name != "eval_generator"]
        if self.eval_generator in training_streams:
            raise ValueError("evaluation stream must be separate from training")
        # Legacy hosts intentionally share direct-draw training streams. Keep
        # their draw order; Forge enforces distinct named bindings separately.
        # The model stream is copied through global RNG in step(), so sharing
        # it with a directly consumed stream would overwrite that stream's draws.
        # A direct global-generator binding would also be rewound by fork_rng.
        global_stream = (torch.default_generator if self.device.type == "cpu"
                         else torch.cuda.default_generators[self.device.index])
        if self.model_generator is not None and (self.model_generator is global_stream or any(
                getattr(self, name) in (self.model_generator, global_stream)
                for name in self._STREAMS if name not in ("eval_generator", "model_generator"))):
            raise ValueError("model stream must be separate from other training streams and global draws")
        self._noisy_D = InputNoise(self.D, 0.0, self.input_noise_generator)
        self.penalty = recipe.make_critic_penalty(self.opt_d, **self.penalty_options)
        self.completed_steps = 0

    @property
    def latent_damping(self):
        return self.opt_g.latent_damping

    @property
    def latent_history(self):
        return self.opt_g.latent_history

    @property
    def ema_D(self):
        """The trainer-owned EMA critic (K3P anchor)."""
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

    def _sample_training_prior(self, n):
        if type(self.prior) is MoGParticlePrior:
            return self.prior.sample(n, generator=self.latent_generator,
                                     noise_generator=self.prior_noise_generator)
        return self.prior.sample(n, generator=self.latent_generator)

    @staticmethod
    def _generate(model, latent, sigma, stream):
        """``model(latent) + sigma * eps``; no draw when sigma == 0."""
        y = model(latent)
        if sigma == 0:
            return y
        return y + sigma * torch.randn(y.shape, generator=stream, device=y.device, dtype=y.dtype)

    def step(self, real, *, generator_real=None, collect_stats=False):
        """Run one public update, optionally isolating model-internal randomness.

        ``model_generator`` controls stochastic layers such as dropout without
        consuming caller-global RNG. It is checkpointed with the other streams.
        """
        if self.model_generator is None:
            return self._step(real, generator_real=generator_real, collect_stats=collect_stats)
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.model_generator.get_state(), self.device)
            else:
                torch.set_rng_state(self.model_generator.get_state())
            try:
                return self._step(real, generator_real=generator_real, collect_stats=collect_stats)
            finally:
                state = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else torch.get_rng_state()
                self.model_generator.set_state(state)

    def _step(self, real, *, generator_real=None, collect_stats=False):
        """Perform one D update and one G/prior update; return detached losses.

        ``step`` in the result is the completed update count. The generator
        loss pairs fakes with ``generator_real`` (a tensor or a callable;
        default: ``real``). ``collect_stats`` additionally
        returns the gradient penalty's synchronized diagnostic dictionary
        (including K3P's blend weight ``s``).
        """
        recipe = self.recipe
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("recipe training budget exhausted")
        real = self._batch(real, "real")
        if generator_real is not None and not callable(generator_real):
            generator_real = self._batch(generator_real, "generator_real")
            if generator_real.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if len(generator_real) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
        network, prior_scale = learning_rate_scales(self.completed_steps, recipe)
        for optimizer, rates, roles in zip((self.opt_g, self.opt_d), self.initial_lrs, self.roles):
            for group, rate, role in zip(optimizer.param_groups, rates, roles):
                group["lr"] = rate * (prior_scale if role == "prior" else network)
        sigma_in = input_noise_std(recipe, self.completed_steps)
        sigma_out = output_noise_std(recipe, self.completed_steps)
        noise = self.noise_generator
        critic = self._noisy_D
        critic.std = sigma_in
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, _ = self._sample_training_prior(len(real))
            fake = self._generate(self.G, latent, sigma_out, noise)
        loss_d = self.loss.d_loss(critic(real), critic(fake))
        self.penalty.collect_stats = collect_stats
        penalty = self.penalty(critic, real, fake)
        penalty_stats = self.penalty.last_stats
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()

        self.D.eval()
        self.G.train()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, indices = self._sample_training_prior(len(real))
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
            # MoG caches its zero-noise fast path. Buffer copies alone must not
            # leave EMA sampling on the previous side of sigma=0.
            if type(self.ema_prior) is MoGParticlePrior:
                self.ema_prior.set_sigma(self.prior.sigma)
        self.completed_steps += 1
        result = {key: value.detach() for key, value in
                  dict(loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
                       prior_regularization=prior_reg, penalty=penalty).items()}
        result["step"] = self.completed_steps
        if collect_stats:
            result["penalty_stats"] = penalty_stats
        return result

    @torch.no_grad()
    def sample(self, n, *, ema=False, generator=None, output_noise=False,
               fixed_first_n=False, offset=0):
        """Draw live or EMA samples without changing modes or training RNGs.

        Samples omit training output noise by default. ``output_noise=True``
        adds its current value using the sampling stream. ``fixed_first_n``
        enumerates component indices from ``offset`` within the table bounds. Learned MoG kernel noise remains part of the prior sampling law.
        Either output-noise choice leaves training trajectories unchanged.
        """
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        if type(output_noise) is not bool:
            raise ValueError("output_noise must be a boolean")
        if type(fixed_first_n) is not bool or type(offset) is not int or offset < 0:
            raise ValueError("fixed_first_n must be boolean and offset a nonnegative integer")
        stream = self.eval_generator if generator is None else self._stream(generator, 0)
        if stream in tuple(getattr(self, name) for name in self._STREAMS if name != "eval_generator"):
            raise ValueError("sampling requires a stream separate from training")
        model, prior = (self.ema_G, self.ema_prior) if ema else (self.G, self.prior)
        modes = [(module, module.training) for root in (model, prior) for module in root.modules()]
        devices = [self.device.index] if self.device.type == "cuda" else []
        try:
            model.eval()
            prior.eval()
            with torch.random.fork_rng(devices=devices):
                latent, _ = prior.sample(n, generator=stream, fixed_first_n=fixed_first_n, offset=offset)
                sigma = output_noise_std(self.recipe, self.completed_steps) if output_noise else 0.0
                return self._generate(model, latent, sigma, stream)
        finally:
            for module, flag in modes:
                module.training = flag

    _STREAMS = ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")

    def extend_execution(self, max_steps):
        """Extend a restored run's execution allowance, preserving its LR/noise horizon.

        Model, optimizer, prior, EMA and RNG state stay untouched. This is an
        explicit continuation operation, not a change to ``recipe.total_steps``.
        """
        if type(max_steps) is not int or max_steps <= self.max_steps:
            raise ValueError("continuation max_steps must exceed the existing execution allowance")
        self.max_steps = max_steps

    def state_dict(self):
        """Return an independent checkpoint; save the caller's data cursor too."""
        names = ("G", "D", "prior", "ema_G", "ema_prior")
        state = {
            "schema": 3, "recipe": self.recipe.to_dict(),
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
        }
        if self.max_steps != self.recipe.total_steps:
            state["max_steps"] = self.max_steps
        return deepcopy(state)

    def load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state.

        Recreate the same parameter freezing and stream bindings before loading.
        Legacy checkpoints do not encode training-stream alias topology.
        Validation of both optimizers (which carry the K3P state) and all RNG states precedes any
        mutation of the live trainer. Schema-2 checkpoints (separate K3P state)
        are upgraded; schema-1 checkpoints (an older formulation) are rejected.
        """
        if isinstance(state, dict) and state.get("schema") == 1:
            raise ValueError("schema-1 GANTrainer checkpoints come from an older formulation and cannot "
                             "resume under K3P; retrain, or pin the old release to continue them")
        if isinstance(state, dict) and state.get("schema") == 2:
            state = _upgrade_schema_2(state)
        expected = self.state_dict()
        # Recipe subclasses can still serialize the historical formulation
        # fields. Preserve an exact current recipe before trying the public
        # recipe's migration, which drops those formerly fixed fields.
        if (isinstance(state, dict) and isinstance(state.get("recipe"), dict)
                and state["recipe"] != expected["recipe"]):
            state = {**state, "recipe": _upgrade_recipe_fields(state["recipe"])}
        if not isinstance(state, dict) or state.keys() != expected.keys() or state.get("schema") != 3:
            raise ValueError("invalid GANTrainer checkpoint schema")
        saved_recipe = state["recipe"]
        if not isinstance(saved_recipe, dict) or saved_recipe != expected["recipe"]:
            raise ValueError("checkpoint recipe does not match trainer")
        for key in ("optimizer_options", "penalty_options", "device", "dtype", "requires_grad"):
            if state[key] != expected[key]:
                raise ValueError(f"checkpoint {key} does not match trainer")
        if state.get("max_steps", self.recipe.total_steps) != self.max_steps:
            raise ValueError("checkpoint execution budget does not match trainer")
        steps = state["completed_steps"]
        if type(steps) is not int or not 0 <= steps <= self.max_steps:
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
                    or tensors[k].dtype != v.dtype for k, v in current.items()
                    if isinstance(v, torch.Tensor)):
                raise ValueError(f"checkpoint model {name} has incompatible tensors")
            # MoG includes validated non-tensor _extra_state. Validate loading on
            # a disposable copy before touching any live parameters or streams.
            try:
                model = deepcopy(getattr(self, name))
                model.load_state_dict(deepcopy(tensors))
                if type(model) is MoGParticlePrior and model.standardize != self.recipe.standardize:
                    raise ValueError("checkpoint prior standardization does not match recipe")
                if type(model) is MoGParticlePrior:
                    model.set_sigma(model.sigma)
            except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
                raise ValueError(f"checkpoint model {name} has incompatible state") from error
        if not isinstance(state["optimizers"], list) or len(state["optimizers"]) != 2:
            raise ValueError("invalid checkpoint optimizer schema")
        try:
            for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
                deepcopy(optimizer).load_state_dict(deepcopy(values))
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid checkpoint optimizer state") from error
        try:
            global_stream = (torch.default_generator if self.device.type == "cpu"
                             else torch.cuda.default_generators[self.device.index])
            global_state = state["cpu_rng"] if self.device.type == "cpu" else state["cuda_rng"]
            shared_states = {id(global_stream): global_state.cpu()}
            for name, value in state["streams"].items():
                value = value.cpu()
                torch.Generator(device=self.device).set_state(value)
                stream_id = id(getattr(self, name))
                if stream_id in shared_states and not torch.equal(shared_states[stream_id], value):
                    raise ValueError("conflicting checkpoint states for a shared training stream")
                shared_states[stream_id] = value
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
        self.initial_lrs, self.completed_steps = deepcopy(rates), steps
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)


# Recipe fields that once named a fixed choice (with the value that choice
# had), and fields added since (with their defaults).
_REMOVED_RECIPE_FIELDS = {"loss_type": "logistic", "gan_mode": "rp",
                          "reg_method": "autograd"}
_ADDED_RECIPE_FIELDS = {"reg_anchor_weight": 1.0, "direct_particle_gain": True, "reg_arm": "k3p"}
# Construction-time init once lived on the recipe; saved weights supersede it.
_INIT_FIELD_VALUES = (None, "batch_feature_zero")


def _upgrade_recipe_fields(recipe):
    """Drop removed recipe fields that held the only supported value; add new defaults."""
    if any(key in recipe and recipe[key] != value for key, value in _REMOVED_RECIPE_FIELDS.items()):
        return recipe  # another formulation: left as is, so the recipe check rejects it
    if recipe.get("initialization") in _INIT_FIELD_VALUES:
        recipe = {key: value for key, value in recipe.items() if key != "initialization"}
    recipe = {key: value for key, value in recipe.items() if key not in _REMOVED_RECIPE_FIELDS}
    return {**_ADDED_RECIPE_FIELDS, **recipe}


def _upgrade_schema_2(state):
    """Move a schema-2 checkpoint's separate K3P state into its optimizer states."""
    try:
        state = dict(state)
        k3p = state.pop("k3p")
        opt_g, opt_d = state["optimizers"]
        critic = k3p["critic"]
        opt_g = {**opt_g, "regularizer": {"latent": k3p["latent"], "direct": None}}
        opt_d = {**opt_d, "regularizer": {"record": critic["penalty"], "ema": critic["ema"],
                                          "guard": critic["guard"]}}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid checkpoint K3P state") from error
    state["optimizers"], state["schema"] = [opt_g, opt_d], 3
    return state
