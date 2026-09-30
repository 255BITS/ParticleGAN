"""A checkpointable training loop using the current recipe formulation."""
from copy import deepcopy
import math

import torch
from torch import nn

from .particle_prior import ParticlePrior
from .recipes import Recipe, learning_rate_scales
from .policy import UpdatePolicy, _state_to_device, _validate_optimizer_state, input_noise_std, output_noise_std


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
def _normalized_recipe(recipe):
    """A recipe dict in the form checkpoints are compared in: added fields filled with the values that
    reproduce older behaviour, removed fixed choices dropped when they hold their only value, and the
    construction-time ``initialization`` (superseded by saved weights) dropped."""
    recipe = {**_ADDED_RECIPE_FIELDS, **recipe}
    if all(recipe.get(key, value) == value for key, value in _REMOVED_RECIPE_FIELDS.items()):
        recipe = {key: value for key, value in recipe.items() if key not in _REMOVED_RECIPE_FIELDS}
    if recipe.get("initialization", None) in (None, "batch_feature_zero"):
        recipe.pop("initialization", None)
    return recipe


# Recipe fields that once named a fixed choice, with the only value they could hold; a saved
# recipe that records them with that value loads, any other value is rejected.
_REMOVED_RECIPE_FIELDS = {"loss_type": "logistic", "gan_mode": "rp", "reg_arm": "k3p",
                          "reg_method": "autograd"}
_ADDED_RECIPE_FIELDS = {"reg_anchor_weight": 1.0, "direct_particle_gain": True, "continuous_policy": None, "amsgrad": False, "critic_r1_real": True,
                        "critic_payoff_damping": True, "output_noise_mode": "fixed",
                        "lr_control": "mobility", "particle_birth_death": False,
                        "row_evidence_gate": False, "table_release_rule": "any",
                        "row_evidence_hot": True, "row_evidence_exclude": True, "row_evidence_hold": True,
                        "birth_death_space": "data", "serve_average": 0.0, "reopen_signal": "data",
                        "row_evidence_null": "theory", "birth_death_isolation": False,
                        "birth_death_feature_scale": "none"}


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
    critic ``ema_D`` (a frozen deep copy). Noise comes from a trainer stream.
    Networks train from the weights they arrive with; initialize them first
    (e.g. ``particlegan.init.deterministic_orthogonal_``). ``sample`` is clean
    by default; ``sample(..., output_noise=True)`` adds the current output
    noise (``output_sigma()``; see ``recipe.output_noise_mode``).

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
        self.loss = recipe.make_loss()
        self.prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
        self.penalty = recipe.make_critic_penalty(self.opt_d, **self.penalty_options)
        self.policy = UpdatePolicy(
            recipe, self.G, self.D, prior=self.prior,
            generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
            seed=seed, streams={"latent_generator": latent_generator,
                                "penalty_generator": penalty_generator},
            penalty=self.penalty,
            schedule=lambda step, config: learning_rate_scales(step, config))
        self._noisy_D = InputNoise(self.D, 0.0, self.noise_generator)

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
        return self.policy._output_sigma(base, detach=detach)

    def output_sigma(self):
        """Current output-noise standard deviation as a float."""
        return self.policy.output_sigma()

    def _generate(self, model, latent, sigma, stream):
        return self.policy.generate(latent, sigma=sigma, stream=stream, model=model)

    def served_snapshot(self):
        """Independent currently served generator/table state, also available in caller-owned loops."""
        return self.policy.served_snapshot()

    def served_model(self, *, generation_factory=None):
        """Independent frozen modules that reproduce currently served samples."""
        return self.policy.served_model(generation_factory=generation_factory)

    def step(self, real, *, generator_real=None, collect_stats=False):
        """Perform one D update and one G/prior update; return detached losses.

        ``step`` in the result is the completed update count. The generator
        loss pairs fakes with ``generator_real`` (a tensor or a callable;
        default: ``real``). ``collect_stats`` additionally
        returns the gradient penalty's synchronized diagnostic dictionary
        (including the penalty's blend weight ``s``).
        """
        try:
            if self.serial_backward:
                # Serialized backward preserves exact CUDA continuation while
                # leaving the caller's autograd execution mode unchanged.
                with torch.autograd.set_multithreading_enabled(False):
                    return self._step(real, generator_real=generator_real, collect_stats=collect_stats)
            return self._step(real, generator_real=generator_real, collect_stats=collect_stats)
        except Exception:
            self.policy.abort_step()
            raise

    def _serve_release(self):
        return self.policy._serve_release()

    def _serve_settled(self):
        return self.policy._serve_settled()

    def _serve_apply(self):
        return self.policy._serve_apply()

    def _served_parameters(self):
        return self.policy._served_parameters()

    def _served_averages(self):
        return self.policy._served_averages()

    def _average_rate(self):
        return self.policy._average_rate()

    def _step(self, real, *, generator_real=None, collect_stats=False):
        self._serve_release()
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
        step_noise = self.policy.begin_step(real, game_record=self.penalty.regularizer.record)
        sigma_in, sigma_out = step_noise.input_sigma, step_noise.output_sigma
        noise = self.noise_generator
        critic = self._noisy_D
        critic.std = sigma_in
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, _ = self.prior.sample(len(real), generator=self.latent_generator)
            self.policy.observe_support(latent)
            fake = self._generate(self.G, latent, sigma_out, noise)
        self.policy.observe_critic_pair(real, fake)
        loss_d = self.loss.d_loss(critic(real), critic(fake))
        self.penalty.collect_stats = collect_stats
        penalty = self.penalty(critic, real, fake)
        penalty_stats = self.penalty.last_stats
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        self.policy.after_critic_step()

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
            self.policy.after_generator_backward(
                loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
            self.opt_g.step()
            self.policy.after_generator_step()
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
        self.policy.finish_step()
        self._serve_apply()
        result = {key: value.detach() for key, value in
                  dict(loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
                       prior_regularization=prior_reg, penalty=penalty).items()}
        result["step"] = self.completed_steps
        if collect_stats:
            result["penalty_stats"] = penalty_stats
        return result

    def _table_tester(self):
        return self.policy._table_tester()

    def _stray_gate(self):
        return self.policy._stray_gate()

    def _settle_observe(self, index):
        return self.policy._settle_observe(index)

    @torch.no_grad()
    def sample(self, n, *, ema=False, generator=None, output_noise=False):
        """Draw live or EMA samples without changing modes or training RNGs.

        Samples are clean by default: output noise is a training regularizer.
        ``output_noise=True`` adds the current training output noise (drawn
        from the sampling stream, as before). Only the sampling stream
        (``generator`` or the trainer's evaluation stream) is consumed, so
        either choice leaves training trajectories unchanged.
        """
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        if type(output_noise) is not bool:
            raise ValueError("output_noise must be a boolean")
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
                sigma = (self._output_sigma(output_noise_std(self.recipe, self.completed_steps))
                         if output_noise else 0.0)
                return self._generate(model, latent, sigma, stream)
        finally:
            for module, flag in modes:
                module.training = flag

    _STREAMS = ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")

    def state_dict(self):
        """Return an independent checkpoint (the training iterate; the served average is derived); save the caller's data cursor too."""
        swapped = self._fast is not None
        self._serve_release()
        try:
            return self._state_dict()
        finally:
            if swapped:
                self._serve_apply()

    def _state_dict(self):
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
            # Compact metadata completes the reusable policy state without
            # duplicating image-sized models or optimizer tensors. Older flat
            # schema-4 checkpoints remain accepted below.
            "policy": {"last_output_sigma": self.last_output_sigma,
                       "roles": self.policy.roles, "row_semantics": self.policy.row_semantics,
                       "served_source": ("averaged" if self.recipe.serve_average > 0
                                         and self._serve_settled() else "fast")},
            "streams": {name: getattr(self, name).get_state() for name in self._STREAMS},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            **({"output_noise": {"log_sigma": self.log_output_sigma.detach()}}
               if self.log_output_sigma is not None else {}),
            **({"lr_settle": self.lr_settle.state_dict()} if self.lr_settle is not None else {}),
            **({"birth_death": self.birth_death.state_dict()} if self.birth_death is not None else {}),
            **({"row_evidence": self.row_evidence.state_dict()} if self.row_evidence is not None else {}),
        })

    def load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state (a rejected checkpoint leaves the served model served)."""
        served = self._fast is not None
        try:
            return self._load_state_dict(state)
        except Exception:
            if served and self._fast is None:
                self._serve_apply()
            raise

    def _load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state.

        Recreate the same parameter freezing before loading. Validation of both
        optimizers (which carry the KA2 state) and all RNG states precedes any
        mutation of the live trainer. Earlier formulations cannot be resumed
        under KA2; use the release that wrote those checkpoints.
        """
        self._serve_release()
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
        if (not isinstance(state, dict) or state.get("schema") != 4
                or set(state) not in (set(expected), set(expected) - {"policy"})):
            raise ValueError("invalid GANTrainer checkpoint schema")
        saved_recipe = state["recipe"]
        if not isinstance(saved_recipe, dict) or _normalized_recipe(saved_recipe) != _normalized_recipe(expected["recipe"]):
            raise ValueError("checkpoint recipe does not match trainer")
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
                _validate_optimizer_state(optimizer, values)
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
            deepcopy(self.controller).load_state_dict(_state_to_device(state["controller"], self.device))
        if self.lr_settle is not None:
            deepcopy(self.lr_settle).load_state_dict(_state_to_device(state["lr_settle"], self.device),
                                                    (self.opt_g, self.opt_d))
        if self.birth_death is not None:
            self.birth_death.check_state(state["birth_death"])
        if self.row_evidence is not None:
            self.row_evidence.check_state(state["row_evidence"])
        metadata = state.get("policy")
        if "policy" in state:
            if (not isinstance(metadata, dict) or metadata.keys() != expected["policy"].keys()
                    or metadata["roles"] != self.policy.roles
                    or metadata["row_semantics"] != self.policy.row_semantics):
                raise ValueError("checkpoint policy topology does not match trainer")
            sigma = metadata["last_output_sigma"]
            if sigma is not None and (type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0):
                raise ValueError("invalid checkpoint policy output sigma")
            settled = any(value is not None and role == "table" and value.get("last_decisive") == -1
                          for row, roles in zip(state.get("lr_settle", []), self.policy.roles)
                          for value, role in zip(row, roles))
            source = "averaged" if self.recipe.serve_average > 0 and settled else "fast"
            if metadata["served_source"] != source:
                raise ValueError("inconsistent checkpoint policy served source")
        for name, values in state["models"].items():
            getattr(self, name).load_state_dict(values)
        if self.log_output_sigma is not None:
            with torch.no_grad():
                self.log_output_sigma.copy_(state["output_noise"]["log_sigma"])
        for optimizer, values in zip((self.opt_g, self.opt_d), state["optimizers"]):
            optimizer.load_state_dict(deepcopy(values))
        if self.controller is not None:
            self.controller.load_state_dict(_state_to_device(state["controller"], self.device))
        if self.lr_settle is not None:
            self.lr_settle.load_state_dict(_state_to_device(state["lr_settle"], self.device),
                                          (self.opt_g, self.opt_d))
        if self.birth_death is not None:
            self.birth_death.load_state_dict(state["birth_death"])
        if self.row_evidence is not None:
            self.row_evidence.load_state_dict(state["row_evidence"])
        self.initial_lrs, self.completed_steps = deepcopy(rates), steps
        self.last_output_sigma = None if metadata is None else metadata["last_output_sigma"]
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        self._serve_apply()


def _policy_property(name):
    def get(trainer):
        return getattr(trainer.policy, name)

    def set_value(trainer, value):
        setattr(trainer.policy, name, value)

    return property(get, set_value)


for _name in ("log_output_sigma", "last_output_sigma", "initial_lrs", "ema_G", "ema_prior",
              "_fast", "completed_steps", "controller", "lr_settle", "birth_death", "row_evidence",
              *UpdatePolicy._STREAMS):
    setattr(GANTrainer, _name, _policy_property(_name))


def _trainer_roles(trainer):
    # Preserve the historical receipt names; the reusable API uses explicit table/noise roles.
    return [["prior" if role == "table" else "generator" if role == "noise" else role
             for role in row] for row in trainer.policy.roles]


GANTrainer.roles = property(_trainer_roles)
