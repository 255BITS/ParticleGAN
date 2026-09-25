"""A small, checkpointable training loop for unconditional particle GANs."""
from copy import copy, deepcopy
import math

import torch

from .particle_prior import ParticlePrior
from .recipes import Recipe, learning_rate_scale


class GANTrainer:
    """Own the update mechanics; callers supply networks and real batches.

    Supports scalar, unconditional GAN recipes with a particle prior. Fresh real
    batches for the generator can be passed as ``generator_real`` tensors or
    zero-argument callables. Data-loader position is caller-owned and must be
    saved separately when checkpointing. Checkpoints restore global PyTorch RNG
    state as well as this trainer's sampling streams for exact continuation on
    the same device. Sampling never advances training RNG streams.
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
        self.loss = recipe.make_loss()
        self.penalty = recipe.make_gradient_penalty(**self.penalty_options)
        self.prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
        self.ema_G, self.ema_prior = deepcopy(self.G).eval(), deepcopy(self.prior).eval()
        for module in (self.ema_G, self.ema_prior):
            module.requires_grad_(False)
        # K3P anchors the critic to its parameter EMA; the trainer owns that
        # copy (the base GradRegularizer never allocates networks).
        self.ema_D = None
        if self.penalty.arm == "k3p" and self.penalty.anchor is None:
            from .k3p import CriticAnchor
            self.ema_D = deepcopy(self.D).eval().requires_grad_(False)
            self.penalty.anchor = CriticAnchor(self.D, self.ema_D)
        self.latent_generator = self._stream(latent_generator, seed + 2)
        self.penalty_generator = self._stream(penalty_generator, seed + 3)
        self.eval_generator = self._stream(None, seed + 4)
        self.completed_steps = 0

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

    def step(self, real, *, generator_real=None, collect_stats=False):
        """Perform one D update and one G/prior update; return detached losses.

        ``step`` in the result is the completed update count. Only ``rp`` and
        ``ra`` invoke a generator-real callback. ``collect_stats`` additionally
        returns the gradient penalty's synchronized diagnostic dictionary.
        """
        if self.completed_steps >= self.recipe.total_steps:
            raise RuntimeError("recipe training budget exhausted")
        real = self._batch(real, "real")
        if self.recipe.gan_mode in ("rp", "ra") and generator_real is not None and not callable(generator_real):
            generator_real = self._batch(generator_real, "generator_real")
            if generator_real.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if self.recipe.gan_mode == "rp" and len(generator_real) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
        scale = learning_rate_scale(self.completed_steps, self.recipe.total_steps,
                                    self.recipe.lr_anneal_start, self.recipe.lr_floor)
        for optimizer, rates in zip((self.opt_g, self.opt_d), self.initial_lrs):
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * scale
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, _ = self.prior.sample(len(real), generator=self.latent_generator)
            fake = self.G(latent)
        loss_d = self.loss.d_loss(self.D(real), self.D(fake))
        penalty, penalty_stats = self.penalty.penalty(
            self.D, real, fake, self.completed_steps + 1,
            generator=self.penalty_generator, collect_stats=collect_stats)
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        self.penalty.after_critic_step(self.opt_d)
        if self.ema_D is not None:
            with torch.no_grad():
                for averaged, current in zip(self.ema_D.buffers(), self.D.buffers()):
                    averaged.copy_(current)

        self.D.eval()
        self.G.train()
        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            latent, indices = self.prior.sample(len(real), generator=self.latent_generator)
            fake_logits = self.D(self.G(latent))
            real_logits = None
            if self.recipe.gan_mode in ("rp", "ra"):
                real_g = generator_real() if callable(generator_real) else generator_real
                real_g = real if real_g is None else self._batch(real_g, "generator_real")
                if real_g.shape[1:] != real.shape[1:]:
                    raise ValueError("generator_real must match the real sample shape")
                if self.recipe.gan_mode == "rp" and len(real_g) != len(real):
                    raise ValueError("RpGAN generator_real must match the real batch size")
                real_logits = self.D(real_g)
            loss_gan = self.loss.g_loss(fake_logits, real_logits)
            prior_reg = loss_gan.new_zeros(())
            if self.prior.z.requires_grad:
                raw = self.prior.z if self.recipe.num_particles <= 1024 else self.prior.z[torch.unique(indices)]
                prior_reg = self.prior_regularizer(raw)
            loss_g = loss_gan + self.recipe.prior_reg * prior_reg
            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
        with torch.no_grad():
            for target, source in ((self.ema_G, self.G), (self.ema_prior, self.prior)):
                for averaged, current in zip(target.parameters(), source.parameters()):
                    averaged.mul_(self.recipe.ema_decay).add_(current, alpha=1 - self.recipe.ema_decay)
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
        """Draw live or EMA samples without changing modes or training RNGs."""
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        stream = self.eval_generator if generator is None else self._stream(generator, 0)
        if stream is self.latent_generator or stream is self.penalty_generator:
            raise ValueError("sampling requires a stream separate from training")
        model, prior = (self.ema_G, self.ema_prior) if ema else (self.G, self.prior)
        modes = [(module, module.training) for root in (model, prior) for module in root.modules()]
        devices = [self.device.index] if self.device.type == "cuda" else []
        try:
            model.eval()
            prior.eval()
            with torch.random.fork_rng(devices=devices):
                latent, _ = prior.sample(n, generator=stream)
                return model(latent)
        finally:
            for module, flag in modes:
                module.training = flag

    def _model_names(self):
        names = ("G", "D", "prior", "ema_G", "ema_prior")
        return names + (("ema_D",) if self.ema_D is not None else ())

    def state_dict(self):
        """Return an independent checkpoint; save the caller's data cursor too."""
        return deepcopy({
            "schema": 1, "recipe": self.recipe.to_dict(),
            "optimizer_options": self.optimizer_options, "penalty_options": self.penalty_options,
            "device": str(self.device), "dtype": str(self.dtype),
            "models": {name: getattr(self, name).state_dict()
                       for name in self._model_names()},
            "requires_grad": {name: {key: p.requires_grad for key, p in getattr(self, name).named_parameters()}
                              for name in self._model_names()},
            "penalty_state": self.penalty.state_dict(),
            "optimizers": [self.opt_g.state_dict(), self.opt_d.state_dict()],
            "initial_lrs": self.initial_lrs, "completed_steps": self.completed_steps,
            "streams": {name: getattr(self, name).get_state() for name in
                        ("latent_generator", "penalty_generator", "eval_generator")},
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        })

    def load_state_dict(self, state):
        """Restore a compatible checkpoint, including global PyTorch RNG state.

        Recreate the same parameter freezing before loading. Validation of both
        optimizers and all RNG states precedes any mutation of the live trainer.
        """
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys() or state.get("schema") != 1:
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
        try:
            copy(self.penalty).load_state_dict(deepcopy(state["penalty_state"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint penalty state") from error
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
        self.penalty.load_state_dict(deepcopy(state["penalty_state"]))
        self.initial_lrs, self.completed_steps = deepcopy(rates), steps
        for name, value in state["streams"].items():
            getattr(self, name).set_state(value.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
