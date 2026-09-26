"""Simultaneous extragradient with Adam for the critic, generator, and particles.

Gidel et al. 2019, Extra-Adam (A Variational Inequality Perspective on GANs).
The neighbor hop is a rotation: the critic and the particles chase each other
onto a mode the critic scores higher, and the emptied mode stays empty. One
Adam step along the current joint field is the lookahead. Gradients taken at
that point are applied from the original point, which cancels the rotational
part of the alternating step.

Both half-steps use the group learning rates already set for this update
(network 0.00425, prior 0.0085 at the constant-LR screen). No new coefficient.

The extrapolation's Adam moments are not committed. The probe requires one
moment update per outer step, and the committed update is the Adam step of
the gradient at the extrapolated point. D, G, and the particles are
differentiated together at the current point, then together at the
extrapolated point (same minibatch: the RNG is replayed).

Declared before the run, and not used: extrapolation from the past (reuse the
previous gradient as the lookahead). That saves the extra backward, but the
hypothesis is that a fresh lookahead of the current field removes the
rotation. A stale gradient is a different method.
"""
from __future__ import annotations

import atexit
import copy
import json
import sys

import torch

receipt = {"mechanism": "extragradient", "steps": 0}
_FINDER = None
_PLAIN = None
_ORIGINALS = {}
_LOGGED = False
_ATEXIT = False


def install() -> None:
    """Install before the probe captures ``Adam.step``. Unset flag never calls this."""
    global _FINDER, _PLAIN, _ATEXIT
    if _FINDER is not None:
        return
    _PLAIN = _unwrap_adam(torch.optim.Adam.step)
    _FINDER = _Finder()
    sys.meta_path.insert(0, _FINDER)
    for name in _Finder._TARGETS:
        module = sys.modules.get(name)
        if module is not None:
            _Finder._apply(name, module)
    if not _ATEXIT:
        atexit.register(_emit)
        _ATEXIT = True
    print(json.dumps({
        "event": "dynamics",
        "name": "extragradient",
        "setting": "simultaneous Extra-Adam, same group LRs, gradient at the extrapolated point applied from the origin",
    }), flush=True)


def uninstall() -> None:
    """Restore the host step. Tests only."""
    global _FINDER
    import benchmarks.locked_shared.mode_hold as mode_hold
    import particlegan.training as training
    if "train_mode_hold" in _ORIGINALS:
        mode_hold.train_mode_hold = _ORIGINALS["train_mode_hold"]
    if "GANTrainer.step" in _ORIGINALS:
        training.GANTrainer.step = _ORIGINALS["GANTrainer.step"]
    if _FINDER is not None and _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    _FINDER = None


def _emit() -> None:
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def _unwrap_adam(step):
    seen = set()
    while getattr(step, "__wrapped__", None) is not None and id(step) not in seen:
        seen.add(id(step))
        step = step.__wrapped__
    return step


def _plain_adam(optimizer) -> None:
    """Unhooked Adam update. The public ``step`` enters ``no_grad``; this does too."""
    if _PLAIN is None:
        raise RuntimeError("extragradient Adam was not captured; call install()")
    with torch.no_grad():
        _PLAIN(optimizer)


def _clone_params(module):
    return [p.detach().clone() for p in module.parameters()]


def _copy_params(module, saved) -> None:
    with torch.no_grad():
        for parameter, value in zip(module.parameters(), saved):
            parameter.copy_(value)


def _clone_buffers(module):
    return [(name, buf.detach().clone()) for name, buf in module.named_buffers()]


def _copy_buffers(module, saved) -> None:
    buffers = dict(module.named_buffers())
    with torch.no_grad():
        for name, value in saved:
            buffers[name].copy_(value)


def _snap_opt(optimizer):
    return copy.deepcopy(optimizer.state_dict())


def _load_opt(optimizer, state) -> None:
    rates = [group["lr"] for group in optimizer.param_groups]
    betas = [group["betas"] for group in optimizer.param_groups]
    optimizer.load_state_dict(copy.deepcopy(state))
    for group, rate, beta in zip(optimizer.param_groups, rates, betas):
        group["lr"] = rate
        group["betas"] = beta


def _generators(*objects):
    found, seen = [], set()

    def add(value) -> None:
        if isinstance(value, torch.Generator) and id(value) not in seen:
            seen.add(id(value))
            found.append(value)

    for obj in objects:
        if obj is None:
            continue
        add(obj)
        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    add(cell.cell_contents)
                except ValueError:
                    pass
        if isinstance(obj, torch.nn.Module):
            for mod in obj.modules():
                for attr in ("noise_stream", "generator"):
                    add(getattr(mod, attr, None))
        for attr in ("input_stream", "output_stream", "noise_stream",
                     "latent_generator", "penalty_generator", "noise_generator",
                     "eval_generator"):
            add(getattr(obj, attr, None))
    return found


def _snap_rng(*objects):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu, cuda, [(gen, gen.get_state()) for gen in _generators(*objects)]


def _load_rng(snap) -> None:
    cpu, cuda, gens = snap
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    for gen, state in gens:
        gen.set_state(state)


def _note_step(optimizers) -> None:
    global _LOGGED
    receipt["steps"] += 1
    step = receipt["steps"]
    if step == 1 or step % 200 == 0:
        _LOGGED = True
        print(json.dumps({
            "event": "dynamics_step",
            "name": "extragradient",
            "steps": step,
            "group_lrs": [[group["lr"] for group in opt.param_groups] for opt in optimizers],
        }), flush=True)


def apply_joint_extragradient(optimizers, modules, evaluate, rng_objects):
    """One simultaneous Extra-Adam update.

    ``evaluate`` backprops every player at the parameters it finds and does
    not step. It is called twice on the same minibatch. The returned value is
    the second call (the gradient that is applied).
    """
    base_params = [(module, _clone_params(module)) for module in modules]
    base_buffers = [(module, _clone_buffers(module)) for module in modules]
    base_opts = [(opt, _snap_opt(opt)) for opt in optimizers]
    rng = _snap_rng(*rng_objects)

    def restore_point() -> None:
        for module, saved in base_params:
            _copy_params(module, saved)
        for module, saved in base_buffers:
            _copy_buffers(module, saved)
        for opt, state in base_opts:
            _load_opt(opt, state)
        _load_rng(rng)

    evaluate()
    try:
        for opt in optimizers:
            _plain_adam(opt)
        predicted = [(module, _clone_params(module)) for module in modules]
    finally:
        restore_point()
    for module, saved in predicted:
        _copy_params(module, saved)
    try:
        result = evaluate()
    finally:
        # Gradients stay on the parameters. The step below starts at the origin.
        for module, saved in base_params:
            _copy_params(module, saved)
    for opt in optimizers:
        opt.step()
    _note_step(optimizers)
    return result


def _freeze(module):
    flags = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    return flags


def _thaw(module, flags) -> None:
    for parameter, flag in zip(module.parameters(), flags):
        parameter.requires_grad_(flag)


def train_mode_hold(recipe=None, *, seed: int = 0, gan_factory=None, cap_factory=None,
                    diagnostics=False, training_recipe=None, log=None, noise_policy=None):
    """Host ``train_mode_hold`` with one simultaneous Extra-Adam update per step."""
    from contextlib import nullcontext
    from dataclasses import replace

    import benchmarks.locked_shared.mode_hold as mode_hold
    from particlegan import ParticlePrior, ParticleRegularizer, learning_rate_scale
    from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output

    recipe = mode_hold.ModeHoldRecipe() if recipe is None else recipe
    if training_recipe is not None:
        recipe = replace(recipe, n_particles=training_recipe.num_particles,
                         particle_l2=0.0, vicreg_weight=training_recipe.prior_reg,
                         beta1=training_recipe.betas[0], beta2=training_recipe.betas[1],
                         ema=training_recipe.ema_decay, d_lr_mult=training_recipe.d_lr_mult,
                         steps=training_recipe.total_steps)
    torch.manual_seed(seed)
    stream = torch.Generator().manual_seed(seed)
    means = mode_hold.ring_means()
    prior = (ParticlePrior(recipe.n_particles, mode_hold.Z_DIM, init_std=0.5, generator=stream)
             if training_recipe is None else training_recipe.make_prior(generator=stream))
    generator = mode_hold.SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
    critic = mode_hold.SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER)
    if noise_policy is not None:
        generator = wrap_output(generator, noise_policy)
        critic = wrap_input(critic, noise_policy)
    gan = (gan_factory or (training_recipe.make_loss if training_recipe else mode_hold.make_gan_loss))()
    regularizer = (cap_factory or (training_recipe.make_gradient_penalty
                                   if training_recipe else mode_hold.make_b_cap))()
    vicreg = ParticleRegularizer(weight=recipe.vicreg_weight)
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(prior.parameters()),
        lr=mode_hold.LR, betas=(recipe.beta1, recipe.beta2),
    )
    opt_d = torch.optim.Adam(
        critic.parameters(), lr=mode_hold.LR * recipe.d_lr_mult,
        betas=(recipe.beta1, recipe.beta2),
    )
    if training_recipe is not None:
        opt_g, opt_d = training_recipe.make_optimizers(generator, critic, prior)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    batch = mode_hold.BATCH if training_recipe is None else training_recipe.batch_size
    ema_g = [p.detach().clone() for p in generator.parameters()]
    ema_z = prior.z.detach().clone()

    @torch.no_grad()
    def measure(step: int):
        context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
        with context:
            latent, _ = prior.sample(mode_hold.EVAL_N, generator=torch.Generator().manual_seed(seed + 9))
            row = mode_hold.diversity(generator(latent), means, detailed=diagnostics)
            if diagnostics:
                row["support"] = mode_hold.diversity(generator(prior.z), means, detailed=True)
                if noise_policy is not None and noise_policy.output_std > 0:
                    row["support_scope"] = "one noisy draw per particle"
        return row

    def snapshot(step: int) -> dict:
        saved_g = [p.detach().clone() for p in generator.parameters()]
        saved_z = prior.z.detach().clone()
        with torch.no_grad():
            for param, ema in zip(generator.parameters(), ema_g):
                param.copy_(ema)
            prior.z.copy_(ema_z)
            if noise_policy is not None and step == recipe.steps:
                noise_policy.capture_final_ema()
            row = measure(step)
        with torch.no_grad():
            for param, saved in zip(generator.parameters(), saved_g):
                param.copy_(saved)
            prior.z.copy_(saved_z)
        row.update(step=step, seed=seed)
        return row

    def evaluate():
        # Same draws as the host: one real/latent pair for D, then another for G.
        # G is scored at the critic parameters evaluate finds, not after a D step.
        real = mode_hold.sample_ring(means, batch, mode_hold.SIGMA, stream)
        latent, _ = prior.sample(batch, generator=stream)
        context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
        with context:
            fake = generator(latent).detach()
        d_loss = gan.d_loss(critic(real), critic(fake))
        d_loss = d_loss + regularizer(critic, real, fake, step=step + 1)
        opt_d.zero_grad()
        d_loss.backward()
        flags = _freeze(critic)
        try:
            latent, _ = prior.sample(batch, generator=stream)
            fake = generator(latent)
            if gan.mode in ("rp", "ra"):
                real_g = mode_hold.sample_ring(means, batch, mode_hold.SIGMA, stream)
                g_loss = gan.g_loss(critic(fake), critic(real_g))
            else:
                g_loss = gan.g_loss(critic(fake))
            if recipe.fm_weight > 0.0:
                real_mean = mode_hold.sample_ring(means, batch, mode_hold.SIGMA, stream).detach().mean(0)
                g_loss = g_loss + recipe.fm_weight * (fake.mean(0) - real_mean).pow(2).sum()
            g_loss = g_loss + recipe.particle_l2 * prior.z.pow(2).mean()
            g_loss = g_loss + vicreg(prior.z)
            opt_g.zero_grad()
            g_loss.backward()
        finally:
            _thaw(critic, flags)

    curve, live_curve = [], []
    for step in range(recipe.steps):
        if noise_policy is not None:
            noise_policy.set_step(step)
        if training_recipe is not None:
            scale = learning_rate_scale(
                step, recipe.steps, training_recipe.lr_anneal_start, training_recipe.lr_floor)
            for opt, rates in zip((opt_g, opt_d), base_lrs):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * scale
        # Once per player. The controller writes the constant group LRs; both
        # half-steps read those same values.
        mode_hold.schedule_optimizer(opt_d, step)
        mode_hold.schedule_optimizer(opt_g, step)
        apply_joint_extragradient(
            (opt_d, opt_g), (critic, generator, prior), evaluate,
            (stream, noise_policy, generator, critic, prior),
        )
        with torch.no_grad():
            for ema, param in zip(ema_g, generator.parameters()):
                ema.mul_(recipe.ema).add_(param, alpha=1.0 - recipe.ema)
            ema_z.mul_(recipe.ema).add_(prior.z, alpha=1.0 - recipe.ema)
        mode_hold.checkpoint(step + 1, lambda: measure(step + 1))
        if diagnostics and (step + 1) % 200 == 0:
            curve.append(snapshot(step + 1))
        if diagnostics and ((step + 1) % 200 == 0 or
                            (step + 1 >= recipe.steps - 200 and (step + 1) % 50 == 0)):
            point = {"step": step + 1, **measure(step + 1)}
            live_curve.append(point)
            if log is not None:
                log(point)
    if noise_policy is not None:
        noise_policy.capture_final_live()
    final = snapshot(recipe.steps)
    final["verdict"] = mode_hold.verdict(final)
    if diagnostics:
        live = measure(recipe.steps)
        final["live"] = {**live, "verdict": mode_hold.verdict(live)}
        final["curve"] = curve
        final["live_curve"] = live_curve
    return final


def _trainer_step(self, real, *, generator_real=None, collect_stats=False):
    """``GANTrainer.step`` with one simultaneous Extra-Adam update."""
    from particlegan.recipes import learning_rate_scales
    from particlegan.training import input_noise_std, output_noise_std

    recipe = self.recipe
    if self.completed_steps >= recipe.total_steps:
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
    self._noisy_D.std = sigma_in
    held = {}

    def evaluate():
        self.D.train()
        self.G.eval()
        with torch.no_grad():
            latent, _ = self.prior.sample(len(real), generator=self.latent_generator)
            fake = self._generate(self.G, latent, sigma_out, self.noise_generator)
        critic = self._noisy_D
        loss_d = self.loss.d_loss(critic(real), critic(fake))
        self.penalty.collect_stats = collect_stats
        penalty = self.penalty(critic, real, fake)
        loss_d = loss_d + penalty
        self.opt_d.zero_grad()
        loss_d.backward()
        self.D.eval()
        self.G.train()
        flags = _freeze(self.D)
        try:
            latent, indices = self.prior.sample(len(real), generator=self.latent_generator)
            fake_logits = critic(self._generate(self.G, latent, sigma_out, self.noise_generator))
            real_g = generator_real() if callable(generator_real) else generator_real
            real_g = real if real_g is None else self._batch(real_g, "generator_real")
            if real_g.shape[1:] != real.shape[1:]:
                raise ValueError("generator_real must match the real sample shape")
            if len(real_g) != len(real):
                raise ValueError("RpGAN generator_real must match the real batch size")
            loss_gan = self.loss.g_loss(fake_logits, critic(real_g))
            prior_reg = loss_gan.new_zeros(())
            if self.prior.z.requires_grad:
                raw = self.prior.z if recipe.num_particles <= 1024 else self.prior.z[torch.unique(indices)]
                prior_reg = self.prior_regularizer(raw)
            loss_g = loss_gan + recipe.prior_reg * prior_reg
            self.opt_g.zero_grad()
            loss_g.backward()
        finally:
            _thaw(self.D, flags)
        held["loss_d"] = loss_d
        held["loss_g"] = loss_g
        held["loss_gan"] = loss_gan
        held["prior_reg"] = prior_reg
        held["penalty"] = penalty

    apply_joint_extragradient(
        (self.opt_d, self.opt_g), (self.D, self.G, self.prior), evaluate,
        (self, generator_real),
    )
    with torch.no_grad():
        for target, source in ((self.ema_G, self.G), (self.ema_prior, self.prior)):
            for averaged, current in zip(target.parameters(), source.parameters()):
                averaged.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
            for averaged, current in zip(target.buffers(), source.buffers()):
                averaged.copy_(current)
    self.completed_steps += 1
    result = {key: value.detach() for key, value in
              dict(loss_d=held["loss_d"], loss_g=held["loss_g"], loss_gan=held["loss_gan"],
                   prior_regularization=held["prior_reg"], penalty=held["penalty"]).items()}
    result["step"] = self.completed_steps
    if collect_stats:
        result["penalty_stats"] = self.penalty.last_stats
    return result


class _Loader:
    def __init__(self, inner, name):
        self.inner = inner
        self.name = name

    def create_module(self, spec):
        create = getattr(self.inner, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _Finder._apply(self.name, module)


class _Finder:
    _TARGETS = {"benchmarks.locked_shared.mode_hold", "particlegan.training"}

    @staticmethod
    def _apply(name, module) -> None:
        if name == "benchmarks.locked_shared.mode_hold":
            if module.train_mode_hold is train_mode_hold:
                return
            _ORIGINALS.setdefault("train_mode_hold", module.train_mode_hold)
            module.train_mode_hold = train_mode_hold
        elif name == "particlegan.training":
            if module.GANTrainer.step is _trainer_step:
                return
            _ORIGINALS.setdefault("GANTrainer.step", module.GANTrainer.step)
            module.GANTrainer.step = _trainer_step

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._TARGETS or fullname in sys.modules:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, fullname)
        return spec
