"""Fixed, GAN-native dampers for hid_q K3P. Nothing here runs unless ``install`` does.

The ring amplifier is the alternating update in the first few steps: a 1e-7
change becomes O(1) by about step 8, and the one-step map there is a
nonlinear kink (singular value grows as the probe epsilon shrinks) living in
G and in D's response to G.

optimistic
    Daskalakis et al. 2018, Algorithm 1, alpha = 1. After the host Adam step,
    subtract ``lr * (u_t - u_{t-1})`` with ``u`` the bias-corrected Adam
    direction. No new coefficient and no extra forward. Installed on
    ``Adam.step`` before the probe captures it, so legacy and K3P each apply
    it once.

extragradient
    Simultaneous extragradient on the current minibatch. A raw Adam lookahead
    (not the probe's counted step) writes a predicted D and, from the restored
    base, a predicted G. The committed gradients are taken at that predicted
    point and the host's real ``opt.step`` writes them onto the base point.
    One counted update, one noise draw, the existing Adam step as the only
    extrapolation length.

ema_fake
    The critic's fake forward uses the generator EMA and EMA particles already
    kept at ``recipe.ema`` / ``ema_decay`` (0.995). The generator step stays
    on the live weights. No new decay and no extra loss.

k3p_pull
    The probe's penalty patch lands on ``particlegan.grad_regularizers``, but
    the ring calls ``benchmarks.legacy.grad_regularizers``. This points the
    legacy ``a_r1r2`` penalty at ``mechanism.scaled_penalty`` (the intended
    EMA-critic pull, spike guard, and handover). Constants stay the ones in
    mechanism.py. At s == 1 the formula is that file's RMS R1 plus one-sided
    fake cap, which is what the mechanism calls bit-exact, not the host's
    symmetric squared R1/R2.

row_damp
    Same A2 row rule as ``latent.py`` (rho = 0.75 + 0.25 cos, at most half the
    step removed, parent v from raw g, rho = 1 with no history). The sparse
    gate is left as written. On a fully dense table, where that gate cannot
    fire, the same rewrite runs anyway. No new coefficient.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext

NAMES = ("optimistic", "extragradient", "ema_fake", "k3p_pull", "row_damp")
_NAME = None
_ORIG_ADAM = None
_PREV = "k3p_optimistic_prev"


def install(name: str) -> None:
    global _NAME
    if name not in NAMES:
        raise SystemExit(f"unknown relief {name}")
    if _NAME is not None:
        raise SystemExit(f"relief already installed: {_NAME}")
    _NAME = name
    print(json.dumps({"event": "relief", "name": name, "setting": _setting(name)}), flush=True)
    if name == "optimistic":
        _install_optimistic()
    else:
        sys.meta_path.insert(0, _Finder())


def _setting(name: str) -> str:
    if name == "optimistic":
        return "Daskalakis Algorithm 1, alpha=1, on every Adam update"
    if name == "extragradient":
        return "simultaneous extragradient, same minibatch, one host Adam corrector"
    if name == "ema_fake":
        return "critic fake uses the existing generator EMA (decay 0.995)"
    if name == "k3p_pull":
        return "legacy a_r1r2 calls mechanism.scaled_penalty (decay 0.999, floor 0.01, guard 5 after 200)"
    return "A2 rho=0.75+0.25*cos on a dense latent table; sparse gate unchanged"


def _install_optimistic() -> None:
    import torch
    global _ORIG_ADAM
    _ORIG_ADAM = torch.optim.Adam.step

    def step(optimizer, closure=None):
        return _optimistic(optimizer, closure)

    step._k3p_optimistic = True
    torch.optim.Adam.step = step


def _count(state) -> int:
    import torch
    value = state["step"]
    return int(value.item() if torch.is_tensor(value) else value)


def _direction(state, group):
    import torch
    t = _count(state)
    if t <= 0:
        raise RuntimeError("optimistic Adam saw a step count of 0")
    beta1, beta2 = group["betas"]
    # beta1 is 0 on this recipe; 0**t is 0 for t > 0.
    b1 = 0.0 if beta1 == 0.0 else beta1 ** t
    b2 = 0.0 if beta2 == 0.0 else beta2 ** t
    mhat = state["exp_avg"] / (1.0 - b1)
    second = state["max_exp_avg_sq"] if group.get("amsgrad", False) else state["exp_avg_sq"]
    vhat = second / (1.0 - b2)
    return mhat / (vhat.sqrt() + group["eps"])


def _optimistic(optimizer, closure):
    import torch
    for group in optimizer.param_groups:
        if group.get("weight_decay", 0) != 0:
            raise RuntimeError("optimistic relief expects zero weight decay")
    loss = _ORIG_ADAM(optimizer, closure)
    with torch.no_grad():
        for group in optimizer.param_groups:
            rate = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = optimizer.state[parameter]
                current = _direction(state, group)
                previous = state.get(_PREV)
                if previous is None:
                    previous = torch.zeros_like(current)
                parameter.add_(current - previous, alpha=-rate)
                state[_PREV] = current.detach().clone()
    return loss


class _Loader:
    def __init__(self, inner, name):
        self.inner = inner
        self.name = name

    def create_module(self, spec):
        create = getattr(self.inner, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        if self.name == "benchmarks.legacy.grad_regularizers" and _NAME == "k3p_pull":
            _patch_legacy_penalty(module)
            print(json.dumps({"event": "relief_hook", "target": "legacy.GradientPenalty.penalty"}), flush=True)
        elif self.name == "latent" and _NAME == "row_damp":
            _patch_latent(module)
            print(json.dumps({"event": "relief_hook", "target": "latent.begin"}), flush=True)
        elif self.name.endswith("mode_hold") and _NAME == "extragradient":
            module.train_mode_hold = train_mode_hold
            print(json.dumps({"event": "relief_hook", "target": "train_mode_hold"}), flush=True)
        elif self.name.endswith("training") and _NAME in ("ema_fake", "extragradient"):
            _patch_trainer(module)
            print(json.dumps({"event": "relief_hook", "target": "GANTrainer.step"}), flush=True)
        elif self.name.endswith("legacy_noise_adapters") and _NAME == "ema_fake":
            _patch_noise(module)
            print(json.dumps({"event": "relief_hook", "target": "NoisePolicy"}), flush=True)
        elif self.name.endswith("particle_prior") and _NAME == "ema_fake":
            _patch_prior(module)
            print(json.dumps({"event": "relief_hook", "target": "ParticlePrior.sample"}), flush=True)


class _Finder:
    """Patch mode_hold and GANTrainer once each is imported.

    Python 3.12 only consults ``find_spec``; a ``find_module`` hook never runs.
    """

    _TARGETS = {
        "benchmarks.locked_shared.mode_hold",
        "particlegan.training",
        "benchmarks.transfer_suite.legacy_noise_adapters",
        "particlegan.particle_prior",
        "benchmarks.legacy.grad_regularizers",
        "latent",
    }

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._TARGETS or fullname in sys.modules:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, fullname)
        return spec


def _patch_legacy_penalty(module) -> None:
    """Send the legacy a_r1r2 arm through mechanism.scaled_penalty.

    ``__call__`` passes ``ema_critic=``; scaled_penalty does not take it.
    Other arms stay on the host method. Importing mechanism also registers
    its spike guard and the LR record the handover weight reads.
    """
    import atexit
    original = module.GradientPenalty.penalty

    def penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True, *, ema_critic=None):
        if self.arm != "a_r1r2":
            return original(self, D, x_real, x_fake, step, generator, collect_stats, ema_critic=ema_critic)
        import mechanism
        out = mechanism.scaled_penalty(self, D, x_real, x_fake, step=step, generator=generator, collect_stats=collect_stats)
        calls = mechanism.receipt["calls"]
        if calls == 1 or calls % 200 == 0:
            trace = mechanism.receipt["s_trace"]
            print(json.dumps({
                "event": "k3p_pull",
                "calls": calls,
                "pure_a": mechanism.receipt["pure_a_calls"],
                "blend": mechanism.receipt["blend_calls"],
                "pure_b": mechanism.receipt["pure_b_calls"],
                "critic_steps": mechanism.receipt["critic_steps"],
                "s": trace[-1] if trace else None,
            }), flush=True)
        return out

    module.GradientPenalty.penalty = penalty

    def _done():
        import mechanism
        rec = mechanism.receipt
        print(json.dumps({
            "event": "k3p_pull_done",
            "calls": rec["calls"],
            "pure_a": rec["pure_a_calls"],
            "blend": rec["blend_calls"],
            "pure_b": rec["pure_b_calls"],
            "critic_steps": rec["critic_steps"],
            "anchor_started_call": rec["anchor_started_call"],
            "guard_clip_step_count": rec.get("guard_clip_step_count"),
            "final_s": rec.get("final_s"),
        }), flush=True)

    atexit.register(_done)


_DENSE_HIST = {}


def _patch_latent(module) -> None:
    """Apply the existing A2 rewrite on a dense table, where the sparse gate cannot."""
    import atexit
    original = module.begin
    module.receipt["dense_calls"] = 0
    warned = {"beta": False}

    def begin(opt):
        saved = original(opt)
        if saved:
            return saved
        extra = _dense_rewrite(module, opt, warned)
        return extra

    module.begin = begin

    def _done():
        print(json.dumps({"event": "row_damp_done", "dense_calls": module.receipt.get("dense_calls", 0),
                           "scoped_calls": module.receipt.get("scoped_calls", 0),
                           "calls": module.receipt.get("calls", 0)}), flush=True)

    atexit.register(_done)


def _dense_rewrite(module, opt, warned):
    import response
    import torch
    saved = []
    for group in opt.param_groups:
        tables = [p for p in group["params"] if id(p) in response.prior_ids and p.grad is not None]
        if len(tables) != 1 or tables[0].dim() != 2:
            continue
        if group["betas"][0] != 0.0:
            if not warned["beta"]:
                warned["beta"] = True
                print(json.dumps({"event": "row_damp_skip", "reason": "beta1!=0"}), flush=True)
            continue
        p = tables[0]
        state = opt.state.get(p)
        if not state or "exp_avg" not in state:
            continue
        g = p.grad.detach()
        with torch.no_grad():
            norm = g.square().sum(-1).sqrt()
            active = norm > 0
            rows = int(active.numel())
            if rows == 0 or int(active.sum()) != rows:
                continue
            h = _DENSE_HIST.get(id(p))
            if h is None or h.shape != p.shape:
                h = torch.zeros_like(p)
                _DENSE_HIST[id(p)] = h
            hn = h.square().sum(-1).sqrt()
            has = hn > 0
            cos = (g * h).sum(-1) / (norm * hn).clamp_min(1e-30)
            rho = torch.where(has, 0.75 + 0.25 * cos, torch.ones_like(cos))
            bc1 = 1.0 - module.B1 ** (float(state["step"]) + 1.0)
            state["exp_avg"].copy_(g * (2.0 * bc1 * rho - 1.0).unsqueeze(-1))
            saved.append((group, group["betas"], state["exp_avg"], g.clone()))
            group["betas"] = (module.B1, group["betas"][1])
            h[active] = g[active]
            module.receipt["dense_calls"] += 1
            n = module.receipt["dense_calls"]
            if n == 1 or n % 200 == 0:
                print(json.dumps({
                    "event": "row_damp",
                    "dense_calls": n,
                    "rows": rows,
                    "rho_mean": float(rho.mean()),
                    "history_rows": int(has.sum()),
                }), flush=True)
    return saved


def _raw_adam():
    fn = getattr(sys.modules.get("__main__"), "original_step", None)
    if fn is None:
        import torch
        fn = torch.optim.Adam.step
    return fn


def _clone_params(module):
    return [p.detach().clone() for p in module.parameters()]


def _copy_params(module, saved) -> None:
    import torch
    with torch.no_grad():
        for parameter, value in zip(module.parameters(), saved):
            parameter.copy_(value)


def _clone_buffers(module):
    return [(name, buf.detach().clone()) for name, buf in module.named_buffers()]


def _copy_buffers(module, saved) -> None:
    import torch
    buffers = dict(module.named_buffers())
    with torch.no_grad():
        for name, value in saved:
            buffers[name].copy_(value)


def _snap_opt(opt):
    import copy
    return copy.deepcopy(opt.state_dict())


def _load_opt(opt, state) -> None:
    import copy
    rates = [group["lr"] for group in opt.param_groups]
    betas = [group["betas"] for group in opt.param_groups]
    opt.load_state_dict(copy.deepcopy(state))
    for group, rate, beta in zip(opt.param_groups, rates, betas):
        group["lr"] = rate
        group["betas"] = beta


def _generators(*objects):
    import torch
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
                add(getattr(mod, "noise_stream", None))
        for attr in ("input_stream", "output_stream", "noise_stream",
                     "latent_generator", "penalty_generator", "noise_generator",
                     "eval_generator"):
            add(getattr(obj, attr, None))
    return found


def _snap_rng(*objects):
    import torch
    gens = _generators(*objects)
    return torch.get_rng_state(), [(gen, gen.get_state()) for gen in gens]


def _load_rng(snap) -> None:
    import torch
    cpu, gens = snap
    torch.set_rng_state(cpu)
    for gen, state in gens:
        gen.set_state(state)


def _policy_counts(policy):
    if policy is None or not hasattr(policy, "_counts"):
        return None
    return dict(policy._counts)


def _load_counts(policy, counts) -> None:
    if counts is not None:
        policy._counts.update(counts)


def _locals(func_name: str):
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code.co_name == func_name:
            return frame.f_locals
        frame = frame.f_back
    return None


def _patch_prior(module) -> None:
    original = module.ParticlePrior.sample

    def sample(self, *args, **kwargs):
        pending = getattr(sample, "pending", False)
        locs = _locals("train_mode_hold") if pending else None
        ema = None if locs is None else locs.get("ema_z")
        if ema is None:
            return original(self, *args, **kwargs)
        sample.pending = False
        import torch
        with torch.no_grad():
            saved = self.z.detach().clone()
            self.z.copy_(ema)
        try:
            latent, idx = original(self, *args, **kwargs)
            # sample indexes z; clone before the live table is restored.
            latent = latent.detach().clone()
        finally:
            with torch.no_grad():
                self.z.copy_(saved)
        return latent, idx

    sample.pending = False
    module.ParticlePrior.sample = sample
    _patch_prior.sample = sample


def _patch_noise(module) -> None:
    original = module.NoisePolicy.discriminator
    original_set = module.NoisePolicy.set_step

    def set_step(self, completed_steps):
        hook = getattr(_patch_prior, "sample", None)
        if hook is not None:
            hook.pending = True
        return original_set(self, completed_steps)

    @contextmanager
    def discriminator(self):
        import torch
        locs = _locals("train_mode_hold")
        generator = None if locs is None else locs.get("generator")
        ema = None if locs is None else locs.get("ema_g")
        saved = None
        if generator is not None and ema is not None:
            saved = [p.detach().clone() for p in generator.parameters()]
            with torch.no_grad():
                for parameter, value in zip(generator.parameters(), ema):
                    parameter.copy_(value)
        try:
            with original(self):
                yield
        finally:
            if saved is not None:
                with torch.no_grad():
                    for parameter, value in zip(generator.parameters(), saved):
                        parameter.copy_(value)

    module.NoisePolicy.set_step = set_step
    module.NoisePolicy.discriminator = discriminator


def _patch_trainer(module) -> None:
    original = module.GANTrainer.step

    def step(self, real, **kwargs):
        if _NAME == "ema_fake":
            return _ema_trainer_step(self, original, real, **kwargs)
        if _NAME == "extragradient":
            return _eg_trainer_step(self, real, **kwargs)
        return original(self, real, **kwargs)

    module.GANTrainer.step = step


def _ema_trainer_step(trainer, original, real, **kwargs):
    """First sample and first G forward in this step read the EMA copies."""
    import torch
    prior_sample = trainer.prior.sample
    generate = trainer._generate
    phase = {"sample": True, "generate": True}

    def sample(*args, **kw):
        if not phase["sample"]:
            return prior_sample(*args, **kw)
        phase["sample"] = False
        with torch.no_grad():
            saved = trainer.prior.z.detach().clone()
            trainer.prior.z.copy_(trainer.ema_prior.z)
        try:
            latent, idx = prior_sample(*args, **kw)
            latent = latent.detach().clone()
        finally:
            with torch.no_grad():
                trainer.prior.z.copy_(saved)
        return latent, idx

    def _generate(model, latent, sigma, stream):
        if model is not trainer.G or not phase["generate"]:
            return generate(model, latent, sigma, stream)
        phase["generate"] = False
        saved = _clone_params(trainer.G)
        with torch.no_grad():
            for parameter, value in zip(trainer.G.parameters(), trainer.ema_G.parameters()):
                parameter.copy_(value)
        try:
            return generate(model, latent, sigma, stream)
        finally:
            _copy_params(trainer.G, saved)

    trainer.prior.sample = sample
    trainer._generate = _generate
    try:
        return original(trainer, real, **kwargs)
    finally:
        trainer.prior.sample = prior_sample
        trainer._generate = generate


def _freeze_d(critic):
    flags = [p.requires_grad for p in critic.parameters()]
    critic.requires_grad_(False)
    return flags


def _thaw_d(critic, flags) -> None:
    for parameter, flag in zip(critic.parameters(), flags):
        parameter.requires_grad_(flag)


def train_mode_hold(recipe=None, *, seed: int = 0, gan_factory=None, cap_factory=None,
                    diagnostics=False, training_recipe=None, log=None, noise_policy=None):
    """Same host as ``mode_hold.train_mode_hold``, with one extragradient corrector."""
    import torch
    from dataclasses import replace
    import benchmarks.locked_shared.mode_hold as mode_hold

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
    prior = (mode_hold.ParticlePrior(recipe.n_particles, mode_hold.Z_DIM, init_std=0.5, generator=stream)
             if training_recipe is None else training_recipe.make_prior(generator=stream))
    generator = mode_hold.SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
    critic = mode_hold.SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER)
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
        generator = wrap_output(generator, noise_policy)
        critic = wrap_input(critic, noise_policy)
    gan = (gan_factory or (training_recipe.make_loss if training_recipe else mode_hold.make_gan_loss))()
    regularizer = (cap_factory or (training_recipe.make_gradient_penalty
                                   if training_recipe else mode_hold.make_b_cap))()
    vicreg = mode_hold.ParticleRegularizer(weight=recipe.vicreg_weight)
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
    adam = _raw_adam()

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

    def d_loss_at(step):
        real = mode_hold.sample_ring(means, batch, mode_hold.SIGMA, stream)
        latent, _ = prior.sample(batch, generator=stream)
        context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
        with context:
            fake = generator(latent).detach()
        loss = gan.d_loss(critic(real), critic(fake))
        return loss + regularizer(critic, real, fake, step=step + 1)

    def g_loss_at():
        latent, _ = prior.sample(batch, generator=stream)
        fake = generator(latent)
        if gan.mode in ("rp", "ra"):
            real_g = mode_hold.sample_ring(means, batch, mode_hold.SIGMA, stream)
            loss = gan.g_loss(critic(fake), critic(real_g))
        else:
            loss = gan.g_loss(critic(fake))
        if recipe.fm_weight > 0.0:
            real_mean = mode_hold.sample_ring(means, batch, mode_hold.SIGMA, stream).detach().mean(0)
            loss = loss + recipe.fm_weight * (fake.mean(0) - real_mean).pow(2).sum()
        loss = loss + recipe.particle_l2 * prior.z.pow(2).mean()
        return loss + vicreg(prior.z)

    curve, live_curve = [], []
    for step in range(recipe.steps):
        if noise_policy is not None:
            noise_policy.set_step(step)
        if training_recipe is not None:
            scale = mode_hold.learning_rate_scale(
                step, recipe.steps, training_recipe.lr_anneal_start, training_recipe.lr_floor)
            for opt, rates in zip((opt_g, opt_d), base_lrs):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * scale
        # One schedule call per player. The probe controller sets the rate from
        # the step index; the lookahead uses that same rate.
        mode_hold.schedule_optimizer(opt_d, step)
        mode_hold.schedule_optimizer(opt_g, step)
        rng = _snap_rng(stream, noise_policy, generator, critic)
        counts = _policy_counts(noise_policy)
        base_d, base_g, base_z = _clone_params(critic), _clone_params(generator), _clone_params(prior)
        buf_g, buf_d = _clone_buffers(generator), _clone_buffers(critic)
        state_d, state_g = _snap_opt(opt_d), _snap_opt(opt_g)

        opt_d.zero_grad()
        d_loss_at(step).backward()
        adam(opt_d)
        pred_d = _clone_params(critic)
        _copy_params(critic, base_d)
        _load_opt(opt_d, state_d)

        flags = _freeze_d(critic)
        try:
            opt_g.zero_grad()
            g_loss_at().backward()
            adam(opt_g)
        finally:
            _thaw_d(critic, flags)
        pred_g, pred_z = _clone_params(generator), _clone_params(prior)
        _copy_params(critic, base_d)
        _copy_params(generator, base_g)
        _copy_params(prior, base_z)
        _copy_buffers(generator, buf_g)
        _copy_buffers(critic, buf_d)
        _load_opt(opt_d, state_d)
        _load_opt(opt_g, state_g)
        _load_rng(rng)
        _load_counts(noise_policy, counts)

        _copy_params(critic, pred_d)
        _copy_params(generator, pred_g)
        _copy_params(prior, pred_z)
        opt_d.zero_grad()
        d_loss_at(step).backward()
        flags = _freeze_d(critic)
        try:
            opt_g.zero_grad()
            g_loss_at().backward()
        finally:
            _thaw_d(critic, flags)
        _copy_params(critic, base_d)
        _copy_params(generator, base_g)
        _copy_params(prior, base_z)
        opt_d.step()
        opt_g.step()

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


def _eg_trainer_step(trainer, real, *, generator_real=None, collect_stats=False):
    """Simultaneous extragradient around one ``GANTrainer`` update."""
    import torch
    from particlegan.recipes import learning_rate_scales
    from particlegan.training import input_noise_std, output_noise_std

    recipe = trainer.recipe
    if trainer.completed_steps >= recipe.total_steps:
        raise RuntimeError("recipe training budget exhausted")
    real = trainer._batch(real, "real")
    if generator_real is not None and not callable(generator_real):
        generator_real = trainer._batch(generator_real, "generator_real")
    network, prior_scale = learning_rate_scales(trainer.completed_steps, recipe)
    for optimizer, rates, roles in zip((trainer.opt_g, trainer.opt_d), trainer.initial_lrs, trainer.roles):
        for group, rate, role in zip(optimizer.param_groups, rates, roles):
            group["lr"] = rate * (prior_scale if role == "prior" else network)
    sigma_in = input_noise_std(recipe, trainer.completed_steps)
    sigma_out = output_noise_std(recipe, trainer.completed_steps)
    trainer._noisy_D.std = sigma_in
    adam = _raw_adam()

    def d_pass():
        trainer.D.train()
        trainer.G.eval()
        with torch.no_grad():
            latent, _ = trainer.prior.sample(len(real), generator=trainer.latent_generator)
            fake = trainer._generate(trainer.G, latent, sigma_out, trainer.noise_generator)
        critic = trainer._noisy_D
        loss_d = trainer.loss.d_loss(critic(real), critic(fake))
        trainer.penalty.collect_stats = collect_stats
        penalty = trainer.penalty(critic, real, fake)
        return loss_d + penalty, penalty

    def g_pass():
        # Caller freezes D so this backward cannot write on top of D's gradient.
        trainer.D.eval()
        trainer.G.train()
        critic = trainer._noisy_D
        latent, indices = trainer.prior.sample(len(real), generator=trainer.latent_generator)
        fake_logits = critic(trainer._generate(trainer.G, latent, sigma_out, trainer.noise_generator))
        real_g = generator_real() if callable(generator_real) else generator_real
        real_g = real if real_g is None else trainer._batch(real_g, "generator_real")
        loss_gan = trainer.loss.g_loss(fake_logits, critic(real_g))
        prior_reg = loss_gan.new_zeros(())
        if trainer.prior.z.requires_grad:
            raw = trainer.prior.z if recipe.num_particles <= 1024 else trainer.prior.z[torch.unique(indices)]
            prior_reg = trainer.prior_regularizer(raw)
        loss_g = loss_gan + recipe.prior_reg * prior_reg
        return loss_g, loss_gan, prior_reg

    rng = _snap_rng(trainer, trainer.G, trainer.D, trainer.prior, generator_real)
    base = {name: _clone_params(getattr(trainer, name)) for name in ("G", "D", "prior")}
    buffers = {name: _clone_buffers(getattr(trainer, name)) for name in ("G", "D", "prior")}
    states = {trainer.opt_d: _snap_opt(trainer.opt_d), trainer.opt_g: _snap_opt(trainer.opt_g)}

    trainer.opt_d.zero_grad()
    d_pass()[0].backward()
    adam(trainer.opt_d)
    pred_d = _clone_params(trainer.D)
    _copy_params(trainer.D, base["D"])
    _load_opt(trainer.opt_d, states[trainer.opt_d])

    flags = _freeze_d(trainer.D)
    try:
        trainer.opt_g.zero_grad()
        g_pass()[0].backward()
        adam(trainer.opt_g)
    finally:
        _thaw_d(trainer.D, flags)
    pred_g = _clone_params(trainer.G)
    pred_z = _clone_params(trainer.prior)

    for name, saved in base.items():
        _copy_params(getattr(trainer, name), saved)
    for name, saved in buffers.items():
        _copy_buffers(getattr(trainer, name), saved)
    for opt, state in states.items():
        _load_opt(opt, state)
    _load_rng(rng)

    _copy_params(trainer.D, pred_d)
    _copy_params(trainer.G, pred_g)
    _copy_params(trainer.prior, pred_z)
    trainer.opt_d.zero_grad()
    loss_d, penalty = d_pass()
    loss_d.backward()
    flags = _freeze_d(trainer.D)
    try:
        trainer.opt_g.zero_grad()
        loss_g, loss_gan, prior_reg = g_pass()
        loss_g.backward()
    finally:
        _thaw_d(trainer.D, flags)
    penalty_stats = trainer.penalty.last_stats
    for name, saved in base.items():
        _copy_params(getattr(trainer, name), saved)
    trainer.opt_d.step()
    trainer.opt_g.step()
    with torch.no_grad():
        for target, source in ((trainer.ema_G, trainer.G), (trainer.ema_prior, trainer.prior)):
            for averaged, current in zip(target.parameters(), source.parameters()):
                averaged.mul_(recipe.ema_decay).add_(current, alpha=1 - recipe.ema_decay)
            for averaged, current in zip(target.buffers(), source.buffers()):
                averaged.copy_(current)
    trainer.completed_steps += 1
    result = {key: value.detach() for key, value in
              dict(loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
                   prior_regularization=prior_reg, penalty=penalty).items()}
    result["step"] = trainer.completed_steps
    if collect_stats:
        result["penalty_stats"] = penalty_stats
    return result
