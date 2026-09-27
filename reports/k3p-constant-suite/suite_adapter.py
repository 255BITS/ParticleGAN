"""Arm adapter: runs one suite arm on the unchanged benchmark hosts (nothing in particlegan/ is edited).

Routes
* GANTrainer hosts (toy100 native, 6 vector, 4 image, ring tasks): ``GANTrainer.__init__`` is
  wrapped so the host's (legacy, pre-K3P) recipe is replaced by a ``particlegan.Recipe`` with the
  host's resource/LR fields and the arm's released-K3P fields. The trainer then builds the package
  K3P optimizers/penalty itself. Instance noise stays in each host's own wrappers (declared by the
  arm config), so the package recipe's noise fields are 0 there.
* Custom-loop hosts (9 legacy transfer hosts, mode_hold hold/shift): the host builds plain Adam and
  a legacy ``GradientPenalty`` with the config's marker arm ``a_r1r2``. Marked penalty calls are
  answered by the package K3P kernel (``particlegan.grad_regularizers.GradientPenalty``); the first
  optimizer stepped after a marked call is the critic, which gets the package spike guard and step
  record, and an EMA anchor evaluated by swapping the critic optimizer's parameter data (works for the
  hosts' closures). Prior tables alone in a beta1=0 group get package A2 ``LatentRowDamping``; direct
  particle groups get ``DirectParticleResponse``. AMSGrad is set on every Adam group.
* simple_B_cap3: the critic objective becomes the simple-critic ``SimpleCriticLoss`` (imported from
  the simple-critic worktree's worker.py, unchanged). GANTrainer hosts: trainer.loss/penalty swapped,
  critic optimizer rebuilt (no EMA critic, guard off, betas (0, d_beta2)). Legacy hosts: GANLoss.d_loss
  returns 0 and marked penalty calls return the simple loss (critic evaluated in batch-sized chunks
  on flattened inputs); critic betas forced to (0, d_beta2) before each critic step.

Receipts (``RECEIPT``): Adam constructions with initial-parameter sha256 (init receipt), applied LR
ranges per optimizer group, AMSGrad flags, noise wrappers, K3P/simple call counts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

import particlegan
from particlegan import GANTrainer, ParticlePrior, Recipe, get_recipe
from particlegan.grad_regularizers import CriticStepRecord, GradientPenalty as K3PKernel, _score_scalar
from particlegan.k3p import CriticSpikeGuard, DirectParticleResponse, LatentRowDamping

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIMPLE_WORKER = Path("/home/martyn/dev/ParticleGAN/.claude/worktrees/simple-critic/reports/simple-critic/worker.py")
MARKER_ARM = "a_r1r2"
ARMS = json.loads((HERE / "arms.json").read_text())["arms"]
RELEASED = get_recipe()
K3P_FIELDS = ("d_guard_ratio", "d_guard_min_steps", "latent_damping_max_rate", "reg_anchor_decay",
              "reg_anchor_weight", "direct_particle_betas", "direct_particle_gain", "amsgrad")

RECEIPT = {"optimizers": [], "lr": {}, "amsgrad_flags": [], "noise": {}, "k3p_legacy": {}, "simple_calls": 0,
           "gantrainer_recipes": [], "route": None}
ARM = None      # arm dict
ARM_NAME = None
LOG = print     # log(line)


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def params_sha(params):
    h = hashlib.sha256()
    for p in params:
        h.update(sha(p).encode())
    return h.hexdigest()


def arm_config(base: dict) -> dict:
    cfg = dict(base)
    cfg["reg_arm"] = MARKER_ARM
    cfg.update(ARM["config"])
    return cfg


def k3p_overlay() -> dict:
    """Released K3P values of the K3P-only fields, then the arm's recipe overrides."""
    values = {f: getattr(RELEASED, f) for f in K3P_FIELDS}
    values.update(ARM["recipe"])
    return values


def load_simple():
    spec = importlib.util.spec_from_file_location("simple_critic_worker", SIMPLE_WORKER)
    saved = list(sys.path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    assert Path(module.particlegan.__file__).resolve().is_relative_to(ROOT), module.particlegan.__file__
    return module


_SIMPLE = None


def simple_loss():
    global _SIMPLE
    if _SIMPLE is None:
        _SIMPLE = load_simple()
    spec = {k: v for k, v in ARM["simple"].items() if k != "d_beta2"}
    return _SIMPLE.SimpleCriticLoss(SimpleNamespace(**spec))


# ------------------------------------------------------------------ GANTrainer route
class SimpleCrit:
    """Stands in for GANTrainer's (loss, penalty): d_loss 0, the whole critic loss in the penalty."""

    def __init__(self, trainer):
        self.trainer, self.loss_fn, self.base_loss = trainer, simple_loss(), trainer.loss
        self.critic_steps, self.collect_stats, self.last_stats, self.last_terms = 0, False, {}, {}

    def d_loss(self, real_logits, fake_logits):
        return real_logits.new_zeros(())

    def g_loss(self, fake_logits, real_logits):
        return self.base_loss.g_loss(fake_logits, real_logits)

    def __call__(self, critic, real, fake):
        self.critic_steps += 1
        RECEIPT["simple_calls"] += 1
        b2 = ARM["simple"]["d_beta2"]
        for g in self.trainer.opt_d.param_groups:
            g["betas"] = (g["betas"][0], b2)
        RECEIPT["simple_d_betas"] = list(self.trainer.opt_d.param_groups[0]["betas"])
        u = torch.rand(len(real), device=real.device, generator=self.trainer.penalty_generator)
        total, terms = chunked_simple(self.loss_fn, critic, real, fake, u, self.critic_steps)
        self.last_terms = terms
        if self.collect_stats:
            self.last_stats = {k: float(v) for k, v in terms.items()}
        return total

    def diagnostics(self):
        return {}


def chunked_simple(loss_fn, D, real, fake, u, step):
    """SimpleCriticLoss on flattened inputs; D sees batch-sized chunks in its own input shape."""
    shape, n = real.shape[1:], len(real)

    def flat_d(x):
        outs = [D(c.reshape(len(c), *shape)) for c in x.split(n)]
        return torch.cat([o.reshape(len(o), -1).mean(1) if o.ndim > 1 else o for o in outs])
    return loss_fn(flat_d, real.reshape(n, -1), fake.reshape(len(fake), -1), u, step)


def package_recipe(host_recipe, *, ring=False):
    """particlegan.Recipe with the host's fields and the arm's K3P fields."""
    names = {f.name for f in dataclasses.fields(Recipe)}
    values = {n: getattr(host_recipe, n) for n in names if hasattr(host_recipe, n)}
    values.update(k3p_overlay())
    if not ring:  # host wrappers apply the declared noise; host policy hooks apply the network horizon
        values.update(input_noise_std=0.0, output_noise_std=0.0)
        cfg = RECEIPT.get("config", {})
        if cfg.get("network_lr_horizon_cap") is not None:
            values.update(network_lr_horizon_cap=cfg["network_lr_horizon_cap"],
                          network_lr_floor=cfg.get("network_lr_floor"))
    return Recipe(**values)


def convert_to_simple(trainer):
    recipe, D = trainer.recipe, trainer.D
    # Built with the recipe's betas so hosts' construction-time optimizer_receipts check passes;
    # beta2 = d_beta2 is applied at every critic step (SimpleCrit.__call__), before the first update.
    opt_d = recipe.make_critic_optimizer(D, ema_critic=None, **trainer.optimizer_options)
    assert opt_d.guard is None and opt_d.ema_critic is None
    trainer.opt_d = opt_d
    trainer.initial_lrs[1] = [g["lr"] for g in opt_d.param_groups]
    crit = SimpleCrit(trainer)
    trainer.loss, trainer.penalty = crit, crit
    return crit


_orig_trainer_init = GANTrainer.__init__


def _trainer_init(self, recipe, generator, discriminator, *args, **kwargs):
    ring = getattr(recipe, "_suite_ring", False) or RECEIPT["route"] == "ring"
    if not ring:
        recipe = package_recipe(recipe)
    _orig_trainer_init(self, recipe, generator, discriminator, *args, **kwargs)
    from benchmarks.toy100.models import InputNoise, OutputNoise
    RECEIPT["noise"].setdefault("wrappers", []).append(
        dict(input_wrapper=isinstance(discriminator, InputNoise), output_wrapper=isinstance(generator, OutputNoise),
             recipe_input_noise=self.recipe.input_noise_std, recipe_output_noise=self.recipe.output_noise_std))
    RECEIPT["gantrainer_recipes"].append({k: getattr(self.recipe, k) for k in (
        "lr", "d_lr_mult", "betas", "reg_coeff", "reg_kappa", "lr_floor", "network_lr_floor", "network_lr_horizon_cap",
        "total_steps", *K3P_FIELDS)})
    if ARM["critic"] == "simple_B_cap3":
        convert_to_simple(self)
    if ring:
        return
    inner, every = self.step, max(1, self.recipe.total_steps // 24)
    t0 = time.monotonic()

    def step(real, **kw):
        kw.setdefault("collect_stats", (self.completed_steps + 1) % every == 0)
        out = inner(real, **kw)
        if out["step"] % every == 0:
            ps = out.get("penalty_stats") or {}
            LOG(f"train {out['step']:6d}/{self.recipe.total_steps} Ld={float(out['loss_d']):+.4f} "
                f"Lg={float(out['loss_g']):+.4f} pen={float(out['penalty']):.3g} s={ps.get('s', float('nan')):.3g} "
                f"lr_d={self.opt_d.param_groups[0]['lr']:.3g} {time.monotonic() - t0:.0f}s")
        return out
    self.step = step


# ------------------------------------------------------------------ legacy custom-loop route
class SwapAnchor:
    """EMA of the critic optimizer's parameters, evaluated by swapping parameter data (closure-safe)."""

    def __init__(self, decay):
        self.decay, self.params, self.ema = float(decay), None, None

    @torch.no_grad()
    def start_(self):
        self.ema = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update_(self):
        for e, p in zip(self.ema, self.params):
            e.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def input_grad(self, D, x):
        """Surrogate s(x) = <x, grad Dbar(x)> per row: grad_x of its score is Dbar's input gradient."""
        saved = [p.data for p in self.params]
        try:
            for p, e in zip(self.params, self.ema):
                p.data = e
            xb = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                gb = torch.autograd.grad(_score_scalar(D(xb)), xb)[0].detach()
        finally:
            for p, v in zip(self.params, saved):
                p.data = v
        return (x * gb).flatten(1).sum(1)


LEG = {"pending": False, "critic": None, "record": None, "anchor": None, "guard": None, "kernel": {},
       "latent": {}, "direct": {}, "prior_ids": set(), "depth": {}, "tokens": {}, "u_gen": None, "enabled": False}


def _legacy_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True, *, ema_critic=None):
    if self.arm != MARKER_ARM:
        return _orig_penalty(self, D, x_real, x_fake, step, generator, collect_stats, ema_critic=ema_critic)
    LEG["pending"] = True
    if ARM["critic"] == "simple_B_cap3":
        if LEG["u_gen"] is None:
            LEG["u_gen"] = torch.Generator(device=x_real.device).manual_seed(3)
        RECEIPT["simple_calls"] += 1
        u = torch.rand(len(x_real), device=x_real.device, generator=LEG["u_gen"])
        total, terms = chunked_simple(_LEGACY_SIMPLE[0], D, x_real, x_fake, u, RECEIPT["simple_calls"])
        return total, ({k: float(v) for k, v in terms.items()} if collect_stats else {})
    key = (self.coeff, self.kappa, self.lazy_k)
    kernel = LEG["kernel"].get(key)
    if kernel is None:
        opts = PKG_RECIPE._penalty_options(coeff=self.coeff, kappa=self.kappa, lazy_k=self.lazy_k)
        kernel = LEG["kernel"][key] = K3PKernel(record=LEG["record"], **opts)
    record = LEG["record"]
    anchor = LEG["anchor"]
    ema = (lambda x: anchor.input_grad(D, x)) if anchor is not None else (lambda x: x.new_zeros(len(x)))
    pen, stats = kernel.penalty(D, x_real, x_fake, record.observed_steps + 1, collect_stats, ema_critic=ema)
    r = RECEIPT["k3p_legacy"]
    r["calls"] = r.get("calls", 0) + 1
    r["s_last"] = kernel.blend_weight()
    return pen, stats


def _identify(opt):
    if LEG["critic"] is None and LEG["pending"]:
        LEG["critic"] = opt
        params = [p for g in opt.param_groups for p in g["params"]]
        if LEG["anchor"] is not None:
            LEG["anchor"].params = params
        RECEIPT["k3p_legacy"]["critic_parameters"] = sum(p.numel() for p in params)


def _pre(opt, args, kwargs):
    d = LEG["depth"][id(opt)] = LEG["depth"].get(id(opt), 0) + 1
    if d != 1 or not LEG["enabled"]:
        return
    _identify(opt)
    tokens = []
    if opt is LEG["critic"]:
        if ARM["critic"] == "simple_B_cap3":
            for g in opt.param_groups:
                g["betas"] = (g["betas"][0], ARM["simple"]["d_beta2"])
        elif LEG["guard"] is not None:
            LEG["guard"].apply_(opt)
    else:
        for index, group in enumerate(opt.param_groups):
            ps = group["params"]
            if (PKG_RECIPE.latent_damping_max_rate > 0 and len(ps) == 1 and id(ps[0]) in LEG["prior_ids"]
                    and ps[0].dim() == 2 and group["betas"][0] == 0.0):
                damp = LEG["latent"].get(id(ps[0]))
                if damp is None:
                    damp = LEG["latent"][id(ps[0])] = LatentRowDamping(
                        ps[0], torch.zeros_like(ps[0], requires_grad=False), PKG_RECIPE.latent_damping_max_rate)
                tokens.append((damp, damp.begin(opt)))
            elif (group.get("_comparison_prior") and not any(id(p) in LEG["prior_ids"] for p in ps)
                  and all(p.grad is not None for p in ps)):
                key = (id(opt), index)
                resp = LEG["direct"].get(key)
                if resp is None:
                    n = sum(p.numel() for p in ps)
                    resp = LEG["direct"][key] = DirectParticleResponse(
                        ps, torch.zeros(n, dtype=ps[0].dtype, device=ps[0].device),
                        betas=PKG_RECIPE.direct_particle_betas, gain=PKG_RECIPE.direct_particle_gain)
                tokens.append((resp, resp.begin(opt)))
    LEG["tokens"][id(opt)] = tokens


def _post(opt, args, kwargs):
    d = LEG["depth"].get(id(opt), 1)
    LEG["depth"][id(opt)] = d - 1
    if d != 1:
        return
    if LEG["enabled"]:
        for obj, token in reversed(LEG["tokens"].pop(id(opt), [])):
            obj.end(token)
        if opt is LEG["critic"] and LEG["record"] is not None:
            LEG["record"].record_step(opt)
    _receipt_post(opt)


def _receipt_post(opt):
    idx = next((i for i, o in enumerate(_OPTS) if o is opt), None)
    if idx is None:
        return
    for gi, g in enumerate(opt.param_groups):
        key = f"opt{idx}.g{gi}"
        lr = float(g["lr"])
        row = RECEIPT["lr"].setdefault(key, {"min": lr, "max": lr, "steps": 0, "amsgrad": bool(g.get("amsgrad"))})
        row["min"], row["max"], row["steps"] = min(row["min"], lr), max(row["max"], lr), row["steps"] + 1
        row["amsgrad"] = row["amsgrad"] and bool(g.get("amsgrad"))


_OPTS = []
_INIT_SEEN = {}
_orig_adam_init = torch.optim.Adam.__init__
_orig_prior_init = ParticlePrior.__init__
_LEGACY_SIMPLE = []
PKG_RECIPE = None


def _adam_init(self, params, *args, **kwargs):
    _orig_adam_init(self, params, *args, **kwargs)
    if ARM["recipe"].get("amsgrad"):
        self.defaults["amsgrad"] = True
        for g in self.param_groups:
            g["amsgrad"] = True
    _OPTS.append(self)
    ps = [p for g in self.param_groups for p in g["params"]]
    for p in ps:  # first sight of each trainable tensor = its initial value (before any update)
        if id(p) not in _INIT_SEEN:
            _INIT_SEEN[id(p)] = (tuple(p.shape), sha(p))
    RECEIPT["optimizers"].append(dict(index=len(_OPTS) - 1, cls=type(self).__name__, groups=len(self.param_groups),
                                      params=sum(p.numel() for p in ps), init_sha256=params_sha(ps),
                                      amsgrad=[bool(g.get("amsgrad")) for g in self.param_groups]))


def _prior_init(self, *args, **kwargs):
    _orig_prior_init(self, *args, **kwargs)
    LEG["prior_ids"].update(id(p) for p in self.parameters())


_orig_penalty = None


# ------------------------------------------------------------------ ablation support (all routes)
ACTIVITY = {"a2_damped_steps": 0, "a2_calls": 0, "direct_steps": 0, "direct_gain_steps": 0, "guard_calls": 0,
            "guard_clips": [], "penalty_calls": 0, "penalty_s_lt_1": 0, "anchor_evals": 0}


def _count_activity():
    """Class-level counters: does each K3P component actually change an update in this task?"""
    orig_a2, orig_dr, orig_guard = LatentRowDamping.begin, DirectParticleResponse.begin, CriticSpikeGuard.apply_

    def a2_begin(self, optimizer):
        token = orig_a2(self, optimizer)
        ACTIVITY["a2_calls"] += 1
        ACTIVITY["a2_damped_steps"] += token is not None
        return token

    def dr_begin(self, optimizer):
        token = orig_dr(self, optimizer)
        ACTIVITY["direct_steps"] += token is not None
        ACTIVITY["direct_gain_steps"] += token is not None and self.last_gain != 1.0
        return token

    def guard_apply(self, optimizer):
        count = orig_guard(self, optimizer)
        ACTIVITY["guard_calls"] += 1
        if torch.is_tensor(count):
            ACTIVITY["guard_clips"].append(count.detach())
        return count
    LatentRowDamping.begin, DirectParticleResponse.begin, CriticSpikeGuard.apply_ = a2_begin, dr_begin, guard_apply


_orig_k3p_penalty = K3PKernel._k3p_penalty


def _counted_k3p_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
    ACTIVITY["penalty_calls"] += 1
    ACTIVITY["penalty_s_lt_1"] += self.blend_weight() < 1.0
    return _orig_k3p_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic)


def _variant_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
    """Ablation penalties, defined for the constant-LR form only (s == 1: K3P's phase A).

    K3P phase A (reference): c/2 * [mean ||g_r||^2/d + mean relu(||g_f||/sqrt d - k)^2]
    r1_only   : c/2 * mean ||g_r||^2/d                          (drop the fake cap)
    cap_only  : c/2 * [mean relu(||g_r||/sqrt d - k)^2 + mean relu(||g_f||/sqrt d - k)^2]  (drop R1; cap both)
    abs_units : c/2 * [mean ||g_r||^2 + mean relu(||g_f|| - k)^2]   (L2 instead of RMS units)
    """
    s = self.blend_weight()
    if s < 1.0:
        raise RuntimeError(f"penalty variant {ARM['penalty']} is defined for s == 1 only (got s={s})")
    ACTIVITY["penalty_calls"] += 1
    self.record.calls += 1
    d = x_real[0].numel()
    mode, k, c = ARM["penalty"], self.kappa, coefficient / 2.0
    if mode == "r1_only":
        pen = c * (self._grad_norm(D, x_real, squared=True) / d).mean()
    elif mode == "cap_only":
        n_r = self._grad_norm(D, x_real) / d ** 0.5
        n_f = self._grad_norm(D, x_fake) / d ** 0.5
        pen = c * ((n_r - k).relu().square().mean() + (n_f - k).relu().square().mean())
    elif mode == "abs_units":
        sq_r = self._grad_norm(D, x_real, squared=True)
        n_f = self._grad_norm(D, x_fake)
        pen = c * (sq_r.mean() + (n_f - k).relu().square().mean())
    else:
        raise ValueError(f"unknown penalty variant {mode}")
    if not collect_stats:
        return pen, {}
    return pen, {"applied": True, "pen": float(pen.detach()), "center": k, "s": s, "prox": 0.0, "phase": mode}


def install(arm_name: str, *, route: str, config: dict | None = None, log=print):
    """Install the arm's patches for this process (one task per process)."""
    global ARM, ARM_NAME, LOG, PKG_RECIPE, _orig_penalty
    ARM, ARM_NAME, LOG = ARMS[arm_name], arm_name, log
    _count_activity()
    K3PKernel._k3p_penalty = _variant_penalty if ARM.get("penalty") else _counted_k3p_penalty
    RECEIPT["penalty_variant"] = ARM.get("penalty")
    RECEIPT["route"], RECEIPT["config"] = route, dict(config or {})
    torch.optim.Adam.__init__ = _adam_init
    ParticlePrior.__init__ = _prior_init
    GANTrainer.__init__ = _trainer_init
    register_optimizer_step_pre_hook(_pre)
    register_optimizer_step_post_hook(_post)
    if route == "gantrainer" and ARM["critic"] == "simple_B_cap3":
        _patch_receipt_check()
    if route == "legacy":
        from benchmarks.legacy import grad_regularizers as legacy_gr
        from benchmarks.legacy.gan_loss import GANLoss
        cfg = config or {}
        base = {n: cfg[n] for n in ("lr", "lr_floor", "lr_anneal_start", "network_lr_floor",
                                     "network_lr_horizon_cap", "reg_coeff", "reg_kappa") if n in cfg}
        PKG_RECIPE = get_recipe(**base, **k3p_overlay())
        _orig_penalty = legacy_gr.GradientPenalty.penalty
        legacy_gr.GradientPenalty.penalty = _legacy_penalty
        LEG["enabled"] = True
        if ARM["critic"] == "simple_B_cap3":
            _LEGACY_SIMPLE.append(simple_loss())
            GANLoss.d_loss = lambda self, real_logits, fake_logits: real_logits.new_zeros(()).sum()
        else:
            LEG["anchor"] = SwapAnchor(PKG_RECIPE.reg_anchor_decay) if PKG_RECIPE.reg_anchor_weight else None
            LEG["record"] = CriticStepRecord(LEG["anchor"])
            LEG["guard"] = (CriticSpikeGuard(PKG_RECIPE.d_guard_ratio, PKG_RECIPE.d_guard_min_steps)
                            if PKG_RECIPE.d_guard_ratio else None)
        RECEIPT["k3p_legacy"]["package_recipe"] = {k: getattr(PKG_RECIPE, k) for k in (
            "reg_coeff", "reg_kappa", "lr_floor", "network_lr_floor", *K3P_FIELDS)}
    return RECEIPT


def finish_receipt():
    """Summaries for the smoke checks: LR constant? AMSGrad everywhere? guard clips."""
    rows = RECEIPT["lr"].values()
    RECEIPT["lr_constant"] = bool(rows) and all(r["min"] == r["max"] for r in rows)
    RECEIPT["amsgrad_all"] = bool(rows) and all(r["amsgrad"] for r in rows)
    if LEG["guard"] is not None:
        RECEIPT["k3p_legacy"]["guard_clipped_tensors"] = int(LEG["guard"].clipped_tensors)
    if LEG["record"] is not None:
        RECEIPT["k3p_legacy"]["record"] = LEG["record"].state_dict()
    act = dict(ACTIVITY)
    act["guard_clips"] = int(sum(int(c) for c in ACTIVITY["guard_clips"]))
    RECEIPT["activity"] = act
    seen, finals = set(), []
    for opt in _OPTS:  # bit-exactness receipt: every trained tensor after the last update
        for g in opt.param_groups:
            for p in g["params"]:
                if id(p) not in seen:
                    seen.add(id(p))
                    finals.append(sha(p))
    RECEIPT["final_sha256"] = params_sha_list(finals) if finals else None
    RECEIPT["init_sha256"] = params_sha_list([h for _, h in _INIT_SEEN.values()])
    RECEIPT["init_tensors"] = len(_INIT_SEEN)
    RECEIPT["particlegan_file"] = particlegan.__file__
    return RECEIPT


def params_sha_list(items):
    return hashlib.sha256("".join(items).encode()).hexdigest()


def _patch_receipt_check():
    """Harness fix: the transfer hosts' optimizer_receipts() asserts critic betas == recipe.betas, but
    simple_B_cap3 deliberately uses critic betas (0, d_beta2). Check against the recipe betas, then restore."""
    from benchmarks.transfer_suite import toy100_compatibility as tc
    original = tc.optimizer_receipts

    def receipts(trainer):
        group = trainer.opt_d.param_groups[0]
        saved = group["betas"]
        group["betas"] = tuple(trainer.recipe.betas)
        try:
            rows = original(trainer)
        finally:
            group["betas"] = saved
        for row in rows:
            if row["role"] == "d":
                row["betas"] = list(saved)
        return rows
    tc.optimizer_receipts = receipts
