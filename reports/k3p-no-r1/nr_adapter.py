"""No-R1 arms on top of the k3p-constant-suite adapter (``../k3p-constant-suite/suite_adapter.py``, unchanged).

Every nr_* arm is gs2_c03_lr2_d05 with R1 on reals removed and replaced by one SPIKE control and one SETTLING
mechanism. Nothing in particlegan/ is edited: the changes are class-level patches installed once per task process
after ``suite_adapter.install``, so they reach every host the same way the suite's own arm overrides do:

* penalty (all routes): ``GradientPenalty._k3p_penalty`` -> ``nr_penalty``. The GANTrainer route (native toy100,
  standard-trainer toys, rings) reaches it through ``CriticPenalty``; the custom-loop route (legacy toys, mode_hold
  hold/shift) through the suite's marked-penalty kernel.
      pen = c/2 * [spike + mean relu(||g_f||/sqrt d - kappa)^2 (+ anchor_weight * prox)]
  No R1 term is ever evaluated.
* hinge: ``particlegan.gan_loss.GANLoss.d_loss`` and ``benchmarks.legacy.gan_loss.GANLoss.d_loss`` (RpGAN logistic
  only) become mean relu(1 - (D(r) - D(f))); g_loss is untouched. Every arm counts its d_loss calls by form.
* oadam: ``torch.optim.Adam.step`` is replaced by a hooked dispatcher. For the critic optimizer (GANTrainer:
  ``trainer.opt_d``; legacy: the critic the suite identifies) it runs ``lib/oadam.py`` ``OptimisticAdam.step`` on
  that optimizer's own param groups/state (amsgrad must be on); every other Adam runs the stock update. The spike
  guard reads ``max_exp_avg_sq`` in these arms (``guard_buffer``).
* anchor: the EMA-anchor prox term, normally gated on s < 1 (LR annealed below peak), is evaluated at s == 1.
  The anchor starts on the first penalty call where it can (the legacy swap anchor needs the critic identified),
  then the critic's step record advances the EMA after every critic step, as in K3P's phase B.

Receipts: ``receipt()`` -> nr_receipt.json in each task dir (and ``nr_receipt`` inside result.json).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / "k3p-constant-suite"
ROOT = HERE.parents[1]
for p in (str(SUITE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import suite_adapter as sa  # noqa: E402
from particlegan import GANTrainer  # noqa: E402
from particlegan import gan_loss as pkg_gan_loss  # noqa: E402
from particlegan.grad_regularizers import GradientPenalty as K3PKernel, _score_scalar  # noqa: E402
from particlegan.k3p import CriticSpikeGuard  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("nr_oadam", ROOT / "lib" / "oadam.py")
_oadam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oadam)
OptimisticAdam = _oadam.OptimisticAdam  # lib/oadam.py of this worktree (by path: lib/ is not a package)

NR_ARMS = json.loads((HERE / "arms.json").read_text())["arms"]
sa.ARMS.update(NR_ARMS)  # the suite's worker/launcher look arms up in sa.ARMS
SPIKES = ("none", "symcap", "pathcap", "pairsec", "dvalcap")
SETTLES = ("none", "hinge", "oadam", "anchor")
PATH_SEED = 7  # pathcap interpolation stream (one per device, per task process)

NR = {"arm": None, "spike": None, "settle": None, "guard_buffer": None, "installed": False,
      "critic_opts": [], "kernels": [], "oadam_steps": 0, "d_loss": {}, "pair_mismatch": 0,
      "sums": {}, "penalty_calls": 0, "anchor_start_step": None, "anchor_deferred": 0}
_PATH_GEN = {}


# ------------------------------------------------------------------ terms (pure functions, unit-tested)
def per_sample(out):
    """One logit per sample (a spatial map is averaged), matching ``_score_scalar``'s per-image mean."""
    return out.flatten(1).mean(dim=1) if out.ndim >= 2 else out


def grad_rms(D, x):
    """||grad_x D(x)|| / sqrt(d) per sample, with create_graph (same kernel as K3P's fake cap)."""
    d = x[0].numel()
    return K3PKernel._grad_norm(D, x, squared=False) / d ** 0.5


def cap(D, x, kappa):
    return (grad_rms(D, x) - kappa).relu().square().mean()


def path_points(x_real, x_fake, u):
    """x_hat = r + u (f - r), pairing by batch index (first n = min(len) rows)."""
    n = min(len(x_real), len(x_fake))
    r, f = x_real[:n].detach(), x_fake[:n].detach()
    u = u[:n].reshape(n, *([1] * (r.dim() - 1))).to(r.dtype)
    return r + u * (f - r)


def nearest_other(x):
    """Index of each row's nearest other row (Euclidean on flattened rows) and that distance."""
    flat = x.detach().flatten(1)
    dist = torch.cdist(flat, flat, compute_mode="donot_use_mm_for_euclid_dist")
    dist.fill_diagonal_(float("inf"))
    j = dist.argmin(dim=1)
    return j, (flat - flat[j]).norm(dim=1)


def secant_slopes(D, x_real):
    """|D(r_i) - D(r_j)| / ||r_i - r_j|| with r_j = nearest other real; gradient flows through D only."""
    j, dist = nearest_other(x_real)
    logits = per_sample(D(x_real.detach()))
    return (logits - logits[j]).abs() / dist.clamp_min(1e-12)


def pairsec(D, x_real, kappa):
    if len(x_real) < 2:
        return x_real.new_zeros(())
    d = x_real[0].numel()
    return (secant_slopes(D, x_real) / d ** 0.5 - kappa).relu().square().mean()


def dvalcap(D, x_real, margin=1.0):
    logits = per_sample(D(x_real.detach()))
    return ((logits - logits.mean()).abs() - margin).relu().square().mean()


def spike_term(spike, D, x_real, x_fake, kappa, u=None):
    if spike == "none":
        return x_real.new_zeros(())
    if spike == "symcap":
        return cap(D, x_real.detach(), kappa)
    if spike == "pathcap":
        if len(x_real) != len(x_fake):
            NR["pair_mismatch"] += 1
        return cap(D, path_points(x_real, x_fake, u), kappa)
    if spike == "pairsec":
        return pairsec(D, x_real, kappa)
    if spike == "dvalcap":
        return dvalcap(D, x_real)
    raise ValueError(f"unknown spike control {spike}")


def hinge_d(real_logits, fake_logits):
    """Relativistic (paired) hinge critic loss."""
    return F.relu(1.0 - (real_logits - fake_logits)).mean()


# ------------------------------------------------------------------ the penalty
def _path_u(x_real):
    dev = x_real.device
    gen = _PATH_GEN.get(str(dev))
    if gen is None:
        gen = _PATH_GEN[str(dev)] = torch.Generator(device=dev).manual_seed(PATH_SEED)
    return torch.rand(len(x_real), device=dev, generator=gen)


def _anchor_ready(anchor):
    return anchor is not None and getattr(anchor, "params", ()) is not None  # legacy SwapAnchor: critic identified


def anchor_prox(kernel, D, x_real, ema_critic, dimension):
    """anchor_weight * mean ||g_r - gbar_r||^2 / d at s == 1 (K3P's phase-B prox, gate bypassed)."""
    record, anchor = kernel.record, kernel.anchor
    if kernel.anchor_weight <= 0:
        raise ValueError("anchor arm needs reg_anchor_weight > 0")
    if not record.anchor_started:
        if not _anchor_ready(anchor):
            NR["anchor_deferred"] += 1
            return x_real.new_zeros(())
        anchor.start_()
        record.anchor_started = True
        NR["anchor_start_step"] = record.observed_steps
        return x_real.new_zeros(())  # EMA == live critic: prox is exactly 0 on the starting call (as in phase B)
    fn = ema_critic if ema_critic is not None else anchor
    x = x_real.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
    xb = x_real.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        gb = torch.autograd.grad(_score_scalar(fn(xb)), xb)[0].detach()
    return kernel.anchor_weight * (g - gb).pow(2).flatten(1).sum(dim=1).mean() / dimension


def _acc(name, value):
    v = value.detach().float()
    s = NR["sums"].setdefault(name, [torch.zeros((), device=v.device), torch.zeros((), device=v.device), 0])
    if s[0].device != v.device:
        s[0], s[1] = s[0].to(v.device), s[1].to(v.device)
    s[0] += v
    s[1] += (v > 0).float()
    s[2] += 1


def nr_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
    """Replaces ``GradientPenalty._k3p_penalty`` in nr_* arms (constant LR only: s == 1)."""
    s = self.blend_weight()
    if s < 1.0:
        raise RuntimeError(f"nr arms are defined at constant LR only (s == 1), got s={s}")
    d = x_real[0].numel()
    if d != x_fake[0].numel():
        raise ValueError("nr penalty needs reals and fakes of the same per-sample size")
    if self not in NR["kernels"]:
        NR["kernels"].append(self)
    self.record.calls += 1
    NR["penalty_calls"] += 1
    sa.ACTIVITY["nr_penalty_calls"] = NR["penalty_calls"]
    k = self.kappa
    u = _path_u(x_real) if NR["spike"] == "pathcap" else None
    spike = spike_term(NR["spike"], D, x_real, x_fake, k, u)
    fake_cap = cap(D, x_fake.detach(), k)
    total = spike + fake_cap
    prox = None
    if NR["settle"] == "anchor":
        prox = anchor_prox(self, D, x_real, ema_critic, d)
        total = total + prox
        _acc("prox", prox)
    _acc("spike", spike)
    _acc("fake_cap", fake_cap)
    pen = (coefficient / 2.0) * total
    if not collect_stats:
        return pen, {}
    return pen, {"applied": True, "pen": float(pen.detach()), "center": k, "s": s,
                 "prox": 0.0 if prox is None else float(prox.detach()), "phase": f"nr:{NR['spike']}+{NR['settle']}",
                 "spike": float(spike.detach()), "fake_cap": float(fake_cap.detach())}


# ------------------------------------------------------------------ loss
_ORIG_PKG_D = pkg_gan_loss.GANLoss.d_loss


def _count_loss(form):
    NR["d_loss"][form] = NR["d_loss"].get(form, 0) + 1


def _pkg_d_loss(self, real_logits, fake_logits):
    if NR["settle"] == "hinge":
        _count_loss("rp_hinge")
        return hinge_d(real_logits, fake_logits)
    _count_loss("rp_softplus")
    return _ORIG_PKG_D(self, real_logits, fake_logits)


def _legacy_d_loss_factory(orig):
    def d_loss(self, real_logits, fake_logits):
        if (self.mode, self.loss_type) != ("rp", "logistic"):
            raise ValueError(f"nr arms expect the RpGAN logistic loss, host uses {self.mode}/{self.loss_type}")
        if NR["settle"] == "hinge":
            _count_loss("rp_hinge")
            return hinge_d(real_logits, fake_logits)
        _count_loss("rp_softplus")
        return orig(self, real_logits, fake_logits)
    return d_loss


# ------------------------------------------------------------------ optimizer
_RAW_OADAM = None
_RAW_ADAM = None


def _unhook(step):
    return step.__wrapped__ if getattr(step, "hooked", False) else step


def _is_critic(opt):
    return any(opt is o for o in NR["critic_opts"]) or (sa.LEG["enabled"] and opt is sa.LEG["critic"])


def oadam_update(opt, closure=None):
    """``OptimisticAdam.step`` on ``opt``'s own groups/state (the critic Adam object keeps its hooks and class)."""
    for g in opt.param_groups:
        if not g.get("amsgrad") or g.get("weight_decay", 0) != 0 or g.get("maximize", False):
            raise ValueError(f"oadam critic needs amsgrad on, no weight decay, no maximize: {g.get('amsgrad')}")
    NR["oadam_steps"] += 1
    return _RAW_OADAM(opt, closure)


def _dispatch(self, closure=None):
    if NR["settle"] == "oadam" and _is_critic(self):
        return oadam_update(self, closure)
    return _RAW_ADAM(self, closure)


def install_oadam():
    global _RAW_OADAM, _RAW_ADAM
    _RAW_OADAM = _unhook(OptimisticAdam.step)
    _RAW_ADAM = _unhook(torch.optim.Adam.step)
    hooked = torch.optim.Optimizer.profile_hook_step(_dispatch)
    hooked.hooked = True
    torch.optim.Adam.step = hooked


def guard_apply_max(self, optimizer):
    """``CriticSpikeGuard.apply_`` reading the second moment the update actually uses (AMSGrad max buffer)."""
    with torch.no_grad():
        flags = []
        for group in optimizer.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                st = optimizer.state.get(p)
                if p.grad is None or not st or "exp_avg_sq" not in st:
                    continue
                buf = st["max_exp_avg_sq"] if "max_exp_avg_sq" in st else st["exp_avg_sq"]
                t = st["step"]
                vhat = buf.mean() / (1.0 - beta2 ** t)
                ratio = p.grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
                clip = torch.as_tensor(t >= self.min_steps, device=ratio.device) & (ratio > self.ratio)
                p.grad.mul_(torch.where(clip, self.ratio / ratio, torch.ones_like(ratio)))
                flags.append(clip)
        if not flags:
            return 0
        count = torch.stack(flags).sum()
        self._clipped = self._clipped + count
        return count


# ------------------------------------------------------------------ install / receipt
def install_nr(arm_name):
    """Patches that must precede ``suite_adapter.install`` (it wraps the guard with its counters)."""
    arm = sa.ARMS[arm_name]
    nr = arm["nr"]
    if nr["spike"] not in SPIKES or nr["settle"] not in SETTLES:
        raise ValueError(f"bad nr arm {arm_name}: {nr}")
    NR.update(arm=arm_name, spike=nr["spike"], settle=nr["settle"], guard_buffer=nr["guard_buffer"])
    if nr["guard_buffer"] == "max_exp_avg_sq":
        CriticSpikeGuard.apply_ = guard_apply_max
    if nr["settle"] == "oadam":
        install_oadam()


def install_post():
    """Patches after ``suite_adapter.install`` (which sets its own penalty variant and trainer wrapper)."""
    K3PKernel._k3p_penalty = nr_penalty
    pkg_gan_loss.GANLoss.d_loss = _pkg_d_loss
    try:
        from benchmarks.legacy import gan_loss as legacy_gan_loss
        legacy_gan_loss.GANLoss.d_loss = _legacy_d_loss_factory(legacy_gan_loss.GANLoss.d_loss)
    except ImportError:
        pass
    inner = GANTrainer.__init__

    def trainer_init(self, *args, **kwargs):
        inner(self, *args, **kwargs)
        NR["critic_opts"].append(self.opt_d)
    GANTrainer.__init__ = trainer_init
    NR["installed"] = True


_ORIG_INSTALL = sa.install


def install(arm_name, *, route, config=None, log=print):
    """Drop-in for ``suite_adapter.install`` (the suite worker calls ``sa.install``)."""
    if arm_name not in NR_ARMS:
        return _ORIG_INSTALL(arm_name, route=route, config=config, log=log)
    install_nr(arm_name)
    receipt = _ORIG_INSTALL(arm_name, route=route, config=config, log=log)
    install_post()
    return receipt


def _has_guard(opt):
    if sa.LEG["enabled"] and opt is sa.LEG["critic"]:
        return sa.LEG["guard"] is not None
    return getattr(opt, "guard", None) is not None


def _critic_rows():
    opts = list(NR["critic_opts"])
    if sa.LEG["enabled"] and sa.LEG["critic"] is not None and not any(sa.LEG["critic"] is o for o in opts):
        opts.append(sa.LEG["critic"])
    rows = []
    for opt in opts:
        keys = sorted({k for st in opt.state.values() for k in st})
        oadam = NR["settle"] == "oadam"
        rows.append(dict(cls=type(opt).__name__,
                         update_rule="lib.oadam.OptimisticAdam.step" if oadam else "torch.optim.Adam.step",
                         amsgrad=[bool(g.get("amsgrad")) for g in opt.param_groups],
                         betas=[list(g["betas"]) for g in opt.param_groups],
                         lr=[float(g["lr"]) for g in opt.param_groups], state_keys=keys,
                         guard=NR["guard_buffer"] if _has_guard(opt) else None))
    return rows


def receipt(cfg):
    sums = {k: dict(mean=float(v[0]) / max(v[2], 1), positive_frac=float(v[1]) / max(v[2], 1), calls=v[2])
            for k, v in NR["sums"].items()}
    kernels = [dict(anchor_weight=k.anchor_weight, kappa=k.kappa, coeff=k.coeff, lazy_k=k.lazy_k,
                    anchor_started=k.record.anchor_started, anchor=type(k.anchor).__name__ if k.anchor else None,
                    blend_s=k.blend_weight(), record_steps=k.record.observed_steps) for k in NR["kernels"]]
    anchor_active = bool(kernels) and all(k["anchor_started"] for k in kernels) and sums.get("prox", {}).get(
        "positive_frac", 0) > 0
    rec = dict(
        arm=NR["arm"], spike=NR["spike"], settle=NR["settle"],
        r1_weight_applied=0.0,
        penalty_patched=K3PKernel._k3p_penalty is nr_penalty,
        penalty_form=(f"c/2*[{NR['spike']} + fake_cap" + (" + anchor_weight*prox" if NR["settle"] == "anchor" else "")
                      + "]"),
        nr_penalty_calls=NR["penalty_calls"], stock_k3p_penalty_calls=sa.ACTIVITY.get("penalty_calls", 0),
        term_stats=sums, pathcap_pair_mismatch=NR["pair_mismatch"],
        loss_form="rp_hinge" if NR["settle"] == "hinge" else "rp_softplus", d_loss_calls=dict(NR["d_loss"]),
        g_loss="RpGAN softplus (unchanged)",
        critic_optimizers=_critic_rows(), oadam_steps=NR["oadam_steps"], guard_buffer=NR["guard_buffer"],
        anchor_active=anchor_active if NR["settle"] == "anchor" else False,
        anchor_start_step=NR["anchor_start_step"], anchor_deferred_calls=NR["anchor_deferred"], kernels=kernels,
        noise=dict(declared={k: cfg.get(k) for k in ("input_noise_std", "output_noise_std")},
                   applied=sa.RECEIPT.get("noise")),
        lr=dict(declared={k: cfg.get(k) for k in ("lr", "d_lr_mult", "lr_floor", "network_lr_floor",
                                                   "prior_lr_mult", "reg_coeff")},
                applied_groups=sa.RECEIPT.get("lr"), lr_constant=sa.RECEIPT.get("lr_constant"),
                gantrainer_recipes=sa.RECEIPT.get("gantrainer_recipes")),
    )
    if not math.isfinite(sum(v["mean"] for v in sums.values())):
        rec["nonfinite_terms"] = True
    return rec
