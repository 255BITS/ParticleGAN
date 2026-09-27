"""Simple-critic arm on the KA2 constant-LR shift protocol (ring of 8, shift at 2400).

Same data, architecture, particle prior, learning rates, shift and evaluator as
reports/ka2-default-candidate/constant-lr-api/worker.py, with three changes:

* no instance noise (critic input noise and generator output noise are 0 from step 0);
* every LR is constant (recipe floors = 1, no transition) and checked every update;
* the critic objective is a configurable simple formulation instead of
  RpGAN + KA2 penalty, and the critic optimizer carries no KA2 controller
  (recipe factory with the spike guard off and no EMA critic, so its ``step()``
  is Adam's update plus read-only bookkeeping).

Critic loss = base + real term + path term + cap term:
  base   wgan: E D(f) - E D(r) | hinge | logistic | rplogistic (the public trainer's base)
  real   drift: E D(r)^2 | r1: E ||grad D(r)||^2 | drift+r1
  path   on xhat = f + u (r - f), u ~ U(0,1):
         lower: E relu(t - ||grad D(xhat)||)^2 | two_sided: E (||grad D(xhat)|| - t)^2
  cap    E relu(||grad D(x)|| - c)^2 over xhat (interp) or real+fake+xhat (all)
  center (optional, --lam-center) lam * (E D(r))^2: pins only D's level (batch mean over reals),
         unlike drift, which also penalizes the spread of D over reals
  lazy   (optional, --lazy-k k) every non-base term is applied only on critic steps with
         step % k == 0, with its weight multiplied by k (grad_regularizers lazy_k semantics)
The generator/prior update is the public trainer's (RpGAN g_loss by default).
One arm = one formulation, seed 0. Tail logs/<arm>.log.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

import particlegan
from particlegan import get_recipe
from particlegan.recipes import learning_rate_scales
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator

SEED = 0
SHIFT_STEP = 2400
OBSERVE_EVERY = 10
PREHOLD = (1210, 2400)
PROBE_N = mode_hold.EVAL_N


def good(point):
    return point["modes"] == 8 and .90 <= point["hq"] <= 1.0


def window(points, begin, end, spacing=OBSERVE_EVERY):
    sel = [p for p in points if begin <= p["step"] <= end and (p["step"] - begin) % spacing == 0]
    return dict(checks=len(sel), expected_checks=len(range(begin, end + 1, spacing)),
                passing_checks=sum(map(good, sel)), failing_steps=[p["step"] for p in sel if not good(p)],
                min_hq=min((p["hq"] for p in sel), default=None))


def recovery(points, end):
    sel = [p for p in points if SHIFT_STEP < p["step"] <= end]
    first = next((p["step"] for p in sel if good(p)), None)
    after = [p for p in sel if first is not None and p["step"] >= first]
    suffix = []
    for p in reversed(sel):
        if not good(p):
            break
        suffix.append(p)
    return dict(first_passing_step=first, updates_to_first_pass=None if first is None else first - SHIFT_STEP,
                checks_from_first_pass=len(after), passing_checks_from_first_pass=sum(map(good, after)),
                failing_steps_after_first_pass=[p["step"] for p in after if not good(p)],
                stable_suffix_start=suffix[-1]["step"] if suffix else None, stable_suffix_checks=len(suffix))


# ---------------------------------------------------------------- critic loss
def grad_norm(g):
    return torch.sqrt(g.pow(2).flatten(1).sum(1) + 1e-12)


class SimpleCriticLoss:
    """``loss(D, real, fake, u)`` -> (scalar, detached term dict)."""

    def __init__(self, args):
        self.a = args
        self.real_terms = set() if args.real == "none" else set(args.real.split("+"))
        self.need_real_grad = "r1" in self.real_terms or args.cap == "all"
        self.need_fake_grad = args.cap == "all"
        self.need_path = args.path in ("lower", "two_sided") or args.cap in ("interp", "all")
        self.u_lo, self.u_hi = (float(v) for v in args.path_u.split(","))

    def base(self, dr, df):
        kind = self.a.loss
        if kind == "wgan":
            return df.mean() - dr.mean()
        if kind == "hinge":
            return F.relu(1 - dr).mean() + F.relu(1 + df).mean()
        if kind == "logistic":
            return F.softplus(-dr).mean() + F.softplus(df).mean()
        return F.softplus(-(dr - df)).mean()  # rplogistic

    def __call__(self, D, real, fake, u, step=1):
        a, n = self.a, len(real)
        k = a.lazy_k
        if k > 1 and step % k != 0:  # lazy skip: base only (grad_regularizers lazy_k semantics)
            out = D(torch.cat([real, fake]).detach())
            terms = {"base": self.base(out[:n], out[n:])}
            zero = out.new_zeros(())
            terms.update({t: zero for t in ("drift", "r1", "path", "cap")})
            if a.lam_center:
                terms["center"] = zero
            return terms["base"], {kk: v.detach() for kk, v in terms.items()}
        parts = [real, fake]
        u = self.u_lo + (self.u_hi - self.u_lo) * u  # identity for the round-1 default 0,1
        if self.need_path:
            parts.append(fake + u[:, None] * (real - fake))
        x = torch.cat(parts).detach()
        need_grad = self.need_real_grad or self.need_fake_grad or self.need_path
        if need_grad:
            x.requires_grad_(True)
        out = D(x)
        dr, df = out[:n], out[n:2 * n]
        terms = {"base": self.base(dr, df)}
        zero = out.new_zeros(())
        if need_grad:
            g = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
            sq = g.pow(2).flatten(1).sum(1)
            norm = torch.sqrt(sq + 1e-12)
            nr, nf = norm[:n], norm[n:2 * n]
            npath = norm[2 * n:] if self.need_path else None
        lam_drift = a.lam_real if a.lam_drift is None else a.lam_drift
        terms["drift"] = lam_drift * dr.pow(2).mean() if "drift" in self.real_terms else zero
        terms["r1"] = a.lam_real * sq[:n].mean() if "r1" in self.real_terms else zero
        if a.path == "lower":
            terms["path"] = a.lam_path * F.relu(a.path_target - npath).pow(2).mean()
        elif a.path == "two_sided":
            terms["path"] = a.lam_path * (npath - a.path_target).pow(2).mean()
        elif a.path == "secant":
            # path from fake: D must rise by >= t*dist from each fake to its nearest real (no slope at real).
            with torch.no_grad():
                nn = torch.cdist(fake.detach(), real.detach()).argmin(1)
                dist = (real[nn] - fake).detach().norm(dim=1)
            terms["path"] = a.lam_path * F.relu(a.path_target * dist - (dr[nn] - df)).pow(2).mean()
        else:
            terms["path"] = zero
        if a.cap == "interp":
            terms["cap"] = a.lam_cap * F.relu(npath - a.cap_target).pow(2).mean()
        elif a.cap == "all":
            terms["cap"] = a.lam_cap * F.relu(norm - a.cap_target).pow(2).mean()
        else:
            terms["cap"] = zero
        if a.lam_center:
            terms["center"] = a.lam_center * dr.mean().pow(2)
        if k > 1:  # lazy application step: proportionally bigger hit
            for t in terms:
                if t != "base":
                    terms[t] = k * terms[t]
        total = sum(terms.values())
        return total, {k: v.detach() for k, v in terms.items()}


def g_loss_fn(kind, recipe_loss, fake_logits, real_logits):
    if kind == "rpgan":
        return recipe_loss.g_loss(fake_logits, real_logits)
    if kind == "wgan":
        return -fake_logits.mean()
    return F.softplus(-fake_logits).mean()  # ns


# ---------------------------------------------------------------- probes
def probe(D, real, fake, u):
    """D values and input-gradient norms at real, fake and path points (no training effect)."""
    was = D.training
    D.eval()
    flags = [p.requires_grad for p in D.parameters()]
    D.requires_grad_(False)
    try:
        path = fake + u[:, None] * (real - fake)
        x = torch.cat([real, fake, path]).detach().requires_grad_(True)
        with torch.enable_grad():
            out = D(x)
            g = torch.autograd.grad(out.sum(), x)[0]
        n = len(real)
        norm = grad_norm(g)
        dr, df = out[:n].detach(), out[n:2 * n].detach()
        return dict(dr_mean=float(dr.mean()), dr_max=float(dr.max()), dr_absmax=float(dr.abs().max()),
                    df_mean=float(df.mean()), gap=float(dr.mean() - df.mean()),
                    g_real=float(norm[:n].mean()), g_fake=float(norm[n:2 * n].mean()),
                    g_path=float(norm[2 * n:].mean()), g_max=float(norm.max()),
                    g_path_min=float(norm[2 * n:].min()))
    finally:
        for p, f in zip(D.parameters(), flags):
            p.requires_grad_(f)
        D.train(was)


# ---------------------------------------------------------------- run
def make_recipe(args):
    return get_recipe(total_steps=args.steps, lr_floor=1.0, network_lr_floor=1.0,
                      input_noise_std=0.0, output_noise_std=0.0, output_noise_warmup=0.0,
                      d_guard_ratio=0.0, reg_anchor_weight=0.0, d_lr_mult=args.d_lr_mult,
                      latent_damping_max_rate=args.latent_damping)


def describe(args):
    parts = [args.loss]
    if args.real != "none":
        lam_d = args.lam_real if args.lam_drift is None else args.lam_drift
        parts.append("+".join(f"{t}({lam_d if t == 'drift' else args.lam_real:g})" for t in args.real.split("+")))
    if args.path != "none":
        win = "" if args.path_u == "0,1" or args.path == "secant" else f",u={args.path_u}"
        parts.append(f"path-{args.path}({args.lam_path:g},t={args.path_target:g}{win})")
    if args.cap != "none":
        parts.append(f"cap-{args.cap}({args.lam_cap:g},c={args.cap_target:g})")
    if args.lam_center:
        parts.append(f"center({args.lam_center:g})")
    extra = []
    if args.g_loss != "rpgan":
        extra.append(f"G={args.g_loss}")
    if args.n_critic != 1:
        extra.append(f"nD={args.n_critic}")
    if args.d_lr_mult != 1:
        extra.append(f"dlr×{args.d_lr_mult:g}")
    if args.d_beta2 is not None:
        extra.append(f"Dβ2={args.d_beta2:g}")
    if args.latent_damping != 0.5:
        extra.append(f"A2={args.latent_damping:g}")
    if args.lazy_k != 1:
        extra.append(f"lazy_k={args.lazy_k}")
    return " + ".join(parts) + ("" if not extra else " [" + ", ".join(extra) + "]")


def rates(opt_g, opt_d, roles):
    return {f"{role}_{i}": float(group["lr"])
            for opt, rs in zip((opt_g, opt_d), roles) for i, (group, role) in enumerate(zip(opt.param_groups, rs))}


def run(args):
    recipe = make_recipe(args)
    out_dir = args.output or HERE / "runs" / args.arm
    log_path = args.log or HERE / "logs" / f"{args.arm}.log"
    if (out_dir / "result.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir}/result.json exists; pass --overwrite or pick a new --arm")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Construction order of the public baseline (GANTrainer builds the prior after G and D).
    torch.manual_seed(SEED)
    G = SimpleMLPGenerator(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2).to(device)
    D = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER).to(device)
    prior = recipe.make_prior().to(device)
    opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=None, foreach=False, fused=False)
    assert opt_d.guard is None and opt_d.ema_critic is None, "critic optimizer must carry no controller"
    if args.d_beta2 is not None:  # constant optimizer constant, not a schedule
        for group in opt_d.param_groups:
            group["betas"] = (group["betas"][0], args.d_beta2)
    prior_ids = {id(p) for p in prior.parameters()}
    roles = [["prior" if any(id(p) in prior_ids for p in g["params"]) else "generator" for g in opt_g.param_groups],
             ["critic"] * len(opt_d.param_groups)]
    base_loss = recipe.make_loss()
    prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
    critic_loss = SimpleCriticLoss(args)
    ema_G, ema_prior = copy.deepcopy(G).eval(), copy.deepcopy(prior).eval()
    for m in (ema_G, ema_prior):
        m.requires_grad_(False)
    latent_stream = torch.Generator(device=device).manual_seed(SEED + 2)
    penalty_stream = torch.Generator(device=device).manual_seed(SEED + 3)
    data_stream = torch.Generator(device=device).manual_seed(SEED)
    means = mode_hold.ring_means().to(device)

    @torch.no_grad()
    def sample(n, ema=False, seed=SEED + 9):
        s = torch.Generator(device=device).manual_seed(seed)
        g, pr = (ema_G, ema_prior) if ema else (G, prior)
        was = [(m, m.training) for m in (*g.modules(), *pr.modules())]
        g.eval()
        pr.eval()
        try:
            z, _ = pr.sample(n, generator=s)
            return g(z)
        finally:
            for m, f in was:
                m.training = f

    def probe_now(fake):
        s = torch.Generator(device=device).manual_seed(SEED + 11)
        idx = torch.randint(0, 8, (PROBE_N,), device=device, generator=s)
        real = means[idx] + mode_hold.SIGMA * torch.randn(PROBE_N, 2, device=device, generator=s)
        u = torch.rand(PROBE_N, device=device, generator=s)
        return probe(D, real, fake, u)

    initial_rates = rates(opt_g, opt_d, roles)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    declaration = {
        "schema": 1, "experiment": "simple_critic_shift", "arm": args.arm, "formulation": describe(args),
        "config": config, "seed": SEED, "recipe": recipe.to_dict(), "initial_rates": initial_rates,
        "noise": "none (input_noise_std=0, output_noise_std=0)", "lr": "constant; checked every update",
        "critic_optimizer": "recipe.make_critic_optimizer, guard off, no EMA critic (Adam + inert bookkeeping)",
        "generator_side": f"public trainer update, g_loss={args.g_loss}",
        "particlegan_file": particlegan.__file__, "torch": torch.__version__,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                     capture_output=True).stdout.strip(),
    }
    (out_dir / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")

    points, started = [], time.monotonic()
    frozen_post_shift = None
    critic_step = 0
    lr_rows = []
    with log_path.open("w", buffering=1) as log, (out_dir / "metrics.jsonl").open("w", buffering=1) as metrics:
        log.write(f"# arm={args.arm} formulation={describe(args)} device={args.device} steps={args.steps}\n")
        log.write("# step phase modes hq pass | Dr mean/max |Dr|max Df | grad-norm real fake path(min) max | "
                  "loss_d terms | loss_g | sec\n")
        for step in range(1, args.steps + 1):
            net_scale, prior_scale = learning_rate_scales(step - 1, recipe)
            if net_scale != 1.0 or prior_scale != 1.0:
                raise RuntimeError(f"recipe LR schedule not constant at {step}: {net_scale}, {prior_scale}")
            idx = torch.randint(0, 8, (recipe.batch_size,), device=device, generator=data_stream)
            real = means[idx] + mode_hold.SIGMA * torch.randn(recipe.batch_size, 2, device=device,
                                                               generator=data_stream)
            # ---- critic step(s)
            D.train()
            G.eval()
            for _ in range(args.n_critic):
                with torch.no_grad():
                    z, _ = prior.sample(len(real), generator=latent_stream)
                    fake = G(z)
                u = torch.rand(len(real), device=device, generator=penalty_stream)
                critic_step += 1
                loss_d, terms = critic_loss(D, real, fake, u, critic_step)
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            # ---- generator / prior step (public trainer's update)
            D.eval()
            G.train()
            D.requires_grad_(False)
            z, indices = prior.sample(len(real), generator=latent_stream)
            fake_logits = D(G(z))
            real_logits = D(real)
            loss_gan = g_loss_fn(args.g_loss, base_loss, fake_logits, real_logits)
            loss_g = loss_gan
            if recipe.prior_reg and prior.z.requires_grad:
                raw = prior.z[torch.unique(indices)]
                loss_g = loss_g + recipe.prior_reg * prior_regularizer(raw)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            D.requires_grad_(True)
            with torch.no_grad():
                for tgt, src in ((ema_G, G), (ema_prior, prior)):
                    for a, c in zip(tgt.parameters(), src.parameters()):
                        a.mul_(recipe.ema_decay).add_(c, alpha=1 - recipe.ema_decay)
                    for a, c in zip(tgt.buffers(), src.buffers()):
                        a.copy_(c)
            actual = rates(opt_g, opt_d, roles)
            if actual != initial_rates:
                raise RuntimeError(f"LR changed at update {step}: {actual} != {initial_rates}")
            lr_rows.append(step)
            # ---- observation
            if step % OBSERVE_EVERY == 0 or step == args.steps:
                fake_eval = sample(mode_hold.EVAL_N)
                point = {"step": step, **mode_hold.diversity(fake_eval, means)}
                point["pass"] = good(point)
                point.update(probe_now(fake_eval))
                losses = {"loss_d": float(loss_d.detach()), "loss_g": float(loss_g.detach()),
                          **{k: float(v) for k, v in terms.items()}}
                if not all(math.isfinite(v) for v in (*losses.values(), point["dr_absmax"], point["g_max"])):
                    log.write(f"# NONFINITE at {step}: {losses}\n")
                    raise RuntimeError(f"Nonfinite values at update {step}: {losses}")
                points.append({k: point[k] for k in ("step", "modes", "hq", "pass", "dr_mean", "dr_max",
                                                     "dr_absmax", "df_mean", "g_real", "g_fake", "g_path",
                                                     "g_path_min", "g_max")})
                metrics.write(json.dumps({**point, "losses": losses, "lr": actual}) + "\n")
                phase = "post" if step > SHIFT_STEP else ("hold" if step >= PREHOLD[0] else "acq")
                tstr = " ".join(f"{k}={v:.3g}" for k, v in terms.items() if k == "base" or float(v) != 0)
                log.write(f"{step:5d} {phase:4s} m={point['modes']} hq={point['hq']:.3f} "
                          f"{'PASS' if point['pass'] else 'fail'} | Dr={point['dr_mean']:+.3f}/{point['dr_max']:+.3f} "
                          f"|Dr|max={point['dr_absmax']:.3f} Df={point['df_mean']:+.3f} | "
                          f"g r={point['g_real']:.3f} f={point['g_fake']:.3f} p={point['g_path']:.3f}"
                          f"({point['g_path_min']:.3f}) max={point['g_max']:.3f} | "
                          f"Ld={float(loss_d.detach()):+.4f} {tstr} | Lg={float(loss_g.detach()):+.4f} | "
                          f"{time.monotonic() - started:.0f}s\n")
            if step == SHIFT_STEP:
                torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict()},
                           out_dir / "shift-models.pt")
                means.add_(means.new_tensor([1., 0.]))
                # A frozen copy's samples are fixed (fixed eval stream), so one check after the shift suffices.
                frozen_post_shift = mode_hold.diversity(sample(mode_hold.EVAL_N), means)
                log.write(f"# shift (1,0) after {step}; frozen copy on new target: m={frozen_post_shift['modes']} "
                          f"hq={frozen_post_shift['hq']:.3f}\n")
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict()},
                   out_dir / "final-models.pt")
        result = {"schema": 1, "status": "COMPLETE", "arm": args.arm, "formulation": describe(args),
                  "config": config, "completed_steps": args.steps, "seconds": time.monotonic() - started,
                  "device": args.device, "torch": torch.__version__,
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  "learning_rates": initial_rates, "constant_lr_verified_updates": len(lr_rows),
                  "stationary": window(points, 1000, 1200, 50),
                  "prehold": window(points, *PREHOLD),
                  "recovery_extended": recovery(points, args.steps) if args.steps > SHIFT_STEP else None,
                  "frozen_post_shift": frozen_post_shift,
                  "final": points[-1], "final_ema": mode_hold.diversity(sample(mode_hold.EVAL_N, ema=True), means),
                  "points": points}
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
        log.write(f"# COMPLETE {args.steps} updates in {result['seconds']:.0f}s; prehold "
                  f"{result['prehold']['passing_checks']}/{result['prehold']['checks']}; "
                  f"arrival={None if not result['recovery_extended'] else result['recovery_extended']['updates_to_first_pass']}\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="arm name; logs/<arm>.log and runs/<arm>/")
    p.add_argument("--loss", choices=("wgan", "hinge", "logistic", "rplogistic"), default="wgan")
    p.add_argument("--real", choices=("none", "drift", "r1", "drift+r1"), default="none")
    p.add_argument("--lam-real", type=float, default=1.0, help="weight of the real term(s)")
    p.add_argument("--lam-drift", type=float, default=None, help="drift weight when it differs from --lam-real")
    p.add_argument("--path", choices=("none", "lower", "two_sided", "secant"), default="none",
                   help="secant: relu(t*|r_nn-f| - (D(r_nn)-D(f)))^2 with r_nn the nearest real of each fake")
    p.add_argument("--path-u", default="0,1", help="lo,hi window for path points u (lower/two_sided/cap)")
    p.add_argument("--d-beta2", type=float, default=None, help="critic Adam beta2 (recipe default .999)")
    p.add_argument("--lam-path", type=float, default=10.0)
    p.add_argument("--path-target", type=float, default=1.0)
    p.add_argument("--cap", choices=("none", "interp", "all"), default="none")
    p.add_argument("--lam-cap", type=float, default=10.0)
    p.add_argument("--cap-target", type=float, default=1.0)
    p.add_argument("--g-loss", choices=("rpgan", "wgan", "ns"), default="rpgan",
                   help="generator loss; rpgan is the public trainer's")
    p.add_argument("--n-critic", type=int, default=1, help="critic updates per G update (fresh fakes)")
    p.add_argument("--d-lr-mult", type=float, default=1.0, help="constant critic LR multiplier")
    p.add_argument("--latent-damping", type=float, default=0.5, help="A2 particle-row damping (recipe default .5)")
    p.add_argument("--lam-center", type=float, default=0.0,
                   help="level pin lam*(E D(real))^2 (batch mean only; 0 = off)")
    p.add_argument("--lazy-k", type=int, default=1,
                   help="apply all non-base terms every k-th critic step with weight x k (1 = every step)")
    p.add_argument("--steps", type=int, default=4600)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.steps <= 0 or args.n_critic < 1 or args.lazy_k < 1:
        p.error("--steps, --n-critic and --lazy-k must be positive")
    run(args)


if __name__ == "__main__":
    main()
