"""secant_r1_b2 + K3P v0.8.0's critic spike guard and/or EMA anchor (ring-8 shift, seed 0).

The critic formulation, data, init, shift, cadence, evaluator, constant LRs, A2 damping
and critic Adam beta2 0.9 are exactly worker.py's secant_r1_b2 (its SimpleCriticLoss,
probe and scoring are imported read-only). The critic optimizer is the same recipe
factory with guard off and no EMA critic; the K3P mechanisms are added outside it:

* guard  particlegan.k3p.CriticSpikeGuard (unchanged since v0.8.0), K3P defaults
         ratio=5, min_steps=200: between backward() and opt_d.step(), a critic tensor
         with >=200 Adam steps whose grad RMS exceeds 5*sqrt(mean bias-corrected v) is
         scaled down to that ratio.
* anchor K3P's EMA-anchor ("prox") term, anchor_weight * mean ||g_r - gbar_r||^2 / d,
         g = grad_x D(real), gbar = grad_x Dbar(real), Dbar = parameter EMA of D
         (RobustCriticAnchor, K3P decay 0.999), entering the loss with K3P's coeff/2
         (reg_coeff 1, anchor_weight 1 -> 0.5 * prox). In K3P the term only exists in
         the annealed-LR phase (s < 1), so under constant LRs (k3p_constant) it is
         inert; here it is active from critic step 1 (Dbar starts as the initial D,
         updated after every critic step).

One arm = one formulation. Tail logs/<arm>.log.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import worker as W  # noqa: E402  (read-only reuse; sets up ROOT on sys.path)

import torch  # noqa: E402

import particlegan  # noqa: E402
from particlegan.k3p import CriticSpikeGuard, RobustCriticAnchor  # noqa: E402
from particlegan.recipes import learning_rate_scales  # noqa: E402
from benchmarks.locked_shared import mode_hold  # noqa: E402
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator  # noqa: E402

SEED, SHIFT_STEP, OBSERVE_EVERY, PREHOLD = W.SEED, W.SHIFT_STEP, W.OBSERVE_EVERY, W.PREHOLD
# secant_r1_b2's resolved config (runs/secant_r1_b2/declaration.json)
SECANT_R1_B2 = dict(loss="wgan", real="r1", lam_real=1.0, lam_drift=None, path="secant", path_u="0.1,0.9",
                    d_beta2=0.9, lam_path=10.0, path_target=0.5, cap="all", lam_cap=10.0, cap_target=1.0,
                    g_loss="rpgan", n_critic=1, d_lr_mult=1.0, latent_damping=0.5, lam_center=0.0, lazy_k=1)
# K3P v0.8.0 recipe defaults (d_guard_ratio, d_guard_min_steps, reg_anchor_decay, reg_anchor_weight, reg_coeff)
K3P = dict(guard_ratio=5.0, guard_min_steps=200, anchor_decay=0.999, anchor_weight=1.0, reg_coeff=1.0)


def describe(args):
    extra = []
    if args.guard:
        extra.append(f"K3P guard(ratio={K3P['guard_ratio']:g},min={K3P['guard_min_steps']})")
    if args.anchor:
        extra.append(f"K3P EMA-anchor({K3P['reg_coeff'] / 2 * K3P['anchor_weight']:g}*prox,decay={K3P['anchor_decay']:g})")
    return W.describe(args) + "".join(" + " + e for e in extra)


def run(args):
    recipe = W.make_recipe(args)  # guard 0, anchor weight 0, noise 0, LR floors 1
    out_dir = args.output or HERE / "runs" / args.arm
    log_path = args.log or HERE / "logs" / f"{args.arm}.log"
    if (out_dir / "result.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir}/result.json exists; pass --overwrite or pick a new --arm")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(SEED)
    G = SimpleMLPGenerator(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2).to(device)
    D = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER).to(device)
    prior = recipe.make_prior().to(device)
    opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=None, foreach=False, fused=False)
    assert opt_d.guard is None and opt_d.ema_critic is None, "recipe critic optimizer must carry no controller"
    for group in opt_d.param_groups:
        group["betas"] = (group["betas"][0], args.d_beta2)
    guard = CriticSpikeGuard(ratio=K3P["guard_ratio"], min_steps=K3P["guard_min_steps"]) if args.guard else None
    anchor = None
    if args.anchor:
        ema_D = copy.deepcopy(D)
        ema_D.requires_grad_(False)
        anchor = RobustCriticAnchor(D, ema_D, decay=K3P["anchor_decay"])
        anchor.start_()
    prox_w = K3P["reg_coeff"] / 2 * K3P["anchor_weight"]
    prior_ids = {id(p) for p in prior.parameters()}
    roles = [["prior" if any(id(p) in prior_ids for p in g["params"]) else "generator" for g in opt_g.param_groups],
             ["critic"] * len(opt_d.param_groups)]
    base_loss = recipe.make_loss()
    prior_regularizer = recipe.make_prior_regularizer(weight=1.0)
    critic_loss = W.SimpleCriticLoss(args)
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
        idx = torch.randint(0, 8, (W.PROBE_N,), device=device, generator=s)
        real = means[idx] + mode_hold.SIGMA * torch.randn(W.PROBE_N, 2, device=device, generator=s)
        u = torch.rand(W.PROBE_N, device=device, generator=s)
        return W.probe(D, real, fake, u)

    # Init: recipe default via recipe.make_prior + recipe.make_optimizers; the anchor's EMA critic is
    # deep-copied from the already-initialized D.
    init = W.init_receipt.ring_receipt(
        recipe, G, D, prior, SimpleMLPGenerator, SimpleMLPDiscriminator, mode_hold,
        ema_D=None if anchor is None else ema_D,
        applied_via="recipe.make_prior + recipe.make_optimizers(G, D, prior); anchor EMA = deepcopy(D) after")
    initial_rates = W.rates(opt_g, opt_d, roles)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    mechanisms = {"guard": None if guard is None else {"class": "particlegan.k3p.CriticSpikeGuard",
                                                       "ratio": guard.ratio, "min_steps": guard.min_steps,
                                                       "applied": "between backward() and opt_d.step()"},
                  "anchor": None if anchor is None else {
                      "class": "particlegan.k3p.RobustCriticAnchor", "decay": K3P["anchor_decay"],
                      "anchor_weight": K3P["anchor_weight"], "reg_coeff": K3P["reg_coeff"], "loss_weight": prox_w,
                      "term": "loss_weight * mean ||grad D(real) - grad Dbar(real)||^2 / d",
                      "start": "critic step 1 (Dbar = initial D), EMA update after every critic step"}}
    declaration = {
        "schema": 1, "experiment": "simple_critic_shift", "arm": args.arm, "formulation": describe(args),
        "config": config, "seed": SEED, "recipe": recipe.to_dict(), "initial_rates": initial_rates,
        "k3p_mechanisms": mechanisms, "noise": "none", "lr": "constant; checked every update",
        "init_receipt": init,
        "particlegan_file": particlegan.__file__, "torch": torch.__version__,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=W.ROOT, text=True,
                                     capture_output=True).stdout.strip(),
    }
    (out_dir / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")

    points, started, frozen_post_shift, critic_step, lr_rows = [], time.monotonic(), None, 0, 0
    dim = 2
    with log_path.open("w", buffering=1) as log, (out_dir / "metrics.jsonl").open("w", buffering=1) as metrics:
        log.write(f"# arm={args.arm} formulation={describe(args)} device={args.device} steps={args.steps}\n")
        log.write("# step phase modes hq pass | Dr mean/max |Dr|max Df | grad-norm real fake path(min) max | "
                  "loss_d terms | loss_g | guard clips (cum) | sec\n")
        log.write(W.init_receipt.summary_line(init))
        for step in range(1, args.steps + 1):
            net_scale, prior_scale = learning_rate_scales(step - 1, recipe)
            if net_scale != 1.0 or prior_scale != 1.0:
                raise RuntimeError(f"recipe LR schedule not constant at {step}")
            idx = torch.randint(0, 8, (recipe.batch_size,), device=device, generator=data_stream)
            real = means[idx] + mode_hold.SIGMA * torch.randn(recipe.batch_size, 2, device=device,
                                                               generator=data_stream)
            D.train()
            G.eval()
            with torch.no_grad():
                z, _ = prior.sample(len(real), generator=latent_stream)
                fake = G(z)
            u = torch.rand(len(real), device=device, generator=penalty_stream)
            critic_step += 1
            loss_d, terms = critic_loss(D, real, fake, u, critic_step)
            if anchor is not None:
                x = real.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(D(x).sum(), x, create_graph=True)[0]
                xb = real.detach().clone().requires_grad_(True)
                gb = torch.autograd.grad(anchor(xb).sum(), xb)[0].detach()
                t = prox_w * (g - gb).pow(2).flatten(1).sum(1).mean() / dim
                loss_d = loss_d + t
                terms["anchor"] = t.detach()
            opt_d.zero_grad()
            loss_d.backward()
            if guard is not None:
                guard.apply_(opt_d)
            opt_d.step()
            if anchor is not None:
                anchor.update_()
            D.eval()
            G.train()
            D.requires_grad_(False)
            z, indices = prior.sample(len(real), generator=latent_stream)
            fake_logits = D(G(z))
            real_logits = D(real)
            loss_g = W.g_loss_fn(args.g_loss, base_loss, fake_logits, real_logits)
            if recipe.prior_reg and prior.z.requires_grad:
                loss_g = loss_g + recipe.prior_reg * prior_regularizer(prior.z[torch.unique(indices)])
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
            actual = W.rates(opt_g, opt_d, roles)
            if actual != initial_rates:
                raise RuntimeError(f"LR changed at update {step}: {actual} != {initial_rates}")
            lr_rows += 1
            if step % OBSERVE_EVERY == 0 or step == args.steps:
                fake_eval = sample(mode_hold.EVAL_N)
                point = {"step": step, **mode_hold.diversity(fake_eval, means)}
                point["pass"] = W.good(point)
                point.update(probe_now(fake_eval))
                clips = None if guard is None else guard.clipped_tensors
                losses = {"loss_d": float(loss_d.detach()), "loss_g": float(loss_g.detach()),
                          **{k: float(v) for k, v in terms.items()}}
                if not all(math.isfinite(v) for v in (*losses.values(), point["dr_absmax"], point["g_max"])):
                    log.write(f"# NONFINITE at {step}: {losses}\n")
                    raise RuntimeError(f"Nonfinite values at update {step}: {losses}")
                keep = {k: point[k] for k in ("step", "modes", "hq", "pass", "dr_mean", "dr_max", "dr_absmax",
                                              "df_mean", "g_real", "g_fake", "g_path", "g_path_min", "g_max")}
                keep["guard_clips"] = clips
                points.append(keep)
                metrics.write(json.dumps({**point, "losses": losses, "lr": actual, "guard_clips": clips}) + "\n")
                phase = "post" if step > SHIFT_STEP else ("hold" if step >= PREHOLD[0] else "acq")
                tstr = " ".join(f"{k}={v:.3g}" for k, v in terms.items() if k == "base" or float(v) != 0)
                log.write(f"{step:5d} {phase:4s} m={point['modes']} hq={point['hq']:.3f} "
                          f"{'PASS' if point['pass'] else 'fail'} | Dr={point['dr_mean']:+.3f}/{point['dr_max']:+.3f} "
                          f"|Dr|max={point['dr_absmax']:.3f} Df={point['df_mean']:+.3f} | "
                          f"g r={point['g_real']:.3f} f={point['g_fake']:.3f} p={point['g_path']:.3f}"
                          f"({point['g_path_min']:.3f}) max={point['g_max']:.3f} | "
                          f"Ld={float(loss_d.detach()):+.4f} {tstr} | Lg={float(loss_g.detach()):+.4f} | "
                          f"clips={'-' if clips is None else clips} | {time.monotonic() - started:.0f}s\n")
            if step == SHIFT_STEP:
                means.add_(means.new_tensor([1., 0.]))
                frozen_post_shift = mode_hold.diversity(sample(mode_hold.EVAL_N), means)
                log.write(f"# shift (1,0) after {step}; frozen copy on new target: m={frozen_post_shift['modes']} "
                          f"hq={frozen_post_shift['hq']:.3f}\n")
        result = {"schema": 1, "status": "COMPLETE", "arm": args.arm, "formulation": describe(args),
                  "config": config, "k3p_mechanisms": mechanisms, "init_receipt": init,
                  "guard_clipped_tensors": None if guard is None else guard.clipped_tensors,
                  "completed_steps": args.steps, "seconds": time.monotonic() - started,
                  "device": args.device, "torch": torch.__version__,
                  "learning_rates": initial_rates, "constant_lr_verified_updates": lr_rows,
                  "stationary": W.window(points, 1000, 1200, 50), "prehold": W.window(points, *PREHOLD),
                  "recovery_extended": W.recovery(points, args.steps) if args.steps > SHIFT_STEP else None,
                  "frozen_post_shift": frozen_post_shift, "final": points[-1],
                  "final_ema": mode_hold.diversity(sample(mode_hold.EVAL_N, ema=True), means), "points": points}
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
        rec = result["recovery_extended"]
        log.write(f"# COMPLETE {args.steps} updates in {result['seconds']:.0f}s; prehold "
                  f"{result['prehold']['passing_checks']}/{result['prehold']['checks']}; "
                  f"arrival={None if not rec else rec['updates_to_first_pass']}; clips={result['guard_clipped_tensors']}\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--guard", action="store_true", help="K3P critic spike guard (ratio 5, min 200 steps)")
    p.add_argument("--anchor", action="store_true", help="K3P EMA-anchor prox term, active from step 1")
    p.add_argument("--steps", type=int, default=4600)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    for k, v in SECANT_R1_B2.items():
        setattr(args, k, v)
    run(args)


if __name__ == "__main__":
    main()
