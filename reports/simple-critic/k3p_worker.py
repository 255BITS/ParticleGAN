"""K3P (public GANTrainer) on the simple-critic ring-8 shift protocol.

Adapted from reports/ka2-default-candidate/constant-lr-api/worker.py. Data, arch,
construction order, seed 0, (1,0) shift after 2400, 4600 updates, observation every 10
updates on 4096 samples (live G, stream seed+9) are unchanged. particlegan and
benchmarks.locked_shared are imported from the K3P checkout named by --k3p-root
(default: .claude/worktrees/k3p-develop = origin/develop, the released K3P formulation with
the #194 batch_feature_zero default init applied by GANTrainer; pass
--k3p-root .claude/worktrees/k3p-master for the archived v0.8.0 random-init replay).
The K3P_ROOT environment variable is honored when the flag is absent.

Arms:
  stock     get_recipe(total_steps=4600): K3P exactly as released, its own noise
            and LR schedules over the 4600-update horizon.
  constant  same recipe with input/output noise std 0 and lr_floor =
            network_lr_floor = 1 (constant LRs, checked every update). K3P's own
            critic penalty, EMA anchor and spike guard are kept.

Observation-only probe: the simple-critic worker's D-value / input-gradient-norm
probe (real, fake, path points; own RNG stream), so max grad-norm is comparable.
Tail logs/<arm>.log. Writes runs/<arm>/{declaration.json,metrics.jsonl,result.json}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
_WORKTREES = Path("/home/martyn/dev/ParticleGAN/.claude/worktrees")


def _k3p_root(argv):
    # Resolved before importing particlegan: the flag picks the package checkout.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--k3p-root", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    return (known.k3p_root or Path(os.environ.get("K3P_ROOT", _WORKTREES / "k3p-develop"))).resolve()


K3P_ROOT = _k3p_root(sys.argv[1:])
sys.path.insert(0, str(K3P_ROOT))
sys.path.insert(1, str(HERE))

import torch

import particlegan
from particlegan import GANTrainer, get_recipe
from particlegan.training import input_noise_std, output_noise_std
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator

import init_receipt  # noqa: E402

assert Path(particlegan.__file__).resolve().is_relative_to(K3P_ROOT), particlegan.__file__
K3P_COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=K3P_ROOT, text=True,
                            capture_output=True).stdout.strip()
K3P_LABEL = f"{K3P_ROOT.name}@{K3P_COMMIT}"

SEED = 0
SHIFT_STEP = 2400
OBSERVE_EVERY = 10
PROBE_N = mode_hold.EVAL_N
FORMULATION = {
    "stock": f"REF: K3P {K3P_LABEL} as released (RpGAN+K3P penalty/anchor/guard, own noise + LR schedules)",
    "constant": f"REF: K3P {K3P_LABEL} critic/penalty, noise off, constant LRs (floors 1)",
}


def make_recipe(arm, steps):
    if arm == "stock":
        return get_recipe(total_steps=steps)
    return get_recipe(total_steps=steps, input_noise_std=0.0, output_noise_std=0.0,
                      lr_floor=1.0, network_lr_floor=1.0)


def make_trainer(recipe, device):
    # Construction order of the public baseline: networks on CPU, then moved.
    torch.manual_seed(SEED)
    generator = SimpleMLPGenerator(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2).to(device)
    critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER).to(device)
    return GANTrainer(recipe, generator, critic, seed=SEED,
                      optimizer_options={"foreach": False, "fused": False})


def measure(trainer, means, *, ema=False):
    stream = torch.Generator(device=trainer.device).manual_seed(SEED + 9)
    return mode_hold.diversity(trainer.sample(mode_hold.EVAL_N, ema=ema, generator=stream), means)


def good(point):
    return point["modes"] == 8 and .90 <= point["hq"] <= 1.0


def grad_norm(g):
    return torch.sqrt(g.pow(2).flatten(1).sum(1) + 1e-12)


def probe(D, real, fake, u):
    """Copied from simple-critic worker.py: D values and input-gradient norms (no training effect)."""
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


def rates(trainer):
    return {f"{role}_{i}": float(group["lr"])
            for optimizer, roles in zip((trainer.opt_g, trainer.opt_d), trainer.roles)
            for i, (group, role) in enumerate(zip(optimizer.param_groups, roles))}


def run(args):
    arm_name = f"k3p_{'stock_ref' if args.arm == 'stock' else 'constant'}"
    out_dir = getattr(args, "output", None) or HERE / "runs" / arm_name
    log_path = getattr(args, "log", None) or HERE / "logs" / f"{arm_name}.log"
    if (out_dir / "result.json").exists():
        raise SystemExit(f"{out_dir}/result.json exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recipe = make_recipe(args.arm, args.steps)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    trainer = make_trainer(recipe, args.device)
    device = trainer.device
    stream = torch.Generator(device=args.device).manual_seed(SEED)
    means = mode_hold.ring_means().to(args.device)
    # Init: whatever the checkout's public path does (GANTrainer -> recipe.make_prior +
    # recipe.make_optimizers with the trainer's EMA critic); the receipt checks it against a fresh
    # public-path build and against an initialization=None build (pre-#194 random init).
    init = init_receipt.ring_receipt(
        recipe, trainer.G, trainer.D, trainer.prior, SimpleMLPGenerator, SimpleMLPDiscriminator, mode_hold,
        ema_D=trainer.ema_D, applied_via=f"GANTrainer ({K3P_LABEL})") \
        if hasattr(recipe, "initialization") else {"initialization": "pre-#194 random (no initialization field)"}
    initial_rates = rates(trainer)
    declaration = {
        "schema": 1, "experiment": "k3p_simple_critic_shift", "arm": arm_name,
        "formulation": FORMULATION[args.arm], "seed": SEED, "recipe": recipe.to_dict(),
        "training_api": f"particlegan.get_recipe + particlegan.GANTrainer.step ({K3P_LABEL})",
        "k3p_root": str(K3P_ROOT), "init_receipt": init,
        "particlegan_file": particlegan.__file__, "torch": torch.__version__,
        "k3p_git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=K3P_ROOT, text=True,
                                         capture_output=True).stdout.strip(),
        "initial_rates": initial_rates, "device": args.device,
    }
    (out_dir / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    points, started = [], time.monotonic()
    lr_ranges = {k: [v, v] for k, v in initial_rates.items()}
    with log_path.open("w", buffering=1) as log, (out_dir / "metrics.jsonl").open("w", buffering=1) as metrics:
        log.write(f"# arm={arm_name} formulation={FORMULATION[args.arm]} device={args.device} steps={args.steps}\n"
                  f"# particlegan={particlegan.__file__}\n"
                  "# step phase modes hq pass | Dr mean/max |Dr|max Df | grad-norm real fake path(min) max | "
                  "Ld pen s | Lg | lr G/D/prior | noise in/out | sec\n")
        if "param_sha256" in init:
            log.write(init_receipt.summary_line(init))
        for step in range(1, args.steps + 1):
            idx = torch.randint(0, 8, (recipe.batch_size,), device=args.device, generator=stream)
            real = means[idx] + mode_hold.SIGMA * torch.randn(
                recipe.batch_size, 2, device=args.device, generator=stream)
            observe = step % OBSERVE_EVERY == 0 or step == args.steps
            stats = trainer.step(real, collect_stats=observe)
            actual = rates(trainer)
            for k, v in actual.items():
                lr_ranges[k] = [min(lr_ranges[k][0], v), max(lr_ranges[k][1], v)]
            if args.arm == "constant" and actual != initial_rates:
                raise RuntimeError(f"LR changed at update {step}: {actual} != {initial_rates}")
            if observe:
                fake_eval = trainer.sample(mode_hold.EVAL_N, generator=torch.Generator(
                    device=device).manual_seed(SEED + 9))
                point = {"step": step, **mode_hold.diversity(fake_eval, means)}
                point["pass"] = good(point)
                s = torch.Generator(device=device).manual_seed(SEED + 11)
                pidx = torch.randint(0, 8, (PROBE_N,), device=device, generator=s)
                preal = means[pidx] + mode_hold.SIGMA * torch.randn(PROBE_N, 2, device=device, generator=s)
                pu = torch.rand(PROBE_N, device=device, generator=s)
                point.update(probe(trainer.D, preal, fake_eval, pu))
                losses = {k: float(v) for k, v in stats.items() if isinstance(v, torch.Tensor)}
                if not all(math.isfinite(v) for v in (*losses.values(), point["dr_absmax"], point["g_max"])):
                    log.write(f"# NONFINITE at {step}: {losses}\n")
                    raise RuntimeError(f"Nonfinite values at update {step}: {losses}")
                pen = stats["penalty_stats"] or {}
                noise = (input_noise_std(recipe, step - 1), output_noise_std(recipe, step - 1))
                points.append({k: point[k] for k in ("step", "modes", "hq", "pass", "dr_mean", "dr_max",
                                                     "dr_absmax", "df_mean", "g_real", "g_fake", "g_path",
                                                     "g_path_min", "g_max")})
                metrics.write(json.dumps({**point, "losses": losses, "penalty": pen,
                                          "controller": trainer.penalty.diagnostics(), "lr": actual,
                                          "input_noise": noise[0], "output_noise": noise[1]},
                                         default=float) + "\n")
                phase = "post" if step > SHIFT_STEP else ("hold" if step >= 1210 else "acq")
                lr = "/".join(f"{actual.get(k, float('nan')):.3g}" for k in ("generator_0", "critic_0", "prior_1"))
                log.write(f"{step:5d} {phase:4s} m={point['modes']} hq={point['hq']:.3f} "
                          f"{'PASS' if point['pass'] else 'fail'} | Dr={point['dr_mean']:+.3f}/{point['dr_max']:+.3f} "
                          f"|Dr|max={point['dr_absmax']:.3f} Df={point['df_mean']:+.3f} | "
                          f"g r={point['g_real']:.3f} f={point['g_fake']:.3f} p={point['g_path']:.3f}"
                          f"({point['g_path_min']:.3f}) max={point['g_max']:.3f} | "
                          f"Ld={losses['loss_d']:+.4f} pen={losses['penalty']:.3g} s={pen.get('s', float('nan')):.3g} | "
                          f"Lg={losses['loss_g']:+.4f} | lr {lr} | noise {noise[0]:.3g}/{noise[1]:.3g} | "
                          f"{time.monotonic() - started:.0f}s\n")
            if step == SHIFT_STEP:
                means.add_(means.new_tensor([1., 0.]))
                log.write(f"# SHIFT (1,0) after update {step}\n")
        result = {"schema": 1, "status": "COMPLETE", "arm": arm_name, "formulation": FORMULATION[args.arm],
                  "completed_steps": trainer.completed_steps, "seconds": time.monotonic() - started,
                  "device": args.device, "torch": torch.__version__, "particlegan_file": particlegan.__file__,
                  "k3p_git_commit": declaration["k3p_git_commit"], "k3p_root": str(K3P_ROOT),
                  "init_receipt": init, "recipe": recipe.to_dict(),
                  "lr_ranges": lr_ranges, "constant_lr_verified_every_step": args.arm == "constant",
                  "final_ema": measure(trainer, means, ema=True), "points": points}
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
        log.write(f"# COMPLETE {time.monotonic() - started:.0f}s\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("stock", "constant"), required=True)
    parser.add_argument("--steps", type=int, default=4600)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--k3p-root", type=Path, default=None,
                        help="K3P checkout to import (default: $K3P_ROOT or .claude/worktrees/k3p-develop)")
    parser.add_argument("--output", type=Path, default=None, help="run dir (default runs/<arm>)")
    parser.add_argument("--log", type=Path, default=None, help="log file (default logs/<arm>.log)")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % OBSERVE_EVERY:
        parser.error("--steps must be a positive multiple of 10")
    run(args)


if __name__ == "__main__":
    main()
