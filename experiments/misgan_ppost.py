#!/usr/bin/env python
"""Particle-posterior imputation (ppost) with trained G_x, and imputation cost.

No new training. For each finished run in results/misgan/runs that has a checkpoint:

* cost: wall-clock of the run's own imputer, 16 draws on the 10k test rows,
  in ms per 1k rows. Runs one at a time on an otherwise idle GPU.
* ppost (runs in PPOST_SOURCES): draw M latents from G_x's EMA prior and
  compute x_k = G_x(z_k). Each test row resamples k with
  w_k ~ exp(-||m (x_k - x_obs)||^2 / 2 sigma^2) and fills its missing
  coordinates from x_k. sigma maximizes the log-likelihood of one held-out
  observed coordinate per training row (4,096 training rows, no test data).
  The grid is M in {256, 4096}, each without and with refinement: 20 Adam steps
  on z (the recipe's generator optimizer, lr 0.01) fitting the observed
  coordinates from the resampled particle.

Writes <run>/posthoc.json and logs one line per (run, config).

    python experiments/misgan_ppost.py [--runs results/misgan/runs]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from particlegan import get_recipe  # noqa: E402

from experiments.misgan_toy import imputer_net, impute  # noqa: E402
from lib.misgan import (DIM, Problem, bayes_posterior, imputation_metrics, ppost_indices,  # noqa: E402
                        refine_latents, select_sigma)
from lib.toy_models import SimpleMLPGenerator  # noqa: E402

PPOST_SOURCES = ("oracle", "misgan", "misgan_realmask", "misgan_detach", "misgan_gauss",
                 "aegan_recon_w0.1", "aegan_recon_w1", "aegan_ce")
SIZES = (256, 4096)
REFINE_STEPS = 20


def timed(fn, device):
    """(result, seconds) with CUDA synchronized around the call."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - start


def load(run_dir, device):
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg = summary["config"]
    arm = cfg["arm"]
    recipe = get_recipe(total_steps=cfg["total_steps"], batch_size=cfg["batch_size"], **cfg["recipe_overrides"])
    learnable = arm != "misgan_gauss"
    state = torch.load(run_dir / "ckpt.pt", map_location=device)
    gx = SimpleMLPGenerator(z_dim=recipe.z_dim, out_dim=DIM).to(device)
    gx.load_state_dict(state["gx"])
    prior_x = recipe.make_prior(z_dim=recipe.z_dim, learnable=learnable).to(device)
    prior_x.load_state_dict(state["prior_x"])
    net, net_z = imputer_net(arm, cfg, recipe)
    net = net.to(device)
    net.load_state_dict(state["imputer"])
    prior_i = None
    if net_z is not None:
        prior_i = recipe.make_prior(z_dim=net_z, learnable=learnable).to(device)
        prior_i.load_state_dict(state["imputer_prior"])
    for module in (gx, prior_x, net, prior_i):
        if module is not None:
            module.eval().requires_grad_(False)
    return cfg, recipe, gx, prior_x, net, prior_i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="results/misgan/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    problems = {}
    for run_dir in sorted(p.parent for p in Path(args.runs).glob("*/ckpt.pt")):
        label = run_dir.name.split("__", 1)[-1]
        cfg, recipe, gx, prior_x, net, prior_i = load(run_dir, device)
        key = (cfg["mechanism"], cfg["n_train"], cfg["n_test"], cfg["data_seed"])
        if key not in problems:
            problem = Problem(cfg["mechanism"], cfg["n_train"], cfg["n_test"], cfg["data_seed"], device)
            problems[key] = problem, bayes_posterior(problem, problem.x_test, problem.m_test)[0]
        problem, post = problems[key]
        x, m, draws = problem.x_test, problem.m_test, cfg["impute_draws"]
        per_1k = 1000.0 / len(x)
        out = {}

        def own():
            gen = torch.Generator(device=device).manual_seed(1)
            with torch.no_grad():
                return torch.stack([impute(cfg["arm"], net, prior_i, gx, prior_x, x, m, gen) for _ in range(draws)])
        own()  # warm-up
        _, seconds = timed(own, device)
        out["cost_ms_1k"] = 1000 * seconds * per_1k
        print(f"cost {run_dir.name} ms_1k={out['cost_ms_1k']:.1f}", flush=True)
        if label in PPOST_SOURCES:
            out["ppost"] = {}
            for M in SIZES:
                gen = torch.Generator(device=device).manual_seed(M)
                with torch.no_grad():
                    z, _ = prior_x.sample(M, generator=gen)
                    table = gx(z)
                (sigma, cv_ll), fit_s = timed(lambda: select_sigma(
                    table, problem.x_train[:4096], problem.m_train[:4096], gen), device)
                for steps in (0, REFINE_STEPS):
                    def run():
                        k = ppost_indices(table, x, m, sigma, draws, gen)
                        if steps == 0:
                            return m * x + (1 - m) * table[k]
                        xr, mr = x.repeat(draws, 1), m.repeat(draws, 1)
                        filled = refine_latents(gx, z[k].reshape(-1, z.shape[1]), xr, mr, steps,
                                                lambda p: recipe.make_generator_optimizer(
                                                    p, lr=0.01, betas=(0.9, 0.999)))
                        return filled.reshape(draws, len(x), DIM)
                    run()  # warm-up
                    imputed, seconds = timed(run, device)
                    metrics = imputation_metrics(problem, imputed, post)
                    metrics.update(sigma=sigma, cv_ll=cv_ll, sigma_fit_s=fit_s, ms_1k=1000 * seconds * per_1k)
                    name = f"M{M}" + ("r" if steps else "")
                    out["ppost"][name] = metrics
                    print(f"ppost {run_dir.name} {name} sigma={sigma:.3g} acc={metrics['acc']:.3f} "
                          f"acc_lo={metrics['acc_lo']:.3f} itv={metrics['itv']:.3f} istd={metrics['istd']:.3f} "
                          f"rmse={metrics['rmse']:.3f} ms_1k={metrics['ms_1k']:.1f}", flush=True)
        (run_dir / "posthoc.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
