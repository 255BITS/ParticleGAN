"""Why does the axis_silu critic collapse vector_unequal_mass's rare 2% component, and what fixes it?

--diagnose [ARM ...]: rerun axis_silu and LeakyReLU (or the named arms) on unequal_mass (the exact suite episode) with a per-observation
  probe of the rare component: mass, particles, core eig, minor/major std ratio, spill, the prior particles'
  z spread vs the generator's local Jacobian, and the critic's value/gradient on real rare samples vs fakes.
  Writes diagnose.log / diagnose.jsonl, and one line per observation to diagnose_<arm>.log.
--arms: targeted arms on unequal_mass (arms.log / arms.jsonl).
--crosscheck ARM [ARM ...]: those arms on all 8 dev vector tasks (crosscheck.log / crosscheck.jsonl).
--v5: protocol v5 rescore (v5.log / v5.jsonl): V5_ARMS on unequal_mass plus LeakyReLU and axis_silu on the
  other 7 dev tasks. Each record carries the v4 and v5 verdict of the same episode.
Same fixed cosine recipe, seed 0, sustained rule as ../anisotropic_core_metric/run.py.
One CPU episode per process, 1 thread each. tail -f the printed log for one line per episode.
Tables for the README: python summarize.py diagnose|arms|crosscheck.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

TASK = "vector_unequal_mass"
RARE = 3  # corner (1.5, 1.5), target mass .02
# name -> (critic card name or None for LeakyReLU, spec overrides, kind)
# kind: "critic" = architecture-only change (promotable), "reg" = regularizer, "recipe" = diagnostic only.
ARMS = {
    "leaky_orig": (None, {}, "ref"),
    "axis_silu": ("axis_silu", {}, "ref"),
    "oriented8_silu": ("oriented8_silu", {}, "critic"),
    "silu_fourier3": ("axis_silu", {"fourier": 3}, "critic"),
    "silu_hidden128": ("axis_silu", {"d_hidden": 128}, "critic"),
    "leaky1_silu": ("axis_leaky1_silu", {}, "critic"),
    "leaky1_silu_fourier3": ("axis_leaky1_silu", {"fourier": 3}, "critic"),
    "silu_cap_kappa0.5": ("axis_silu", {"reg_kappa": .5}, "reg"),
    "silu_cap_kappa2.5": ("axis_silu", {"reg_kappa": 2.5}, "reg"),
    "silu_prior_reg0.5": ("axis_silu", {"prior_reg": .5}, "reg"),
    # Round 2, chosen after the diagnosis: a zero-initialized raw-coordinate linear score beside the
    # MLP, and the rare_focus Softplus5 winner (D96 + the same raw linear skip) as a v4 reference.
    "silu_raw_skip": ("axis_silu_raw_skip", {}, "critic"),
    "silu_raw_skip_d96": ("axis_silu_raw_skip", {"d_hidden": 96}, "critic"),
    "linear_skip_d96_beta5": ("linear_skip_d96_beta5", {"d_hidden": 96}, "critic"),
    "silu_particles1024": ("axis_silu", {"particles": 1024}, "recipe"),
    "silu_steps2x": ("axis_silu", {"steps": 2400}, "recipe"),
    "silu_batch512": ("axis_silu", {"batch": 512}, "recipe"),
    "silu_dlr3": ("axis_silu", {"d_lr_mult": 3.}, "recipe"),
}
ARM_LIST = list(ARMS)
V5_ARMS = ["leaky_orig", "axis_silu", "silu_dlr3", "linear_skip_d96_beta5"]
SUITE_ARMS = ["leaky_orig", "axis_silu"]


def install(arm):
    """Point the suite's critic constructor at the arm's card and return the task spec with overrides."""
    from benchmarks.transfer_suite import vector_tasks as vt
    from benchmarks.transfer_suite import linear_skip_refinement_research as skip, smooth_critic_research as smooth
    card_name, overrides, _ = ARMS[arm]
    for module in (smooth, skip):
        card = next((c for c in module.ARCHITECTURES if c["name"] == card_name), None)
        if card is not None:
            vt.SimpleMLPDiscriminator = module.constructor(card)
    return vt, overrides


def spec_for(vt, task, overrides):
    from copy import deepcopy
    return deepcopy(next(t for t in vt.TASKS if t["name"] == task)) | overrides


# ---------------------------------------------------------------- diagnosis
def probe(model_g, prior, critic, spec, completed, metrics):
    """Rare-component probe on the live generator/prior/critic at one observation."""
    import torch
    from benchmarks.transfer_suite import vector_tasks as vt
    means = torch.tensor(spec["means"])
    sigma = float(torch.tensor(spec["covariances"])[RARE, 0, 0].sqrt())
    z = prior.z.detach()
    with torch.no_grad():
        out = model_g(z)
    assign = torch.cdist(out, means).argmin(1)
    member = assign == RARE
    n = int(member.sum())
    rec = dict(step=completed, mass=metrics["component_mass"][RARE], particles=n,
               core_eig=metrics["component_core_min_eigen_ratio"], rare_core_err=metrics["component_core_covariance_errors"][RARE],
               rare_spill=metrics["component_spill"][RARE], spill=metrics["max_component_spill"], sw1=metrics["sw1_normalized"])
    zs = z.std(0).norm()  # whole-cloud z spread (RMS radius)
    rec["z_cloud_rms"] = float(zs)
    if n >= 2:
        pts, zr = out[member], z[member]
        d = (pts - means[RARE]).norm(dim=1)
        rec["core_particles"] = int((d <= 4 * sigma).sum())
        c = pts[d <= 4 * sigma] if (d <= 4 * sigma).sum() >= 2 else pts
        ev = torch.linalg.eigvalsh(torch.cov(c.T, correction=0)).clamp_min(0)
        rec["out_major_over_sigma"] = float(ev[-1].sqrt() / sigma)
        rec["out_minor_over_sigma"] = float(ev[0].sqrt() / sigma)
        rec["minor_major"] = float((ev[0] / ev[-1].clamp_min(1e-12)).sqrt())
        zc = zr - zr.mean(0)
        rec["z_rare_rms"] = float(zc.square().sum(1).mean().sqrt())
        rec["z_rare_min_pair"] = float(torch.pdist(zr).min())
        # Local Jacobian of G at the rare particles: operator norm, and the output spread
        # it would give the observed z spread (J @ (z_i - z_bar)) vs the actual spread.
        jac = torch.stack([torch.autograd.functional.jacobian(model_g, zi[None])[0, :, 0] for zi in zr])  # n,2,zd
        rec["jac_op"] = float(torch.linalg.matrix_norm(jac, ord=2).mean())
        lin = torch.einsum("nij,nj->ni", jac, zc)
        rec["linear_out_rms_over_sigma"] = float(lin.square().sum(1).mean().sqrt() / sigma)
        rec["out_rms_over_sigma"] = float((pts - pts.mean(0)).square().sum(1).mean().sqrt() / sigma)
    # Same Jacobian for the largest component, as a reference scale.
    big = out[assign == 0]
    if len(big) >= 2:
        zb = z[assign == 0][:16]
        jb = torch.stack([torch.autograd.functional.jacobian(model_g, zi[None])[0, :, 0] for zi in zb])
        rec["jac_op_big"] = float(torch.linalg.matrix_norm(jb, ord=2).mean())
        rec["z_big_rms"] = float((z[assign == 0] - z[assign == 0].mean(0)).square().sum(1).mean().sqrt())
    # Critic: value and gradient on real rare samples vs fake rare points; radial pull toward the mean.
    g = torch.Generator().manual_seed(7)
    real = means[RARE] + sigma * torch.randn(512, 2, generator=g)
    def val_grad(x):
        x = x.clone().requires_grad_(True)
        v = critic(x)
        (gx,) = torch.autograd.grad(v.sum(), x)
        return v.detach(), gx
    vr, gr = val_grad(real)
    rec["D_real_rare"] = float(vr.mean())
    rec["gradD_real_rare"] = float(gr.norm(dim=1).mean())
    inward = -(real - means[RARE]) / (real - means[RARE]).norm(dim=1, keepdim=True)
    rec["inward_pull_real"] = float((gr * inward).sum(1).mean())  # >0: D rises toward the center
    rec["D_center"] = float(critic(means[RARE][None]).detach())
    ring = means[RARE] + sigma * torch.stack([torch.cos(torch.linspace(0, 6.283, 17)[:-1]),
                                              torch.sin(torch.linspace(0, 6.283, 17)[:-1])], 1)
    rec["D_center_minus_1sigma_ring"] = float(rec["D_center"] - critic(ring).detach().mean())
    if n >= 1:
        vf, gf = val_grad(out[member])
        rec["D_fake_rare"] = float(vf.mean())
        rec["gradD_fake_rare"] = float(gf.norm(dim=1).mean())
    vb, _ = val_grad(means[0] + sigma * torch.randn(512, 2, generator=g))
    rec["D_real_big"] = float(vb.mean())
    return rec


def diagnose(arm):
    import torch
    torch.set_num_threads(1)
    vt, overrides = install(arm)
    spec = spec_for(vt, TASK, overrides)
    made = {}
    for name in ("ParticlePrior", "SimpleMLPGenerator", "SimpleMLPDiscriminator"):
        cls = getattr(vt, name)
        def factory(*a, _cls=cls, _name=name, **k):
            obj = _cls(*a, **k)
            made.setdefault(_name, obj)
            return obj
        setattr(vt, name, factory)
    score, calls, rows = vt.score_samples, [0], []
    log = open(HERE / f"diagnose_{arm}.log", "w")

    def scored(samples, cfg, completed):
        metrics = score(samples, cfg, completed)
        calls[0] += 1
        if calls[0] % 2 == 1:  # live measurement precedes the EMA one
            with torch.enable_grad():
                rec = probe(made["SimpleMLPGenerator"], made["ParticlePrior"], made["SimpleMLPDiscriminator"], cfg, completed, metrics)
            rows.append(rec)
            log.write(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()) + "\n")
            log.flush()
        return metrics
    vt.score_samples = scored
    res = vt.run_episode(spec, vt.fixed_policy("cosine"), fixed=True)
    conv = res.get("convergence", {})
    return dict(arm=arm, status=res.get("status"), error=res.get("error"), sustained=conv.get("stable_from_step") is not None,
                passing_suffix=conv.get("passing_suffix"), observations=rows)


# ---------------------------------------------------------------- episodes
def run(job):
    group, task, arm = job
    import torch
    torch.set_num_threads(1)
    vt, overrides = install(arm)
    spec = spec_for(vt, task, overrides)
    res = vt.run_episode(spec, vt.fixed_policy("cosine"), fixed=True)
    rec = dict(group=group, task=task, arm=arm, kind=ARMS[arm][2], overrides=overrides, status=res.get("status"),
               seconds=round(res["seconds"], 1), error=res.get("error"), thresholds=spec["thresholds"])
    if res.get("status") != "ERROR":
        conv = res["convergence"]
        rec.update(sustained=conv["stable_from_step"] is not None, passing_suffix=conv["passing_suffix"],
                   failing=[k for k, op, b in spec["thresholds"] if not vt.passes(res["live"], [[k, op, b]])],
                   observation_failing=[[i for i, t in enumerate(spec["thresholds"]) if not vt.passes(o, [t])]
                                        for o in res["observations"]],
                   live={k: v for k, v in res["live"].items() if k not in ("target_mass", "sample_count")})
        if group == "v5":
            rec["v4"] = v4_verdict(vt, spec, res["observations"], res["live"])
    return rec


def v4_verdict(vt, spec, observations, live):
    """The same episode under protocol v4: resolved_* shape/spill aggregates back to all-component ones."""
    import math
    back = dict(zip(vt.RESOLVED_SHAPE_METRICS, ("component_core_covariance_error", "component_core_min_eigen_ratio",
                                                "max_component_spill")))
    thresholds = [[back.get(k, k), op, b] for k, op, b in spec["thresholds"]]
    expected = {math.ceil(i * spec["steps"] / vt.OBSERVATIONS) for i in range(1, vt.OBSERVATIONS + 1)}
    conv = vt.sustained(observations, thresholds, expected_steps=expected)
    return dict(thresholds=thresholds, sustained=conv["stable_from_step"] is not None, passing_suffix=conv["passing_suffix"],
                failing=[k for k, op, b in thresholds if not vt.passes(live, [[k, op, b]])])


def fmt(r):
    if r["status"] == "ERROR":
        return f"{r['task']:<22} {r['arm']:<22} ERROR"
    live = r["live"]
    s = f"{r['task']:<22} {r['arm']:<22} "
    if "v4" in r:
        s += f"v4 {'SUST' if r['v4']['sustained'] else 'fail'}({r['v4']['passing_suffix']:2d}) -> v5 "
    s += f"{'SUST' if r['sustained'] else 'fail'} suffix={r['passing_suffix']:2d} sw1={live['sw1_normalized']:.3f}"
    if "mass_tv" in live:
        s += (f" tv={live['mass_tv']:.3f} core={live['component_core_covariance_error']:.2f}"
              f" eig={live['component_core_min_eigen_ratio']:.2f} spill={live['max_component_spill']:.3f}")
        if "resolved_core_min_eigen_ratio" in live:
            s += f" resolved_eig={live['resolved_core_min_eigen_ratio']:.2f} resolved_spill={live['resolved_max_component_spill']:.3f}"
    if r["task"] == TASK:
        s += (f" minmass={live['min_mass_ratio']:.2f} rare_mass={live['component_mass'][RARE]:.3f}"
              f" comp_eigs={'/'.join(f'{e:.2f}' for e in live['component_core_eigen_ratios'])}")
    return s + f" failing={','.join(r['failing']) or '-'} {r['seconds']}s"


def pool_run(fn, todo, jobs, log, results, line):
    print(f"{len(todo)} episodes; tail -f {log}", flush=True)
    started = time.time()
    with open(log, "w") as lf, open(results, "w") as rf, mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        lf.write(f"# {len(todo)} episodes, {jobs} workers\n")
        lf.flush()
        for rec in pool.imap_unordered(fn, todo):
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            lf.write(f"[{time.time()-started:6.0f}s] {line(rec)}\n")
            lf.flush()
        lf.write(f"# done in {time.time()-started:.0f}s\n")
    print(f"done; results in {results}", flush=True)


def diag_line(r):
    last = r["observations"][-1] if r["observations"] else {}
    return (f"{r['arm']:<12} {'SUST' if r['sustained'] else 'fail'} suffix={r['passing_suffix']} "
            f"final particles={last.get('particles')} core_eig={last.get('core_eig', float('nan')):.2f}; "
            f"per-observation lines in diagnose_{r['arm']}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnose", nargs="*", metavar="ARM", help="default: axis_silu leaky_orig")
    mode.add_argument("--arms", action="store_true")
    mode.add_argument("--crosscheck", nargs="+", metavar="ARM")
    mode.add_argument("--v5", action="store_true")
    args = ap.parse_args()
    if args.diagnose is not None:
        arms = args.diagnose or ["axis_silu", "leaky_orig"]
        pool_run(diagnose, arms, len(arms), HERE / "diagnose.log", HERE / "diagnose.jsonl", diag_line)

    elif args.v5:
        from benchmarks.transfer_suite import vector_tasks as vt
        todo = [("v5", TASK, a) for a in V5_ARMS] + [("v5", t["name"], a) for t in vt.TASKS if t["name"] != TASK
                                                    for a in SUITE_ARMS]
        pool_run(run, todo, args.jobs, HERE / "v5.log", HERE / "v5.jsonl", fmt)
    elif args.arms:
        pool_run(run, [("arms", TASK, a) for a in ARM_LIST], args.jobs, HERE / "arms.log", HERE / "arms.jsonl", fmt)
    else:
        from benchmarks.transfer_suite import vector_tasks as vt
        todo = [("cross", t["name"], a) for t in vt.TASKS for a in args.crosscheck]
        pool_run(run, todo, args.jobs, HERE / "crosscheck.log", HERE / "crosscheck.jsonl", fmt)


if __name__ == "__main__":
    main()
