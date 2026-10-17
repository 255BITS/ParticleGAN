"""Rescore vector_anisotropic with the core/spill metric and screen the axis_silu critic.

A: vector_anisotropic, 7 critic arms, old and new sustained verdicts.
B: axis_silu vs original LeakyReLU critic on every other development vector task.
v4 (--v4): unequal_width and unequal_mass under protocol v4, LeakyReLU vs axis_silu,
old (whole-component) and new (core/spill) verdicts; writes run_v4.log / results_v4.jsonl.
One CPU episode per (task, arm), 1 thread each, run in parallel processes.
Usage: python run.py [--v4] [--jobs N]; tail -f run.log (run_v4.log) for one line per episode.
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

ARMS_A = ["leaky_orig", "axis_softplus1", "axis_softplus5", "axis_softplus20", "axis_softplus50",
          "axis_tanh", "axis_silu"]
ARMS_B = ["leaky_orig", "axis_silu"]
V4_TASKS = ["vector_unequal_width", "vector_unequal_mass"]


def old_bounds(task):
    """Thresholds each core/spill task gated on before its protocol change (v2 whole-component rule)."""
    from benchmarks.transfer_suite import vector_tasks as vt
    extra = [["min_mass_ratio", ">=", .25]] if task == "vector_unequal_mass" else []
    return vt.SEPARATED_BOUNDS + extra if task in ("vector_anisotropic", *V4_TASKS) else None


def architecture(arm):
    from benchmarks.transfer_suite.smooth_critic_research import ARCHITECTURES
    cards = {x["name"]: x for x in ARCHITECTURES}
    for beta in (20., 50.):
        cards[f"axis_softplus{int(beta)}"] = dict(name=f"axis_softplus{int(beta)}", features="axis",
                                                  activation="softplus", beta=beta)
    return None if arm == "leaky_orig" else cards[arm]


def jobs(v4=False):
    from benchmarks.transfer_suite import vector_tasks as vt
    if v4:
        return [("v4", task, arm) for task in V4_TASKS for arm in ARMS_B]
    out = [("A", "vector_anisotropic", arm) for arm in ARMS_A]
    out += [("B", t["name"], arm) for t in vt.TASKS if t["name"] != "vector_anisotropic" for arm in ARMS_B]
    return out


def run(job):
    group, task, arm = job
    import torch
    torch.set_num_threads(1)
    from benchmarks.locked_shared.observation import sustained
    from benchmarks.transfer_suite import vector_tasks as vt
    from benchmarks.transfer_suite.smooth_critic_research import constructor
    card = architecture(arm)
    if card is not None:
        vt.SimpleMLPDiscriminator = constructor(card)
    spec = next(t for t in vt.TASKS if t["name"] == task)
    res = vt.run_episode(spec, vt.fixed_policy("cosine"), fixed=True)
    rec = dict(group=group, task=task, arm=arm, status=res.get("status"), seconds=round(res["seconds"], 1),
               error=res.get("error"), thresholds=spec["thresholds"])
    if res.get("status") != "ERROR":
        conv = res["convergence"]
        rec.update(sustained=conv["stable_from_step"] is not None, passing_suffix=conv["passing_suffix"],
                   passing_observations=conv["passing_observations"],
                   failing=[k for k, op, b in spec["thresholds"] if not vt.passes(res["live"], [[k, op, b]])],
                   # per-observation failing threshold indices (into thresholds), for where the suffix breaks
                   observation_failing=[[i for i, t in enumerate(spec["thresholds"]) if not vt.passes(o, [t])]
                                        for o in res["observations"]],
                   live={k: v for k, v in res["live"].items() if k not in ("target_mass", "sample_count")})
        bounds = old_bounds(task)
        if bounds is not None:
            old = sustained(res["observations"], bounds, expected_steps=[o["step"] for o in res["observations"]])
            rec.update(old_sustained=old["stable_from_step"] is not None and len(res["observations"]) == vt.OBSERVATIONS,
                       old_passing_suffix=old["passing_suffix"], old_final=vt.passes(res["live"], bounds),
                       old_failing=[k for k, op, b in bounds if not vt.passes(res["live"], [[k, op, b]])])
    return rec


def fmt(r):
    if r["status"] == "ERROR":
        return f"{r['group']} {r['task']:<22} {r['arm']:<16} ERROR"
    live = r["live"]
    s = (f"{r['group']} {r['task']:<22} {r['arm']:<16} {'SUST' if r['sustained'] else 'fail'} "
         f"suffix={r['passing_suffix']:2d} sw1={live['sw1_normalized']:.3f}")
    if "mass_tv" in live:
        s += (f" tv={live['mass_tv']:.3f} core={live['component_core_covariance_error']:.2f}"
              f" spill={live['max_component_spill']:.3f} oldcov={live['component_covariance_error']:.2f}")
    if "old_sustained" in r:
        s += f" old={'SUST' if r['old_sustained'] else 'fail'}({r['old_passing_suffix']})"
    if r["task"] == "vector_unequal_mass":
        s += f" minmass={live['min_mass_ratio']:.2f}"
    return s + f" failing={','.join(r['failing']) or '-'} {r['seconds']}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=21)
    ap.add_argument("--v4", action="store_true", help="protocol v4 rescore of unequal_width/unequal_mass")
    args = ap.parse_args()
    todo = jobs(args.v4)
    suffix = "_v4" if args.v4 else ""
    log, results = HERE / f"run{suffix}.log", HERE / f"results{suffix}.jsonl"
    print(f"{len(todo)} episodes; tail -f {log}", flush=True)
    started = time.time()
    with open(log, "w") as lf, open(results, "w") as rf, \
            mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        lf.write(f"# {len(todo)} episodes, {args.jobs} workers\n")
        lf.flush()
        for rec in pool.imap_unordered(run, todo):
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            lf.write(f"[{time.time()-started:6.0f}s] {fmt(rec)}\n")
            lf.flush()
        lf.write(f"# done in {time.time()-started:.0f}s\n")
    print(f"done; results in {results}", flush=True)


if __name__ == "__main__":
    main()
