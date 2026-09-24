"""Fail-fast warm / cold / stay gates for PR84 and one GAN-native follow-up.

warm:  scheduled prefix to 1000, constant-rate fork 1001-1200 (identity + method).
cold:  constant-rate trajectory then ring (1200 updates), stops at first FAIL.
stay:  constant-rate cold ring continued to 2400; 10-step checks 1210-2400.
Every phase prints one JSON line per event so ``tail -f`` on the log is readable.

Ranking runs pin ATen/MKL/oneDNN to AVX2 before torch is imported. A warm rank
is refused when the identity fork is not 200/200. ``delayed_g125`` also stops
when its own warm fork regresses below #107's 200/200.
"""

import argparse
import functools
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

# Must run before the first torch import. Callers can still select avx512
# for the build check by exporting ATEN_CPU_CAPABILITY=avx512.
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
if os.environ["ATEN_CPU_CAPABILITY"].lower() == "avx2":
    os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
    os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FACTORIES = {
    "baseline": ("reports.toy100.pr84_smoothed_candidate", "pr84_smoothed_candidate"),
    "reach": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate"),
    "reach1": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate", dict(reach=1.)),
    "reachsat": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate",
                 dict(ramp="saturating")),
    "reachstall": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate",
                   dict(ramp="stall")),
    "reachstall_g125": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate",
                        dict(ramp="stall", g_curvature_bound=.125)),
    "reachstall_game": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate",
                        dict(ramp="stall", game_bound=True)),
    "reachstall_game2": ("reports.toy100.pr84_reach_candidate", "pr84_reach_candidate",
                         dict(ramp="stall", game_bound=True, game_steps=2)),
    "delayed_g125": ("reports.toy100.pr84_delayed_g_bound", "delayed_budget_g_bound"),
    "stall_cf25": ("reports.toy100.pr84_stall_counterfactual", "stall_counterfactual_g25"),
}
# These must hold #107's 200/200 warm fork. A shortfall is a kill, not a sweep.
WARM_HOLD_METHODS = frozenset({"delayed_g125", "stall_cf25"})
SOURCES = (
    "reports/toy100/gan_followup_probe.py",
    "reports/toy100/pr84_stall_counterfactual.py",
    "reports/toy100/pr84_delayed_g_bound.py",
    "reports/toy100/pr84_reach_candidate.py",
    "reports/toy100/pr84_smoothed_candidate.py",
    "reports/toy100/alternating_curvature_scratch.py",
    "reports/toy100/extra_adam_scratch.py",
    "benchmarks/toy100/warm_equilibrium_probe.py",
    "benchmarks/toy100/continuous_probe.py",
    "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/locked_shared/trajectory.py",
)


def factory(method):
    module, name, *kwargs = FACTORIES[method]
    return functools.partial(getattr(importlib.import_module(module), name), **(kwargs or [{}])[0])


def emit(**row):
    import torch
    row.setdefault("cpu", torch.backends.cpu.get_cpu_capability())
    row.setdefault("aten_cpu_capability", os.environ.get("ATEN_CPU_CAPABILITY"))
    print(json.dumps(row, default=float), flush=True)


def declare(output, phase, method):
    output.mkdir(parents=True, exist_ok=False)
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in SOURCES if (ROOT / name).exists()}
    import torch
    row = dict(phase=phase, method=method, seed=0, host="neural", torch=torch.__version__,
               cpu=torch.backends.cpu.get_cpu_capability(),
               aten_cpu_capability=os.environ.get("ATEN_CPU_CAPABILITY"),
               mkl_enable_instructions=os.environ.get("MKL_ENABLE_INSTRUCTIONS"),
               onednn_max_cpu_isa=os.environ.get("ONEDNN_MAX_CPU_ISA"),
               dnnl_max_cpu_isa=os.environ.get("DNNL_MAX_CPU_ISA"),
               shared_gate_eligible=False,
               purity="GAN dynamics only: no coverage, likelihood, anchor, assignment or clip ladder",
               source=source)
    (output / "declaration.json").write_text(json.dumps(row, indent=2) + "\n")
    emit(event="DECLARED", **{k: v for k, v in row.items() if k != "source"})


def _dynamics(recorder):
    value = recorder.receipt() if hasattr(recorder, "receipt") else {}
    return {k: v for k, v in value.items() if not isinstance(v, (list, dict))}


def warm(output, method):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    @contextmanager
    def variant(name, state, prefix):
        emit(event="VARIANT_START", variant=name)
        recorder, _ = prefix
        recorder.enabled = name == method
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + (target - completed - outer), moment_updates=target)
        receipt = dict(method=name, shared_gate_eligible=False)
        if name == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            receipt.update(_dynamics(recorder))
        emit(event="VARIANT_DONE", variant=name)

    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    result = run_warm_variants(
        config, {n: (lambda s, p, n=n: variant(n, s, p)) for n in ("identity", method)},
        output_dir=output / "forks", prefix_context=lambda: factory(method)(start_step=1000))
    compact = {k: dict(status=v["status"], local=v["local_stability"], final=v["final"])
               for k, v in result["variants"].items()}
    (output / "summary.json").write_text(json.dumps(compact, indent=2, default=float) + "\n")
    for name, row in compact.items():
        loc = row["local"]
        emit(event="WARM", variant=name, status=row["status"],
             **{k: loc.get(k) for k in ("checks", "passing_checks", "min_modes", "min_hq",
                                        "failing_steps") if k in loc})
    return compact


def warm_rank_ok(method, compact):
    """Refuse a warm rank when identity is not 200/200 on AVX2.

    ``delayed_g125`` and ``stall_cf25`` also die if the method fork drops below
    #107's 200/200. That is a kill, not a coefficient sweep.
    """
    import torch
    cpu = torch.backends.cpu.get_cpu_capability()
    identity = compact["identity"]["local"]
    if str(cpu).upper() != "AVX2":
        emit(event="KILL", reason="warm rank requires AVX2; refuse rank", cpu=cpu,
             identity_checks=identity.get("checks"),
             identity_passing=identity.get("passing_checks"))
        return False
    if identity.get("checks") != 200 or identity.get("passing_checks") != 200:
        emit(event="KILL", reason="warm identity/control is not 200/200; refuse rank",
             cpu=cpu,
             identity_checks=identity.get("checks"),
             identity_passing=identity.get("passing_checks"),
             identity_min_hq=identity.get("min_hq"))
        return False
    if method not in WARM_HOLD_METHODS:
        return True
    local = compact[method]["local"]
    passing = local.get("passing_checks")
    if passing is None or passing < 200:
        emit(event="KILL", reason="warm regresses vs #107 200/200; stop, no coefficient sweep",
             cpu=cpu,
             passing_checks=passing, checks=local.get("checks"),
             min_hq=local.get("min_hq"), min_modes=local.get("min_modes"),
             pr84_reference="196/200", stall_reach_reference="200/200")
        return False
    return True


def cold(output, method, tasks):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name=f"{method}_followup", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    for task in tasks:
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        began = time.perf_counter()
        with factory(method)(task=task) as (recorder, _source):
            result, context = run_legacy(spec, recipe, noise,
                                         model_policy=declared_model_policy(config))
        verdict = test_verdict(spec, result)
        obs = result.get("observations") or []
        (output / f"{task}.json").write_text(json.dumps(dict(
            result=result, dynamics=recorder.receipt(), verdict=verdict,
            records=getattr(recorder, "records", None)), default=float) + "\n")
        emit(event="COLD", task=task, passed=verdict["passed"], status=verdict["status"],
             seconds=round(time.perf_counter() - began, 1),
             tail=[{k: r.get(k) for k in ("step", "modes", "hq", "mse") if k in r} for r in obs[-5:]],
             dynamics=_dynamics(recorder))
        if not verdict["passed"]:
            break


def stay(output, method, steps):
    from benchmarks.toy100.continuous_probe import run_probe

    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    began = time.perf_counter()
    with factory(method)(task="mode_hold") as (recorder, _source):
        def hook(state):
            declare_calls = state["declare_optimizer_accounting"]
            recorder.accounting = lambda calls, outer: declare_calls(
                calls=calls + (steps - outer), moment_updates=steps)
            recorder.accounting(recorder.rows[recorder.optimizers[0]]["calls"], recorder.outer_steps)
        evidence = run_probe(config, mode="constant", steps=steps, diagnostic_every=10,
                             checkpoint_hook_step=1, checkpoint_hook=hook)
    diag = evidence.get("diagnostic") or []
    late = [p for p in diag if p["step"] > 1200]
    fails = [p for p in late if not (p["modes"] == 8 and p["hq"] >= .9)]
    terminal = [p for p in diag if p["step"] in range(1000, 1201, 50)]
    row = dict(event="STAY", steps=steps, seconds=round(time.perf_counter() - began, 1),
               terminal_1000_1200=[(p["step"], p["modes"], round(p["hq"], 4)) for p in terminal],
               checks=len(late), passing=len(late) - len(fails),
               min_modes=min((p["modes"] for p in late), default=None),
               min_hq=round(min((p["hq"] for p in late), default=float("nan")), 4),
               failing_steps=[p["step"] for p in fails][:40],
               final=(diag[-1]["step"], diag[-1]["modes"], round(diag[-1]["hq"], 4)) if diag else None,
               dynamics=_dynamics(recorder))
    records = [dict(step=r["outer_step"], sharp=r.get("critic_sharpness"), width=r.get("critic_width"),
                    adv=r.get("critic_advantage"), g_factor=r["g"]["factor"], d_factor=r["d"]["factor"],
                    **({"delayed_g_bound": r["delayed_g_bound"]} if "delayed_g_bound" in r else {}),
                    **({"stall_score": r["stall_score"]} if "stall_score" in r else {}))
               for r in getattr(recorder, "records", [])]
    (output / "stay.json").write_text(json.dumps(dict(summary=row, diagnostic=diag, records=records),
                                                 default=float) + "\n")
    emit(**row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "cold", "stay"), required=True)
    parser.add_argument("--method", choices=tuple(FACTORIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="trajectory,mode_hold")
    parser.add_argument("--steps", type=int, default=2400)
    args = parser.parse_args()
    declare(args.output, args.phase, args.method)
    if args.phase == "warm":
        compact = warm(args.output, args.method)
        if not warm_rank_ok(args.method, compact):
            emit(event="DONE", ranked=False)
            sys.exit(2)
    elif args.phase == "cold":
        cold(args.output, args.method, [t for t in args.tasks.split(",") if t])
    else:
        stay(args.output, args.method, args.steps)
    emit(event="DONE", ranked=True)


if __name__ == "__main__":
    main()
