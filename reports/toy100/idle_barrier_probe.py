"""Fail-fast gates for the idle critic field against the PR84 host.

Logs one JSON line per stage. Cold order is trajectory then ring. Warm is the
forked 200-check constant-rate continuation. Stay is a 2400-update mode hold.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.idle_barrier_field import METHOD, idle_barrier_field
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


CONTEXTS = {
    "pr84": pr84_smoothed_candidate,
    "idle_barrier": idle_barrier_field,
}


def _cold(name, output):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe

    torch.set_num_threads(1)
    context = CONTEXTS[name]
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name=name, lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    output.mkdir(parents=True, exist_ok=False)
    declaration = dict(candidate=name, method=METHOD if name == "idle_barrier" else "pr84",
                       tasks=["trajectory", "mode_hold"], seed=0, shared_gate_eligible=False,
                       purity="GAN dynamics only — no coverage/likelihood term")
    (output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    print(json.dumps(dict(event="DECLARED", **declaration)), flush=True)
    stages = []
    started = time.perf_counter()
    for task in declaration["tasks"]:
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        with context(task=task) as (recorder, _):
            result, details = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        verdict = test_verdict(spec, result)
        live = result.get("live", {})
        curve = result.get("live_curve") or []
        tail = [{k: row.get(k) for k in ("step", "modes", "hq", "mse")} for row in curve[-5:]]
        row = dict(task=task, passed=verdict["passed"], live=live, tail=tail,
                   seconds=result.get("seconds"), fires=getattr(recorder, "field_fires", None))
        (output / f"{task}.json").write_text(json.dumps(
            dict(verdict=verdict, live=live, tail=tail, dynamics=recorder.receipt()),
            allow_nan=False) + "\n")
        stages.append(row)
        print(json.dumps(dict(event="STAGE", candidate=name, **row)), flush=True)
        if not verdict["passed"] and task == "trajectory":
            break
    status = dict(candidate=name, stages=stages, seconds=time.perf_counter() - started)
    (output / "status.json").write_text(json.dumps(status) + "\n")
    print(json.dumps(dict(event="DONE", **status)), flush=True)


def _warm(name, output):
    from contextlib import contextmanager
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    import torch
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    context = CONTEXTS[name]
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == "candidate"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        receipt = dict(method=method, candidate=name, shared_gate_eligible=False)
        print(json.dumps(dict(event="VARIANT_START", candidate=name, variant=method)), flush=True)
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", candidate=name, variant=method)), flush=True)

    factories = {label: (lambda state, prefix, label=label: activate(label, state, prefix))
                 for label in ("identity", "constant", "candidate")}
    print(json.dumps(dict(event="WARM_START", candidate=name)), flush=True)
    summary = run_warm_variants(
        config, factories, output_dir=output / "forks",
        prefix_context=lambda: context(start_step=1000))
    (output / "summary.json").write_text(json.dumps(summary) + "\n")
    brief = {key: dict(status=row["status"], local=row["local_stability"])
             for key, row in summary["variants"].items()}
    print(json.dumps(dict(event="WARM_DONE", candidate=name, variants=brief)), flush=True)


def _stay(output):
    import torch
    from benchmarks.toy100.continuous_probe import run_probe

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    print(json.dumps(dict(event="STAY_START", steps=2400)), flush=True)
    with idle_barrier_field(task="mode_hold") as (recorder, _):
        evidence = run_probe(config, mode="constant", steps=2400, diagnostic_every=50,
                             log=lambda row: print(json.dumps(dict(event="STAY", **{
                                 k: row.get(k) for k in ("event", "step", "modes", "hq")})), flush=True))
    kept = {k: evidence.get(k) for k in ("status", "stationary", "final")}
    (output / "stay.json").write_text(json.dumps(dict(evidence=kept, fires=recorder.field_fires)) + "\n")
    print(json.dumps(dict(event="STAY_DONE", **kept, fires=recorder.field_fires)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("warm", "cold", "stay"), required=True)
    parser.add_argument("--candidate", choices=tuple(CONTEXTS), default="idle_barrier")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gate == "warm":
        _warm(args.candidate, args.output)
    elif args.gate == "cold":
        _cold(args.candidate, args.output)
    else:
        _stay(args.output)


if __name__ == "__main__":
    main()
