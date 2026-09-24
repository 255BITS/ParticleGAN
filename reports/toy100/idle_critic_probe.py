"""Fail-fast gates for the idle-when-covered critic response, plus a PR84 control."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.pr84_idle_critic_response import METHOD, pr84_idle_critic_response
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


FACTORIES = {
    "baseline": pr84_smoothed_candidate,
    "idle": pr84_idle_critic_response,
}


def _declaration(output, phase, method):
    names = (
        "reports/toy100/pr84_idle_critic_response.py",
        "reports/toy100/idle_critic_probe.py",
        "reports/toy100/pr84_smoothed_candidate.py",
        "reports/toy100/alternating_curvature_scratch.py",
        "reports/toy100/extra_adam_scratch.py",
        "benchmarks/toy100/warm_equilibrium_probe.py",
        "benchmarks/toy100/continuous_probe.py",
        "benchmarks/locked_shared/mode_hold.py",
        "benchmarks/locked_shared/trajectory.py",
    )
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source"
    archive.mkdir()
    for name in names:
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    row = dict(phase=phase, method_name=method, seed=0, shared_gate_eligible=False,
               scratch_optimizer_policy=METHOD if method == "idle" else "pr84_smoothed",
               hypothesis=("G keeps the PR84 stencil unless the real batch has a "
                           "sharp-critic peak beyond the particle cloud and the nearest "
                           "particle points away; only then nearby rows read the critic "
                           "past that lip."),
               purity="GAN dynamics only — no coverage/likelihood term",
               host="neural",
               nominal_rates=dict(g=0.00425, d=0.00425, prior=0.0085),
               source=source)
    (output / "declaration.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def _tail(result):
    obs = result.get("observations") or []
    tail = obs[-5:]
    live = result.get("live") or (obs[-1] if obs else {})
    return dict(live=live, tail=[{k: row.get(k) for k in ("step", "modes", "hq", "mse")}
                                 for row in tail], seconds=result.get("seconds"))


def warm(output, method):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    factory = FACTORIES[method]

    @contextmanager
    def variant(name, state, prefix):
        print(json.dumps(dict(event="VARIANT_START", variant=name)), flush=True)
        recorder, _ = prefix
        recorder.enabled = name == method
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + (target - completed - outer),
            moment_updates=target)
        receipt = dict(method=name, shared_gate_eligible=False)
        if name == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", variant=name,
                              status=receipt.get("method"))), flush=True)

    factories = {name: (lambda state, prefix, name=name: variant(name, state, prefix))
                 for name in ("identity", method)}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    result = run_warm_variants(
        config, factories, output_dir=output / "forks",
        prefix_context=lambda: factory(start_step=1000))
    compact = dict(method=method, identity_cold_parity=result["identity_cold_parity"],
                   variants={k: dict(status=v["status"],
                                     local_stability=v["local_stability"], final=v["final"])
                             for k, v in result["variants"].items()})
    (output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(dict(event="WARM_DONE", **compact)), flush=True)


def cold(output, method, tasks):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    factory = FACTORIES[method]
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name=f"{method}_idle_critic_screen", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stages = []
    began = time.perf_counter()
    recorder = None
    for task in tasks:
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        try:
            with factory(task=task) as (recorder, _source):
                result, context = run_legacy(
                    spec, recipe, noise, model_policy=declared_model_policy(config))
            verdict = test_verdict(spec, result)
            data = dict(result=result, applied=context["applied"],
                        noise=context["noise_receipt"], dynamics=recorder.receipt(),
                        spec=spec, verdict=verdict, shared_gate_eligible=False)
            (output / f"{task}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
            row = dict(task=task, verdict=verdict, **_tail(result))
            if hasattr(recorder, "records"):
                armed = [bool(r.get("response_armed", False)) for r in recorder.records]
                uses = [r.get("response_uses", 0) for r in recorder.records]
                row["response_steps_armed"] = int(sum(armed))
                row["response_uses_total"] = int(sum(uses))
            stages.append(row)
            print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
            if not verdict["passed"]:
                break
        except Exception as error:
            import traceback
            (output / f"{task}.error.json").write_text(json.dumps(
                dict(task=task, error=repr(error), traceback=traceback.format_exc(),
                     completed_outer_steps=getattr(recorder, "outer_steps", None)), indent=2) + "\n")
            stages.append(dict(task=task, error=repr(error)))
            print(json.dumps(dict(event="STAGE_FAIL", task=task, error=repr(error))), flush=True)
            break
    status = dict(stages=stages, seconds=time.perf_counter() - began)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **status)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "cold"), required=True)
    parser.add_argument("--method", choices=tuple(FACTORIES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="trajectory,mode_hold")
    args = parser.parse_args()
    _declaration(args.output, args.phase, args.method)
    print(json.dumps(dict(event="DECLARED", phase=args.phase, method=args.method,
                          output=str(args.output))), flush=True)
    if args.phase == "warm":
        warm(args.output, args.method)
    else:
        cold(args.output, args.method, [task for task in args.tasks.split(",") if task])


if __name__ == "__main__":
    main()
