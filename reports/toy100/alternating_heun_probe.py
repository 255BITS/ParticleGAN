"""Fail-fast warm and cold gates for alternating G trapezoid correction."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.alternating_heun_scratch import alternating_heun, METHOD


def _declaration(output, phase, scope):
    names = (
        "reports/toy100/alternating_heun_scratch.py",
        "reports/toy100/alternating_heun_probe.py",
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
    row = dict(phase=phase, seed=0, shared_gate_eligible=False,
               scratch_optimizer_policy=METHOD, method=METHOD,
               d_curvature_bound=2.0, heun_weight=0.5, scope=scope,
               nominal_rates=dict(g=0.00425, d=0.00425, prior=0.0085),
               update_order="D bound, then G proposal and same-sample trapezoid correction",
               source=source)
    (output / "declaration.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def warm(output, scope):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    @contextmanager
    def variant(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == "heun"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + (target - completed - outer),
            moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False,
                       scratch_optimizer_policy=METHOD if recorder.enabled else "control")
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())

    factories = {name: (lambda state, prefix, name=name: variant(name, state, prefix))
                 for name in ("identity", "constant", "heun")}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    result = run_warm_variants(config, factories, output_dir=output / "forks",
        prefix_context=lambda: alternating_heun(start_step=1000, scope=scope))
    compact = dict(identity_cold_parity=result["identity_cold_parity"],
                   variants={k: dict(status=v["status"],
                                     local_stability=v["local_stability"], final=v["final"])
                             for k, v in result["variants"].items()})
    (output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(dict(event="WARM_DONE", **compact)), flush=True)


def cold(output, scope):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="alternating_heun", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stages = []
    began = time.perf_counter()
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        try:
            with alternating_heun(task=task, scope=scope) as (recorder, source):
                result, context = run_legacy(spec, recipe, noise,
                    model_policy=declared_model_policy(config))
            verdict = test_verdict(spec, result)
            data = dict(result=result, applied=context["applied"],
                        noise=context["noise_receipt"], dynamics=recorder.receipt(),
                        spec=spec, verdict=verdict, shared_gate_eligible=False,
                        scratch_optimizer_policy=METHOD)
            (output / f"{task}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
            row = dict(task=task, verdict=verdict, live=result["live"],
                       seconds=result["seconds"])
            stages.append(row)
            print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
            if not verdict["passed"]:
                break
        except Exception as error:
            import traceback
            (output / f"{task}.error.json").write_text(json.dumps(
                dict(task=task, error=repr(error), traceback=traceback.format_exc(),
                     completed_outer_steps=recorder.outer_steps), indent=2) + "\n")
            stages.append(dict(task=task, error=repr(error)))
            break
    status = dict(stages=stages, seconds=time.perf_counter() - began)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **status)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "cold"), required=True)
    parser.add_argument("--scope", choices=("joint", "network"), default="joint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _declaration(args.output, args.phase, args.scope)
    if args.phase == "warm":
        warm(args.output, args.scope)
    else:
        cold(args.output, args.scope)


if __name__ == "__main__":
    main()
