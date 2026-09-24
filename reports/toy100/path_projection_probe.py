"""Fail-fast gates for PR84 plus a path-crossing G direction."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.path_smoothed_candidate import METHOD, path_smoothed_candidate


def _declaration(output, phase):
    names = (
        "reports/toy100/path_smoothed_candidate.py",
        "reports/toy100/path_acquisition.py",
        "reports/toy100/path_projection_probe.py",
        "reports/toy100/alternating_curvature_scratch.py",
        "reports/toy100/extra_adam_scratch.py",
        "reports/toy100/pr84_smoothed_candidate.py",
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
               d_curvature_bound=3.0, g_curvature_bound=0.25, smooth_cap=0.15,
               operator="Frozen PR84 update with path-crossing critic direction on G",
               acquisition_signal="ray that dips then rises in an empty higher basin; direction only",
               nominal_rates=dict(g=0.00425, d=0.00425, prior=0.0085),
               torch_note="local interpreter may differ from the archived cu126/Python 3.12.13 replay",
               source=source)
    (output / "declaration.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def warm(output):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    @contextmanager
    def variant(method, state, prefix):
        print(json.dumps(dict(event="VARIANT_START", variant=method,
                              completed_steps=state["completed_steps"])), flush=True)
        recorder, _ = prefix
        recorder.enabled = method in ("original", "path")
        recorder.path = method == "path"
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
        print(json.dumps(dict(event="VARIANT_DONE", variant=method,
                              status=receipt.get("method"))), flush=True)

    factories = {name: (lambda state, prefix, name=name: variant(name, state, prefix))
                 for name in ("identity", "constant", "original", "path")}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    result = run_warm_variants(config, factories, output_dir=output / "forks",
                               prefix_context=lambda: path_smoothed_candidate(start_step=1000))
    compact = dict(method=METHOD,
                   source=json.loads((output / "declaration.json").read_text())["source"],
                   identity_cold_parity=result["identity_cold_parity"],
                   variants={k: dict(status=v["status"],
                                     local_stability=v["local_stability"], final=v["final"])
                             for k, v in result["variants"].items()})
    (output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(dict(event="WARM_DONE", **{
        name: dict(status=row["status"], stability=row["local_stability"])
        for name, row in compact["variants"].items()}), flush=True)


def _first_eight(result):
    for row in result.get("observations") or []:
        if row.get("modes", 0) >= 8:
            return row.get("step")
    return None


def cold(output):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="path_smoothed_candidate", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stages = []
    began = time.perf_counter()
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        try:
            with path_smoothed_candidate(task=task) as (recorder, source):
                result, context = run_legacy(spec, recipe, noise,
                                              model_policy=declared_model_policy(config))
            verdict = test_verdict(spec, result)
            data = dict(result=result, applied=context["applied"],
                        noise=context["noise_receipt"], dynamics=recorder.receipt(),
                        spec=spec, verdict=verdict, shared_gate_eligible=False,
                        scratch_optimizer_policy=METHOD,
                        first_eight_mode_update=_first_eight(result))
            (output / f"{task}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
            row = dict(task=task, verdict=verdict["status"], live=result["live"],
                       first_eight_mode_update=data["first_eight_mode_update"],
                       path_redirects_total=recorder.receipt()["path_redirects_total"],
                       seconds=result["seconds"])
            stages.append(row)
            print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
            if not verdict["passed"]:
                break
        except Exception as error:
            import traceback
            (output / f"{task}.error.json").write_text(json.dumps(
                dict(task=task, error=repr(error), traceback=traceback.format_exc()), indent=2) + "\n")
            stages.append(dict(task=task, error=repr(error)))
            print(json.dumps(dict(event="STAGE_ERROR", task=task, error=repr(error))), flush=True)
            break
    status = dict(stages=stages, seconds=time.perf_counter() - began)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **status)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "cold"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.phase == "cold":
        if args.previous is None:
            raise ValueError("cold gate requires a passing warm summary")
        previous = json.loads(args.previous.read_text())
        path = previous["variants"]["path"]
        if (not previous["identity_cold_parity"] or path["status"] != "PASS"
                or not path["local_stability"]["pass_all"]
                or path["local_stability"]["checks"] != 200):
            raise RuntimeError("warm gate did not pass; cold acquisition forbidden")
    declaration = _declaration(args.output, args.phase)
    print(json.dumps(dict(event="DECLARED", phase=args.phase, method=METHOD,
                          output=str(args.output))), flush=True)
    if args.phase == "cold":
        if previous.get("method") != METHOD or previous.get("source") != declaration["source"]:
            raise RuntimeError("warm gate method/source differs from this frozen candidate")
        (args.output / "previous-gate.json").write_bytes(args.previous.read_bytes())
        cold(args.output)
    else:
        warm(args.output)


if __name__ == "__main__":
    main()
