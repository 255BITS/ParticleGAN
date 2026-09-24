"""Ordered same-dataset stability and acquisition gates for opponent prediction."""

import argparse
import ast
from contextlib import contextmanager
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.pr84_opponent_prediction import METHOD, pr84_opponent_prediction


def dense_warm_runner():
    """Change only diagnostic cadence in the audited fork helper, from10 to1."""
    from benchmarks.toy100 import warm_equilibrium_probe as warm
    tree = ast.parse(inspect.getsource(warm.run_warm_variants))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "cadence"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError("warm helper cadence assignment changed")
    value = assignments[0].value
    if not (isinstance(value, ast.IfExp) and isinstance(value.orelse, ast.Constant)
            and value.orelse.value == 10):
        raise RuntimeError("warm helper long diagnostic cadence changed")
    value.orelse.value = 1
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    namespace = {}
    exec(compile(tree, "<per-update-warm-observer>", "exec"), warm.__dict__, namespace)
    return namespace["run_warm_variants"], source


def untimed(value):
    if isinstance(value, dict):
        return {key: untimed(item) for key, item in value.items() if "seconds" not in key}
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def declaration(output, phase):
    names = (
        "reports/toy100/pr84_opponent_prediction.py",
        "reports/toy100/pr84_opponent_prediction_probe.py",
        "reports/toy100/pr84_smoothed_candidate.py",
        "reports/toy100/pr84_smoothed_parity.py",
        "reports/toy100/alternating_curvature_scratch.py",
        "reports/toy100/extra_adam_scratch.py",
        "benchmarks/toy100/warm_equilibrium_probe.py",
        "benchmarks/toy100/continuous_probe.py",
        "benchmarks/locked_shared/mode_hold.py",
        "benchmarks/locked_shared/trajectory.py",
        "configs/toy100/constraints_simple_regularization.json",
    )
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    row = dict(method=METHOD, phase=phase, shared_gate_eligible=False, seed=0,
               source=source, nominal_rates=dict(g=.00425, d=.00425, prior=.0085),
               prediction="G sees 2*D_accepted-D_base; materialized D unchanged",
               g_curvature_bound=.25, d_curvature_bound=3., smooth_width_cap=.15,
               noise_horizon=1200, distribution_shift=False,
               long_check_every=1, prediction_stencil_width="phase1 frozen through phase2",
               generated_observer_sha256=hashlib.sha256(dense_warm_runner()[1].encode()).hexdigest())
    return row


def continuation(output, phase, previous):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context
    from reports.toy100.pr84_smoothed_parity import _compare_records

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method in ("original", "prediction")
        recorder.prediction = method == "prediction"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False)
        print(json.dumps(dict(event="VARIANT_START", variant=method)), flush=True)
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", variant=method)), flush=True)

    names = ("identity", "constant", "original", "prediction") if phase == "warm" else (
        "identity", "original", "prediction")
    variants = {name: (lambda state, prefix, name=name: activate(name, state, prefix)) for name in names}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    runner, _ = dense_warm_runner()
    result = runner(config, variants, output_dir=output / "forks",
        steps=1200 if phase == "warm" else 2400,
        prefix_context=lambda: pr84_opponent_prediction(start_step=1000))
    reference_name = ("pr84-independent-audit/smooth40-warm/alternating_curvature.json.gz"
                      if phase == "warm" else "pr84-stationary-hold/forks/original.json.gz")
    reference = ROOT / "reports/toy100/continuous-evidence" / reference_name
    expected = json.loads(gzip.decompress(reference.read_bytes()))
    original = json.loads((output / "forks/original.json").read_text())
    for key in ("warm_state_sha256", "final_state_sha256", "observations", "noise"):
        if untimed(original[key]) != untimed(expected[key]):
            raise RuntimeError(f"disabled prediction changed original PR84 {key}")
    original_diagnostic = {point["step"]: point for point in original["diagnostic"]}
    if any(untimed(original_diagnostic[point["step"]]) != untimed(point)
           for point in expected["diagnostic"]):
        raise RuntimeError("disabled prediction changed archived diagnostic checks")
    _compare_records(original["dynamics_receipt"]["records"], expected["dynamics_receipt"]["records"])
    if phase == "hold":
        old = json.loads((previous.parent / "forks/prediction.json").read_text())
        new = json.loads((output / "forks/prediction.json").read_text())
        if old["warm_state_sha256"] != new["warm_state_sha256"]:
            raise RuntimeError("long continuation changed warm state")
        by_step = {row["step"]: row for row in new["diagnostic"]}
        if any(untimed(by_step[row["step"]]) != untimed(row) for row in old["diagnostic"]):
            raise RuntimeError("long continuation changed warm diagnostics")
        if untimed(new["dynamics_receipt"]["records"][:200]) != untimed(old["dynamics_receipt"]["records"]):
            raise RuntimeError("long continuation changed first200 update records")
    result.update(method=METHOD, phase=phase,
                  source=json.loads((output / "declaration.json").read_text())["source"],
                  original_control_exact_parity=True,
                  original_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  first200_parity=True if phase == "hold" else None)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="CONTINUATION_DONE", **result)), flush=True)


def cold(output):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="pr84_opponent_prediction", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stages = []
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        with pr84_opponent_prediction(task=task) as (recorder, _):
            result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        verdict = test_verdict(spec, result)
        data = dict(result=result, applied=context["applied"], noise=context["noise_receipt"],
                    dynamics=recorder.receipt(), spec=spec, verdict=verdict,
                    shared_gate_eligible=False, scratch_optimizer_policy=METHOD)
        (output / f"{task}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
        row = dict(task=task, verdict=verdict, live=result["live"], seconds=result["seconds"])
        stages.append(row)
        print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
        if not verdict["passed"]:
            break
    (output / "summary.json").write_text(json.dumps(dict(stages=stages), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "hold", "cold"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    row = declaration(args.output, args.phase)
    if args.phase != "warm":
        if args.previous is None:
            raise ValueError("a passing previous gate is required")
        previous = json.loads(args.previous.read_text())
        variant = previous["variants"]["prediction"]
        if (previous["method"] != METHOD or previous["source"] != row["source"]
                or not previous["identity_cold_parity"] or not previous["original_control_exact_parity"]
                or variant["status"] != "PASS" or not variant["local_stability"]["pass_all"]
                or variant["local_stability"]["checks"] != 200):
            raise RuntimeError("previous gate or source binding failed")
        if args.phase == "hold" and previous["phase"] != "warm":
            raise RuntimeError("hold requires the warm gate")
        if args.phase == "cold" and (previous["phase"] != "hold"
                or not variant["long_hold"]["pass_all"] or variant["long_hold"]["checks"] != 1200):
            raise RuntimeError("cold requires the complete same-dataset hold")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in row["source"]:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / "declaration.json").write_text(json.dumps(row, indent=2) + "\n")
    (args.output / "source/generated_warm_observer.py").write_text(dense_warm_runner()[1])
    if args.previous is not None:
        (args.output / "previous-gate.json").write_bytes(args.previous.read_bytes())
    print(json.dumps(dict(event="DECLARED", **row)), flush=True)
    if args.phase == "cold":
        cold(args.output)
    else:
        continuation(args.output, args.phase, args.previous)


if __name__ == "__main__":
    main()
