"""Fail-fast GAN-4 gates: warm, then cold trajectory, ring, stay.

Warm regression versus the same-process PR84 pin stops the lane. Later gates
run only after the earlier one passes. Logs are one JSON object per line.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.pr84_optimistic_game import ALPHA, METHOD, pr84_optimistic_game


def _passes(point):
    from benchmarks.locked_shared.mode_hold import N_MODES, PASS_HQ
    return point.get("modes") == N_MODES and point.get("hq", 0) >= PASS_HQ


def warm_window(payload):
    points = [row for row in payload.get("diagnostic", []) if 1000 < row.get("step", 0) <= 1200]
    passing = [row["step"] for row in points if _passes(row)]
    return dict(checks=len(points), passing=len(passing),
                terminal=points[-1] if points else None)


def run_warm(output: Path):
    from contextlib import contextmanager
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    @contextmanager
    def activate(name, state, prefix):
        recorder, _ = prefix
        recorder.optimism = name == "optimistic"
        recorder.enabled = name in ("original", "optimistic")
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        print(json.dumps(dict(event="VARIANT_START", variant=name)), flush=True)
        receipt = dict(variant=name, optimism=recorder.optimism)
        if name == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(optimistic_updates=recorder.optimistic_updates)
        print(json.dumps(dict(event="VARIANT_DONE", variant=name,
                              optimistic_updates=recorder.optimistic_updates)), flush=True)

    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    variants = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                for name in ("identity", "original", "optimistic")}
    started = time.perf_counter()
    result = run_warm_variants(
        config, variants, output_dir=output / "forks",
        prefix_context=lambda: pr84_optimistic_game(start_step=1000, optimism=False))
    rows = {}
    for name in ("original", "optimistic"):
        payload = json.loads((output / "forks" / f"{name}.json").read_text())
        rows[name] = warm_window(payload)
        print(json.dumps(dict(event="WARM_WINDOW", variant=name, **rows[name])), flush=True)
    pin = rows["original"]["passing"]
    trial = rows["optimistic"]["passing"]
    status = dict(phase="warm", pin_passing=pin, optimistic_passing=trial,
                  regress=trial < pin, seconds=time.perf_counter() - started,
                  method=METHOD, alpha=ALPHA, forks=result)
    (output / "warm_status.json").write_text(json.dumps(status) + "\n")
    print(json.dumps(dict(event="WARM_DONE", pin_passing=pin, optimistic_passing=trial,
                          regress=trial < pin)), flush=True)
    return status


def run_cold(output: Path, task: str, *, optimism: bool, total_steps=None):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="pr84_optimistic_game", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    if total_steps is not None:
        config["total_steps"] = total_steps
    recipe, noise, _ = declared_recipe(config)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
    label = f"{task}_{'optimistic' if optimism else 'pr84'}"
    started = time.perf_counter()
    with pr84_optimistic_game(task=task, optimism=optimism) as (recorder, _):
        result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    verdict = test_verdict(spec, result)
    data = dict(result=result, verdict=verdict, live=result.get("live"),
                seconds=time.perf_counter() - started, optimism=optimism,
                dynamics=recorder.receipt(), noise=context["noise_receipt"])
    (output / f"{label}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
    row = dict(event="COLD_DONE", task=task, optimism=optimism, passed=verdict["passed"],
               verdict=verdict.get("verdict", verdict), live=result.get("live"),
               seconds=data["seconds"], optimistic_updates=recorder.optimistic_updates)
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "traj", "ring", "stay"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimism", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "warm":
        status = run_warm(args.output)
        if status["regress"]:
            raise SystemExit(2)
        return
    task = "trajectory" if args.phase == "traj" else "mode_hold"
    steps = 2400 if args.phase == "stay" else None
    row = run_cold(args.output, task, optimism=args.optimism, total_steps=steps)
    if not row["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
