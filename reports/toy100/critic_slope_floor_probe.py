"""Fail-fast gates for the occupied smoothed-slope floor."""

from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100.residual_curvature import METHOD, residual_curvature

CANDIDATE = "residual_curvature"


@contextmanager
def audited(task="mode_hold", start_step=0):
    with residual_curvature(task=task, start_step=start_step) as (recorder, source):
        yield recorder, source


def variants():
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context

    def factory(method):
        @contextmanager
        def activate(state, prefix):
            recorder, _ = prefix
            recorder.enabled = method == CANDIDATE
            completed, target = state["completed_steps"], state["target_steps"]

            def accounting(calls, outer):
                state["declare_optimizer_accounting"](
                    calls=completed + calls + target - completed - outer, moment_updates=target)
                if outer % 20 == 0:
                    print(json.dumps(dict(event="WARM_PROGRESS", method=method, update=completed + outer)), flush=True)

            recorder.accounting = accounting
            receipt = dict(method=method, shared_gate_eligible=False, scratch_optimizer_policy=METHOD)
            if method == "identity":
                yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates)
                    yield receipt
                if recorder.enabled:
                    receipt.update(recorder.receipt())
        return activate
    return {name: factory(name) for name in ("identity", "constant", CANDIDATE)}


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["warm", "cold", "hold"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase != "warm" and args.previous is None:
        raise ValueError("consumed preceding gate evidence required")
    if args.phase == "cold":
        prior = json.loads(args.previous.read_text())
        if not prior["identity_cold_parity"] or prior["variants"][CANDIDATE]["status"] != "PASS":
            raise RuntimeError("warm gate failed; cold forbidden")
    elif args.phase == "hold":
        prior = json.loads(args.previous.read_text())
        if len(prior["stages"]) != 2 or any(not row.get("verdict", {}).get("passed") for row in prior["stages"]):
            raise RuntimeError("trajectory and ring must pass before hold")
    if args.previous is not None:
        (args.output / "previous-gate.json").write_bytes(args.previous.read_bytes())
    if args.phase == "warm":
        from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants
        result = run_warm_variants(
            config, variants(), output_dir=args.output / "run",
            prefix_context=lambda: audited(start_step=1000))
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(event="WARM_DONE", candidate=result["variants"][CANDIDATE])), flush=True)
        return
    if args.phase == "hold":
        from benchmarks.toy100.continuous_probe import run_probe
        with audited() as (recorder, _source):
            def hook(state):
                recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
                    calls=calls + 2400 - outer, moment_updates=2400)
            result = run_probe(
                config, mode="constant", steps=2400, noise_horizon=1200, diagnostic_every=10,
                dense_after=999, dense_until=1200, checkpoint_hook_step=1, checkpoint_hook=hook,
                log=lambda event: print(json.dumps(event), flush=True))
        result.update(dynamics_receipt=recorder.receipt(), shared_gate_eligible=False,
                      scratch_optimizer_policy=METHOD)
        (args.output / "hold.json").write_text(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps(dict(event="HOLD_DONE", status=result["status"])), flush=True)
        return
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    config.update(name="critic_slope_floor", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    stages, started = [], time.perf_counter()
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        with audited(task=task) as (recorder, _source):
            def progress(calls, outer, task=task):
                if outer % 50 == 0:
                    print(json.dumps(dict(event="COLD_PROGRESS", task=task, update=outer)), flush=True)
            recorder.accounting = progress
            result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        verdict = test_verdict(spec, result)
        (args.output / f"{task}.json").write_text(json.dumps(dict(
            result=result, verdict=verdict, dynamics=recorder.receipt(),
            shared_gate_eligible=False, scratch_optimizer_policy=METHOD), allow_nan=False) + "\n")
        stages.append(dict(task=task, verdict=verdict["status"], live=result["live"],
                           passed=verdict["passed"]))
        print(json.dumps(dict(event="STAGE_DONE", **stages[-1])), flush=True)
        if not verdict["passed"]:
            break
    status = dict(stages=stages, seconds=time.perf_counter() - started)
    (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **status)), flush=True)


if __name__ == "__main__":
    main()
