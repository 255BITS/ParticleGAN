"""Fail-fast GAN-5 gates against a same-process PR84 pin.

Order: warm constant-rate continuation, then cold trajectory, then cold ring.
A warm regression stops the run. Logs are one JSON object per line.
"""

from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.gan5_spectral_basin import BASIN_COEFF, BASIN_EPS, METHOD, gan5_spectral_basin

PR84 = "pr84"
BASIN = "basin"


@contextmanager
def host(task="mode_hold", start_step=0):
    with gan5_spectral_basin(task=task, start_step=start_step, basin=True) as yielded:
        yield yielded


def variants():
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context

    def factory(method):
        @contextmanager
        def activate(state, prefix):
            recorder, _ = prefix
            recorder.enabled = method != "identity"
            recorder._basin = method == BASIN
            completed, target = state["completed_steps"], state["target_steps"]

            def accounting(calls, outer):
                state["declare_optimizer_accounting"](
                    calls=completed + calls + target - completed - outer,
                    moment_updates=target)
                if outer % 50 == 0:
                    print(json.dumps(dict(event="WARM_PROGRESS", method=method,
                                          update=completed + outer, field_calls=calls)), flush=True)

            recorder.accounting = accounting
            receipt = dict(method=method, shared_gate_eligible=False)
            if method == "identity":
                yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates)
                    yield receipt
                if recorder.enabled:
                    receipt.update(recorder.receipt())
                    receipt["basin"] = method == BASIN
            print(json.dumps(dict(event="WARM_CHILD_DONE", method=method,
                                  status=receipt.get("method"))), flush=True)
        return activate

    return {name: factory(name) for name in ("identity", PR84, BASIN)}


def _passing(summary, name):
    row = summary["variants"][name]
    local = row["local_stability"]
    return dict(status=row["status"], passing=local["passing_checks"],
                checks=local["checks"], min_modes=local["min_modes"],
                min_hq=local["min_hq"], failing=local["failing_steps"],
                final_modes=row["final"].get("modes"), final_hq=row["final"].get("hq"))


def run_warm(output: Path):
    import torch
    from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(dict(event="WARM_START", method=METHOD, coeff=BASIN_COEFF, eps=BASIN_EPS)), flush=True)
    with host(start_step=1000) as (_, source):
        (output / "mode_hold_transformed.py").write_text(source)
    started = time.perf_counter()
    summary = run_warm_variants(
        config, variants(), output_dir=output / "run",
        prefix_context=lambda: host(start_step=1000))
    summary["seconds"] = time.perf_counter() - started
    pin = _passing(summary, PR84)
    cand = _passing(summary, BASIN)
    summary["pin_pr84"] = pin
    summary["candidate"] = cand
    summary["warm_regressed"] = cand["passing"] < pin["passing"]
    (output / "summary.json").write_text(json.dumps(
        {k: summary[k] for k in ("pin_pr84", "candidate", "warm_regressed", "seconds",
                                 "identity_cold_parity", "warm_state_sha256")},
        indent=2) + "\n")
    print(json.dumps(dict(event="WARM_DONE", pin=pin, candidate=cand,
                          regressed=summary["warm_regressed"], seconds=summary["seconds"])), flush=True)
    return summary


def run_cold(output: Path, tasks):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="gan5_spectral_basin", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    output.mkdir(parents=True, exist_ok=True)
    stages = []
    started = time.perf_counter()
    for task in tasks:
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        print(json.dumps(dict(event="COLD_START", task=task)), flush=True)
        with host(task=task, start_step=0) as (recorder, source):
            (output / f"{task}_transformed.py").write_text(source)

            def progress(calls, outer, task=task):
                if outer % 100 == 0:
                    print(json.dumps(dict(event="COLD_PROGRESS", task=task, update=outer,
                                          field_calls=calls)), flush=True)

            recorder.accounting = progress
            result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        verdict = test_verdict(spec, result)
        live = result.get("live", {})
        row = dict(task=task, passed=verdict["passed"], verdict=verdict, live=live)
        (output / f"{task}.json").write_text(json.dumps(
            dict(verdict=verdict, live=live, dynamics=recorder.receipt()), allow_nan=False) + "\n")
        stages.append(row)
        print(json.dumps(dict(event="STAGE_DONE", task=task, passed=verdict["passed"], live=live)), flush=True)
        if not verdict["passed"]:
            break
    status = dict(stages=[{k: r[k] for k in ("task", "passed", "live")} for r in stages],
                  seconds=time.perf_counter() - started)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(dict(event="COLD_DONE", **status)), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["warm", "cold", "ring"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.phase == "warm":
        run_warm(args.output)
        return
    if args.previous is None or json.loads(args.previous.read_text()).get("warm_regressed") is not False:
        raise RuntimeError("warm gate did not pass; cold forbidden")
    if args.phase == "cold":
        run_cold(args.output, ("trajectory",))
        return
    prior = json.loads(args.previous.read_text())
    # ring uses the trajectory status file as --previous after a second check
    if "stages" in prior and not all(row.get("passed") for row in prior["stages"]):
        raise RuntimeError("trajectory gate failed; ring forbidden")
    run_cold(args.output, ("mode_hold",))


if __name__ == "__main__":
    main()
