"""Pin PR84, then mode-exit projection. Fail-fast. No merge.

Order: warm 1001-1200, stay window 1201-1600, then cold trajectory and cold ring.
Stop before cold if the warm window is worse than the same-process PR84 pin.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STAY_UNTIL = 1600


def _tail(event, **payload):
    print(json.dumps(dict(event=event, **payload)), flush=True)


def _eight(points):
    hits = [p for p in points if p["modes"] >= 8 and p["hq"] >= .9]
    return dict(checks=len(points), ge8=len(hits),
                min_modes=min((p["modes"] for p in points), default=None),
                min_hq=min((p["hq"] for p in points), default=None),
                last=None if not points else dict(step=points[-1]["step"],
                                                   modes=points[-1]["modes"],
                                                   hq=points[-1]["hq"]))


def run_warm(output):
    import torch
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
    from reports.toy100.pr84_mode_exit_projection import pr84_mode_exit_projection

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.projection = method == "projected"
        recorder.enabled = method != "identity"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        _tail("VARIANT_START", variant=method, completed=completed, projection=recorder.projection)
        if method == "identity":
            yield dict(method=method, shared_gate_eligible=False)
        else:
            with constant_rate_context(state) as rates:
                receipt = dict(method=method, shared_gate_eligible=False,
                               projection=recorder.projection)
                receipt.update(rates)
                yield receipt
            receipt.update(recorder.receipt())
            receipt.pop("records", None)
        _tail("VARIANT_DONE", variant=method)

    factories = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                 for name in ("identity", "pr84", "projected")}
    run_warm_variants(
        config, factories, output_dir=output / "warm",
        steps=STAY_UNTIL, prefix_context=lambda: pr84_mode_exit_projection(start_step=1000, projection=False))
    summary = json.loads((output / "warm" / "summary.json").read_text())
    rows = {}
    for name in ("pr84", "projected"):
        evidence = json.loads((output / "warm" / f"{name}.json").read_text())
        diag = evidence["diagnostic"]
        warm = [p for p in diag if 1000 < p["step"] <= 1200]
        stay = [p for p in diag if 1200 < p["step"] <= STAY_UNTIL]
        rows[name] = dict(warm=_eight(warm), stay=_eight(stay),
                          local=evidence["local_stability"], hold=evidence["long_hold"],
                          projected_updates=evidence.get("dynamics_receipt", {}).get("projected_updates"))
        _tail("WARM_ROW", variant=name, **rows[name])
    (output / "warm_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def run_cold(output):
    import torch
    from unittest.mock import patch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    from reports.toy100.pr84_mode_exit_projection import pr84_mode_exit_projection
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="mode_exit_projection_cold", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    policy = declared_model_policy(config)
    specs = {row["spec"]["name"]: row["spec"] for row in plan()}
    rows = {}
    for task in ("trajectory", "mode_hold"):
        for variant, factory in (
            ("pr84", pr84_smoothed_candidate),
            ("projected", lambda **kw: pr84_mode_exit_projection(projection=True, **kw)),
        ):
            _tail("COLD_START", task=task, variant=variant, steps=specs[task]["steps"])
            with factory(task=task) as (recorder, _):
                def progress(calls, outer, task=task, variant=variant, recorder=recorder):
                    if outer % 50 == 0:
                        _tail("COLD_PROGRESS", task=task, variant=variant, update=outer)
                recorder.accounting = progress
                result, _details = run_legacy(specs[task], recipe, noise, model_policy=policy)
            verdict = test_verdict(specs[task], result)
            live = result.get("live", {})
            row = dict(task=task, variant=variant, verdict=verdict.get("passed"),
                       live_modes=live.get("modes"), live_hq=live.get("hq"),
                       mse=result.get("mse"), seconds=result.get("seconds"),
                       projected_updates=recorder.receipt().get("projected_updates"))
            rows[f"{task}:{variant}"] = row
            _tail("COLD_DONE", **row)
            if variant == "projected" and task == "trajectory" and rows.get("trajectory:pr84", {}).get("verdict") and not row["verdict"]:
                _tail("KILL", reason="cold trajectory regresses versus PR84")
                break
        else:
            continue
        break
    (output / "cold_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-warm", action="store_true")
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args.output.mkdir(parents=True, exist_ok=True)
    decision = dict(
        mechanism="project G step off mode-exit direction using D directional derivative",
        purity="GAN dynamics only — D gradient, no coverage/likelihood term")
    _tail("DECLARED", stay_until=STAY_UNTIL, **decision)
    warm = None if args.skip_warm else run_warm(args.output)
    if warm is not None:
        worse = warm["projected"]["warm"]["ge8"] < warm["pr84"]["warm"]["ge8"]
        decision["warm"] = warm
        decision["warm_worse"] = worse
        if worse:
            decision["verdict"] = "KILL"
            decision["reason"] = "warm 8-mode checks regressed versus PR84; cold not run"
            (args.output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
            _tail("KILL", reason=decision["reason"])
            return
    if not args.skip_cold:
        decision["cold"] = run_cold(args.output)
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    _tail("PROBE_DONE", verdict=decision.get("verdict", "SCORED"))


if __name__ == "__main__":
    main()
