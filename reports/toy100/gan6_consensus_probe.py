"""Fail-fast GAN-6 gates: warm, cold trajectory, cold ring, then stay.

Warm regression against the PR84 pin stops the ladder. Logs are one JSON
object per line.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.gan6_particle_consensus import METHOD, gan6_particle_consensus
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


def _log(event, **payload):
    print(json.dumps(dict(event=event, **payload)), flush=True)


def _constant_config():
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="gan6_consensus", lr_floor=1.0, lr_anneal_start=0.0)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    return config


def _context(arm, task, start_step=0):
    if arm == "pr84":
        return pr84_smoothed_candidate(task=task, start_step=start_step)
    if arm == "gan6":
        return gan6_particle_consensus(task=task, start_step=start_step, correction=True)
    raise ValueError(arm)


def run_warm(arm, output: Path):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == arm
        completed, target = state["completed_steps"], state["target_steps"]

        def accounting(calls, outer):
            state["declare_optimizer_accounting"](
                calls=completed + calls + target - completed - outer, moment_updates=target)
            if outer % 50 == 0:
                _log("WARM_PROGRESS", arm=arm, variant=method, update=completed + outer)

        recorder.accounting = accounting
        receipt = dict(method=method, arm=arm, shared_gate_eligible=False)
        if method == "identity":
            yield receipt
            return
        with constant_rate_context(state) as rates:
            receipt.update(rates)
            yield receipt
        if recorder.enabled:
            receipt.update(recorder.receipt())

    factories = {
        "identity": lambda state, prefix: activate("identity", state, prefix),
        arm: lambda state, prefix: activate(arm, state, prefix),
    }
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    _log("WARM_START", arm=arm)
    run_warm_variants(
        config, factories, output_dir=output / "forks",
        prefix_context=lambda: _context(arm, "mode_hold", start_step=1000))
    summary = json.loads((output / "forks" / "summary.json").read_text())
    variant = json.loads((output / "forks" / f"{arm}.json").read_text())
    row = dict(arm=arm, status=variant["status"], local=variant["local_stability"],
               warm_state_sha256=variant["warm_state_sha256"], summary_status=summary["status"])
    _log("WARM_DONE", **row)
    (output / "warm.json").write_text(json.dumps(row) + "\n")
    return row


def run_cold(arm, output: Path):
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe

    config = _constant_config()
    recipe, noise, _ = declared_recipe(config)
    model_policy = declared_model_policy(config)
    stages = []
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        _log("COLD_START", arm=arm, task=task, steps=spec["steps"])
        with _context(arm, task, start_step=0) as (recorder, _source):
            def progress(calls, outer, task=task):
                if outer % 100 == 0:
                    _log("COLD_PROGRESS", arm=arm, task=task, update=outer, field_calls=calls)
            recorder.accounting = progress
            result, _context_details = run_legacy(spec, recipe, noise, model_policy=model_policy)
        verdict = test_verdict(spec, result)
        live = result.get("live", {})
        support = (live.get("support") or {}) if isinstance(live, dict) else {}
        row = dict(task=task, passed=bool(verdict.get("passed")), verdict=verdict,
                   modes=live.get("modes", support.get("modes")), hq=live.get("hq", support.get("hq")),
                   live_modes=live.get("modes"), live_hq=live.get("hq"))
        if hasattr(recorder, "consensus_records"):
            actions = {}
            for item in recorder.consensus_records:
                actions[item["action"]] = actions.get(item["action"], 0) + 1
            row["consensus_actions"] = actions
        stages.append(row)
        _log("STAGE_DONE", arm=arm, **{k: row[k] for k in row if k != "verdict"},
             verdict_passed=row["passed"])
        (output / f"{task}.json").write_text(json.dumps(dict(stage=row, live=live), default=str) + "\n")
        if not row["passed"]:
            break
    (output / "cold.json").write_text(json.dumps(dict(arm=arm, stages=stages)) + "\n")
    return stages


def run_stay(output: Path):
    from benchmarks.toy100.continuous_probe import run_probe

    config = _constant_config()
    _log("STAY_START", steps=2400)
    with gan6_particle_consensus(task="mode_hold", start_step=0) as (recorder, _source):
        def hook(state):
            recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
                calls=calls + 2400 - outer, moment_updates=2400)
        result = run_probe(
            config, mode="constant", steps=2400, noise_horizon=1200, diagnostic_every=50,
            checkpoint_hook_step=1, checkpoint_hook=hook,
            log=lambda event: _log("STAY_TICK", **event) if isinstance(event, dict) else None)
    tail = [dict(step=row["step"], modes=row.get("modes"), hq=row.get("hq"))
            for row in result.get("diagnostic", []) if row["step"] >= 1200]
    summary = dict(status=result.get("status"), stationary=result.get("stationary"),
                   continued_hold=result.get("continued_hold"), tail=tail[-8:])
    _log("STAY_DONE", status=summary["status"], tail=summary["tail"])
    (output / "stay.json").write_text(json.dumps(summary, default=str) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["pr84", "gan6"], required=True)
    parser.add_argument("--phase", choices=["warm", "cold", "stay"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-gate", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.phase == "cold" and args.arm == "gan6":
        if args.warm_gate is None:
            raise SystemExit("gan6 cold requires --warm-gate")
        gate = json.loads(args.warm_gate.read_text())
        if gate.get("status") != "PASS":
            _log("STOP", reason="warm regression", gate=gate)
            raise SystemExit(2)
    if args.phase == "warm":
        run_warm(args.arm, args.output)
    elif args.phase == "cold":
        run_cold(args.arm, args.output)
    else:
        run_stay(args.output)
    _log("PHASE_SECONDS", arm=args.arm, phase=args.phase, seconds=round(time.perf_counter() - started, 1))


if __name__ == "__main__":
    main()
