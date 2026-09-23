"""Frozen learned-LR policy versus fixed schedules on the full behavioral suite.

The controller is fitted on separate synthetic distributions. Nothing in this
module feeds held-out scores back into policy fitting or checkpoint selection.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
import traceback
from unittest.mock import patch

import torch

from particlegan import learning_rate_scale
from benchmarks.locked_shared import baseline


CONTROLLERS = ("constant", "cosine", "learned", "time_only")


class FixedSchedule:
    def __init__(self, total_steps, *, cosine):
        self.total_steps, self.cosine = total_steps, cosine
        self.base_rates, self.trace = {}, []

    def step(self, optimizer, completed_updates, role):
        rates = self.base_rates.setdefault(optimizer, [g["lr"] for g in optimizer.param_groups])
        scale = learning_rate_scale(completed_updates, self.total_steps, .6, .05) if self.cosine else 1.
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate * scale
        if completed_updates % 20 == 0:
            self.trace.append({"step": completed_updates, "role": role, "multiplier": scale})


def optimizer_role(optimizer, local_variables):
    """Bridge the existing hosts' phase names to the generic G/D contract.

    Single-optimizer identity pretraining and direct particle updates are G
    phases. No task identity or evaluation quantity reaches the controller.
    """
    if local_variables.get("opt_d") is optimizer:
        return "d"
    if any(local_variables.get(name) is optimizer for name in ("opt_g", "opt_p", "opt")):
        return "g"
    raise RuntimeError("host optimizer has no declared generator/discriminator role")


@contextmanager
def control_host_schedules(controller):
    timing = {"seconds": 0., "calls": 0}

    def before_step(optimizer, completed_updates):
        started = time.perf_counter()
        frame = inspect.currentframe().f_back
        try:
            role = optimizer_role(optimizer, frame.f_locals)
        finally:
            del frame
        controller.step(optimizer, completed_updates, role=role)
        timing["seconds"] += time.perf_counter() - started
        timing["calls"] += 1

    with ExitStack() as stack:
        for name in baseline.METRICS:
            stack.enter_context(patch.object(getattr(baseline, name), "schedule_optimizer", before_step))
        yield timing


def fingerprint(policy_path, policy):
    result = baseline.protocol()
    root = Path(__file__).resolve().parents[1]
    dependencies = [Path(__file__), *sorted((root / "benchmarks" / "learned_lr").glob("*.py"))]
    for path in dependencies:
        result["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result.update(version="learned-lr-heldout-v1", policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                  policy=policy, controllers=CONTROLLERS,
                  torch_git_revision=torch.version.git_version, torch_build=torch.__config__.show(),
                  cpu_capability=torch.backends.cpu.get_cpu_capability(), machine=platform.machine(),
                  split="All nine behavioral hosts held out from controller fitting; policy frozen before this run.",
                  schedule_policy="Replace host schedules before every optimizer update; fixed initial group ratios.",
                  phase_bridge="opt_d -> D; opt_g/opt_p/opt -> G, including identity pretraining",
                  controller_timing="CPU callback time including feature extraction and research integration bridge")
    return result


def render(report, path):
    lines = ["# Learned LR adapter — held-out behavioral comparison", "",
             "One frozen policy; all nine hosts are held out from policy fitting. Live weights determine PASS. "
             "The ten shared checks are reported separately and do not rank controllers. "
             "`time_only` keeps the learned bias/progress coefficients and suppresses feedback features.", "",
             "| Controller | Live toys | Bounds | Sustained toys | Ring modes / HQ | Ring confirmed | Total seconds | Controller seconds | Overall |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in report["rows"]:
        score = baseline.score_row(row, report["shared"])
        toys = row["toys"]
        stable = sum(v.get("convergence", {}).get("stable_from_step") is not None for v in toys.values())
        ring = toys.get("mode_hold", {})
        live = ring.get("live", {})
        hq = live.get("hq")
        ring_text = f"{live.get('modes', '—')}/8 / {hq:.2%}" if isinstance(hq, (float, int)) else "—"
        confirmed = ring.get("convergence", {}).get("confirmed_step")
        seconds = sum(v.get("seconds", 0) for v in toys.values())
        overhead = sum(v.get("controller_timing", {}).get("seconds", 0) for v in toys.values())
        lines.append(f"| {row['controller']} | {score['passed_toys']}/9 | {score['passed_metrics']}/29 | {stable}/9 | "
                     f"{ring_text} | {confirmed if confirmed is not None else '—'} | {seconds:.2f} | {overhead:.2f} | {score['status']} |")
    lines += ["", "Sustained success requires a complete 24-point curve and at least five final passing observations. "
              "The ring additionally requires 8/8 modes and HQ≥90%. No incomplete row can pass. "
              "Times are single observations, not replicated speed estimates.", "",
              "| Controller | " + " | ".join(baseline.METRICS) + " |",
              "| --- | " + " | ".join("---" for _ in baseline.METRICS) + " |"]
    for row in report["rows"]:
        score = baseline.score_row(row, report["shared"])
        lines.append("| " + row["controller"] + " | " + " | ".join(score["toys"][k]["status"] for k in baseline.METRICS) + " |")
    lines += ["", "EMA remains separate:", "", "| Controller / toy | EMA metrics |", "| --- | --- |"]
    for row in report["rows"]:
        for name, toy in row["toys"].items():
            if "ema" in toy:
                lines.append(f"| {row['controller']} / {name} | `{json.dumps(toy['ema'], sort_keys=True)}` |")
    passed = sum(v.get("status") == "PASS" for v in report["shared"].values())
    lines += ["", f"Shared checks: {passed}/10 PASS. Raw bounds, curves, errors, action traces, source hashes "
              "and the exact frozen policy are in [results.json](results.json).", ""]
    path.write_text("\n".join(lines))


def run(policy_path, candidate_path, reference, output):
    from benchmarks.learned_lr.controller import OptimizerLRAdapter

    if (output / "results.json").exists():
        raise FileExistsError("use a new output directory; held-out results cannot be silently overwritten")
    torch.set_num_threads(1)
    policy = json.loads(policy_path.read_text())
    cards = json.loads(candidate_path.read_text())
    if len(cards) != 1:
        raise ValueError("supply one fixed formulation for a schedule-only comparison")
    candidate = baseline.Candidate(**cards[0])
    # Scheduling belongs exclusively to the compared controller.
    candidate = replace(candidate, lr_schedule="host")
    protocol = fingerprint(policy_path, policy)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
              "protocol_sha256": baseline.digest(protocol), "rows": [], "shared": {}}
    for kind in CONTROLLERS:
        config = asdict(replace(candidate, name=kind))
        report["rows"].append({"controller": kind, "config": config,
                               "config_sha256": baseline.digest(config), "toys": {}})
    output.mkdir(parents=True, exist_ok=True)

    def save():
        baseline.write_json(output / "results.json", report)
        render(report, output / "README.md")

    save()
    for row in report["rows"]:
        for toy, budget in baseline.BUDGETS.items():
            kind = row["controller"]
            controller = (FixedSchedule(budget, cosine=kind == "cosine") if kind in ("constant", "cosine") else
                          OptimizerLRAdapter(policy, budget, ablation="time_only" if kind == "time_only" else "none", interval=20))
            print(f"START heldout controller={kind} toy={toy} steps={budget}", flush=True)
            started = time.perf_counter()
            timing = {}
            try:
                with control_host_schedules(controller) as timing:
                    result = baseline.run_toy(toy, baseline.Candidate(**row["config"]))
                result["controller_trace"] = controller.trace
                json.dumps(result, allow_nan=False)
            except Exception:
                result = {"error": traceback.format_exc()}
            result["seconds"] = time.perf_counter() - started
            result["controller_timing"] = dict(timing)
            row["toys"][toy] = result
            save()
            print(json.dumps({"event": "HELDOUT_DONE", "controller": kind, "toy": toy,
                              "live": result.get("live"), "error": result.get("error"),
                              "seconds": result["seconds"]}, allow_nan=False), flush=True)
    if reference:
        from benchmarks.locked_shared.shared_checks import run_shared
        run_shared(reference, report, save)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("reports/behavioral_baseline/convergence/leading_config.json"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.policy, args.candidate, args.reference, args.output)
    return 0 if all(len(r["toys"]) == 9 and all("error" not in t for t in r["toys"].values()) for r in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
