"""Run an explicit configuration search without changing behavioral thresholds.

python -u -m benchmarks.transfer_suite.solvability_search --plan plan.json --output /tmp/search
Plans contain candidates (name, overrides, optional steps_multiplier) and task names.
All tasks are now inspected development cases, including the former reserved cases.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from . import suite, vector_tasks
from .protocol import test_verdict


VECTOR_OPTIONS = set("hidden layers d_hidden d_layers fourier z_dim particles batch steps lr d_lr_mult "
                     "prior_lr_mult prior_reg betas ema_decay reg_arm reg_coeff reg_kappa loss_type gan_mode "
                     "d_every g_every".split())
IMAGE_OPTIONS = set("architecture width z_dim particles batch_size steps lr_g lr_d adam_betas "
                    "gradient_penalty penalty_coeff kappa prior_weight ema_decay".split())


def jobs(plan, declared):
    """Resolve every job before training; reject ignored keys and altered targets."""
    result, seen = [], set()
    for candidate in plan["candidates"]:
        name = candidate["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_-]+", name) or name in seen:
            raise ValueError("candidate names must be unique lowercase letters, digits, _ or -")
        seen.add(name)
        if set(candidate) - {"name", "overrides", "steps_multiplier", "tasks", "task_overrides", "schedule"}:
            raise ValueError("unknown candidate option")
        selected = candidate.get("tasks", plan.get("tasks", []))
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("task lists must be nonempty and unique")
        if set(candidate.get("task_overrides", {})) - set(selected):
            raise ValueError("task override is not in this candidate's task list")
        multiplier = candidate.get("steps_multiplier", 1)
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("steps_multiplier must be finite and positive")
        if candidate.get("schedule", "cosine") not in ("cosine", "constant"):
            raise ValueError("unsupported schedule")
        for task in selected:
            original = declared[task]
            if original["runner"] not in ("vector", "stress", "image"):
                raise ValueError("legacy hosts need explicit config translation, not ignored overrides")
            allowed = IMAGE_OPTIONS if original["runner"] == "image" else VECTOR_OPTIONS
            overrides = candidate.get("overrides", {})
            per_task = candidate.get("task_overrides", {}).get(task, {})
            if (set(overrides) | set(per_task)) - allowed:
                raise ValueError(f"unsupported training option or attempted target/metric change for {task}")
            spec = deepcopy(original)
            spec.update(overrides)
            spec["steps"] = round(spec["steps"] * multiplier)
            spec.update(per_task)
            if type(spec["steps"]) is not int or spec["steps"] < 24:
                raise ValueError("at least 24 integer training steps required")
            if original["runner"] == "image" and spec["architecture"] not in (
                    "transpose", "residual_upsample", "mean_discriminator", "uniform_generator"):
                raise ValueError("unsupported image architecture")
            result.append((candidate, original, spec, vector_tasks.fixed_policy(candidate.get("schedule", "cosine"))))
    if not result:
        raise ValueError("at least one candidate required")
    return result


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def render(output, records):
    lines = ["# Solvability search", "", "Seed 0, live weights, unchanged thresholds; 24 measurements and five final passing checks. "
             "These are inspected development tasks. Different training budgets and capacities are explicit. "
             "A per-task witness establishes solvability, not a shared default.", "",
             "| Candidate | Task | Sustained | Confirmed step | Budget | Seconds | Final failing metrics |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for record in records:
        v = record["verdict"]
        failures = ", ".join(f"{m['metric']}={m['value']:.4g}" if isinstance(m.get("value"), (int, float))
                             else str(m) for m in v.get("metrics", []) if m["status"] != "PASS")
        lines.append(f"| {record['candidate']['name']} | {record['spec']['name']} | {v['status']} | "
                     f"{v.get('convergence', {}).get('confirmed_step')} | {record['spec']['steps']} | "
                     f"{record['seconds']:.2f} | {failures or '—'} |")
    (output / "README.md").write_text("\n".join(lines) + "\n")


def run(plan, output):
    declared = {spec["name"]: spec for spec in suite.manifest()["tasks"]}
    prepared = jobs(plan, declared)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    torch.set_num_threads(1)
    write(output / "plan.json", plan)
    protocol = suite.snapshot(output)
    write(output / "protocol.json", protocol)
    records = []
    for candidate, original, spec, policy in prepared:
        task = original["name"]
        print(json.dumps(dict(event="START", candidate=candidate["name"], task=task, steps=spec["steps"])), flush=True)
        suite.verify_source(protocol)
        result = suite.run_episode(spec, policy, fixed=True, allow_reserved=True)
        verdict = test_verdict(spec, result)
        record = dict(candidate=candidate, original_spec=original, spec=spec, policy=policy,
                      verdict=verdict, seconds=result["seconds"])
        file = f"episodes/{candidate['name']}__{task}.json.gz"
        payload = dict(**record, result=result, source_sha256=protocol["source_sha256"])
        raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
        (output / file).write_bytes(gzip.compress(raw, mtime=0))
        record.update(artifact=file, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                      live=result.get("live"), ema=result.get("ema"))
        records.append(record)
        write(output / "index.json", dict(records=records))
        render(output, records)
        print(json.dumps(dict(event="DONE", candidate=candidate["name"], task=task,
                              status=verdict["status"], live=result.get("live"), seconds=result["seconds"],
                              error=result.get("error"))), flush=True)
    suite.verify_source(protocol)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(json.loads(args.plan.read_text()), args.output)
