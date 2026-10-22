"""Reduce certified saved trajectories and actual public returns; never train.

The parent publisher owns the complete comparison/leaderboard and stream/cost
audit. This diagnostic separates Gaussian endpoints from temporal gates and
checks the unchanged controls against their original first-round artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
import torch

torch.set_num_threads(1)

SOURCE = "a03373da3421303b89c99a39898b091e22910134"
DIGEST = "668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8"
ROOT = Path("/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022")
OLD_ROOT = Path("/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022")
ARMS = ("bispf-optimized-quintic-v1", "bispf-output-energy-v1",
        "bispf-secant-extragradient-v1", "bispf-bregman-game-v1")
TASKS = ("gaussian1d_smoke", "gaussian1d_stability", "vector_overlap")
CHECKED = {}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify(path, expected):
    path = Path(path)
    if str(path) not in CHECKED:
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"artifact changed: {path}")
        CHECKED[str(path)] = actual
    elif CHECKED[str(path)] != expected:
        raise ValueError("conflicting artifact identities")
    return path


def tree_hash(value):
    h = hashlib.sha256()
    def append(item):
        if isinstance(item, torch.Tensor):
            h.update(f"tensor:{item.dtype}:{tuple(item.shape)}".encode())
            h.update(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in sorted(item.items()):
                h.update(json.dumps(key).encode())
                append(child)
        elif isinstance(item, (list, tuple)):
            h.update(f"sequence:{len(item)}".encode())
            for child in item:
                append(child)
        else:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    append(value)
    return h.hexdigest()


def requests(state):
    return {item["request"]["candidate"]["id"]: (key, item["request"])
            for key, item in state["submissions"].items()}


def cell(state, request_id, task):
    jobs = [j for j in state["jobs"].values() if request_id in j["subscribers"]
            and task in j["definition"]["task_ids"]]
    if len(jobs) != 1:
        raise ValueError(f"expected one job for {request_id}/{task}")
    job = jobs[0]
    if not job.get("result"):
        return job, None
    rows = [r for r in job["result"]["task_results"] if r["task_id"] == task]
    if len(rows) != 1:
        raise ValueError("ambiguous task result")
    return job, rows[0]


def failures(row, thresholds):
    failed = []
    for name, op, bound in thresholds:
        value = row.get(name)
        good = type(value) in (int, float) and math.isfinite(value)
        if good:
            good = {">=": value >= bound, "<=": value <= bound, "==": value == bound}[op]
        if not good:
            failed.append(f"{name}{op}{bound}")
    return failed


def summary(values):
    if not values:
        return None
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite retained diagnostic")
    return dict(count=len(values), min=min(values), median=statistics.median(values),
                mean=statistics.fmean(values), max=max(values), last=values[-1])


def curve(rows, thresholds):
    tests = [not failures(r, thresholds) for r in rows]
    longest = suffix = 0
    for passed in tests:
        suffix = suffix + 1 if passed else 0
        longest = max(longest, suffix)
    return dict(observations=len(rows), passing=sum(tests), longest=longest, terminal_suffix=suffix,
                first_passing_step=next((r["step"] for r, p in zip(rows, tests) if p), None),
                passing_steps=[r["step"] for r, p in zip(rows, tests) if p],
                failure_counts=dict(Counter(f for r in rows for f in failures(r, thresholds))),
                metric_ranges={name: summary([r[name] for r in rows])
                               for name in ("cdf_ks", "mean_error_sigma", "std_ratio", "mean_error", "covariance_error", "sw1_normalized")
                               if rows and name in rows[0]})


def trajectory(row, task):
    evidence = row["evidence"]
    rows = evidence["observations"]
    thresholds = task["evaluation"]["thresholds"]
    output = dict(status=row["gate_status"], endpoint=row["metrics"],
                  endpoint_pass=not failures(row["metrics"], thresholds),
                  endpoint_failures=failures(row["metrics"], thresholds), whole_curve=curve(rows, thresholds),
                  original_thresholds=thresholds)
    if row["task_id"] == "gaussian1d_stability":
        if [r["step"] for r in rows] != [base + math.ceil(i * 1000 / 24) for base in range(1000, 6000, 1000) for i in range(1, 25)]:
            raise ValueError("Gaussian phase observations differ")
        stationary, recovery, hold = rows[:72], rows[72:96], rows[96:]
        output["phases"] = {name: curve(part, thresholds) for name, part in
                            (("stationary", stationary), ("recovery", recovery), ("shifted_hold", hold))}
        output["deadline"] = dict(required_last_five_steps=[r["step"] for r in recovery[-5:]],
            passing_last_five=sum(not failures(r, thresholds) for r in recovery[-5:]),
            status="PASS" if all(not failures(r, thresholds) for r in recovery[-5:]) else "FAIL")
        output["frozen_control"] = curve(evidence["frozen_observations"], thresholds)
        recomputed = dict(stationary_passes=output["phases"]["stationary"]["passing"], stationary_checks=72,
                          reacquisition_status=output["deadline"]["status"], shift_hold_passes=output["phases"]["shifted_hold"]["passing"],
                          shift_hold_checks=24, frozen_passes=output["frozen_control"]["passing"], frozen_checks=48)
        if recomputed != row["evaluator_result"]:
            raise ValueError("phase counts disagree with certified evaluator")
    elif row["task_id"] == "gaussian1d_smoke":
        confirmations = evidence["confirmations"]
        hits = [a["step"] for a, b in zip(rows, confirmations)
                if not failures(a, thresholds) and not failures(b["metrics"], thresholds)]
        output["same_state_independent_confirmed_steps"] = hits
        if hits != row["evaluator_result"]["confirmed_steps"]:
            raise ValueError("smoke confirmation differs")
    return output


def phase(task, step):
    if task == "gaussian1d_stability":
        return "stationary" if step <= 4000 else "recovery" if step <= 5000 else "shifted_hold"
    return "acquisition" if task == "gaussian1d_smoke" else "training"


def trace(row):
    receipt = row["evidence"]["public_update_trace"]
    path = verify(receipt["path"], receipt["sha256"])
    values = defaultdict(lambda: defaultdict(list))
    counters = defaultdict(Counter)
    alphas = defaultdict(Counter)
    first, last, count = None, None, 0
    for line in path.open():
        item = json.loads(line)
        step = item["step"]
        if last is not None and step != last + 1:
            raise ValueError("update trace has a gap")
        first = step if first is None else first
        last, count = step, count + 1
        for key in ("all", phase(row["task_id"], step)):
            arrays, counts = values[key], counters[key]
            counts["updates"] += 1
            for section, names in {
                "penalty_stats": ("s_thermal", "ema_residual", "ema_data_scale"),
                "generator_replay": ("applied_scale", "trials", "field_evaluations", "base_proposal_l2", "corrected_proposal_l2", "raw_corrected_proposal_l2", "protection_delta_l2", "applied_l2", "consistency_lhs", "consistency_rhs", "nonfinite_trials"),
                "bregman_stats": ("canonical_force_rms", "canonical_force_max", "canonical_thermal_force_rms", "used_target_fraction", "route_entropy_nats", "price_std", "selected_cost_mean"),
            }.items():
                for name in names:
                    value = item.get(section, {}).get(name)
                    if type(value) in (float, int):
                        arrays[f"{section}.{name}"].append(value)
            for name in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "output_energy", "bregman_game"):
                if name in item:
                    arrays[name].append(item[name])
            thermal = item["penalty_stats"]["s_thermal"]
            counts["thermal_exactly_one"] += thermal == 1.
            counts["thermal_below_one"] += thermal < 1.
            replay = item.get("generator_replay")
            if replay:
                alphas[key][str(replay["applied_scale"])] += 1
                counts["accepted"] += replay["accepted"]
                counts["rejected"] += not replay["accepted"]
                counts["actual_zero_move"] += replay["applied_l2"] == 0.
                lhs, rhs = replay["consistency_lhs"], replay["consistency_rhs"]
                counts["accepted_consistency_violations"] += bool(replay["accepted"] and lhs is not None and lhs > rhs)
                counts["nonzero_final_protection"] += bool(replay["protection_delta_l2"] is not None and replay["protection_delta_l2"] > 0)
                if replay["base_proposal_l2"] > 0:
                    arrays["generator_replay.applied_over_base_norm"].append(replay["applied_l2"] / replay["base_proposal_l2"])
    if count != receipt["rows"]:
        raise ValueError("update trace row count differs")
    expected_first = 1001 if row["task_id"] == "gaussian1d_stability" else 1
    if first != expected_first or last != row["evidence"].get("completed_steps", row["metrics"]["step"]):
        raise ValueError("update trace scope differs")
    return dict(receipt=receipt, first_step=first, last_step=last,
                phases={key: dict(counts=dict(counters[key]), alpha_histogram=dict(alphas[key]),
                                  scalars={name: summary(items) for name, items in arrays.items()})
                        for key, arrays in values.items()})


def saved_samples(job, row):
    receipt = row["evidence"]["saved_observer_outputs"]
    base = Path(row["evidence"].get("artifact_root") or job["attempts"][-1]["path"])
    data = torch.load(verify(base / receipt["path"], receipt["sha256"]), map_location="cpu", weights_only=False)
    return {item["step"]: tree_hash({k: item[k] for k in ("step", "samples", "confirmation_samples", "frozen_samples") if k in item})
            for item in data}


def final_state(row):
    receipt = row["evidence"]["provenance_checkpoint"]
    state = torch.load(verify(Path(receipt["artifact_root"]) / receipt["path"], receipt["sha256"]),
                       map_location="cpu", weights_only=False)["trainer"]
    return dict(models={name: tree_hash(value) for name, value in state["models"].items()},
                optimizers=[tree_hash(value) for value in state["optimizers"]])


def parity(old_job, old, new_job, new):
    old_samples, new_samples = saved_samples(old_job, old), saved_samples(new_job, new)
    if old_samples.keys() != new_samples.keys():
        raise ValueError("repeat observation cadence changed")
    changed = [step for step in old_samples if old_samples[step] != new_samples[step]]
    a, b = final_state(old), final_state(new)
    return dict(gate_equal=old["gate_status"] == new["gate_status"], final_metrics_exact=old["metrics"] == new["metrics"],
                scored_observations_exact=old["evidence"]["observations"] == new["evidence"]["observations"],
                frozen_observations_exact=old["evidence"].get("frozen_observations") == new["evidence"].get("frozen_observations"),
                sample_sets=len(old_samples), saved_sample_tensors_exact=not changed, first_different_sample_step=min(changed) if changed else None,
                final_model_tensors_exact=a["models"] == b["models"], final_optimizer_states_exact=a["optimizers"] == b["optimizers"],
                original_checkpoint_fingerprints=a, fresh_checkpoint_fingerprints=b,
                gaussian_data_digests_exact=old["evidence"].get("data_sha256") == new["evidence"].get("data_sha256"),
                scope="sample tensors include saved independent confirmations and frozen controls; exclude source-specific state metadata hashes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    began = time.process_time()
    state_path, old_path = ROOT / "queue/queue/state.json", OLD_ROOT / "queue/queue/state.json"
    state_bytes, old_bytes = state_path.read_bytes(), old_path.read_bytes()
    state, old = json.loads(state_bytes), json.loads(old_bytes)
    reqs, previous = requests(state), requests(old)
    result = dict(schema_version=1, scope="saved_trajectories_and_actual_update_returns_only", source_commit=SOURCE,
                  source_digest=DIGEST, tasks=list(TASKS), arms={}, repeated_controls={}, sampling_draws_added=0,
                  optimizer_updates_added=0, model_forwards_added=0, qualification_input=False)
    pending = []
    for arm in ARMS:
        request_id, request = reqs[arm]
        if request["source"]["origin_commit"] != SOURCE or request["source"]["digest"] != DIGEST:
            raise ValueError("source binding differs")
        cells = {}
        for task in TASKS:
            job, row = cell(state, request_id, task)
            if row is None:
                dependencies = request["tasks"][task].get("dependencies", [])
                dependencies = [d["task"] if isinstance(d, dict) else d for d in dependencies]
                failed = []
                for dependency in dependencies:
                    _, parent = cell(state, request_id, dependency)
                    if parent is not None and parent["gate_status"] != "PASS":
                        failed.append(dict(task=dependency, status=parent["gate_status"]))
                if failed:
                    if job["attempts"]:
                        raise ValueError("unsatisfied own checkpoint nevertheless executed")
                    cells[task] = dict(status="BLOCKED", attempted=False, failed_own_prerequisites=failed,
                                       scope="unexecuted declared cell, not a scientific receipt")
                else:
                    cells[task] = dict(status=job["status"].upper())
                    pending.append([arm, task])
            elif row["gate_status"] in ("PASS", "FAIL"):
                cells[task] = dict(trajectory=trajectory(row, request["tasks"][task]), update_trace=trace(row))
                if arm in ARMS[:2]:
                    old_job, old_row = cell(old, previous[arm][0], task)
                    result["repeated_controls"].setdefault(arm, {})[task] = parity(old_job, old_row, job, row)
            else:
                cells[task] = dict(status=row["gate_status"], reasons=row.get("reasons"), attempted=bool(job["attempts"]))
        result["arms"][arm] = dict(request_id=request_id, tasks=cells)
    result.update(complete=not pending, pending=pending, verified_artifacts=CHECKED,
                  raw_state_receipts={str(state_path): hashlib.sha256(state_bytes).hexdigest(),
                                      str(old_path): hashlib.sha256(old_bytes).hexdigest()},
                  reduction_cpu_seconds=time.process_time() - began, process_cpu_seconds=time.process_time())
    if pending and not args.allow_incomplete:
        raise ValueError(f"{len(pending)} required diagnostic cells pending")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), complete=result["complete"], pending=len(pending),
                          cpu_seconds=result["reduction_cpu_seconds"], verified_artifacts=len(CHECKED))))


if __name__ == "__main__":
    main()
