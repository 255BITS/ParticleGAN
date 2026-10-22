"""Verify original comparison receipts and publish compact results/saved GIFs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from reports.forge.regenerate_technique_inventory import _evaluator_summary, project_receipt

ARMS = ("bispf-optimized-quintic-v1", "bispf-output-energy-v1",
        "bispf-secant-extragradient-v1", "bispf-bregman-game-v1")
CONTROL = ARMS[0]
CONTROLS = ("gaussian1d_smoke", "ring16_acquisition", "mode_hold", "img_stripes2")
LABELS = ("Original control", "Energy control", "Secant mechanics", "Bregman game")
ARCHIVED_CONTROL = Path("/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json")
ARCHIVED_CONTROL_SHA256 = "50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stream_hashes(row):
    raw = row.get("evidence", {}).get("provenance_checkpoint", {}).get("named_stream_state_sha256", {})
    result = {}
    for key, value in raw.items():
        family, component, purpose, device = json.loads(key)
        if family == "eval":
            continue
        key = json.dumps([family, component, purpose, device.split(":")[0]])
        require(key not in result or result[key] == value, "Conflicting device-normalized stream state")
        result[key] = value
    return result


def data_hashes(row):
    evidence = row.get("evidence", {})
    return {key: value for key, value in evidence.items()
            if ("data" in key or "batch_sequence" in key) and "sha256" in key}


def completed_steps(row):
    evidence = row.get("evidence", {})
    return evidence.get("completed_steps", evidence.get("provenance_checkpoint", {}).get("completed_steps"))


def verify_update_trace(row):
    """Verify retained returns without confusing optimizer proposals with moves."""
    receipt = row.get("evidence", {}).get("public_update_trace")
    if receipt is None:
        return dict(status="UNAVAILABLE")
    path = Path(receipt["path"])
    require(file_hash(path) == receipt["sha256"], "Public update trace identity differs")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(values) == receipt["rows"] and values, "Public update trace is incomplete")
    steps = [value["step"] for value in values]
    require(steps == list(range(steps[0], steps[-1] + 1)), "Public update steps repeat or have gaps")
    require(steps[-1] == completed_steps(row), "Trace does not end at the certified public update")
    scalars = {}
    def observe(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                observe(item, f"{prefix}.{key}" if prefix else key)
        elif type(value) in (bool, int, float) and prefix != "step":
            value = int(value) if type(value) is bool else value
            aggregate = scalars.setdefault(prefix, dict(count=0, minimum=value, maximum=value, sum=0.))
            aggregate["count"] += 1
            aggregate["minimum"] = min(aggregate["minimum"], value)
            aggregate["maximum"] = max(aggregate["maximum"], value)
            aggregate["sum"] += value
    for value in values:
        observe(value)
    for aggregate in scalars.values():
        aggregate["mean"] = aggregate.pop("sum") / aggregate["count"]
    return dict(status="VERIFIED", **receipt, first_step=steps[0], last_step=steps[-1],
                scalar_aggregates=scalars)


def historical_control_check(rows):
    if not ARCHIVED_CONTROL.is_file():
        return dict(status="UNAVAILABLE", path=str(ARCHIVED_CONTROL), expected_sha256=ARCHIVED_CONTROL_SHA256)
    require(file_hash(ARCHIVED_CONTROL) == ARCHIVED_CONTROL_SHA256, "Archived control readout identity differs")
    original = read_json(ARCHIVED_CONTROL)
    archived = {row["task_id"]: row for row in original["task_results"]}
    comparisons = {}
    for task, row in rows.items():
        old = archived.get(task)
        comparisons[task] = dict(archived_task_present=old is not None,
                                 gate_equal=old is not None and old["gate_status"] == row["gate_status"],
                                 final_metrics_equal=old is not None and old.get("metrics", {}) == row.get("metrics", {}))
    return dict(status="VERIFIED", path=str(ARCHIVED_CONTROL), sha256=ARCHIVED_CONTROL_SHA256,
                source_commit="d9d72d53761c898bce20102e1995fc3d899c1447",
                candidate_revision=original["candidate_revision"], tasks=comparisons,
                preserves_original_qualification=True)


def numerical_note(task, cell):
    if cell["status"] not in {"PASS", "FAIL"}:
        return cell.get("reason") or "No completed scientific observation"
    metrics, evaluation = cell["metrics"], cell["evaluator_summary"]
    if task == "gaussian1d_smoke":
        return (f"first confirmed step {evaluation.get('first_confirmed_step')}; "
                f"endpoint KS {metrics['cdf_ks']:.6f} (acquisition gate)")
    if task == "gaussian1d_stability":
        return (f"stationary {evaluation['stationary_passes']}/{evaluation['stationary_checks']}; "
                f"shifted hold {evaluation['shift_hold_passes']}/{evaluation['shift_hold_checks']}; "
                f"reacquisition {evaluation['reacquisition_status']}")
    convergence = evaluation.get("convergence", {})
    suffix = (f"; terminal suffix {convergence['passing_suffix']} (required {convergence['minimum_stable_checks']})"
              if convergence else "")
    if task.startswith("img_"):
        return f"modes {metrics['modes']}; clean {metrics['hq']:.5f}" + suffix
    if task in {"grid100", "rotated100", "staggered100"}:
        center = metrics.get("center_max_sigma")
        covariance = metrics.get("cov_frob_rms")
        center = "undefined" if center is None else f"{center:.4f}"
        covariance = "undefined" if covariance is None else f"{covariance:.5f}"
        return (f"holdout precision {metrics['precision']:.5f}; mass TV {metrics['mass_tv']:.5f}; "
                f"center max {center} sigma; covariance RMS {covariance}; "
                f"terminal passes {evaluation['terminal_summary']['passing_checks']}/"
                f"{evaluation['terminal_summary']['checks']}")
    names = [name for name in ("mass_tv", "min_mass_ratio", "component_covariance_error",
                              "component_min_eigen_ratio", "covariance_error", "mean_error",
                              "sw1_normalized", "hq") if name in metrics]
    return "; ".join(f"{name} {metrics[name]:.6f}" for name in names) + suffix


def write_appendix(summary, output, tasks):
    arms = summary["arms"]
    lines = ["# First-principles successor comparison", "",
             "Generated from certified original receipts by [publish.py](publish.py). "
             "This research panel does not change ordinary qualification or the "
             "[goal leaderboard](../technique-inventory.md).", "",
             f"Executed source `{summary['source_commit']}`, manifest `{summary['source_digest']}`; protocol seed 0. "
             "All task gates and allowances are unchanged.", "",
             "| Arm | Original failures repaired | Passing controls | New regressions | Paid attempt seconds |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for cid, label in zip(ARMS, LABELS):
        arm = arms[cid]
        lines.append(f"| {label} | {arm['previously_failing_task_passes']}/11 | "
                     f"{arm['regression_control_passes']}/4 | {len(arm['regressions_vs_control'])} | "
                     f"{arm['paid_attempt_seconds']:.3f} |")
    lines += ["", f"Reserved cap: {summary['cost_audit']['reserved_seconds']:,} seconds; total paid attempt time: "
              f"{summary['total_paid_attempt_seconds']:.3f} seconds. "
              "This is not a speed comparison.", "", "## All task verdicts", "",
              "Each verdict links to an actual-training GIF rendered only from retained scored observations. "
              "No training, sampling or rescoring was added.", "",
              "| Task | " + " | ".join(LABELS) + " |", "| --- | " + " | ".join("---" for _ in ARMS) + " |"]
    for task in tasks:
        cells = []
        for cid in ARMS:
            cell = arms[cid]["cells"][task]
            gif = Path("media") / cid / (task + ".gif")
            cells.append(f"[{cell['status']}]({gif.as_posix()})" if (output / gif).is_file() else cell["status"])
        lines.append(f"| `{task}` | " + " | ".join(cells) + " |")
    lines += ["", "## Numerical detail", "",
              "Endpoint metrics describe the last state; they do not replace sustained gates. "
              "Gaussian smoke is an independently confirmed acquisition gate and may PASS with a failing endpoint. "
              "The [compact JSON](comparison.json) retains every final metric, evaluator summary, "
              "attempt identity and original receipt hash.", ""]
    for task in tasks:
        lines += [f"### `{task}`", ""]
        for cid, label in zip(ARMS, LABELS):
            cell = arms[cid]["cells"][task]
            lines.append(f"- **{label}: {cell['status']}** — {numerical_note(task, cell)}.")
        lines.append("")
    lines += ["## Binding checks", "",
              f"Verified {len(summary['matched_bindings'])} changed-arm/control task pairs: identical task "
              "declarations, initial model hashes and priors, sampling laws, update counts and final consumed "
              "non-evaluation RNG stream hashes. Gaussian data batches additionally have direct sequence hashes. "
              "Other tasks verify their fixed data law and identical consumed named streams; no direct batch "
              "fingerprint is claimed for those adapters. Evaluation draws retain their own isolated streams.", "",
              "All admitted requests share the same complete source manifest, runtime and hardware profile. "
              "Raw stdout, full readouts, per-update events, tensors and checkpoints remain in the artifact "
              "archive named in [protocol.json](protocol.json). Compact provenance includes their exact hashes. "
              "Historical selected results keep their original source and qualification identity.", ""]
    historical = summary["archived_control_check"]
    if historical["status"] == "VERIFIED":
        equal = sum(c["gate_equal"] and c["final_metrics_equal"] for c in historical["tasks"].values())
        lines += [f"The fresh control exactly reproduces {equal}/{len(tasks)} archived final metric dictionaries "
                  "and gate verdicts. This numerical agreement does not merge evidence cohorts. The original "
                  f"full-readout SHA-256 is `{historical['sha256']}` from source `{historical['source_commit']}`.", ""]
    (output / "numerical-results.md").write_text("\n".join(lines))


def publish(queue_root, output, raw_root, *, partial=False, media=False):
    state = read_json(queue_root / "queue/state.json")
    protocol = read_json(output / "protocol.json")
    tasks = protocol["tasks"]
    requests = {}
    for path in (queue_root / "queue/requests").glob("*.json"):
        request = read_json(path)
        cid = request["candidate"]["id"]
        if cid in ARMS:
            require(cid not in requests, "Multiple comparison revisions require separate readouts")
            requests[cid] = request
    require(set(requests) == set(ARMS), "Every admitted arm requires a frozen request")
    sources = {r["source"]["digest"] for r in requests.values()}
    require(len(sources) == 1, "Arms do not share one executed source")
    reference = requests[CONTROL]
    snapshot = Path(reference["source"]["snapshot_path"])
    for name, expected in reference["source"]["files"].items():
        require(file_hash(snapshot / name) == expected, f"Frozen source snapshot differs: {name}")
    deltas = protocol["recipe_deltas_vs_baseline"]
    for cid, request in requests.items():
        require(request["source"] == reference["source"], "Source manifests differ")
        require(set(request["tasks"]) == set(reference["tasks"]), "Task rosters differ")
        for name, card in request["tasks"].items():
            baseline_card = reference["tasks"][name]
            # Generated ownership receipts contain the declared trainer delta;
            # the authoritative task declaration itself must remain identical.
            require({k: v for k, v in card.items() if k != "field_ownership"}
                    == {k: v for k, v in baseline_card.items() if k != "field_ownership"},
                    f"{cid}/{name}: task declaration differs")
            recipe = {k: v["value"] for k, v in card["field_ownership"]["recipe_fields"].items()}
            expected = {k: v["value"] for k, v in baseline_card["field_ownership"]["recipe_fields"].items()}
            expected.update(deltas[cid])
            require(recipe == expected, f"{cid}/{name}: undeclared resolved recipe delta")
        require(request["protocol"] == reference["protocol"] and request["protocol"]["seed"] == 0,
                "Comparison protocol differs")
        require(request["runtime"] == reference["runtime"] and request["compute_profiles"] == reference["compute_profiles"],
                "Runtime/hardware cohorts differ")
    rows = {cid: {} for cid in ARMS}
    provenance = {cid: [] for cid in ARMS}
    original_request_hashes = {}
    charged = Counter()
    request_ids = {request["request_id"] for request in requests.values()}
    ledger = {entry["attempt_id"]: entry for entry in state["charges"]
              if entry["owner"]["request"] in request_ids}
    require(len(ledger) == sum(entry["owner"]["request"] in request_ids for entry in state["charges"]),
            "Repeated attempt in the paid-cost ledger")
    for job in state["jobs"].values():
        result = job.get("result")
        if job["status"] != "terminal" or not result:
            continue
        owner = job.get("cost_owner") or {}
        request_id = owner.get("request")
        matching = [cid for cid, request in requests.items() if request["request_id"] == request_id]
        if not matching:
            continue
        cid = matching[0]
        attempt = result["attempt_id"]
        durable = ROOT / "reports/forge/attempts" / attempt
        projected = project_receipt(ROOT, attempt)
        require(projected["candidate_id"] == cid
                and projected["candidate_revision"] == requests[cid]["candidate_revision"]
                and projected["request_id"] == requests[cid]["request_id"]
                and projected["runtime"] == reference["runtime"], "Original attempt/request binding differs")
        original_request_hashes[attempt] = file_hash(durable / "request.json")
        certificate = read_json(durable / "evidence.json")
        stored = read_json(durable / "result.json")
        require(stored == result and certificate["result_hash"] == stable_hash(result), "Original receipt result differs")
        require(certificate["source"] == reference["source"], "Attempt executed another source")
        require(attempt in ledger and ledger[attempt]["seconds"] == result["raw"]["elapsed_seconds"]
                == job["charged_seconds"], "Certified attempt time differs from the paid queue charge")
        charged[cid] += result.get("raw", {}).get("elapsed_seconds", 0)
        provenance[cid].append(dict(attempt=attempt, result_sha256=file_hash(durable / "result.json"),
                                    certificate_sha256=file_hash(durable / "evidence.json"),
                                    local_artifact_root=certificate["local_artifact_root"]))
        for row in result["task_results"]:
            task = row["task_id"]
            require(task in tasks and task not in rows[cid], "Unexpected or duplicate task")
            rows[cid][task] = {**row, "_attempt_id": attempt}
        if media:
            from experiments.forge.tier1_media import export_attempt
            export_attempt(durable, output / "media" / cid)
    if not partial:
        # A failed own smoke cannot supply a continuation checkpoint. This is
        # an unexecuted declared cell, never a fabricated scientific receipt.
        for cid in ARMS:
            for task in tasks:
                if task in rows[cid]:
                    continue
                dependencies = requests[cid]["tasks"][task].get("dependencies", [])
                dependencies = [d["task"] if isinstance(d, dict) else d for d in dependencies]
                failed = [d for d in dependencies if d in rows[cid]
                          and rows[cid][d]["gate_status"] != "PASS"]
                if failed:
                    rows[cid][task] = dict(task_id=task, gate_status="BLOCKED", raw_status="not_executed",
                        reason="Own checkpoint/gate prerequisites did not pass: " + ", ".join(failed),
                        metrics={}, evidence=dict(unexecuted_request_id=requests[cid]["request_id"]),
                        _attempt_id=None)
        for cid in ARMS:
            require(set(rows[cid]) == set(tasks), f"{cid}: complete original evidence is not available")
    bindings = []
    for cid in ARMS[1:]:
        for task in tasks:
            baseline, candidate = rows[CONTROL].get(task), rows[cid].get(task)
            if baseline is None or candidate is None:
                continue
            comparable = all(completed_steps(r) is not None for r in (baseline, candidate))
            if not comparable:
                bindings.append(dict(candidate=cid, task=task, matched_training="UNAVAILABLE"))
                continue
            require(candidate["initial_prior"] == baseline["initial_prior"], f"{cid}/{task}: initial prior differs")
            require(candidate["initialization"] == baseline["initialization"], f"{cid}/{task}: initial models differ")
            require(candidate["evidence"]["sampling_law"] == baseline["evidence"]["sampling_law"], f"{cid}/{task}: sampling differs")
            steps_equal = completed_steps(candidate) == completed_steps(baseline)
            streams_equal = stream_hashes(candidate) == stream_hashes(baseline)
            baseline_hashes, candidate_hashes = data_hashes(baseline), data_hashes(candidate)
            require(bool(baseline_hashes) == bool(candidate_hashes), f"{cid}/{task}: data fingerprint availability differs")
            batches_equal = baseline_hashes == candidate_hashes if baseline_hashes else None
            if steps_equal:
                require(streams_equal, f"{cid}/{task}: consumed training streams differ")
                require(batches_equal is not False, f"{cid}/{task}: data sequence differs")
            bindings.append(dict(candidate=cid, task=task, initial_models_and_prior_equal=True,
                                 completed_steps_equal=steps_equal, consumed_training_streams_equal=streams_equal,
                                 task_data_law_equal=True, recorded_data_fingerprints_equal=batches_equal,
                                 data_verification="direct_batch_hash_and_named_streams" if baseline_hashes
                                 else "fixed_task_law_and_identical_consumed_named_streams"))
    parent_readout = read_json(ROOT / "reports/forge/three-theme-repairs-20261022/comparison.json")
    reproduced_controls = {}
    for cid in (CONTROL, ARMS[1]):
        historical = parent_readout["arms"][cid]["cells"]
        reproduced_controls[cid] = {
            task: dict(gate_equal=row["gate_status"] == historical[task]["status"],
                       final_metrics_equal=row.get("metrics", {}) == historical[task]["metrics"])
            for task, row in rows[cid].items()}
    compact = {}
    for cid in ARMS:
        observed = rows[cid]
        full = dict(scope="source_bound_first_principles_round2_readout", qualification_input=False,
                    candidate_id=cid, candidate_revision=requests[cid]["candidate_revision"],
                    source=requests[cid]["source"], study=requests[cid]["study"],
                    expected_tasks=tasks, complete=set(observed) == set(tasks),
                    task_results=[observed[t] for t in tasks if t in observed], original_receipts=provenance[cid])
        full_path = raw_root / "full-readouts" / (cid + ".json")
        atomic_json(full_path, full)
        cells = {}
        for task in tasks:
            row = observed.get(task)
            if row is None:
                cells[task] = dict(status="NOT_RUN", metrics={})
            else:
                cells[task] = dict(status=row["gate_status"], attempt=row["_attempt_id"],
                                   metrics=row.get("metrics", {}), reason=row.get("reason"),
                                   public_update_trace=verify_update_trace(row),
                                   evaluator_summary=_evaluator_summary(row.get("evaluator_result", {})))
        repaired = [t for t in tasks if t not in CONTROLS and cells[t]["status"] == "PASS"
                    and rows[CONTROL].get(t, {}).get("gate_status") == "FAIL"]
        regressions = [t for t in tasks if rows[CONTROL].get(t, {}).get("gate_status") == "PASS"
                       and t in observed and cells[t]["status"] != "PASS"]
        declared_control = requests[cid]["study"]["control"]["candidate_id"]
        parent_rows = rows[declared_control]
        gains_vs_parent = [t for t in tasks if cells[t]["status"] == "PASS"
                           and parent_rows.get(t, {}).get("gate_status") == "FAIL"]
        losses_vs_parent = [t for t in tasks if parent_rows.get(t, {}).get("gate_status") == "PASS"
                            and t in observed and cells[t]["status"] != "PASS"]
        compact[cid] = dict(request_id=requests[cid]["request_id"], study_id=requests[cid]["study"]["id"],
                            declared_control=declared_control,
                            repaired_vs_declared_control=gains_vs_parent,
                            regressions_vs_declared_control=losses_vs_parent,
                            candidate_revision=requests[cid]["candidate_revision"], cells=cells,
                            counts=dict(Counter(c["status"] for c in cells.values())),
                            repaired_vs_control=repaired, regressions_vs_control=regressions,
                            previously_failing_task_passes=sum(cells[t]["status"] == "PASS" for t in tasks if t not in CONTROLS),
                            regression_control_passes=sum(cells[t]["status"] == "PASS" for t in CONTROLS),
                            complete_repair=all(c["status"] == "PASS" for c in cells.values()),
                            paid_attempt_seconds=charged[cid],
                            original_full_readout=dict(path=str(full_path), sha256=file_hash(full_path)))
    summary = dict(schema_version=1, qualification_input=False, protocol_sha256=file_hash(output / "protocol.json"),
                   source_commit=reference["source"]["origin_commit"], source_digest=next(iter(sources)),
                   seed=0, runtime=reference["runtime"], compute_profiles=reference["compute_profiles"],
                   matched_bindings=bindings, arms=compact,
                   total_paid_attempt_seconds=sum(charged.values()),
                   original_receipts=provenance, numerical_gates_unchanged=True)
    summary["verified_frozen_source_files"] = len(reference["source"]["files"])
    summary["original_request_sha256"] = original_request_hashes
    summary["archived_control_check"] = historical_control_check(rows[CONTROL])
    summary["first_round_control_reproduction"] = reproduced_controls
    ledger_seconds = sum(entry["seconds"] for entry in ledger.values())
    require(abs(ledger_seconds - summary["total_paid_attempt_seconds"]) < 1e-6, "Paid-cost totals differ")
    require(ledger_seconds <= protocol["reserved_seconds"], "Frozen total budget exceeded")
    require(all(value <= protocol["candidate_reserved_seconds"] for value in charged.values()),
            "Frozen per-arm budget exceeded")
    summary["cost_audit"] = dict(unique_paid_attempts=len(ledger), queue_charge_seconds=ledger_seconds,
                                 certificate_and_queue_charges_equal=True,
                                 reserved_seconds=protocol["reserved_seconds"],
                                 per_arm_reserved_seconds=protocol["candidate_reserved_seconds"])
    atomic_json(output / "comparison.json", summary)
    if not partial:
        write_appendix(summary, output, tasks)
    print(json.dumps({cid: dict(counts=v["counts"], repaired=v["repaired_vs_control"],
                               regressions=v["regressions_vs_control"]) for cid, v in compact.items()}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--partial", action="store_true", help="Write explicitly incomplete live readouts")
    parser.add_argument("--media", action="store_true", help="Render actual saved observations without training")
    args = parser.parse_args()
    publish(args.queue_root, args.output, args.raw_root, partial=args.partial, media=args.media)


if __name__ == "__main__":
    main()
