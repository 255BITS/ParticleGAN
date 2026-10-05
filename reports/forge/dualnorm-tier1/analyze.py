"""Read the frozen BCAP screen and plot certified, non-grading diagnostics.

Run ``run.py report`` first. This reader never trains, grades, selects a default,
or changes the single goal leaderboard. Raw observations remain in the queue or
the optional archive; the JSON output contains final metrics and compact audits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.knowledge import _resolve_retries

CAMPAIGN = "bcap-dualnorm-tier1-v1"
SOURCE_COMMIT = "15eb7cb0911905e401bdfcd7e264945a7ea64d97"
SOURCE_DIGEST = "c5c60a8018e144c447325f3705dd398d04069a91bb294eaaa76d37084ade67cc"
ARMS = ("adam", "sgda", "nsgda-global", "nsgda-layer", "ada-nsgda", "dualnorm-zero",
        "dualnorm-momentum", "dualnorm-d-only", "particle-rownorm-only")
REQUIRED = ("gaussian1d_acquisition", "two_pole", "unused_token_hold", "ae_gan_hold",
            "ring16_acquisition", "five_word_joint_acquisition")
DIAGNOSTIC = "clockfree_audit_measurement_v1"
TRACED = ("gaussian1d_acquisition", "ring16_acquisition", "five_word_joint_acquisition")
TERMINAL = {"PASS", "FAIL", "INVALID", "INCOMPLETE", "BLOCKED"}
TERMINAL_SUBMISSIONS = {"completed", "blocked", "terminal", "concluded"}


def task_map(trial):
    return {task["task"]: task for task in trial["tasks"] if task["qualification_tier"] == 1}


def execution_complete(trial):
    tasks = task_map(trial)
    return (trial.get("submission_status", "").lower() in TERMINAL_SUBMISSIONS
            and set(tasks) == {*REQUIRED, DIAGNOSTIC}
            and all(task["gate_status"] in TERMINAL for task in tasks.values()))


def whole_selection(trials):
    """Retain Forge's objective, requiring every independent peer to finish."""
    selection = select_configuration(trials, 1)
    if not all(execution_complete(trial) for trial in trials):
        selection.update(selection_complete=False, all_trials_terminal=False,
                         selected_candidate_id=None, selected_configuration_id=None,
                         qualified=False, selection_kind="pending", required_pass_count=None,
                         required_total=None, required_passes_by_tier=None,
                         source_digest=None, required_task_bindings=[])
    return selection


def load_reports(root, allow_partial=False):
    reports, trials = {}, []
    for arm in ARMS:
        path = root / "reports/forge/configuration-search" / f"bcap-optim-{arm}-tier1-v1.json"
        report = read_json(path)
        if report.get("input_digest") != stable_hash({k: v for k, v in report.items() if k != "input_digest"}):
            raise ValueError(f"search report digest mismatch: {path}")
        if set(report.get("source_digests", [])) != {SOURCE_DIGEST}:
            raise ValueError(f"search report differs from the executed source: {path}")
        reports[arm] = report
        for original in report["trials"]:
            trial = deepcopy(original)
            trial["arm"] = arm
            if trial.get("source_digest") != SOURCE_DIGEST:
                raise ValueError("a trial differs from the executed source")
            if set(task_map(trial)) != {*REQUIRED, DIAGNOSTIC}:
                raise ValueError("the six required tasks and separate diagnostic must remain intact")
            trials.append(trial)
    if len(trials) != 41 or len({trial["candidate_id"] for trial in trials}) != 41:
        raise ValueError("this screen requires its exact 41 distinct whole configurations")
    if len({report["policy_fingerprint"] for report in reports.values()}) != 1:
        raise ValueError("the studies do not share one frozen policy")
    if not allow_partial and not all(execution_complete(trial) for trial in trials):
        raise ValueError("screen unfinished; run the final report after all peers finish, or use --allow-partial")
    return reports, trials


class Artifacts:
    """Read local artifacts or exact named archive members, without extraction."""

    def __init__(self, root, queue_root, archive=None):
        self.root, self.queue_root = root, queue_root
        self.archive = tarfile.open(archive, "r:gz") if archive else None

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, attempt, name, *, durable=False):
        local = (self.root / "reports/forge/attempts" / attempt if durable
                 else self.queue_root / CAMPAIGN / attempt) / name
        if local.is_file():
            return local.read_bytes()
        if self.archive:
            member = ("durable/" if durable else "attempts/") + attempt + "/" + name
            try:
                stream = self.archive.extractfile(member)
            except KeyError:
                stream = None
            if stream:
                with stream:
                    return stream.read()
        raise FileNotFoundError(f"original artifact unavailable: {attempt}/{name}")

    def receipt(self, trial, task):
        try:
            return self._receipt(trial, task)
        except (ValueError, FileNotFoundError):
            # A recorded INVALID cell stays in the denominator, but damaged or
            # unavailable originals cannot supply numerical evidence or audits.
            if task["gate_status"] == "INVALID":
                return None
            raise

    def _receipt(self, trial, task):
        attempt = task.get("attempt_id")
        if not attempt:
            return None
        resolved = json.loads(self.read(attempt, "request.json", durable=True))
        request = resolved.get("request", resolved)
        result = json.loads(self.read(attempt, "result.json", durable=True))
        certificate = json.loads(self.read(attempt, "evidence.json", durable=True))
        source = request.get("source", {})
        if (certificate.get("result_hash") != stable_hash(result)
                or certificate.get("source") != source
                or certificate.get("runtime") != request.get("runtime")
                or source.get("origin_commit") != SOURCE_COMMIT
                or source.get("digest") != SOURCE_DIGEST
                or stable_hash(source.get("files", {})) != SOURCE_DIGEST
                or request.get("protocol", {}).get("seed") != 0
                or request.get("candidate_revision") != trial["candidate_revision"]
                or result.get("candidate_revision") != trial["candidate_revision"]
                or request.get("candidate", {}).get("id") != trial["candidate_id"]
                or request.get("campaign_id") != CAMPAIGN
                or result.get("attempt_id") != attempt
                or result.get("retry_of") != resolved.get("retry_of")):
            raise ValueError(f"durable receipt binding mismatch: {attempt}")
        matches = [row for row in result.get("task_results", []) if row["task_id"] == task["task"]]
        if len(matches) != 1:
            raise ValueError(f"missing or duplicate task result: {attempt}/{task['task']}")
        row = matches[0]
        if (row.get("compatibility_key") != task["compatibility_key"]
                or row.get("gate_status", row.get("status")) != task["gate_status"]):
            raise ValueError(f"search task differs from certified result: {attempt}/{task['task']}")
        return request, result, certificate, row

    def trace(self, attempt, receipt):
        if (receipt.get("qualification_input") is not False
                or receipt.get("sampling_draws_added") != 0
                or receipt.get("optimizer_updates_added") != 0):
            raise ValueError("diagnostic receipt must declare no grading input or extra draws/updates")
        name = receipt["path"]
        if Path(name).name != name:
            raise ValueError("diagnostic path must be an artifact basename")
        payload = self.read(attempt, name)
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
            raise ValueError("diagnostic trace differs from its certified receipt")
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if len(rows) != receipt["rows"]:
            raise ValueError("diagnostic row count differs from its receipt")
        return valid_trace(rows)


def canonical_rng(rng):
    """Compare device classes while retaining CPU/CUDA state distinctions."""
    bindings = []
    for binding in rng.get("bindings", {}).values():
        device = binding.get("device", "")
        device = "cuda" if device.startswith("cuda:") else device
        bindings.append({k: binding[k] for k in ("family", "component", "purpose", "seed", "initial_state_sha256")}
                        | {"device": device})
    return {"seed": rng.get("seed"), "version": rng.get("version"),
            "bindings": sorted(bindings, key=lambda binding: json.dumps(binding, sort_keys=True))}


def valid_trace(rows):
    """Whitelist usable observations, excluding the frozen input-label bug."""
    result, previous = [], {}
    for raw in rows:
        role, step = raw["optimizer"], raw["step"]
        if role not in {"D", "G"} or step <= previous.get(role, 0):
            raise ValueError("diagnostic steps must increase separately for each optimizer")
        previous[role] = step
        row = {key: deepcopy(raw[key]) for key in ("step", "optimizer", "layers", "players")}
        for key in ("prior", "critic_weight_spectral_norms", "critic_log_spectral_product", "spectral_proxy_scope"):
            if key in raw:
                row[key] = deepcopy(raw[key])
        def finite(value):
            if isinstance(value, dict):
                return all(finite(child) for child in value.values())
            if isinstance(value, list):
                return all(finite(child) for child in value)
            return not isinstance(value, float) or math.isfinite(value)
        if not finite(row):
            raise ValueError("nonfinite retained diagnostic values")
        result.append(row)
    return result


def curve_stats(values):
    if not values:
        return None
    return {"observations": len(values), "first": values[0], "final": values[-1],
            "minimum": min(values), "maximum": max(values),
            "total_variation": sum(abs(b - a) for a, b in zip(values, values[1:]))}


def trace_summary(rows):
    players, weights, spectral, prior = defaultdict(list), defaultdict(list), [], defaultdict(list)
    shapes = {}
    for row in rows:
        for player, value in row["players"].items():
            players[player].append(value["relative_update_sum"])
        for layer in row["layers"]:
            key = (row["optimizer"], layer["index"], layer["player"], layer["component"])
            weights[key].append(layer["weight_norm"])
            shapes[key] = layer["shape"]
        if "critic_log_spectral_product" in row:
            spectral.append(row["critic_log_spectral_product"])
        for key in ("mean_support_row_displacement", "max_outside_support_row_displacement"):
            if key in row.get("prior", {}):
                prior[key].append(row["prior"][key])
    layers = []
    for key, values in sorted(weights.items()):
        ratio = values[-1] / values[0] if values[0] > 0 else None
        layers.append({"optimizer": key[0], "index": key[1], "player": key[2], "component": key[3],
                       "shape": shapes[key], "weight_norm": curve_stats(values),
                       "final_over_first": ratio, "growth_ge_10x": ratio is not None and ratio >= 10})
    sampled_rows = [row["prior"] for row in rows if row.get("prior", {}).get("row_support_kind") == "sampled_indices"]
    return {"rows": len(rows), "relative_update_sum": {k: curve_stats(v) for k, v in players.items()},
            "layer_weight_norms": layers, "critic_log_spectral_product": curve_stats(spectral),
            "particle_displacement": {k: curve_stats(v) for k, v in prior.items()},
            "particle_support_kinds": sorted({row["prior"]["row_support_kind"] for row in rows if "prior" in row}),
            "maximum_certified_unsampled_row_displacement": max((row["max_outside_support_row_displacement"] for row in sampled_rows), default=None),
            "growth_flag_scope": "A >=10x endpoint increase is a descriptive finite-budget flag; it does not prove unbounded growth."}


def numeric_delta(actual, reference):
    return {key: actual[key] - reference[key] for key in actual.keys() & reference.keys()
            if isinstance(actual[key], (int, float)) and not isinstance(actual[key], bool)
            and isinstance(reference[key], (int, float)) and not isinstance(reference[key], bool)}


def aggregate_state_hash(binding):
    """Use only an actually emitted SHA-256 of the complete component state."""
    value = binding.get("initial_state_sha256")
    return value if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) else None


def compact_task(task, receipt):
    result = {key: deepcopy(task[key]) for key in ("task", "importance", "gate_status", "raw_status",
              "metrics", "attempt_id", "compatibility_key", "cost", "reason") if key in task}
    if not receipt and task["gate_status"] == "INVALID":
        result["metrics"] = {}
        result["receipt_limitation"] = "Original certificate is unavailable or invalid; no numerical evidence or training audit attributed."
    if receipt:
        request, _, certificate, row = receipt
        evidence = row.get("evidence", {})
        applied = row.get("applied", {})
        initialization = row.get("initialization", applied.get("initialization", {}))
        rng = row.get("rng", applied.get("rng", {}))
        result["receipt"] = {"result_hash": certificate["result_hash"], "source_digest": SOURCE_DIGEST,
                             "source_commit": SOURCE_COMMIT, "seed": request["protocol"]["seed"]}
        result["convergence"] = deepcopy(row.get("evaluator_result", {}).get("convergence"))
        result["guards"] = {key: deepcopy(evidence.get("guards", {})[key])
                            for key in ("all_finite", "hooks_exercised", "optimizer_updates", "unintended_rng_deviations")
                            if key in evidence.get("guards", {})}
        result["initialization"] = {name: {"aggregate_state_hash_available": aggregate_state_hash(item) is not None,
                                           **({"initial_state_sha256": aggregate_state_hash(item)} if aggregate_state_hash(item) else {}),
                                           **({"initializer": item["initializer"]} if "initializer" in item else {})}
                                    for name, item in initialization.items()}
        result["rng_start_digest"] = stable_hash(canonical_rng(rng)) if rng else None
        result["initializer_contract"] = row.get("initializer", applied.get("initializer"))
        result["declared_fixed_initialization"] = request["tasks"][task["task"]]["execution"].get("fixed_initialization")
        if not evidence:
            result["receipt_limitation"] = "Execution error retained; training initialization, stream and guard audit unavailable."
        if row.get("error"):
            result["error"] = deepcopy(row["error"])
    return result


def audit_task(name, observed, expected):
    """Summarize matches without treating absent error receipts as matches."""
    initializations, streams, contracts = defaultdict(list), defaultdict(list), defaultdict(list)
    component_receipts, missing_hashes = Counter(), defaultdict(list)
    guards, missing, updates, fixtures = Counter(), [], defaultdict(list), []
    for candidate, task, receipt in observed:
        request, _, _, row = receipt
        applied, evidence = row.get("applied", {}), row.get("evidence", {})
        initialization = row.get("initialization", applied.get("initialization", {}))
        rng = row.get("rng", applied.get("rng", {}))
        for component, binding in initialization.items():
            component_receipts[component] += 1
            state_hash = aggregate_state_hash(binding)
            if state_hash:
                initializations[component].append(state_hash)
            else:
                missing_hashes[component].append(candidate)
        if rng:
            canonical = canonical_rng(rng)
            for binding in canonical["bindings"]:
                key = json.dumps([binding[k] for k in ("family", "component", "purpose", "device")])
                streams[key].append(binding)
        task_contract = request["tasks"][name]
        fixed = task_contract["execution"].get("fixed_initialization")
        if fixed:
            fixtures.append(fixed)
        for key in ("execution", "evaluation"):
            contracts[key].append(stable_hash(task_contract[key]))
        guard = evidence.get("guards", {})
        for role, count in guard.get("optimizer_updates", {}).items():
            updates[role].append(count)
        for key in ("all_finite", "hooks_exercised", "unintended_rng_deviations"):
            if key in guard:
                guards[f"{key}={guard[key]}"] += 1
        if ((not initialization or any(aggregate_state_hash(binding) is None for binding in initialization.values()))
                and not fixed) or not rng or not guard:
            missing.append(candidate)
    return {"task": name, "expected_configurations": expected, "certified_receipts": len(observed),
            "initialization": {component: {"receipts": len(initializations[component]),
                                           "component_metadata_receipts": component_receipts[component],
                                           "state_hashes": sorted(set(initializations[component])),
                                           "matched_present": bool(initializations[component]) and len(set(initializations[component])) == 1,
                                           "matched_all_component_receipts": bool(initializations[component])
                                               and len(initializations[component]) == component_receipts[component]
                                               and len(set(initializations[component])) == 1,
                                           "complete_for_expected_configurations": len(initializations[component]) == expected,
                                           "missing_aggregate_hash_candidates": missing_hashes[component]}
                               for component in sorted(component_receipts)},
            "initialization_audit_scope": "Only emitted complete-component initial_state_sha256 values establish state matches. Seeds, initializer names and per-tensor metadata are not substituted for aggregate state hashes.",
            "stream_starts": {key: {"receipts": len(values), "distinct_starts": len({stable_hash(v) for v in values}),
                                    "matched_present": len({stable_hash(v) for v in values}) == 1,
                                    "binding": values[0] if len({stable_hash(v) for v in values}) == 1 else None}
                              for key, values in sorted(streams.items())},
            "task_contracts": {key: {"distinct_hashes": sorted(set(values)), "matched_present": len(set(values)) == 1}
                               for key, values in contracts.items()},
            "declared_fixed_initialization": fixtures[0] if fixtures and len({stable_hash(item) for item in fixtures}) == 1 else None,
            "fixed_initialization_scope": "Stored host weights / zero coordinates retain their explicit task cohort; component initialization hashes were not emitted for this fixture." if fixtures else None,
            "optimizer_update_counts": {role: {"receipts": len(counts), "distinct_counts": sorted(set(counts))}
                                        for role, counts in sorted(updates.items())},
            "guards": dict(guards), "missing_training_audit_candidates": missing,
            "batch_sequence_scope": "Task laws, budgets and named stream starts are bound; no direct consumed-batch digest was recorded. Early errors have incomplete consumption."}


def history_summary(trials, artifacts, current_receipts):
    """Keep original execution failures and costs outside final cell counts."""
    declared, records = {}, {}
    selected = {task["attempt_id"] for trial in trials for task in task_map(trial).values() if task.get("attempt_id")}
    for trial in trials:
        for history in trial.get("attempt_history", []):
            identity = history["attempt_id"]
            if identity in declared:
                if declared[identity] != history:
                    raise ValueError("shared attempt history differs between configurations")
                continue
            declared[identity] = deepcopy(history)
            if not history["valid_receipt"]:
                continue
            receipt = current_receipts.get(identity)
            if receipt is None:
                original = json.loads(artifacts.read(identity, "result.json", durable=True))
                row = original["task_results"][0]
                task = {"attempt_id": identity, "task": row["task_id"],
                        "gate_status": row["gate_status"], "compatibility_key": row["compatibility_key"]}
                receipt = artifacts.receipt(trial, task)
            if receipt is None:
                raise ValueError("attempt history claims a certificate that cannot be validated")
            request, result, certificate, _ = receipt
            if certificate["result_hash"] != history["result_hash"]:
                raise ValueError("attempt history differs from its original result certificate")
            records[identity] = {"attempt_id": identity, "valid_receipt": True,
                                 "retry_of": deepcopy(result.get("retry_of")),
                                 "result_hash": certificate["result_hash"], "request": request,
                                 "attempt_status": result["raw"]["attempt_status"],
                                 "task_results": deepcopy(result["task_results"])}
    errors = _resolve_retries(list(records.values()))
    if errors:
        raise ValueError("invalid execution-repair lineage: " + json.dumps(errors, sort_keys=True))
    for identity, record in records.items():
        if record.get("superseded_by") != declared[identity].get("superseded_by"):
            raise ValueError("reported supersession differs from certified retry lineage")
    repairs = [{"original_attempt": identity, "replacement_attempt": record["superseded_by"],
                "authorization": records[record["superseded_by"]]["retry_of"],
                "original_gate_statuses": declared[identity]["gate_statuses"],
                "replacement_gate_statuses": declared[record["superseded_by"]]["gate_statuses"],
                "original_wall_seconds": declared[identity]["wall_seconds"],
                "replacement_wall_seconds": declared[record["superseded_by"]]["wall_seconds"],
                "scientific_identity_preserved": True}
               for identity, record in sorted(records.items()) if record.get("superseded_by")]
    if selected - declared.keys():
        raise ValueError("current selected attempts are missing from the recorded history")
    return {"current_cells": sum(len(task_map(trial)) for trial in trials),
            "current_selected_attempts": len(selected), "recorded_attempts": len(declared),
            "superseded_original_attempts": len(repairs), "execution_repairs": repairs,
            "all_attempt_wall_seconds": sum(row["wall_seconds"] for row in declared.values()),
            "superseded_attempt_wall_seconds": sum(row["original_wall_seconds"] for row in repairs),
            "counting_scope": "Each request/task's final selected result is counted once. Original repaired attempts remain separate history and cost evidence; numerical SGDA errors are not replaced.",
            "attempts": [{**row, "selected_current": identity in selected}
                         for identity, row in sorted(declared.items())]}


def analyze(root, artifacts, *, allow_partial=False):
    reports, trials = load_reports(root, allow_partial)
    selections = {arm: whole_selection([trial for trial in trials if trial["arm"] == arm]) for arm in ARMS}
    observed, compact, trace_rows, trace_receipts, current_receipts = defaultdict(list), [], {}, {}, {}
    for trial in trials:
        tasks = []
        for task in task_map(trial).values():
            receipt = artifacts.receipt(trial, task)
            if receipt:
                current_receipts[task["attempt_id"]] = receipt
            tasks.append(compact_task(task, receipt))
            if receipt and task["task"] in REQUIRED:
                observed[task["task"]].append((trial["candidate_id"], task, receipt))
            diagnostic = receipt[3].get("evidence", {}).get("optimizer_diagnostics") if receipt else None
            if diagnostic:
                rows = artifacts.trace(task["attempt_id"], diagnostic)
                key = trial["candidate_id"], task["task"]
                trace_rows[key] = rows
                trace_receipts[key] = {"attempt": task["attempt_id"], "sha256": diagnostic["sha256"],
                                       "path": diagnostic["path"], "summary": trace_summary(rows)}
        required = [task for task in tasks if task["importance"] == "required"]
        compact.append({"arm": trial["arm"], "candidate_id": trial["candidate_id"],
                        "configuration_id": trial["configuration_id"], "candidate_revision": trial["candidate_revision"],
                        "settings": trial["settings"], "recipe_overrides": trial["recipe_overrides"],
                        "submission_status": trial["submission_status"], "execution_complete": execution_complete(trial),
                        "required_pass_count": sum(task["gate_status"] == "PASS" for task in required),
                        "required_total": len(required), "tasks": tasks})
    by_id = {trial["candidate_id"]: trial for trial in compact}
    adam_id = selections["adam"]["selected_candidate_id"]
    baselines = [trial for trial in trials if trial["arm"] == "adam"
                 and trial["resolved_recipe"]["lr"] == .00425]
    if len(baselines) != 1:
        raise ValueError("screen must contain the exact incumbent Adam control")
    baseline_id = baselines[0]["candidate_id"] if execution_complete(baselines[0]) else None
    def comparisons(reference_id):
        rows = []
        if not reference_id:
            return rows
        adam = {task["task"]: task for task in by_id[reference_id]["tasks"]}
        for arm, selection in selections.items():
            candidate_id = selection["selected_candidate_id"]
            if not candidate_id or arm == "adam":
                continue
            rows.append({"arm": arm, "candidate_id": candidate_id, "adam_reference": reference_id,
                         "task_comparisons": [{"task": task["task"], "candidate_status": task["gate_status"],
                             "adam_status": adam[task["task"]]["gate_status"],
                             "final_metric_delta_candidate_minus_adam": numeric_delta(task["metrics"], adam[task["task"]]["metrics"])}
                             for task in by_id[candidate_id]["tasks"] if task["task"] in REQUIRED]})
        return rows
    graft_pairs = []
    adam_rates = {trial["resolved_recipe"]["lr"]: trial for trial in trials if trial["arm"] == "adam"}
    for trial in trials:
        if trial["arm"] == "ada-nsgda" and trial["resolved_recipe"]["lr"] in adam_rates:
            reference = adam_rates[trial["resolved_recipe"]["lr"]]
            actual_tasks = {task["task"]: task for task in by_id[trial["candidate_id"]]["tasks"]}
            reference_tasks = {task["task"]: task for task in by_id[reference["candidate_id"]]["tasks"]}
            graft_pairs.append({"base_step_size": trial["resolved_recipe"]["lr"],
                                "graft_candidate": trial["candidate_id"], "adam_candidate": reference["candidate_id"],
                                "complete": execution_complete(trial) and execution_complete(reference),
                                "tasks": [{"task": name, "graft_status": actual_tasks[name]["gate_status"],
                                           "adam_status": reference_tasks[name]["gate_status"],
                                           "final_metric_delta_graft_minus_adam": numeric_delta(actual_tasks[name]["metrics"], reference_tasks[name]["metrics"])}
                                          for name in REQUIRED]})
    momentum = {str(mu): whole_selection([trial for trial in trials
                 if trial["arm"] in {"dualnorm-zero", "dualnorm-momentum"}
                 and trial["resolved_recipe"].get("optimizer_momentum", 0) == mu]) for mu in (0., .5, .9)}
    edges, tradeoffs = [], []
    for arm, selection in selections.items():
        selected_id = selection["selected_candidate_id"]
        if not selected_id:
            continue
        selected = next(trial for trial in trials if trial["candidate_id"] == selected_id)
        rates = sorted({trial["resolved_recipe"]["lr"] for trial in trials if trial["arm"] == arm})
        rate = selected["resolved_recipe"]["lr"]
        edges.append({"arm": arm, "candidate_id": selected_id, "base_step_size": rate,
                      "tested_minimum": rates[0], "tested_maximum": rates[-1],
                      "at_lower_edge": rate == rates[0], "at_upper_edge": rate == rates[-1],
                      "interpretation": "An edge under the registered PASS/hash objective is not a calibrated optimum; any extension requires a separate finite declaration."})
        if baseline_id:
            actual = {task["task"]: task for task in by_id[selected_id]["tasks"]}
            baseline = {task["task"]: task for task in by_id[baseline_id]["tasks"]}
            for name, quality in (("ring16_acquisition", "hq"), ("five_word_joint_acquisition", "quality_fraction")):
                values, reference = actual[name]["metrics"], baseline[name]["metrics"]
                if quality in values and quality in reference and "modes" in values and "modes" in reference:
                    if values[quality] > reference[quality] and values["modes"] < reference["modes"]:
                        tradeoffs.append({"arm": arm, "candidate_id": selected_id, "task": name,
                                          "quality_metric": quality, "candidate_quality": values[quality],
                                          "baseline_quality": reference[quality], "candidate_modes": values["modes"],
                                          "baseline_modes": reference["modes"],
                                          "scope": "Final sampled metrics only; sustained task grades remain unchanged."})
    # One whole recipe per arm, followed by the same count/hash objective across
    # arms. Task metrics and trace availability never enter this selection.
    candidates = [by_id[selection["selected_candidate_id"]] for arm, selection in selections.items()
                  if arm != "adam" and selection["selected_candidate_id"]]
    top = sorted(candidates, key=lambda trial: (-trial["required_pass_count"], trial["configuration_id"]))[:3]
    plotted = ([by_id[baseline_id]] if baseline_id else []) + top
    traces = [{"candidate_id": candidate, "task": task, **value}
              for (candidate, task), value in sorted(trace_receipts.items())]
    all_tasks = [task for trial in compact for task in trial["tasks"]]
    report = {"schema_version": 1, "campaign": CAMPAIGN, "qualification_input": False,
              "status": "complete_execution" if all(execution_complete(trial) for trial in trials) else "partial_provisional",
              "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST, "protocol_seed": 0,
              "policy_fingerprint": reports["adam"]["policy_fingerprint"],
              "input_search_reports": {arm: reports[arm]["input_digest"] for arm in ARMS},
              "required_tasks": list(REQUIRED), "separate_diagnostic_task": DIAGNOSTIC,
              "counts_required": dict(Counter(task["gate_status"] for task in all_tasks if task["importance"] == "required")),
              "counts_diagnostic": dict(Counter(task["gate_status"] for task in all_tasks if task["importance"] != "required")),
              "execution_history": history_summary(trials, artifacts, current_receipts),
              "configurations": compact, "whole_arm_selections": selections, "dualnorm_momentum_selections": momentum,
              "adam_baseline_candidate_id": baseline_id, "adam_selected_candidate_id": adam_id,
              "comparisons_to_incumbent_adam": comparisons(baseline_id),
              "comparisons_to_selected_adam": comparisons(adam_id), "matched_adam_magnitude_graft_pairs": graft_pairs,
              "step_size_grid_edges": edges, "quality_improves_coverage_drops": tradeoffs,
              "matching_audits": [audit_task(name, observed[name], len(trials)) for name in REQUIRED],
              "diagnostics": traces, "plot_candidates": [trial["candidate_id"] for trial in plotted],
              "plot_selection": "Exact .00425 incumbent Adam control and top three non-Adam arm selections by required PASS count then configuration hash; task metrics are not selection inputs. Swept Adam selection is reported separately.",
              "limitations": ["Protocol seed 0 only; all original P1-P6 native/five-seed/scale predictions remain unscored.",
                  "Pure BCAP only; R1/R2, native 7k, sparse177 and width/depth transfer are deferred.",
                  "Clean live metrics and frozen task gates; no EMA or current-implementation qualification inferred.",
                  "Terminal errors and incomplete training stay in the six-task denominator with missing metrics.",
                  "Frozen critic real/fake input-gradient probes are excluded because phase labels were unreliable.",
                  "Spectral proxy is the product of matrix weight norms; Fourier/input maps and nonlinearities are excluded.",
                  "Weight norms are before the observed update; spectral products are after it. Traces contain declared checkpoints only.",
                  "Native Adam prior support means nonzero gradient support; only explicit sampled-index receipts measure unsampled-row drift.",
                  "No task-by-task recipe mixing, default promotion, calibrated equivalence, robustness or speed ranking is supplied."],
              "predictions": {f"P{i}": "unscored: original scope not executed" for i in range(1, 7)}}
    return report, trace_rows


def plot_diagnostics(report, traces, output):
    if not report["plot_candidates"]:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_id = {trial["candidate_id"]: trial for trial in report["configurations"]}
    candidates, receipts = report["plot_candidates"], []
    output.mkdir(parents=True, exist_ok=True)
    for task in TRACED:
        figure, axes = plt.subplots(max(1, len(candidates)), 4, figsize=(17, 3.1 * max(1, len(candidates))), squeeze=False)
        for row_index, candidate in enumerate(candidates):
            trial, rows = by_id[candidate], traces.get((candidate, task), [])
            row_axes = axes[row_index]
            if not rows:
                for axis in row_axes:
                    axis.text(.5, .5, "Trace unavailable", ha="center", va="center", transform=axis.transAxes)
                continue
            player_values, layer_values, spectral = defaultdict(list), defaultdict(list), []
            for row in rows:
                for player, values in row["players"].items():
                    player_values[player].append((row["step"], values["relative_update_sum"]))
                for layer in row["layers"]:
                    if layer["player"] in {"G", "D"}:
                        key = layer["player"], layer["component"], layer["index"], tuple(layer["shape"])
                        layer_values[key].append((row["step"], layer["weight_norm"]))
                if "critic_log_spectral_product" in row:
                    spectral.append((row["step"], row["critic_log_spectral_product"]))
            for player, values in sorted(player_values.items()):
                x, y = zip(*values)
                row_axes[0].plot(x, [v if v > 0 else np.nan for v in y], label=player)
            row_axes[0].set_yscale("log")
            for (player, component, index, shape), values in sorted(layer_values.items()):
                axis = row_axes[1 if player == "G" else 2]
                x, y = zip(*values)
                axis.plot(x, [v if v > 0 else np.nan for v in y], label=f"{component}:{index} {shape}", alpha=.8)
                axis.set_yscale("log")
            if spectral:
                x, y = zip(*spectral)
                row_axes[3].plot(x, y, color="tab:purple")
            for axis in row_axes:
                axis.grid(alpha=.2)
                axis.set_xlabel("Optimizer update")
            for axis in row_axes[:3]:
                axis.legend(fontsize=6, ncol=2)
            recipe = trial["settings"]
            row_axes[0].set_ylabel(f"{trial['arm']}\n{candidate.rsplit('--', 1)[-1][:8]}\nη={recipe.get('lr', '?')}")
        titles = ("Σ layer ‖ΔW‖ / ‖W‖", "G/E parameter norms (before step)",
                  "D parameter norms (before step)", "D Σ log(matrix spectral norm) (after step)")
        for index, title in enumerate(titles):
            axes[0, index].set_title(title, fontsize=10)
            # Comparable scales across selected whole configurations in each task.
            limits = [axis.get_ylim() for axis in axes[:, index] if axis.lines]
            if limits:
                bounds = min(lo for lo, _ in limits), max(hi for _, hi in limits)
                for axis in axes[:, index]:
                    axis.set_ylim(bounds)
        figure.suptitle(task + " — frozen BCAP seed-0 screen", fontsize=12)
        figure.text(.5, .008, "Top whole recipes by frozen PASS/hash objective; no per-task selection. Spectral proxy excludes Fourier maps/nonlinearities. Input-gradient probes excluded.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0, .025, 1, .96))
        path = output / (task + ".png")
        figure.savefig(path, dpi=140, metadata={"Software": "ParticleGAN BCAP frozen-source diagnostic analysis"})
        plt.close(figure)
        receipts.append({"task": task, "path": path.name, "sha256": file_hash(path),
                         "candidates": candidates,
                         "unavailable_candidates": [candidate for candidate in candidates if (candidate, task) not in traces]})
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge/bcap-dualnorm-tier1-v1-queue")
    parser.add_argument("--archive", type=Path, help="original run.py archive, if local raw artifacts were removed")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/forge/dualnorm-tier1/analysis.json")
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "reports/forge/dualnorm-tier1/diagnostics")
    parser.add_argument("--allow-partial", action="store_true", help="explicit provisional readout; never imputes pending task outcomes")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    artifacts = Artifacts(args.root.resolve(), args.queue_root.resolve(), args.archive)
    try:
        report, traces = analyze(args.root.resolve(), artifacts, allow_partial=args.allow_partial)
        report["plots"] = [] if args.no_plots else plot_diagnostics(report, traces, args.plot_dir)
        report["input_digest"] = stable_hash(report)
        atomic_json(args.output, report)
        print(json.dumps({"event": "analyzed", "status": report["status"], "configurations": len(report["configurations"]),
                          "required_counts": report["counts_required"], "plots": len(report["plots"]), "output": str(args.output)}, sort_keys=True), flush=True)
    finally:
        artifacts.close()


if __name__ == "__main__":
    main()
