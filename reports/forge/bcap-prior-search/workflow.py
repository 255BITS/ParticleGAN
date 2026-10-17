"""Bounded prior searches through the unchanged ordinary Forge gate ladder.

Copy this reproduction source into each feature checkout before preparation.
Prepare writes declarations without admission; run requires their committed
source and explicit execution. Bulk envelopes remain in the artifact archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import (
    enqueue_search, materialize_search, plan_search, report_search,
)
from experiments.forge.contracts import atomic_json, atomic_text, file_hash, file_lock, read_json, stable_hash
from experiments.forge.decision_contracts import OUTCOMES
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.queue import Queue, device_slots, drain
from experiments.forge.studies import validate_study
from experiments.forge.views import load_view
from particlegan import get_recipe

LEADER = "recipe-prior-refactor-bcap-pure-v1"
VIEW = "discriminator_stability"
PER_CANDIDATE = 45_000
SEARCH_BUDGET = 135_000
FULL_RESERVATION = 43_020


def require(condition, message):
    if not condition:
        raise ValueError(message)


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed_source(source, declarations):
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    files = set(source["files"]) | set(declarations)
    require(files <= tracked, "Commit every frozen source and declaration before admission.")
    check = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *sorted(files)], cwd=ROOT)
    require(check.returncode == 0, "Frozen source/declarations differ from the selected commit.")


def paths(name):
    stem = f"bcap-prior-{name}"
    return dict(stem=stem, control=f"{stem}-control-v1", base=f"{stem}-base-v1",
                study=f"{stem}-control-study-v1", search=f"{stem}-search-v1",
                report=ROOT / "reports/forge/bcap-prior-search" / name)


def resolved_recipe(card):
    overrides = deepcopy(card["recipe_overrides"])
    recorded_name = overrides.pop("name", card["recipe_preset"])
    recipe = get_recipe(card["recipe_preset"], **overrides)
    require(recipe.name == recorded_name, "Preserve the leader's recipe name.")
    return asdict(recipe)


def prepare(args):
    p = paths(args.name)
    require(not (args.archive / "progress.json").exists(), "Admitted experiments are immutable.")
    view = load_view(ROOT, VIEW)
    require(view["revision"] == 9, "Freeze the current recipe-owned revision-9 view.")
    required = [a for a in view["assignments"] if a["importance"] == "required"]
    require(Counter(a["qualification_tier"] for a in required) == {1: 6, 2: 21, 3: 2},
            "Preserve every required task and gate.")
    leader = read_json(ROOT / f"configs/forge/ideas/{LEADER}.json")
    values = json.loads(args.values)
    delta = json.loads(args.delta)
    require(len(values) == 3 and len(set(values)) == 3 and all(v > 0 for v in values),
            "Exactly three distinct positive strengths; zero is the separate structural control.")
    require(args.axis in delta and delta[args.axis] == values[0], "Base strength must be the first grid value.")
    require(leader["recipe_overrides"]["constraint_geometry_mode"] == "direction_blend",
            "Use the actual direction-blend leader.")
    cards = {}
    for role in ("control", "base"):
        card = deepcopy(leader)
        card.update(id=p[role], parent=LEADER,
                    changed_factors=["Matched current-source BCAP leader control" if role == "control" else args.hypothesis],
                    mechanism_class="structural", mechanism_rationale=args.hypothesis,
                    guide=f"reports/forge/bcap-prior-search/{args.name}/spec.json")
        if role == "base":
            card["recipe_overrides"].update(delta)
        cards[role] = card
        atomic_json(ROOT / f"configs/forge/ideas/{p[role]}.json", card)
    recipes = {role: resolved_recipe(card) for role, card in cards.items()}
    actual_delta = {k: {"control": recipes["control"][k], "candidate": recipes["base"][k]}
                    for k in recipes["control"] if recipes["control"][k] != recipes["base"][k]}
    require(set(actual_delta) == set(delta), "A prior comparison must have only its declared trainer delta.")
    signature = dict(task_id="mode_hold", metric="modes", threshold=8, phase="final")
    evidence_path = "reports/forge/recipe-prior-refactor/publication/results.json"
    evidence = read_json(ROOT / evidence_path)
    study = dict(schema_version=1, id=p["study"], status="ready", candidate=p["control"],
                 control=dict(candidate_id=p["base"], task_map={}), max_rounds=1,
                 hypothesis="Reproduce the unchanged BCAP leader on this feature's frozen source before comparing prior changes.",
                 competing_explanation="Source or host integration may change the control; historical 9/21 evidence cannot fill current cells.",
                 scope=dict(view=VIEW, through_tier=2, execution_backend="cuda", cuda_model="NVIDIA RTX A6000"),
                 campaign=dict(id=f"{p['stem']}-control-v1", budget_seconds=PER_CANDIDATE,
                               candidate_budget_seconds=PER_CANDIDATE),
                 prior_evidence=[dict(path=evidence_path, selector=[],
                                      identity=dict(source_digest=evidence["source_digest"]), use="motivation_only")],
                 prediction={**signature, "op": "=="}, falsifier={**signature, "op": "<"}, terminal_rules=OUTCOMES)
    validate_study(study)
    atomic_json(ROOT / f"configs/forge/studies/{p['study']}.json", study)
    spec = dict(schema_version=2, id=p["search"], trainer_family=leader["trainer_family"],
                base_candidate=p["base"], grid={args.axis: values}, tuning_through_tier=2,
                view=VIEW, execution_backend="cuda", cuda_model="NVIDIA RTX A6000", protocol="screening",
                campaign=dict(schema_version=1, id=p["search"], budget_seconds=SEARCH_BUDGET,
                              candidate_budget_seconds=PER_CANDIDATE, accept_shared_cost_transfer=False,
                              purpose="Three global positive prior strengths, all six Tier1 tasks then all21 Tier2 if eligible; complete_current_tier. Seed0, public initializer, original task architecture/data/prior/capacity/sampling/updates/cadence/gates. Separate matched leader control. No scientific retries, seed variants, Tier3, per-task recipes or default adoption. GPU sharing is intentional; costs are not a speed ranking."),
                hypothesis=args.hypothesis,
                rationale="Convert at least one matched-control Tier2 FAIL to a sustained full-gate PASS. Retain every loss of a prior passing gate and all blocked/unmeasured cells. Search evidence is tuning, not independent confirmation.",
                guide=f"reports/forge/bcap-prior-search/{args.name}/spec.json")
    atomic_json(ROOT / f"configs/forge/searches/{p['search']}.json", spec)
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    materialize_search(ROOT, spec)
    search = plan_search(ROOT, queue.root, spec, queue=queue)
    control = resolve_idea(ROOT, p["control"], study=p["study"], queue_root=queue.root)
    planned = plan_summary(control)
    require(not control["preflight_blockers"] and control["study_review"]["status"] == "READY",
            f"Control is not ready: {control.get('preflight_blockers')}")
    require(all(t["submission_status"] == "READY" for t in search["trials"]), "Every configuration must preflight.")
    require(planned["worst_case_seconds"] == FULL_RESERVATION
            and search["declared_worst_case_seconds"] == 3 * FULL_RESERVATION,
            "Retain the complete original task reservation, including the optional clock audit.")
    require(search["source_digest"] == control["source"]["digest"], "Control and trials need identical source.")
    output = dict(schema_version=1, name=args.name, axis=args.axis, values=values, trainer_delta=actual_delta,
                  hypothesis=args.hypothesis, falsifier=args.falsifier, protocol_seed=0, view_revision=9,
                  baseline=LEADER, required_counts=dict(tier1=6, tier2=21, tier3=2), through_tier=2,
                  full_reservation_seconds=4 * FULL_RESERVATION, maximum_reserved_paid_seconds=180_000,
                  source_digest=search["source_digest"], task_contracts={name: stable_hash(card) for name, card in control["tasks"].items()},
                  search_id=p["search"], control_id=p["control"], control_study_id=p["study"],
                  archive=str(args.archive), acceptance=dict(minimum_matched_fail_to_pass=1,
                      ordinary_tier1_passes_required=6, report_all_regressions=True,
                      preserve_all_control_tier2_passes_for_global_repair=True),
                  stop="One finite round. Complete current-tier peers, then required non-PASS blocks higher tiers. No automatic refinement or scientific retry.")
    atomic_json(p["report"] / "spec.json", output)
    atomic_json(args.archive / "preflight.json", dict(control_plan=planned, search_plan=search))
    atomic_json(p["report"] / "registration.json", dict(spec_sha256=stable_hash(output), source_digest=search["source_digest"],
                preparation_commit=head(), control_candidate_revision=control["candidate_revision"],
                control_worst_case_seconds=planned["worst_case_seconds"],
                search_worst_case_seconds=search["declared_worst_case_seconds"],
                trials=[{k: trial[k] for k in ("candidate_id", "configuration_id", "candidate_revision", "settings",
                                              "submission_status", "submission_blockers", "declared_worst_case_seconds")}
                        for trial in search["trials"]]))
    print(json.dumps(dict(event="prepared", name=args.name, source_digest=search["source_digest"],
                         full_reservation_seconds=output["full_reservation_seconds"])), flush=True)


def drain_shared_pool(queue, pool, devices):
    """Share physical admission across independent, source-specific queues.

    Existing workers keep their leases, states and deadlines. New GPU jobs run
    one per physical GPU so the unchanged tight word allowance is meaningful.
    This changes scheduling only, never training updates or evaluation cadence.
    """
    with file_lock(queue.root / "coordinator.lock", blocking=False):
        processes = []
        while True:
            processes = [p for p in processes if p.poll() is None]
            queue.collect()
            with file_lock(pool / "physical-admission.lock"):
                busy = set()
                for state_path in pool.glob("*/queue/queue/state.json"):
                    for job in read_json(state_path)["jobs"].values():
                        if job["status"] == "running":
                            busy.add(str(job["worker"]["device"]))
                slots = [slot for slot in device_slots(devices) if slot["device"] not in busy]
                # claim([cpu]) treats CUDA requests as impossible allocations;
                # occupied external GPU capacity must instead remain queued.
                claim = queue.claim(slots) if any(s["device"] != "cpu" for s in slots) else None
                if claim:
                    try:
                        processes.append(queue.launch(claim))
                    except (OSError, RuntimeError) as error:
                        atomic_json(Path(claim["worker"]["directory"]) / "terminal.json",
                                    dict(schema_version=1, token=claim["worker"]["token"], attempt_status="error",
                                         reason=f"worker launch failed: {error}", elapsed_seconds=0, result={}))
                        queue.collect()
            queue.flush_events()
            state = queue.inspect()
            if (not any(e["status"] in ("queued", "running") for e in state["submissions"].values())
                    and not any(j["status"] == "running" for j in state["jobs"].values())):
                break
            time.sleep(2)


def run(args):
    p = paths(args.name)
    require(head() == args.source_commit, "Run only the explicitly frozen committed source.")
    require(not args.archive.is_relative_to(ROOT), "Bulk artifacts must live outside Git.")
    spec = read_json(p["report"] / "spec.json")
    registration = read_json(p["report"] / "registration.json")
    require(stable_hash(spec) == registration["spec_sha256"], "Prepared study changed.")
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    progress = args.archive / "progress.json"
    if args.submit:
        require(not progress.exists(), "Do not duplicate an admitted scientific round.")
        control = resolve_idea(ROOT, p["control"], study=p["study"], queue_root=queue.root, freeze_source=True)
        require(control["source"]["digest"] == registration["source_digest"], "Prepared source changed.")
        require({name: stable_hash(card) for name, card in control["tasks"].items()} == spec["task_contracts"], "Prepared task contracts changed.")
        declarations = [f"configs/forge/{kind}/{p[key]}.json" for kind, key in (
            ("ideas", "control"), ("ideas", "base"), ("studies", "study"), ("searches", "search"))]
        declarations += [f"configs/forge/tasks/{name}.json" for name in control["tasks"]]
        committed_source(control["source"], declarations)
        entry = queue.submit(control, control["study"]["campaign"])
        search = enqueue_search(ROOT, queue.root, p["search"], queue=queue)
        require(search["source_digest"] == registration["source_digest"] and search["submitted_count"] == 3,
                "All three trials must share the prepared source.")
        atomic_json(progress, dict(source_commit=args.source_commit, source_digest=registration["source_digest"],
                                  phase="admitted", control_request=entry["request"]["request_id"],
                                  search_requests=[t["request_id"] for t in search["trials"]]))
    if args.drain:
        require(progress.exists(), "Drain needs prior explicit admission.")
        data = read_json(progress)
        require(data["source_commit"] == args.source_commit, "Admission source differs.")
        if args.shared_pool:
            require(args.workers_per_gpu == 1, "The shared pool fixes one new GPU job per physical device.")
            drain_shared_pool(queue, args.shared_pool.resolve(), args.gpus.split(","))
        else:
            drain(queue, args.gpus.split(","), workers_per_gpu=args.workers_per_gpu, allow_sharing=True, watch=False,
                  poll_seconds=2)
        state = queue.inspect()
        unfinished = (any(e["status"] in ("queued", "running", "paused") for e in state["submissions"].values())
                      or any(j["status"] == "running" for j in state["jobs"].values()))
        data["phase"] = "paused" if unfinished else "complete"
        atomic_json(progress, data)
    print(json.dumps(dict(event="execution_state", name=args.name, progress=read_json(progress))), flush=True)


def fmt(value):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def measurement(metric, score, limits=()):
    operators = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
                 "==": lambda a, b: a == b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}
    available = score is not None and (not isinstance(score, float) or math.isfinite(score))
    status = ("UNAVAILABLE" if not available else "DIAGNOSTIC" if not limits else
              "PASS" if all(operators[op](score, target) for op, target in limits) else "FAIL")
    return dict(metric=metric, score=score, target=" and ".join(f"{op} {fmt(target)}" for op, target in limits) or "No declared gate",
                limits=list(limits), status=status)


def metric_rows(metrics, thresholds):
    limits = {}
    for key, op, target in thresholds:
        limits.setdefault(key, []).append((op, target))
    rows = [measurement(key, metrics.get(key), bounds) for key, bounds in limits.items()]

    def diagnostic(key, value):
        if isinstance(value, dict):
            for child, item in sorted(value.items()):
                diagnostic(f"{key}.{child}", item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                diagnostic(f"{key}[{index}]", item)
        elif value is None or isinstance(value, (int, float, bool)):
            rows.append(measurement(key, value))

    for key in sorted(set(metrics) - set(limits)):
        diagnostic(key, metrics[key])
    return rows


def native_thresholds(card, n, *, accuracy=False):
    evaluation = card["evaluation"]
    c = evaluation["coverage_thresholds"]
    common = [("n", ">=", c["min_samples"]), ("valid_n", "==", n),
              ("precision", ">=", c["min_precision"])]
    if accuracy:
        return common + [("all_finite", "==", True)] + [
            (k, "<=", v) for k, v in evaluation["accuracy_limits"].items()] + [
            (k, "==", True) for k in ("accuracy_pass", "frozen_pass", "passed")]
    return common + [("modes", "==", c["min_modes"]),
                     ("min_hq_mode_mass", ">=", c["min_hq_mode_mass"]),
                     ("mass_tv", "<=", c["max_mass_tv"]), ("max_mode_mass", "<=", c["max_mode_mass"]),
                     ("min_cov_eig_ratio", ">=", c["min_cov_eig_ratio"]),
                     ("max_cov_eig_ratio", "<=", c["max_cov_eig_ratio"]),
                     ("min_radial_median_ratio", ">=", c["min_radial_median_ratio"]),
                     ("max_radial_median_ratio", "<=", c["max_radial_median_ratio"]),
                     ("passed", "==", True)]


def compact_row(row):
    keep = ("task_id", "gate_status", "raw_status", "reason", "reasons", "metrics", "evaluator_result", "cost",
            "compatibility_key", "recipe", "initializer", "initialization", "initial_prior", "prior", "prior_policy",
            "field_ownership", "mechanism_applicability", "sampling", "receipt_binding")
    projected = {k: deepcopy(row[k]) for k in keep if k in row}
    applied = row.get("applied", {})
    for key in ("recipe", "initializer", "initialization", "initial_prior", "prior", "field_ownership", "mechanism_applicability"):
        if key not in projected and key in applied:
            projected[key] = deepcopy(applied[key])
    evaluator = projected.get("evaluator_result", {})
    terminal = evaluator.pop("terminal_checks", None)
    if terminal is not None:
        evaluator["terminal_summary"] = dict(checks=len(terminal), passing_checks=sum(bool(p.get("passed")) for p in terminal))
    evidence = row.get("evidence", {})
    if "prior_policy" not in projected and "prior_policy" in evidence:
        projected["prior_policy"] = deepcopy(evidence["prior_policy"])
    raw = row.get("raw")
    if isinstance(raw, dict):
        scalars = {key: deepcopy(value) for key, value in raw.items()
                   if key not in (row.get("metrics") or {})
                   and (value is None or isinstance(value, (int, float, bool)))}
        if scalars:
            projected["raw_endpoint_diagnostics"] = scalars
    projected["provenance_checkpoint"] = deepcopy(evidence.get("provenance_checkpoint"))
    projected["sampling"] = {k: evidence.get(k) for k in ("sampling_contract_version", "sampling_law", "eval_output_noise")}
    projected["guards"] = deepcopy(evidence.get("guards", {}))
    if "comparisons" in evidence:
        projected["clock_parity"] = [dict(condition=point["condition"],
            exact_state_equal=bool(point.get("changed_sha256") and point.get("changed_sha256") == point.get("reference_sha256")))
            for point in evidence["comparisons"]]
    return projected


def display_outcome(row, cell=None, trial=None):
    """Retain prerequisite BLOCKED states without inventing a measured result."""
    if row.get("gate_status"):
        return row["gate_status"], row.get("reason")
    cell, trial = cell or {}, trial or {}
    if cell.get("gate_status") == "BLOCKED":
        blockers = cell.get("blockers", []) + trial.get("submission_blockers", [])
        reason = (cell.get("reason") or trial.get("submission_reason")
                  or "; ".join(str(value) for value in blockers) or "Blocked by frozen prerequisites")
        return "BLOCKED", reason
    failed_tier1 = [task for task in trial.get("tasks", [])
                    if task.get("qualification_tier") == 1 and task.get("importance") == "required"
                    and task.get("attempt_id")
                    and task.get("gate_status") in {"FAIL", "INVALID", "INCOMPLETE", "BLOCKED", "ERROR"}]
    if (trial.get("submission_status") == "blocked" and cell.get("qualification_tier", 0) > 1
            and failed_tier1):
        reason = trial.get("submission_reason") or "; ".join(
            f"{task['task']}: {task['gate_status']} {task.get('reason') or ''}".strip() for task in failed_tier1)
        return "BLOCKED", reason
    return "UNMEASURED", None


def validated_submission_rows(state, entry, attempts):
    """Use original certificates for the control as well as search trials.

    Queue result projections alone cannot establish a matched-control grade.
    This follows report_search's receipt/cohort checks and preserves observed
    scores when provenance is invalid, without granting a PASS or repair.
    """
    from experiments.forge.views import grade_result
    request = entry["request"]
    rows = []
    for definition in request["jobs"]:
        result = state["jobs"][definition["compatibility_key"]].get("result")
        if not result:
            continue
        original = attempts.get(result.get("attempt_id"))
        valid = bool(original and original["valid_receipt"]
                     and original["result_hash"] == stable_hash(result)
                     and not original.get("superseded_by")
                     and original["request"]["candidate_revision"] == request["candidate_revision"]
                     and original["request"]["source"]["digest"] == request["source"]["digest"]
                     and original["request"]["runtime"] == request["runtime"]
                     and original["request"]["compute_profiles"] == request["compute_profiles"])
        binding = dict(attempt_id=result.get("attempt_id"), result_hash=stable_hash(result),
                       valid_receipt=valid, candidate_revision=request["candidate_revision"],
                       source_digest=request["source"]["digest"])
        if not valid:
            binding["receipt_error"] = ((original or {}).get("receipt_error")
                                        or "Missing, superseded or mismatched original receipt/cohort")
        for row in result.get("task_results", []):
            task_id = row["task_id"]
            require(task_id in definition.get("task_ids", [definition["task_id"]]),
                    "Queue result contains a task outside its frozen execution group.")
            grade = grade_result(request["tasks"][task_id], row) if valid else {}
            projected = {**deepcopy(row), **grade, "receipt_binding": deepcopy(binding)}
            projected["gate_status"] = grade.get("gate_status", grade.get("status", "INVALID")) if valid else "INVALID"
            if not valid:
                projected["reason"] = binding["receipt_error"]
            rows.append(projected)
    return rows


def word_hold_failure_witnesses(row, card):
    """Summarize mandatory hold checks without publishing observation streams."""
    if card["id"] != "five_word_joint_hold" or row.get("gate_status") != "FAIL":
        return []
    bounds = {}
    for name, op, target in card["evaluation"]["thresholds"]:
        bounds.setdefault(name, []).append((op, target))
    evidence = row.get("evidence", {})
    cohorts = [("primary", evidence.get("observations", []))]
    confirmations = evidence.get("confirmations", [])
    if confirmations and all(isinstance(point.get("metrics"), dict) for point in confirmations):
        cohorts.append(("independent confirmation", confirmations))
    summaries = []
    for label, points in cohorts:
        def values(point):
            return point["metrics"] if label != "primary" else point

        failed = [point for point in points if any(
            measurement(name, values(point).get(name), limits)["status"] != "PASS"
            for name, limits in bounds.items())]
        if not failed:
            continue
        witnesses = []
        for name, limits in bounds.items():
            def badness(point):
                score = values(point).get(name)
                if score is None or (isinstance(score, float) and not math.isfinite(score)):
                    return math.inf
                return max(((target - score) if op in (">=", ">") else
                            (score - target) if op in ("<=", "<") else abs(score - target))
                           / (abs(target) or 1) for op, target in limits)

            witness = max(points, key=badness)
            metric = measurement(name, values(witness).get(name), limits)
            metric.update(witness_step=witness["step"], witness_observation_sha256=stable_hash(witness))
            witnesses.append(metric)
        summaries.append(dict(name=f"Worst mandatory word-hold {label} checks (saved failure witnesses)",
                              cohort=label, qualification_credit=False, observation_count=len(points),
                              numerical_failed_checks=len(failed), original_observations_sha256=stable_hash(points),
                              rows=witnesses))
    return summaries


def sections(row, card):
    evaluator, evaluation = row.get("evaluator_result", {}), card["evaluation"]
    result = [dict(name="Recorded endpoint", rows=metric_rows(row.get("metrics", {}), evaluation.get("thresholds", [])))]
    temporal = []
    convergence = evaluator.get("convergence", {})
    if evaluation["kind"] == "clockfree_parity":
        parity = {point["condition"]: point["exact_state_equal"] for point in row.get("clock_parity", [])}
        temporal = [measurement(f"{condition}: exact learning-state parity", parity.get(condition), [("==", True)])
                    for condition in evaluation["conditions"]]
    elif evaluation["kind"] == "transfer_sustained":
        temporal = [measurement("terminal passing suffix", convergence.get("passing_suffix"), [(">=", evaluation["minimum_stable_checks"])]),
                    measurement("passing scheduled observations", convergence.get("passing_observations")),
                    measurement("complete declared execution", convergence.get("complete"), [("==", True)])]
    elif evaluation["kind"] in ("gaussian_smoke", "word_smoke"):
        confirmed = evaluator.get("confirmed_steps")
        temporal = [measurement("independently confirmed scheduled states", len(confirmed) if confirmed is not None else None, [(">=", 1)])]
    elif card["id"] == "gaussian1d_stability":
        reacquisition = evaluator.get("reacquisition_status")
        temporal = [measurement("stationary passing checks", evaluator.get("stationary_passes"), [("==", evaluation["stationary_checks"])]),
                    measurement("deadline reacquisition", reacquisition == "PASS" if reacquisition is not None else None, [("==", True)]),
                    measurement("shifted hold passing checks", evaluator.get("shift_hold_passes"), [("==", evaluation["shift_hold_checks"])])]
    elif card["id"] == "five_word_joint_hold":
        count = evaluation["observations"] + 1
        temporal = [measurement("passing primary checks", evaluator.get("passing_observations"), [("==", count)]),
                    measurement("confirmed checks", evaluator.get("confirmed_checks"), [("==", count)])]
    elif card["adapter"] == "native100":
        coverage_status = evaluator.get("coverage_status")
        temporal = [measurement("coverage decision", coverage_status == "PASS" if coverage_status is not None else None, [("==", True)]),
                    measurement("terminal live accuracy passing checks", evaluator.get("terminal_summary", {}).get("passing_checks"), [("==", evaluation["minimum_stable_checks"])])]
        result = [dict(name="Live endpoint coverage, 20000 saved draws", rows=metric_rows({}, native_thresholds(card, 20000))),
                  dict(name="Live endpoint accuracy, 20000 saved draws", rows=metric_rows({}, native_thresholds(card, 20000, accuracy=True))),
                  dict(name="Independent live holdout accuracy, 100000 saved draws", rows=metric_rows(row.get("metrics", {}), native_thresholds(card, 100000, accuracy=True))),
                  dict(name="Independent holdout coverage from the same saved draws", rows=metric_rows({}, native_thresholds(card, 100000)))]
    if (card["adapter"] == "native100" and "final_metrics" in evaluator
            and "_artifact_root" in row and "_artifact_manifest" in row):
        artifact_root = Path(row["_artifact_root"])
        manifest = row["_artifact_manifest"]["files"]
        summary_file = artifact_root / card["id"] / "summary.json"
        expected = manifest[f"{card['id']}/summary.json"]
        require(file_hash(summary_file) == expected["sha256"] and summary_file.stat().st_size == expected["size"], "Native final summary changed.")
        native = read_json(summary_file)
        result = [dict(name="Live endpoint coverage, 20000 saved draws",
                       rows=metric_rows(native["final"]["live"], native_thresholds(card, 20000))),
                  dict(name="Live endpoint accuracy, 20000 saved draws",
                       rows=metric_rows(evaluator["final_metrics"], native_thresholds(card, 20000, accuracy=True))),
                  dict(name="Independent live holdout accuracy, 100000 saved draws",
                       rows=metric_rows(row["metrics"], native_thresholds(card, 100000, accuracy=True)))]
        sample_file = artifact_root / card["id"] / "holdout_samples.npz"
        expected = manifest[f"{card['id']}/holdout_samples.npz"]
        require(file_hash(sample_file) == expected["sha256"] and sample_file.stat().st_size == expected["size"], "Native holdout artifact changed.")
        from benchmarks.toy100.metrics import evaluate_samples
        import numpy as np
        import torch
        with np.load(sample_file) as samples:
            coverage = evaluate_samples(torch.from_numpy(samples["live"]), card["id"])
        require(coverage["passed"] == row["metrics"]["frozen_pass"], "Saved holdout coverage contradicts the recorded grade.")
        result.append(dict(name="Independent holdout coverage recomputed from the same saved draws",
                           rows=metric_rows(coverage, native_thresholds(card, 100000))))
    if temporal:
        result.insert(0, dict(name="Sustained gate", rows=temporal))
    result.extend(row.get("_word_hold_failure_witnesses", []))
    if row.get("raw_endpoint_diagnostics"):
        result.append(dict(name="Additional scalar endpoint diagnostics (no declared gate)",
                           qualification_credit=False, rows=metric_rows(row["raw_endpoint_diagnostics"], [])))
    # Keep compact evaluator counts and numerical diagnostics visible. Full
    # terminal observation packets remain external; compact_row summarizes them.
    omitted = {"terminal_checks"}
    if card["adapter"] == "native100":
        omitted |= {"final_metrics", "holdout_metrics", "holdout_ema_metrics", "oracle_metrics"}
    diagnostics = metric_rows({key: value for key, value in evaluator.items() if key not in omitted}, [])
    if diagnostics:
        result.append(dict(name="Recorded evaluator diagnostics (no additional qualification credit)",
                           qualification_credit=False, rows=diagnostics))
    if card["adapter"] == "native100":
        count = evaluation["holdout_samples"]
        for key, label in (("holdout_ema_metrics", "EMA holdout"), ("oracle_metrics", "Oracle")):
            if key in evaluator:
                result.append(dict(name=f"{label} diagnostic cohort, {count} saved draws (no qualification credit)",
                                   cohort=key, qualification_credit=False,
                                   rows=metric_rows(evaluator[key] or {}, native_thresholds(card, count, accuracy=True))))
    return result


def publish(args):
    p = paths(args.name)
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    progress = read_json(args.archive / "progress.json")
    require(progress["phase"] == "complete", "Finish the registered work before concluding it.")
    state = queue.inspect()
    require(not any(e["status"] in ("queued", "running", "paused") for e in state["submissions"].values())
            and not any(j["status"] == "running" for j in state["jobs"].values()),
            "A paused drain is not a completed scientific round.")
    summary = report_search(ROOT, queue.root, p["search"], queue=queue)
    state = queue.inspect()
    from experiments.forge.knowledge import _attempts
    originals, _ = _attempts(ROOT)
    attempts = {attempt["attempt_id"]: attempt for attempt in originals}
    control_entry = state["submissions"][progress["control_request"]]
    control_results = [compact_row(row) for row in validated_submission_rows(state, control_entry, attempts)]
    control = {r["task_id"]: r for r in control_results}
    comparisons = []
    for trial in summary["trials"]:
        repaired, regressed, unresolved = [], [], []
        for cell in trial["tasks"]:
            if cell["qualification_tier"] != 2 or cell["importance"] != "required":
                continue
            before, after = control.get(cell["task"], {}).get("gate_status", "UNKNOWN"), cell["gate_status"]
            if (before, after) == ("FAIL", "PASS"):
                repaired.append(cell["task"])
            if before == "PASS" and after == "FAIL":
                regressed.append(cell["task"])
            if before == "PASS" and after not in ("PASS", "FAIL"):
                unresolved.append(cell["task"])
        comparisons.append(dict(configuration_id=trial["configuration_id"], settings=trial["settings"],
                                repaired=repaired, regressed=regressed,
                                unresolved_control_passes=unresolved,
                                preserves_recorded_control_passes=not (regressed or unresolved),
                                qualification=trial["qualification"]))
    measured = []
    trials_by_request = {trial["request_id"]: trial for trial in summary["trials"]}
    for role, request_id in [("control", progress["control_request"])] + [
            (str(trial["settings"]), trial["request_id"]) for trial in summary["trials"]]:
        entry = state["submissions"][request_id]
        for file in ("benchmarks/toy100/metrics.py", "benchmarks/toy100/problems.py"):
            require(file_hash(ROOT / file) == entry["request"]["source"]["files"][file], "Saved-only native scorer must match frozen source.")
        raw_rows = {r["task_id"]: r for r in validated_submission_rows(state, entry, attempts)}
        trial = trials_by_request.get(request_id, {})
        cells = {cell["task"]: cell for cell in trial.get("tasks", [])}
        for assignment in entry["request"]["view"]["assignments"]:
            if assignment["qualification_tier"] > 2:
                continue
            name = assignment["task"]
            raw_row = raw_rows.get(name, {})
            projected = compact_row(raw_row)
            projected["_word_hold_failure_witnesses"] = word_hold_failure_witnesses(
                raw_row, entry["request"]["tasks"][name])
            evidence = raw_row.get("evidence", {})
            if (entry["request"]["tasks"][name]["adapter"] == "native100"
                    and evidence.get("artifact_root") and evidence.get("artifact_manifest")):
                projected["_artifact_root"] = evidence["artifact_root"]
                projected["_artifact_manifest"] = evidence["artifact_manifest"]
            rows = sections(projected, entry["request"]["tasks"][name])
            status, reason = display_outcome(projected, cells.get(name), trial)
            measured.append(dict(role=role, task_id=name, tier=assignment["qualification_tier"],
                                 status=status, reason=reason, measurements=rows))
    result = dict(schema_version=1, name=args.name, source_commit=progress["source_commit"],
                  source_digest=progress["source_digest"], scope="ordinary_tuning_comparison",
                  default_adoption=False, independent_confirmation="not_performed",
                  control_results=control_results, trials=summary["trials"], comparisons=comparisons, task_measurements=measured,
                  cost=dict(control={k: state["campaigns"][control_entry["request"]["campaign_id"]][k]
                                     for k in ("spent_seconds", "reserved_seconds")},
                            search=summary.get("campaign_accounting")))
    atomic_json(p["report"] / "results.json", result)
    text = [f"# BCAP prior {args.name} search", "", f"Executed source `{progress['source_commit']}`; protocol seed 0; ordinary revision 9.", "",
            "Matched BCAP leader control and three global strengths. All six Tier 1 gates govern Tier 2 admission. This is tuning evidence.", "",
            "| Strength | Tier 1 passes | Tier 2 passes | Repaired | Regressed | Unresolved control passes |", "| --- | --- | --- | --- | --- | --- |"]
    for trial, comparison in zip(summary["trials"], comparisons):
        tiers = {r["tier"]: r for r in trial["qualification"]["tiers"]}
        text.append(f"| {trial['settings']} | {tiers[1]['required_passed']}/6 | {tiers[2]['required_passed']}/21 | {', '.join(comparison['repaired']) or 'none'} | {', '.join(comparison['regressed']) or 'none'} | {', '.join(comparison['unresolved_control_passes']) or 'none'} |")
    text += ["", "Every task metric and its frozen numerical gate are retained below and in [results.json](results.json). Unreached tasks remain unmeasured.", "", "The goal's [current leaderboard](../../technique-inventory.md) remains the sole generated leaderboard; this table is a source-bound experimental comparison.", ""]
    for task in measured:
        text += [f"## {task['role']}: {task['task_id']} — {task['status']}", ""]
        if task.get("reason"):
            text += [str(task["reason"]), ""]
        for section in task["measurements"]:
            text += [section["name"], ""]
            witnesses = "original_observations_sha256" in section
            if witnesses:
                text += [f"{section['numerical_failed_checks']}/{section['observation_count']} mandatory checks failed numerical bounds. "
                         f"Original cohort SHA256 `{section['original_observations_sha256']}`; no observations rescored for qualification.", ""]
            text += ["| Measurement | Score | Target | Status |" + (" Witness step | Original observation SHA256 |" if witnesses else ""),
                     "| --- | ---: | --- | --- |" + (" ---: | --- |" if witnesses else "")]
            for row in section["rows"]:
                text.append(f"| {row['metric']} | {fmt(row['score'])} | {row['target']} | {row['status']} |" +
                            (f" {row['witness_step']} | `{row['witness_observation_sha256']}` |" if witnesses else ""))
            text.append("")
    atomic_text(p["report"] / "README.md", "\n".join(text))
    print(json.dumps(dict(event="published", name=args.name, comparisons=comparisons)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "publish"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--axis")
    parser.add_argument("--values")
    parser.add_argument("--delta")
    parser.add_argument("--hypothesis")
    parser.add_argument("--falsifier")
    parser.add_argument("--source-commit")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--shared-pool", type=Path)
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    if args.action == "prepare":
        require(all(getattr(args, k) for k in ("axis", "values", "delta", "hypothesis", "falsifier")), "Finish the hypothesis, delta, grid and falsifier before preparation.")
    if args.action == "run":
        require(args.source_commit and (args.submit or args.drain), "Choose explicit source and submit/drain.")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
