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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import (
    enqueue_search, materialize_search, plan_search, report_search,
)
from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from experiments.forge.decision_contracts import OUTCOMES
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.queue import Queue, drain
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
        drain(queue, args.gpus.split(","), workers_per_gpu=args.workers_per_gpu, allow_sharing=True, watch=False)
        data["phase"] = "complete"
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
    return [measurement(key, metrics.get(key), limits.get(key, ()))
            for key in list(limits) + sorted(set(metrics) - set(limits))
            if key in limits or not isinstance(metrics[key], (str, dict, list))]


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
            "compatibility_key", "recipe", "initialization", "initial_prior", "prior_policy", "sampling")
    projected = {k: deepcopy(row[k]) for k in keep if k in row}
    evaluator = projected.get("evaluator_result", {})
    terminal = evaluator.pop("terminal_checks", None)
    if terminal is not None:
        evaluator["terminal_summary"] = dict(checks=len(terminal), passing_checks=sum(bool(p.get("passed")) for p in terminal))
    evidence = row.get("evidence", {})
    projected["provenance_checkpoint"] = deepcopy(evidence.get("provenance_checkpoint"))
    projected["sampling"] = {k: evidence.get(k) for k in ("sampling_contract_version", "sampling_law", "eval_output_noise")}
    projected["guards"] = deepcopy(evidence.get("guards", {}))
    return projected


def sections(row, card):
    evaluator, evaluation = row.get("evaluator_result", {}), card["evaluation"]
    result = [dict(name="Recorded endpoint", rows=metric_rows(row.get("metrics", {}), evaluation.get("thresholds", [])))]
    temporal = []
    convergence = evaluator.get("convergence", {})
    if evaluation["kind"] == "transfer_sustained":
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
    return result


def publish(args):
    p = paths(args.name)
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    progress = read_json(args.archive / "progress.json")
    require(progress["phase"] == "complete", "Finish the registered work before concluding it.")
    summary = report_search(ROOT, queue.root, p["search"], queue=queue)
    state = queue.inspect()
    control_entry = state["submissions"][progress["control_request"]]
    control_results = [compact_row(row) for row in queue._results(state, control_entry)]
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
    for role, request_id in [("control", progress["control_request"])] + [
            (str(trial["settings"]), trial["request_id"]) for trial in summary["trials"]]:
        entry = state["submissions"][request_id]
        for file in ("benchmarks/toy100/metrics.py", "benchmarks/toy100/problems.py"):
            require(file_hash(ROOT / file) == entry["request"]["source"]["files"][file], "Saved-only native scorer must match frozen source.")
        raw_rows = {r["task_id"]: r for r in queue._results(state, entry)}
        for assignment in entry["request"]["view"]["assignments"]:
            if assignment["qualification_tier"] > 2:
                continue
            name = assignment["task"]
            raw_row = raw_rows.get(name, {})
            projected = compact_row(raw_row)
            evidence = raw_row.get("evidence", {})
            if (entry["request"]["tasks"][name]["adapter"] == "native100"
                    and evidence.get("artifact_root") and evidence.get("artifact_manifest")):
                projected["_artifact_root"] = evidence["artifact_root"]
                projected["_artifact_manifest"] = evidence["artifact_manifest"]
            rows = sections(projected, entry["request"]["tasks"][name])
            measured.append(dict(role=role, task_id=name, tier=assignment["qualification_tier"],
                                 status=projected.get("gate_status", "UNMEASURED"), measurements=rows))
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
        for section in task["measurements"]:
            text += [section["name"], "", "| Measurement | Score | Target | Status |", "| --- | ---: | --- | --- |"]
            for row in section["rows"]:
                text.append(f"| {row['metric']} | {fmt(row['score'])} | {row['target']} | {row['status']} |")
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
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    if args.action == "prepare":
        require(all(getattr(args, k) for k in ("axis", "values", "delta", "hypothesis", "falsifier")), "Finish the hypothesis, delta, grid and falsifier before preparation.")
    if args.action == "run":
        require(args.source_commit and (args.submit or args.drain), "Choose explicit source and submit/drain.")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
