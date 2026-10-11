"""BCAP leader vs the opt-in generator idle rule through the ordinary gate ladder.

Reuses the prior-search runner (shared physical GPU admission, receipt-bound
publication) with a single declared trainer delta, generator_idle_se None -> 1.0.
prepare writes declarations; run needs their committed source; publish reads
only saved receipts. Bulk queue state lives in the git-ignored archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "prior_search_workflow", ROOT / "reports/forge/bcap-prior-search/workflow.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from experiments.forge.configuration_search import enqueue_search, materialize_search, plan_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.decision_contracts import OUTCOMES
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.queue import Queue
from experiments.forge.studies import validate_study
from experiments.forge.views import load_view

require = base.require
LEADER, VIEW, PER_CANDIDATE, FULL_RESERVATION = base.LEADER, base.VIEW, base.PER_CANDIDATE, base.FULL_RESERVATION
NAME = "generator-idle"
REPORT = ROOT / "reports/forge/bcap-generator-idle"
AXIS, VALUE = "generator_idle_se", 1.0
HYPOTHESIS = ("Skipping the G/prior step while the critic's paired real-fake gap is within one standard "
              "error of zero stops the constant-rate BCAP leader drifting during holds, repairing "
              "gaussian1d_stability and mode_hold without losing any control pass.")
FALSIFIER = "gaussian1d_stability final std_ratio < .8 (the control's collapse persists)."
EVIDENCE = "reports/forge/bcap-generator-idle/prior_evidence.json"


def paths(name=NAME):
    require(name == NAME, "This workflow registers only the generator idle comparison.")
    stem = "bcap-generator-idle"
    return dict(stem=stem, control=f"{stem}-control-v1", base=f"{stem}-base-v1",
                study=f"{stem}-control-study-v1", search=f"{stem}-search-v1", report=REPORT)


base.paths = paths


def prepare(args):
    p = paths(args.name)
    require(not (args.archive / "progress.json").exists(), "Admitted experiments are immutable.")
    view = load_view(ROOT, VIEW)
    require(view["revision"] == 9, "Freeze the current recipe-owned revision-9 view.")
    required = [a for a in view["assignments"] if a["importance"] == "required"]
    require(Counter(a["qualification_tier"] for a in required) == {1: 6, 2: 21, 3: 2},
            "Preserve every required task and gate.")
    leader = read_json(ROOT / f"configs/forge/ideas/{LEADER}.json")
    require(leader["recipe_overrides"]["constraint_geometry_mode"] == "direction_blend",
            "Use the actual direction-blend leader.")
    cards = {}
    for role in ("control", "base"):
        card = deepcopy(leader)
        card.update(id=p[role], parent=LEADER,
                    changed_factors=["Matched current-source BCAP leader control" if role == "control" else HYPOTHESIS],
                    mechanism_class="structural", mechanism_rationale=HYPOTHESIS,
                    guide=f"reports/forge/bcap-generator-idle/spec.json")
        if role == "base":
            card["recipe_overrides"][AXIS] = VALUE
        cards[role] = card
        atomic_json(ROOT / f"configs/forge/ideas/{p[role]}.json", card)
    recipes = {role: base.resolved_recipe(card) for role, card in cards.items()}
    actual_delta = {k: {"control": recipes["control"][k], "candidate": recipes["base"][k]}
                    for k in recipes["control"] if recipes["control"][k] != recipes["base"][k]}
    require(actual_delta == {AXIS: {"control": None, "candidate": VALUE}},
            "The comparison must have only its declared trainer delta.")
    evidence = read_json(ROOT / EVIDENCE)
    signature = dict(task_id="gaussian1d_stability", metric="std_ratio", threshold=.8, phase="final")
    study = dict(schema_version=1, id=p["study"], status="ready", candidate=p["control"],
                 control=dict(candidate_id=p["base"], task_map={}), max_rounds=1,
                 hypothesis="Reproduce the unchanged BCAP leader on this feature's frozen source before scoring the idle rule.",
                 competing_explanation="Source or host integration may change the control; historical 9/21 evidence cannot fill current cells.",
                 scope=dict(view=VIEW, through_tier=2, execution_backend="cuda", cuda_model="NVIDIA RTX A6000"),
                 campaign=dict(id=f"{p['stem']}-control-v1", budget_seconds=PER_CANDIDATE,
                               candidate_budget_seconds=PER_CANDIDATE),
                 prior_evidence=[dict(path=EVIDENCE, selector=["pull_requests", index],
                                      identity=dict(number=item["number"], head_sha=item["head_sha"]),
                                      use="motivation_only")
                                 for index, item in enumerate(evidence["pull_requests"])],
                 prediction={**signature, "op": ">="}, falsifier={**signature, "op": "<"}, terminal_rules=OUTCOMES)
    validate_study(study)
    atomic_json(ROOT / f"configs/forge/studies/{p['study']}.json", study)
    spec = dict(schema_version=2, id=p["search"], trainer_family=leader["trainer_family"],
                base_candidate=p["base"], grid={AXIS: [VALUE]}, tuning_through_tier=2,
                view=VIEW, execution_backend="cuda", cuda_model="NVIDIA RTX A6000", protocol="screening",
                campaign=dict(schema_version=1, id=p["search"], budget_seconds=PER_CANDIDATE,
                              candidate_budget_seconds=PER_CANDIDATE, accept_shared_cost_transfer=False,
                              purpose="One declared idle threshold (PR #366's planned 1.0 SE), all six Tier1 tasks then all21 Tier2 if eligible; complete_current_tier. Seed0, public initializer, original task architecture/data/prior/capacity/sampling/updates/cadence/gates. Separate matched leader control (45000 s; 90000 s total). No tuning grid, scientific retries, seed variants, Tier3, per-task recipes or default adoption. GPU sharing is intentional; costs are not a speed ranking."),
                hypothesis=HYPOTHESIS,
                rationale="Convert gaussian1d_stability and/or mode_hold FAIL to a sustained full-gate PASS while keeping 6/6 Tier1 and every control Tier2 pass, including five_word_joint_hold. Retain every regression and blocked/unmeasured cell.",
                guide="reports/forge/bcap-generator-idle/spec.json")
    atomic_json(ROOT / f"configs/forge/searches/{p['search']}.json", spec)
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    materialize_search(ROOT, spec)
    search = plan_search(ROOT, queue.root, spec, queue=queue)
    control = resolve_idea(ROOT, p["control"], study=p["study"], queue_root=queue.root)
    planned = plan_summary(control)
    require(not control["preflight_blockers"] and control["study_review"]["status"] == "READY",
            f"Control is not ready: {control.get('preflight_blockers')} {control['study_review']}")
    task_blockers = {name: task.get("preflight_blockers") for name, task in control["tasks"].items()
                     if task.get("preflight_blockers")}
    require(not task_blockers, f"Control tasks are blocked: {task_blockers}")
    require(all(t["submission_status"] == "READY" for t in search["trials"]),
            f"Every configuration must preflight: {[t.get('submission_blockers') for t in search['trials']]}")
    require(planned["worst_case_seconds"] == FULL_RESERVATION
            and search["declared_worst_case_seconds"] == FULL_RESERVATION,
            "Retain the complete original task reservation, including the optional clock audit.")
    require(search["source_digest"] == control["source"]["digest"], "Control and trial need identical source.")
    output = dict(schema_version=1, name=NAME, axis=AXIS, values=[VALUE], trainer_delta=actual_delta,
                  hypothesis=HYPOTHESIS, falsifier=FALSIFIER,
                  prediction="gaussian1d_stability final std_ratio >= .8 and cdf_ks <= .05",
                  target_gates=dict(gaussian1d_stability="FAIL -> PASS", mode_hold="FAIL -> PASS"),
                  protocol_seed=0, view_revision=9, baseline=LEADER,
                  required_counts=dict(tier1=6, tier2=21, tier3=2), through_tier=2,
                  full_reservation_seconds=2 * FULL_RESERVATION, maximum_reserved_paid_seconds=2 * PER_CANDIDATE,
                  source_digest=search["source_digest"],
                  task_contracts={name: stable_hash(card) for name, card in control["tasks"].items()},
                  search_id=p["search"], control_id=p["control"], control_study_id=p["study"],
                  archive=str(args.archive), acceptance=dict(minimum_matched_fail_to_pass=1,
                      ordinary_tier1_passes_required=6, report_all_regressions=True,
                      preserve_all_control_tier2_passes_for_global_repair=True),
                  stop="One finite round. Complete current-tier peers, then required non-PASS blocks higher tiers. No automatic refinement or scientific retry.")
    atomic_json(REPORT / "spec.json", output)
    atomic_json(args.archive / "preflight.json", dict(control_plan=planned, search_plan=search))
    atomic_json(REPORT / "registration.json", dict(spec_sha256=stable_hash(output), source_digest=search["source_digest"],
                preparation_commit=base.head(), control_candidate_revision=control["candidate_revision"],
                control_worst_case_seconds=planned["worst_case_seconds"],
                search_worst_case_seconds=search["declared_worst_case_seconds"],
                trials=[{k: trial[k] for k in ("candidate_id", "configuration_id", "candidate_revision", "settings",
                                              "submission_status", "submission_blockers", "declared_worst_case_seconds")}
                        for trial in search["trials"]]))
    print(json.dumps(dict(event="prepared", name=NAME, source_digest=search["source_digest"],
                          configurations=[t["configuration_id"] for t in search["trials"]])), flush=True)


def run(args):
    p = paths(args.name)
    require(base.head() == args.source_commit, "Run only the explicitly frozen committed source.")
    require(not args.archive.is_relative_to(ROOT), "Bulk artifacts must live outside this checkout.")
    spec, registration = read_json(REPORT / "spec.json"), read_json(REPORT / "registration.json")
    require(stable_hash(spec) == registration["spec_sha256"], "Prepared study changed.")
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    progress = args.archive / "progress.json"
    if args.submit:
        require(not progress.exists(), "Do not duplicate an admitted scientific round.")
        control = resolve_idea(ROOT, p["control"], study=p["study"], queue_root=queue.root, freeze_source=True)
        require(control["source"]["digest"] == registration["source_digest"], "Prepared source changed.")
        require({name: stable_hash(card) for name, card in control["tasks"].items()} == spec["task_contracts"],
                "Prepared task contracts changed.")
        declarations = [f"configs/forge/{kind}/{p[key]}.json" for kind, key in (
            ("ideas", "control"), ("ideas", "base"), ("studies", "study"), ("searches", "search"))]
        declarations += [f"configs/forge/tasks/{name}.json" for name in control["tasks"]]
        base.committed_source(control["source"], declarations)
        entry = queue.submit(control, control["study"]["campaign"])
        search = enqueue_search(ROOT, queue.root, p["search"], queue=queue)
        require(search["source_digest"] == registration["source_digest"] and search["submitted_count"] == 1,
                "The single trial must share the prepared source.")
        atomic_json(progress, dict(source_commit=args.source_commit, source_digest=registration["source_digest"],
                                   phase="admitted", control_request=entry["request"]["request_id"],
                                   search_requests=[t["request_id"] for t in search["trials"]]))
    if args.drain:
        base.run(argparse.Namespace(**{**vars(args), "submit": False, "workers_per_gpu": 1}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "publish"])
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shared-pool", type=Path)
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    args.workers_per_gpu = 1
    if args.action == "run":
        require(args.source_commit and (args.submit or args.drain), "Choose explicit source and submit/drain.")
    {"prepare": prepare, "run": run, "publish": base.publish}[args.action](args)


if __name__ == "__main__":
    main()
