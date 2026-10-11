"""Single-arm stall-reach critic read (closed PR #107) against a fresh BCAP leader control.

Adapted from ``reports/forge/bcap-prior-search/workflow.py``: same revision-9
discriminator_stability tasks, protocol seed 0, deterministic_orthogonal
initializer and frozen task architecture/data/prior/sampling/budget/cadence.
The sole trainer delta is ``generator_critic_read: "stall_reach"``. One finite
round, no strength sweep, no seeds, no retries.

    python reports/forge/stall-reach/workflow.py prepare --archive ARCHIVE
    git commit ...   # freeze every declaration
    python reports/forge/stall-reach/workflow.py run --archive ARCHIVE --source-commit SHA --submit --drain
    python reports/forge/stall-reach/workflow.py publish --archive ARCHIVE

``run`` appends one line per finished task to ``ARCHIVE/run.log`` (easy to tail).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("prior_search_workflow", ROOT / "reports/forge/bcap-prior-search/workflow.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from experiments.forge.contracts import atomic_json, atomic_text, read_json, stable_hash  # noqa: E402
from experiments.forge.decision_contracts import OUTCOMES  # noqa: E402
from experiments.forge.planning import plan_summary, resolve_idea  # noqa: E402
from experiments.forge.queue import Queue, drain  # noqa: E402
from experiments.forge.studies import validate_study  # noqa: E402
from experiments.forge.views import load_view  # noqa: E402

require, head, committed_source = base.require, base.head, base.committed_source
LEADER, VIEW = base.LEADER, base.VIEW
PER_CANDIDATE, FULL_RESERVATION = 45_000, 43_020
REPORT = ROOT / "reports/forge/stall-reach"
DELTA = {"generator_critic_read": "stall_reach"}
IDS = dict(control="bcap-stall-reach-control-v1", base="bcap-stall-reach-base-v1",
           control_study="bcap-stall-reach-control-study-v1", base_study="bcap-stall-reach-study-v1")
HYPOTHESIS = ("Closed PR #107 'stall reach': the generator reads the critic through a 2d+1 axis stencil whose width "
              "widens with the critic's measured input slope (max(.15, .5 min(u, 1/u))/kappa) and jumps to .5/kappa "
              "after 50 consecutive saturated (u >= .6) critic updates. Widening the generator's critic read when "
              "acquisition stalls lets particles see the missing mode. On the PR84 line it gave warm 200/200 and "
              "cold ring 8; here it should convert mode_hold FAIL (suffix 3) to a sustained PASS (suffix >= 5) "
              "under the unchanged BCAP direction-blend leader.")
FALSIFIER = ("No matched Tier 2 FAIL->PASS, or any lost Tier 1 or control Tier 2 pass. One finite round: no width or "
             "window tuning and no retries.")


def prepare(args):
    require(not (args.archive / "progress.json").exists(), "Admitted experiments are immutable.")
    view = load_view(ROOT, VIEW)
    require(view["revision"] == 9, "Freeze the current recipe-owned revision-9 view.")
    required = [a for a in view["assignments"] if a["importance"] == "required"]
    require(Counter(a["qualification_tier"] for a in required) == {1: 6, 2: 21, 3: 2}, "Preserve every task.")
    leader = read_json(ROOT / f"configs/forge/ideas/{LEADER}.json")
    require(leader["recipe_overrides"]["constraint_geometry_mode"] == "direction_blend", "Use the actual leader.")
    cards = {}
    for role in ("control", "base"):
        card = deepcopy(leader)
        card.update(id=IDS[role], parent=LEADER,
                    changed_factors=["Matched current-source BCAP leader control" if role == "control"
                                     else "Recipe.generator_critic_read='stall_reach' (closed PR #107)"],
                    mechanism_class="structural", mechanism_rationale=HYPOTHESIS,
                    guide="reports/forge/stall-reach/spec.json")
        if role == "base":
            card["recipe_overrides"].update(DELTA)
        cards[role] = card
        atomic_json(ROOT / f"configs/forge/ideas/{IDS[role]}.json", card)
    recipes = {role: base.resolved_recipe(card) for role, card in cards.items()}
    delta = {k: {"control": recipes["control"][k], "candidate": recipes["base"][k]}
             for k in recipes["control"] if recipes["control"][k] != recipes["base"][k]}
    require(set(delta) == set(DELTA), "The comparison must have only its declared trainer delta.")
    signature = dict(task_id="mode_hold", metric="modes", threshold=8, phase="final")
    evidence_path = "reports/forge/bcap-prior-search/covariance/results.json"
    evidence = read_json(ROOT / evidence_path)
    for role, study_key, other in (("control", "control_study", "base"), ("base", "base_study", "control")):
        study = dict(schema_version=1, id=IDS[study_key], status="ready", candidate=IDS[role],
                     control=dict(candidate_id=IDS[other], task_map={}), max_rounds=1,
                     hypothesis=("Reproduce the unchanged BCAP leader on this feature's frozen source." if role == "control"
                                 else HYPOTHESIS),
                     competing_explanation=("Source/host integration may change the control; archived controls are bound to "
                                            "their own source and cannot fill current cells." if role == "control" else
                                            "A wider generator read may blur mode resolution under BCAP's saturated slope, "
                                            "trading acquisition for density/hold quality."),
                     scope=dict(view=VIEW, through_tier=2, execution_backend="cuda", cuda_model="NVIDIA RTX A6000"),
                     campaign=dict(id=f"{IDS[role]}", budget_seconds=PER_CANDIDATE, candidate_budget_seconds=PER_CANDIDATE),
                     prior_evidence=[dict(path=evidence_path, selector=[],
                                          identity=dict(source_digest=evidence["source_digest"]), use="motivation_only")],
                     prediction={**signature, "op": "=="}, falsifier={**signature, "op": "<"}, terminal_rules=OUTCOMES)
        validate_study(study)
        atomic_json(ROOT / f"configs/forge/studies/{IDS[study_key]}.json", study)
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    plans = {}
    for role, study_key in (("control", "control_study"), ("base", "base_study")):
        resolved = resolve_idea(ROOT, IDS[role], study=IDS[study_key], queue_root=queue.root)
        require(not resolved["preflight_blockers"] and resolved["study_review"]["status"] == "READY",
                f"{role} is not ready: {resolved.get('preflight_blockers')}")
        plans[role] = resolved
        require(plan_summary(resolved)["worst_case_seconds"] == FULL_RESERVATION, "Retain the full task reservation.")
    require(plans["control"]["source"]["digest"] == plans["base"]["source"]["digest"], "Identical source for both arms.")
    contracts = {name: stable_hash(card) for name, card in plans["control"]["tasks"].items()}
    require(contracts == {name: stable_hash(card) for name, card in plans["base"]["tasks"].items()}, "Identical tasks.")
    spec = dict(schema_version=1, name="stall-reach", trainer_delta=delta, hypothesis=HYPOTHESIS, falsifier=FALSIFIER,
                protocol_seed=0, initializer="deterministic_orthogonal", view=VIEW, view_revision=9, baseline=LEADER,
                closed_pr=107, required_counts=dict(tier1=6, tier2=21, tier3=2), through_tier=2,
                full_reservation_seconds=2 * FULL_RESERVATION, maximum_reserved_paid_seconds=2 * PER_CANDIDATE,
                source_digest=plans["base"]["source"]["digest"], task_contracts=contracts,
                candidate_id=IDS["base"], control_id=IDS["control"], study_ids=[IDS["control_study"], IDS["base_study"]],
                control_reuse=("Archived leader controls (e.g. bcap-prior-covariance-control-v1) are bound to source "
                               f"{evidence['source_digest'][:12]}; this source differs, so one fresh unchanged-leader control runs."),
                archive=str(args.archive),
                acceptance=dict(ordinary_tier1_passes_required=6, minimum_matched_fail_to_pass=1,
                                preserve_all_control_tier2_passes=True, report_all_regressions=True,
                                primary="mode_hold FAIL (suffix 3) -> PASS (suffix >= 5)"),
                diagnostics=["ring16 first confirmed acquisition update vs control",
                             "fraction of updates widened/stalled and max width",
                             "per-task wall cost of the 2d+1 stencil"],
                stop="One finite round. Complete current-tier peers, then required non-PASS blocks higher tiers.")
    atomic_json(REPORT / "spec.json", spec)
    atomic_json(args.archive / "preflight.json", {role: plan_summary(plan) for role, plan in plans.items()})
    atomic_json(REPORT / "registration.json", dict(
        spec_sha256=stable_hash(spec), source_digest=spec["source_digest"], preparation_commit=head(),
        candidate_revisions={role: plans[role]["candidate_revision"] for role in plans},
        worst_case_seconds={role: plan_summary(plans[role])["worst_case_seconds"] for role in plans}))
    print(json.dumps(dict(event="prepared", source_digest=spec["source_digest"], delta=delta)), flush=True)


def _tail_log(queue, path, stop):
    """One line per finished task for both arms; bulk events stay in the queue tree."""
    seen = set()
    with open(path, "a", buffering=1) as log:
        while True:
            state = queue.inspect()
            for key, job in sorted(state["jobs"].items()):
                if key in seen or job["status"] in ("queued", "running"):
                    continue
                seen.add(key)
                result = job.get("result") or {}
                rows = result.get("task_results", [])
                for row in rows or [{"task_id": job.get("task_id", key[:12])}]:
                    cost = (row.get("cost") or result.get("cost") or {})
                    log.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), candidate=job.get("candidate_id", "")[:40],
                                              task=row.get("task_id"), status=job["status"],
                                              raw=row.get("raw_status") or row.get("status"),
                                              wall=round(cost.get("wall_seconds", 0) or 0, 1))) + "\n")
            if stop.is_set():
                return
            time.sleep(10)


def run(args):
    require(head() == args.source_commit, "Run only the explicitly frozen committed source.")
    require(not args.archive.is_relative_to(ROOT), "Bulk artifacts must live outside Git.")
    spec, registration = read_json(REPORT / "spec.json"), read_json(REPORT / "registration.json")
    require(stable_hash(spec) == registration["spec_sha256"], "Prepared study changed.")
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    progress = args.archive / "progress.json"
    if args.submit:
        require(not progress.exists(), "Do not duplicate an admitted scientific round.")
        requests = {}
        for role, study_key in (("control", "control_study"), ("base", "base_study")):
            resolved = resolve_idea(ROOT, IDS[role], study=IDS[study_key], queue_root=queue.root, freeze_source=True)
            require(resolved["source"]["digest"] == registration["source_digest"], "Prepared source changed.")
            require({n: stable_hash(c) for n, c in resolved["tasks"].items()} == spec["task_contracts"], "Tasks changed.")
            declarations = [f"configs/forge/ideas/{IDS[role]}.json", f"configs/forge/studies/{IDS[study_key]}.json"]
            declarations += [f"configs/forge/tasks/{name}.json" for name in resolved["tasks"]]
            committed_source(resolved["source"], declarations)
            entry = queue.submit(resolved, resolved["study"]["campaign"])
            requests[role] = entry["request"]["request_id"]
        atomic_json(progress, dict(source_commit=args.source_commit, source_digest=registration["source_digest"],
                                   phase="admitted", requests=requests))
    if args.drain:
        data = read_json(progress)
        require(data["source_commit"] == args.source_commit, "Admission source differs.")
        stop = threading.Event()
        logger = threading.Thread(target=_tail_log, args=(queue, args.archive / "run.log", stop), daemon=True)
        logger.start()
        try:
            if args.shared_pool:
                base.drain_shared_pool(queue, args.shared_pool.resolve(), args.gpus.split(","))
            else:
                drain(queue, args.gpus.split(","), workers_per_gpu=1, allow_sharing=True, watch=False, poll_seconds=2)
        finally:
            stop.set()
            logger.join()
        state = queue.inspect()
        unfinished = (any(e["status"] in ("queued", "running", "paused") for e in state["submissions"].values())
                      or any(j["status"] == "running" for j in state["jobs"].values()))
        data["phase"] = "paused" if unfinished else "complete"
        atomic_json(progress, data)
    print(json.dumps(dict(event="execution_state", progress=read_json(progress))), flush=True)


def _status(rows, entry, state, name):
    if name in rows:
        return rows[name]["gate_status"], rows[name].get("reason")
    for definition in entry["request"]["jobs"]:
        if name in definition.get("task_ids", [definition["task_id"]]):
            job = state["jobs"].get(definition["compatibility_key"], {})
            if job.get("status") in ("blocked", "skipped", "cancelled") or entry["status"] in ("blocked", "complete"):
                return "BLOCKED", job.get("reason") or "Blocked by a required Tier 1 non-PASS (complete_current_tier)."
    return "UNMEASURED", None


def _wall(row):
    return (row.get("cost") or {}).get("wall_seconds")


def publish(args):
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    progress = read_json(args.archive / "progress.json")
    require(progress["phase"] == "complete", "Finish the registered work before concluding it.")
    state = queue.inspect()
    from experiments.forge.knowledge import _attempts
    attempts = {a["attempt_id"]: a for a in _attempts(ROOT)[0]}
    view = load_view(ROOT, VIEW)
    tiers = {a["task"]: a["qualification_tier"] for a in view["assignments"]
             if a["importance"] == "required" and a["qualification_tier"] <= 2}
    roles, measured = {}, []
    for role, request_id in progress["requests"].items():
        entry = state["submissions"][request_id]
        raw = {r["task_id"]: r for r in base.validated_submission_rows(state, entry, attempts)}
        compact = {name: base.compact_row(row) for name, row in raw.items()}
        cells = {}
        for name, tier in tiers.items():
            status, reason = _status(compact, entry, state, name)
            row = raw.get(name, {})
            guards = (row.get("evidence") or {}).get("guards", {})
            read = guards.get("generator_critic_read")
            cells[name] = dict(tier=tier, status=status, reason=reason, wall_seconds=_wall(row),
                               critic_read=read, hooks_exercised=guards.get("hooks_exercised"))
            if name in raw:
                projected = deepcopy(compact[name])
                projected["_word_hold_failure_witnesses"] = base.word_hold_failure_witnesses(row, entry["request"]["tasks"][name])
                measured.append(dict(role=role, task_id=name, tier=tier, status=status, reason=reason,
                                     measurements=base.sections(projected, entry["request"]["tasks"][name])))
        roles[role] = dict(request_id=request_id, candidate_revision=entry["request"]["candidate_revision"], cells=cells,
                           cost={k: state["campaigns"][entry["request"]["campaign_id"]][k]
                                 for k in ("spent_seconds", "reserved_seconds")})
    control, candidate = roles["control"]["cells"], roles["base"]["cells"]
    passes = {role: {t: sum(c["status"] == "PASS" for c in roles[role]["cells"].values() if c["tier"] == t) for t in (1, 2)}
              for role in roles}
    tier2 = [n for n, c in control.items() if c["tier"] == 2]
    repaired = [n for n in tier2 if (control[n]["status"], candidate[n]["status"]) == ("FAIL", "PASS")]
    lost_tier1 = [n for n, c in control.items() if c["tier"] == 1 and c["status"] == "PASS" and candidate[n]["status"] != "PASS"]
    regressed = [n for n in tier2 if control[n]["status"] == "PASS" and candidate[n]["status"] == "FAIL"]
    unresolved = [n for n in tier2 if control[n]["status"] == "PASS" and candidate[n]["status"] not in ("PASS", "FAIL")]
    accepted = passes["base"][1] == 6 and bool(repaired) and not (regressed or unresolved)
    comparison = dict(tier1_passes=passes, repaired=repaired, lost_tier1=lost_tier1, regressed=regressed,
                      unresolved_control_passes=unresolved, accepted=accepted,
                      verdict="ACCEPTED" if accepted else "FALSIFIED")
    result = dict(schema_version=1, name="stall-reach", source_commit=progress["source_commit"],
                  source_digest=progress["source_digest"], scope="ordinary_single_arm_comparison", closed_pr=107,
                  default_adoption=False, independent_confirmation="not_performed", trainer_delta=DELTA,
                  roles=roles, comparison=comparison)
    atomic_json(REPORT / "results.json", result)
    text = ["# Stall-reach critic read: measurements", "",
            f"Executed source `{progress['source_commit']}`; protocol seed 0; ordinary revision 9. "
            "Every task metric and its frozen numerical gate. Unreached tasks remain unmeasured.", ""]
    for task in measured:
        text += [f"## {task['role']}: {task['task_id']} — {task['status']}", ""]
        if task.get("reason"):
            text += [str(task["reason"]), ""]
        for section in task["measurements"]:
            text += [section["name"], "", "| Measurement | Score | Target | Status |", "| --- | ---: | --- | --- |"]
            text += [f"| {r['metric']} | {base.fmt(r['score'])} | {r['target']} | {r['status']} |" for r in section["rows"]]
            text.append("")
    atomic_text(REPORT / "measurements.md", "\n".join(text))
    print(json.dumps(dict(event="published", comparison=comparison)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["prepare", "run", "publish"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shared-pool", type=Path)
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    if args.action == "run":
        require(args.source_commit and (args.submit or args.drain), "Choose explicit source and submit/drain.")
    globals()[args.action](args)


if __name__ == "__main__":
    main()
