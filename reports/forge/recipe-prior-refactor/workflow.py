"""Fresh selected-leader measurement under recipe-owned priors.

Preparation never allocates a worker. Run requires explicit admission/drain and
a committed scientific source. Diagnostic requests cover unreached questions;
they cannot replace a failed ordinary result or confer qualification credit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import (atomic_json, atomic_text, file_hash,
    identifier, read_json, stable_hash, utc_now, validate_idea)
from experiments.forge.decision_contracts import OUTCOMES
from experiments.forge.planning import load_idea, plan_summary, resolve_idea
from experiments.forge.studies import validate_study
from experiments.forge.trainer_families import current_family_candidates
from experiments.forge.views import load_tasks, load_view, validate_view

VIEW = "discriminator_stability"
CAMPAIGN = "recipe-prior-refactor-selected-leaders-v1"
PER_CANDIDATE = 44_040
CAMPAIGN_CAP = 308_280
ORDINARY_ALLOWANCE = 43_020
PRIOR_FIELDS = dict(prior_update="learned", prior_regularizer="vicreg",
                    prior_reg_target_std=1.0, prior_reg_eps=1e-4, prior_l2=0.0)
ACTIVE = {"queued", "running", "paused"}
MIN_FREE_BYTES = 20 * 1024**3
MAX_ARCHIVE_BYTES = 18 * 1024**3
STOP_FREE_BYTES = 5 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def head(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def baseline():
    data = read_json(OUT / "baseline.json")
    require(data["source_commit"] == "03466efa4de8271b9a2c964406dc6f4f0f260792",
            "Preserve the exact selected pre-refactor develop lineage")
    require(len(data["leaders"]) == 7 and len(data["task_cards"]) == 30,
            "Preserve seven selected leaders and the revision-8 task catalog")
    for name, descriptor in data["task_cards"].items():
        path = OUT / "legacy-task-cards" / f"{name}.json"
        require(file_hash(path) == descriptor["sha256"] and path.stat().st_size == descriptor["bytes"],
                f"{name}: exact original task bytes changed")
    return data


def leader_id(family):
    return identifier(f"recipe-prior-refactor-{family}-v1")


def measured_candidate_id(leader):
    family = leader["selection"]["family"]
    # These immutable legacy declarations own an explicit reference cloud and
    # continuous-policy preset. Fresh studies bind the new task contract without
    # stripping their scientific requirements to fit a new declaration schema.
    return leader["selection"]["candidate_id"] if family in {"atlas", "e22"} else leader_id(family)


def control_task(name):
    return identifier(f"recipe-prior-refactor-legacy-{name}")


def question(task):
    """Project only explicitly declared ownership/source metadata changes."""
    result = deepcopy(task)
    result["execution"].pop("prior_contract", None)
    result["execution"]["prior"].pop("learnable", None)
    for key in ("sources", "evaluator_revision"):
        result["evaluation"].pop(key, None)
    if "requires_capabilities" in result:
        result["requires_capabilities"] = [x for x in result["requires_capabilities"] if x != "learned_locations"]
    return result


def ordinary_contract(root):
    old = baseline()
    view = load_view(root, VIEW)
    require(view["revision"] == 9 and view.get("evidence_scope", "ordinary") == "ordinary",
            "Use the explicit revision-9 recipe-owned ordinary view")
    require(view["assignments"] == old["view"]["assignments"],
            "Retain all original placements/order, including the optional clock audit")
    tasks = {}
    for assignment in view["assignments"]:
        name = assignment["task"]
        task = read_json(root / f"configs/forge/tasks/{name}.json")
        original = read_json(OUT / "legacy-task-cards" / f"{name}.json")
        require(task["execution"].get("prior_contract") == "recipe_owned_v1"
                and "learnable" not in task["execution"]["prior"], f"{name}: new ownership contract not bound")
        require(question(task) == question(original),
                f"{name}: architecture, law, initial prior, initialization, budget, cadence, gate or dependency changed")
        tasks[name] = task
    active = [a for a in view["assignments"] if a["qualification_tier"] <= 2]
    require(Counter(a["qualification_tier"] for a in active if a["importance"] == "required") == {1: 6, 2: 21},
            "Keep the full six/21 required denominator")
    require(sum(tasks[a["task"]]["resources"]["timeout_seconds"] for a in active) == ORDINARY_ALLOWANCE,
            "Original full allowances plus the optional 300-second clock audit must be retained")
    return view, tasks


def candidate_for(leader):
    from experiments.forge.taskrecipes import ADAPTABLE_FIELDS, RESOURCE_FIELDS
    original = leader["declaration"]
    if leader["selection"]["family"] in {"atlas", "e22"}:
        return deepcopy(original)
    result = {key: deepcopy(original[key]) for key in (
        "extensions", "requires_capabilities", "api_changes", "execution_path",
        "initializer", "claim_contract", "host_adaptation") if key in original}
    recipe = {**deepcopy(leader["portable_recipe"]), **PRIOR_FIELDS}
    # v3 resource fields live solely in the task. Preserve the reference's
    # objective/host delegation while removing its redundant resource entries;
    # the exact old declaration and full reference Recipe remain in baseline.
    # An explicit whole reference also makes previously implicit host defaults
    # explicit. Declare their existing task ownership rather than ask frozen
    # behavior hosts to change model/objective settings. recipe_owned_v1 removes
    # prior_reg from delegation, so the preserved weight remains global.
    result["host_adaptation"] = dict(schema_version=1,
        recipe_fields=sorted((ADAPTABLE_FIELDS - RESOURCE_FIELDS) & recipe.keys()))
    require(recipe["prior_reg"] == leader["prior_reg"], "Do not retune selected prior weights")
    result.update(schema_version=3, id=leader_id(leader["selection"]["family"]),
        trainer_family=leader["selection"]["configuration_family"],
        parent=leader["selection"]["candidate_id"], api_version="forge-api-v1",
        recipe_preset=original.get("recipe_preset"), recipe_overrides=recipe,
        changed_factors=["Recipe owns prior learn/freeze and regularization under recipe_owned_v1; selected global settings and prior_reg weight retained."],
        mechanism_class="structural",
        mechanism_rationale="Remeasure the selected recipe under an explicit prior ownership contract. Legacy host VICReg/L2 and task trainability are not silently inherited; this is not a superiority comparison with archived grades.",
        guide="reports/forge/recipe-prior-refactor/README.md")
    validate_idea(result)
    return result


def signature(tasks, name):
    thresholds = tasks[name]["evaluation"]["thresholds"]
    metric, op, threshold = next((x for x in thresholds if x[0] not in {"sample_count", "finite_fraction"}), thresholds[0])
    inverse = {"<=": ">", ">=": "<", "<": ">=", ">": "<=", "==": "!="}
    # Study signatures support ordered comparisons. Equality's missing outcome
    # is represented by a strict lower bound only where that is meaningful.
    if op == "==":
        op = ">="
    return (dict(task_id=name, metric=metric, op=op, threshold=threshold, phase="final"),
            dict(task_id=name, metric=metric, op=inverse[op], threshold=threshold, phase="final"))


def study_for(leader, view_id, tasks, names, *, diagnostic=False):
    cid = leader_id(leader["selection"]["family"])
    name = names[0] if diagnostic else "gaussian1d_smoke"
    prediction, falsifier = signature(tasks, name)
    study = dict(schema_version=1, id=f'{cid}-{"diagnostic" if diagnostic else "ordinary"}-study',
        status="ready", candidate=measured_candidate_id(leader),
        control=dict(candidate_id=leader["selection"]["candidate_id"],
                     task_map={n: control_task(n) for n in names}),
        hypothesis="The selected global recipe remains runnable and reaches the original numerical acquisition/hold gates under recipe-owned learned priors. Diagnose unreached questions separately; archived outcomes supply lineage only.",
        competing_explanation="Hidden host prior penalties or trainability previously altered the selected recipe. The refactor can change outcomes even when prior_reg is unchanged; source-bound fresh results are required.",
        scope=dict(view=view_id, through_tier=1 if diagnostic else 2,
                   execution_backend="cuda", cuda_model="NVIDIA RTX A6000"),
        campaign=dict(id=CAMPAIGN, budget_seconds=CAMPAIGN_CAP, candidate_budget_seconds=PER_CANDIDATE),
        max_rounds=1, prior_evidence=[dict(path="reports/forge/recipe-prior-refactor/baseline.json",
            selector=[], identity=dict(source_commit=baseline()["source_commit"],
                                      scope="exact_pre_refactor_selected_leader_settings"), use="motivation_only")],
        prediction=prediction, falsifier=falsifier, terminal_rules=deepcopy(OUTCOMES))
    validate_study(study)
    return study


def declarations_for(root):
    view, tasks = ordinary_contract(root)
    old = baseline()
    current = current_family_candidates(root)
    require([(c["family"], c["candidate_id"]) for c in current] ==
            [(x["selection"]["family"], x["selection"]["candidate_id"]) for x in old["leaders"]],
            "Selected roster changed; explicitly review a fresh baseline/campaign")
    declarations = {}
    for name in old["task_cards"]:
        task = read_json(OUT / "legacy-task-cards" / f"{name}.json")
        task["id"] = control_task(name)
        # Binding-only aliases retain their exact internal producer references.
        # Canonical hold validators require these identities; no alias is ever
        # assigned to an execution view or used as a checkpoint producer.
        declarations[f'configs/forge/tasks/{task["id"]}.json'] = task
    active = [a["task"] for a in view["assignments"] if a["qualification_tier"] <= 2]
    for leader in old["leaders"]:
        candidate = candidate_for(leader)
        study = study_for(leader, VIEW, tasks, active)
        require(file_hash(root / leader["declaration_path"]) == leader["declaration_sha256"],
                f'{leader["selection"]["family"]}: original selected declaration changed')
        if leader["selection"]["family"] not in {"atlas", "e22"}:
            declarations[f'configs/forge/ideas/{candidate["id"]}.json'] = candidate
        declarations[f'configs/forge/studies/{study["id"]}.json'] = study
    return declarations


def write_fresh(path, value):
    if path.exists():
        require(read_json(path) == value, f"Immutable declaration changed: {path}")
    else:
        atomic_json(path, value)


def install(root, declarations):
    for relative, value in declarations.items():
        write_fresh(root / relative, value)


def request_summary(request):
    summary = plan_summary(request)
    active = [r for r in summary["tasks"] if r["permitted_by_tier_cap"]]
    return dict(candidate_id=request["candidate"]["id"], study_id=request["study"]["id"],
        source_digest=request["source"]["digest"], candidate_revision=request["candidate_revision"],
        admission=request["study_review"]["status"], study_admission=request["study_admission"],
        preflight_blockers=request["preflight_blockers"], tasks=active,
        worst_case_seconds=summary["worst_case_seconds"], execution_policy=request["execution_policy"],
        protocol_seed=request["protocol"]["seed"],
        task_keys={n: j["compatibility_key"] for j in request["jobs"] for n in j.get("task_ids", [j["task_id"]])
                   if n in {a["task"] for a in active}})


def assert_selected_recipe(request, leader):
    from dataclasses import asdict
    from experiments.forge.api import resolve_public_recipe
    actual = asdict(resolve_public_recipe(request["candidate"]))
    for key, value in leader["portable_recipe"].items():
        require(stable_hash(actual[key]) == stable_hash(value),
                f'{leader["selection"]["family"]}: selected field {key} changed')
    require(all(actual[k] == v for k, v in PRIOR_FIELDS.items()), "New prior policy defaults changed")
    require(actual["prior_reg"] == leader["prior_reg"], "Selected prior_reg weight changed")


def saved_grading(attempt):
    """Validate saved independent grades; never execute an evaluator here."""
    request, result = attempt["request"], attempt["result"]
    require(request.get("requires_independent_grading") is True, "Require independent frozen grading")
    raw = result["raw"]
    rows = result["task_results"]
    require(len({r["task_id"] for r in rows}) == len(rows), "Duplicate certified task result")
    if raw["attempt_status"] != "completed":
        require(all(r["gate_status"] in {"INCOMPLETE", "BLOCKED"} for r in rows),
                "Noncompleted attempts cannot have passing or failed scientific grades")
        return dict(attempt_status=raw["attempt_status"], independently_graded=False)
    grading = raw.get("grading", {})
    require(grading.get("raw_hash") == stable_hash(raw["result"]), "Independent grading raw result hash mismatch")
    require(grading.get("source_digest") == request["source"]["digest"], "Independent grading source mismatch")
    require(set(grading.get("grades", {})) == {r["task_id"] for r in rows}, "Missing/extra independent task grade")
    for row in rows:
        grade = grading["grades"][row["task_id"]]
        expected = grade.get("gate_status", grade.get("status", "INVALID"))
        require(row["gate_status"] == expected, "Independent gate status differs from recorded task result")
        require(all(row.get(k) == v for k, v in grade.items()), "Independent grader method/result differs from task result")
        member_raw = raw["result"].get("task_results", {}).get(row["task_id"], raw["result"])
        require(row.get("evidence") == grade.get("evidence", member_raw.get("evidence")),
                "Recorded evidence differs from independently graded input")
        require(row.get("metrics") == grade.get("metrics", member_raw.get("metrics")),
                "Recorded metrics differ from independently graded output")
    return dict(attempt_status="completed", independently_graded=True,
        raw_result_sha256=grading["raw_hash"], source_digest=grading["source_digest"],
        grading_sha256=stable_hash(grading), grade_sha256={n: stable_hash(g) for n, g in grading["grades"].items()})


def resolve_plans(root, registration):
    ordinary_contract(root)
    requests = {}
    for family, plan in registration["families"].items():
        request = resolve_idea(root, plan["candidate_id"], study=plan["study_id"])
        leader = next(x for x in baseline()["leaders"] if x["selection"]["family"] == family)
        assert_selected_recipe(request, leader)
        require(request_summary(request) == plan, f"{family}: source/runtime/task/study binding changed")
        requests[family] = request
    require(len({r["source"]["digest"] for r in requests.values()}) <= 1,
            "Every candidate must run one common frozen scientific source")
    return requests


def prepare(options):
    root = options.repository.resolve()
    declarations = declarations_for(root)
    if not options.install:
        stage = options.artifacts / "ordinary-declarations" / stable_hash(declarations)[:16]
        install(stage, declarations)
        print(f"Staged {len(declarations)} declarations at {stage}; no admission or training.", flush=True)
        return
    install(root, declarations)
    old = baseline()
    plans = {}
    for leader in old["leaders"]:
        family = leader["selection"]["family"]
        request = resolve_idea(root, measured_candidate_id(leader), study=f'{leader_id(family)}-ordinary-study')
        assert_selected_recipe(request, leader)
        require(request["protocol"]["seed"] == 0 and request["execution_policy"]["mode"] == "complete_current_tier",
                "Seed0 and complete-current-tier ordinary policy are mandatory")
        plans[family] = request_summary(request)
        require(plans[family]["worst_case_seconds"] == ORDINARY_ALLOWANCE, f"{family}: full allowance changed")
    registration = dict(schema_version=1, scope="recipe_prior_refactor_ordinary_measurement",
        qualification_input=False, prepared_commit=head(root), campaign_id=CAMPAIGN,
        campaign_budget_seconds=CAMPAIGN_CAP, candidate_budget_seconds=PER_CANDIDATE,
        baseline_sha256=file_hash(OUT / "baseline.json"), view=VIEW, view_revision=9, through_tier=2,
        families=plans, declarations={p: stable_hash(v) for p, v in declarations.items()},
        lineage_declarations={l["declaration_path"]: l["declaration_sha256"] for l in old["leaders"]},
        diagnostic_policy="Only supported unmeasured Tier2 tasks. Same-scope producers may duplicate ordinary PASS prefixes; failed/incomplete prefixes cannot be retried. Diagnostic outcomes never grant ordinary credit.",
        storage=dict(estimated_bytes=15 * 1024**3, max_archive_bytes=MAX_ARCHIVE_BYTES,
                     admission_min_free_bytes=MIN_FREE_BYTES, stop_min_free_bytes=STOP_FREE_BYTES),
        interpretation="Remeasure every selected leader at the new prior contract; no archived qualification is rewritten and no superiority or default selection follows automatically.")
    write_fresh(options.registration, registration)
    print({f: dict(admission=p["admission"], task_blockers=sum(bool(t["blockers"] or t["execution_group_blockers"]) for t in p["tasks"]))
           for f, p in plans.items()}, flush=True)


def committed_source(root, source, declarations):
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    files = set(source["files"]) | set(declarations)
    require(files <= tracked, "Commit scientific source and all declarations before admission")
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *sorted(files)], cwd=root)
    require(result.returncode == 0, "Scientific source/declarations differ from committed HEAD")


def archive_size(path):
    if not path.exists():
        return 0
    return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])


def storage_check(artifacts, registration, *, admission=False):
    artifacts.mkdir(parents=True, exist_ok=True)
    policy = registration["storage"]
    used = archive_size(artifacts)
    free = shutil.disk_usage(artifacts).free
    require(used < policy["max_archive_bytes"], f"Archive storage cap reached: {used} bytes")
    require(free >= policy["admission_min_free_bytes" if admission else "stop_min_free_bytes"],
            f"Insufficient archive headroom: {free} bytes; retain historical archives and resolve capacity")
    return dict(archive_bytes=used, free_bytes=free)


def run(options):
    from experiments.forge.queue import Queue, drain
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    require(not artifacts.is_relative_to(root), "Bulk artifacts belong outside Git")
    require(head(root) == options.source_commit, "Run only the explicitly frozen committed source")
    registration = read_json(options.registration)
    require(registration["baseline_sha256"] == file_hash(OUT / "baseline.json"), "Original settings receipt changed")
    requests = resolve_plans(root, registration)
    source = next(iter(requests.values()))["source"] if requests else None
    if source:
        committed_source(root, source, set(registration["declarations"]) | set(registration.get("ordinary_declarations", {})) | set(registration["lineage_declarations"]))
    queue = Queue(artifacts / "queue", report_root=root / "reports/forge")
    lane = "diagnostic" if registration["scope"] == "recipe_prior_refactor_research_diagnostic" else "ordinary"
    progress_path = artifacts / f"{lane}-progress.json"
    if options.submit:
        require(not progress_path.exists(), "No unchanged campaign re-admission or scientific retry")
        storage = storage_check(artifacts, registration, admission=True)
        progress = dict(schema_version=1, phase=f"{lane}_submitted", requests={}, blocked={},
            registration_sha256=file_hash(options.registration), source_commit=options.source_commit,
            source_digest=source["digest"] if source else registration["source_digest"],
            storage_at_admission=storage, created_at=utc_now())
        # Persist the no-retry receipt before the first admission. A partial
        # administrative submission must be recovered explicitly, never replayed.
        atomic_json(progress_path, progress)
        for family, request in requests.items():
            plan = registration["families"][family]
            runnable = [t for t in plan["tasks"] if not (t["blockers"] or t["execution_group_blockers"])]
            if plan["admission"] != "READY" or plan["preflight_blockers"] or not runnable:
                progress["blocked"][family] = dict(admission=plan["admission"],
                    reasons=plan["preflight_blockers"], tasks=plan["tasks"])
                atomic_json(progress_path, progress)
                continue
            frozen = resolve_idea(root, plan["candidate_id"], study=plan["study_id"],
                                  queue_root=queue.root, freeze_source=True)
            require(request_summary(frozen) == plan, f"{family}: source changed during freezing")
            receipt = queue.submit(frozen, frozen["study"]["campaign"])
            progress["requests"][family] = receipt["request"]["request_id"]
            atomic_json(progress_path, progress)
            print(f'{family}: admitted {progress["requests"][family]}', flush=True)
    else:
        require(progress_path.is_file(), "Drain requires prior explicit admission")
        progress = read_json(progress_path)
        require(progress["registration_sha256"] == file_hash(options.registration)
                and progress["source_commit"] == options.source_commit, "Admitted source/registration changed")
    if options.drain and progress["requests"]:
        storage_check(artifacts, registration)
        stop = threading.Event()
        guard_errors = []
        def guard_storage():
            while not stop.wait(30):
                try:
                    storage_check(artifacts, registration)
                except Exception as exc:
                    guard_errors.append(str(exc))
                    queue.pause(CAMPAIGN, True)
                    atomic_json(artifacts / "storage-stop.json", dict(reason=str(exc), at=utc_now(),
                        action="Campaign paused; current bounded workers retain their leases, no new allocation."))
                    return
        thread = threading.Thread(target=guard_storage, daemon=True)
        thread.start()
        try:
            drain(queue, options.gpus.split(","), workers_per_gpu=options.workers_per_gpu,
                  allow_sharing=True, watch=False, campaign=CAMPAIGN)
        finally:
            stop.set()
            thread.join(timeout=2)
        require(not guard_errors, f"Storage guard paused campaign: {guard_errors}")
    state = queue.inspect()
    active = {f: state["submissions"][rid]["status"] for f, rid in progress["requests"].items()
              if state["submissions"][rid]["status"] in ACTIVE}
    if options.drain and not active:
        progress.update(phase=f"{lane}_complete", completed_at=utc_now())
        atomic_json(progress_path, progress)
    print(dict(phase=progress["phase"], active=active, blocked=list(progress["blocked"])), flush=True)


def read_ordinary(root, artifacts, registration):
    require(registration["scope"] == "recipe_prior_refactor_ordinary_measurement", "Expected ordinary registration")
    progress = read_json(artifacts / "ordinary-progress.json")
    require(progress["phase"] == "ordinary_complete" and progress["registration_sha256"] == file_hash(artifacts / "ordinary-registration.json"),
            "Diagnostic preparation requires the original completed ordinary registration")
    state = read_json(artifacts / "queue/queue/state.json")
    require(not any(state["submissions"][rid]["status"] in ACTIVE for rid in progress["requests"].values()),
            "Ordinary work must be terminal before choosing unreached diagnostic questions")
    publisher = load_publisher(root)
    for family, rid in progress["requests"].items():
        request = state["submissions"][rid]["request"]
        require(request_summary(request) == registration["families"][family],
                f"{family}: original admitted bindings differ")
        for job in request["jobs"]:
            result = state["jobs"][job["compatibility_key"]].get("result")
            if result is None:
                continue
            certificate = publisher.certified_attempt(root, result["attempt_id"])
            saved_grading(certificate)
            require(result == certificate["result"], "Diagnostic selection cannot trust an edited queue result")
            require(certificate["request"]["source"]["digest"] == request["source"]["digest"]
                    and certificate["result"]["candidate_revision"] == request["candidate_revision"],
                    "Diagnostic producer/result source differs from the admitted ordinary leader")
    return progress, state


def diagnostic_selection(request, results):
    """Never repeat a completed/failed task; only namespace-required PASS parents."""
    assignments = [a for a in request["view"]["assignments"]
                   if a["qualification_tier"] == 2 and a["importance"] == "required"]
    selected, producers, blocked = [], set(), {}
    for assignment in assignments:
        name = assignment["task"]
        if name in results:
            continue
        task = request["tasks"][name]
        if task.get("preflight_blockers"):
            blocked[name] = task["preflight_blockers"]
            continue
        dependencies = task.get("dependencies", [])
        missing = [d["task"] for d in dependencies if results.get(d["task"], {}).get("gate_status") != "PASS"]
        if missing:
            blocked[name] = [f"Own ordinary producer did not PASS: {p}; no failed/incomplete producer retry" for p in missing]
            continue
        selected.append(name)
        producers.update(d["task"] for d in dependencies)
    require(producers <= {"gaussian1d_smoke", "five_word_joint_smoke"}, "Review any new producer dependency before spending")
    return sorted(producers), selected, blocked


def prepare_diagnostic(options):
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    ordinary_path = artifacts / "ordinary-registration.json"
    ordinary = read_json(ordinary_path)
    progress, state = read_ordinary(root, artifacts, ordinary)
    requests = resolve_plans(root, ordinary)
    declarations, summaries, plans = {}, {}, {}
    source_digest = next(iter(requests.values()))["source"]["digest"]
    for leader in baseline()["leaders"]:
        family = leader["selection"]["family"]
        request = requests[family]
        if family not in progress["requests"]:
            summaries[family] = dict(selected=[], producers=[], blocked="Ordinary family unsupported; no diagnostic host substitution")
            continue
        rid = progress["requests"][family]
        results = {r["task_id"]: r for j in request["jobs"]
                   if state["jobs"][j["compatibility_key"]].get("result")
                   for r in state["jobs"][j["compatibility_key"]]["result"]["task_results"]}
        producers, selected, blocked = diagnostic_selection(request, results)
        summaries[family] = dict(selected=selected, producers=producers, blocked=blocked,
            ordinary_request_id=rid, ordinary_result_sha256=stable_hash(results),
            duplicate_producer_allowance_seconds=sum(request["tasks"][p]["resources"]["timeout_seconds"] for p in producers))
        if not selected:
            continue
        names = producers + selected
        view_id = f"recipe-prior-refactor-{family}-diagnostic-v1"
        view = deepcopy(request["view"])
        view.update(id=view_id, revision=1, evidence_scope="research_diagnostic", eligibility={},
            assignments=[dict(task=n, qualification_tier=1, importance="diagnostic", order=i) for i, n in enumerate(names)],
            ranking=dict(policy="diagnostic_outcomes_only", cost_separate=True),
            policy_change_reason="Measure only original Tier2 questions unreached by the terminal recipe-owned ordinary run. Own namespace-specific successful prefix duplicates are charged separately; no failed result retry and no ordinary qualification credit.")
        validate_view(view, load_tasks(root))
        study = study_for(leader, view_id, request["tasks"], names, diagnostic=True)
        declarations[f"configs/forge/views/{view_id}.json"] = view
        declarations[f'configs/forge/studies/{study["id"]}.json'] = study
    if not options.install:
        stage = artifacts / "diagnostic-declarations" / stable_hash(declarations)[:16]
        install(stage, declarations)
        atomic_json(stage / "selection.json", summaries)
        print(f"Staged {len(declarations)} diagnostic declarations; no admission/training.", flush=True)
        return
    install(root, declarations)
    for family, selected in summaries.items():
        if not selected["selected"]:
            continue
        leader = next(l for l in baseline()["leaders"] if l["selection"]["family"] == family)
        request = resolve_idea(root, measured_candidate_id(leader), study=f"{leader_id(family)}-diagnostic-study")
        require(request["source"]["digest"] == source_digest, "Diagnostics must preserve the ordinary scientific source")
        require(request["study_review"]["status"] == "READY" and not request["preflight_blockers"],
                f"{family}: diagnostic binding not READY")
        require(all(not(t["blockers"] or t["execution_group_blockers"]) for t in request_summary(request)["tasks"]),
                f"{family}: diagnostic host blocked")
        plan = request_summary(request)
        oldkeys = set(ordinary["families"][family]["task_keys"].values())
        require(not oldkeys.intersection(plan["task_keys"].values()), "Diagnostic namespace must not alias ordinary credit")
        plans[family] = plan
    registration = dict(schema_version=1, scope="recipe_prior_refactor_research_diagnostic", qualification_input=False,
        prepared_commit=head(root), source_digest=source_digest, campaign_id=CAMPAIGN,
        campaign_budget_seconds=CAMPAIGN_CAP, candidate_budget_seconds=PER_CANDIDATE,
        baseline_sha256=file_hash(OUT / "baseline.json"), ordinary_registration_sha256=file_hash(ordinary_path),
        families=plans, selection=summaries, storage=ordinary["storage"],
        declarations={p: stable_hash(v) for p, v in declarations.items()},
        ordinary_declarations=ordinary["declarations"], lineage_declarations=ordinary["lineage_declarations"],
        interpretation="Separate unreached-task diagnostic measurements; prefixes are duplicate namespace dependencies, not replications. All ordinary failures and blocks remain unchanged.")
    write_fresh(options.registration, registration)
    print(summaries, flush=True)


def load_publisher(root):
    spec = importlib.util.spec_from_file_location("prior_refactor_saved_publication", root / "reports/forge/bcap-develop-integration/publish.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(options):
    """Read only certified saved results. Optional media adds no science."""
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    ordinary = read_json(artifacts / "ordinary-registration.json")
    progress, state = read_ordinary(root, artifacts, ordinary)
    publisher = load_publisher(root)
    requirements = [a for a in baseline()["view"]["assignments"] if a["qualification_tier"] <= 2]
    attempts, rows, cost, media = {}, [], {}, []
    lanes = [("ordinary", ordinary, progress)]
    diagnostic_path = artifacts / "diagnostic-registration.json"
    if diagnostic_path.exists():
        diagnostic = read_json(diagnostic_path)
        dp = read_json(artifacts / "diagnostic-progress.json")
        require(dp["phase"] == "diagnostic_complete" and dp["registration_sha256"] == file_hash(diagnostic_path),
                "Collect only completed diagnostic work")
        lanes.append(("research_diagnostic", diagnostic, dp))
    renderer = publisher._saved_renderer(root) if options.media else None
    for scope, registration, lane_progress in lanes:
        for family, plan in registration["families"].items():
            rid = lane_progress["requests"].get(family)
            if rid is None:
                continue
            sub = state["submissions"][rid]
            require(sub["status"] not in ACTIVE, f"{family}/{scope}: active work")
            request = sub["request"]
            require(request_summary(request) == plan, f"{family}/{scope}: admitted bindings differ")
            for job in state["jobs"].values():
                if rid not in job["subscribers"]:
                    continue
                require(len(job["attempts"]) <= 1, "This measurement campaign admits no scientific retries")
                for history in job["attempts"]:
                    aid = history["attempt_id"]
                    if aid not in attempts:
                        attempt = publisher.certified_attempt(root, aid)
                        require(job["result"] == attempt["result"], f"{aid}: queue/certificate mismatch")
                        require(attempt["request"]["candidate_revision"] == plan["candidate_revision"]
                                and attempt["request"]["source"]["digest"] == plan["source_digest"], f"{aid}: scientific source mismatch")
                        attempts[aid] = attempt
                        saved_grading(attempt)
                        charges = [c for c in state["charges"] if c["attempt_id"] == aid]
                        require(len(charges) == 1, f"{aid}: missing/duplicate charge")
                        seconds = sum(r.get("cost", {}).get("wall_seconds", 0) for r in attempt["result"]["task_results"])
                        require(math.isclose(seconds, charges[0]["seconds"], abs_tol=1e-7), f"{aid}: cost mismatch")
                        cost[aid] = dict(scope=scope, family=family, seconds=seconds,
                            full_allowance_seconds=job["definition"]["budget_seconds"])
                    attempt = attempts[aid]
                    for row in attempt["result"]["task_results"]:
                        name = row["task_id"]
                        require(row["compatibility_key"] == plan["task_keys"][name], f"{aid}: task identity mismatch")
                        compact = next(x for x in attempt["compact"]["task_results"] if x["task_id"] == name)
                        saved, proof = publisher.checkpoint(row)
                        item = dict(scope=scope, family=family, task_id=name, attempt_id=aid,
                            candidate_revision=plan["candidate_revision"], source_digest=plan["source_digest"],
                            task_contract_sha256=stable_hash(request["tasks"][name]), provenance_checkpoint=proof)
                        item.update(compact)
                        item["qualification_input"] = False
                        item["recorded_evidence_scope"] = scope
                        item["independent_grading"] = saved_grading(attempt)
                        rows.append(item)
                        if options.media and row.get("evidence", {}).get("saved_observer_outputs"):
                            output = options.output / "media" / f"{family}-{scope}-{name}.gif"
                            entry = dict(task=request["tasks"][name], row=row, attempt=attempt, request=request)
                            receipt = publisher.render_saved(entry, output, renderer)
                            media.append(dict(family=family, scope=scope, task_id=name, path=str(output), sha256=file_hash(output), receipt=receipt))
    ordinary_lookup = {(r["family"], r["task_id"]): r for r in rows if r["scope"] == "ordinary"}
    diagnostic_lookup = {(r["family"], r["task_id"]): r for r in rows if r["scope"] == "research_diagnostic"}
    cells, standings = [], []
    for leader in baseline()["leaders"]:
        family = leader["selection"]["family"]
        plan = ordinary["families"][family]
        plans = {t["task"]: t for t in plan["tasks"]}
        for assignment in requirements:
            name = assignment["task"]
            recorded = ordinary_lookup.get((family, name))
            diagnostic = diagnostic_lookup.get((family, name))
            reasons = plans[name]["blockers"] + plans[name]["execution_group_blockers"]
            if not reasons and recorded is None:
                sub = state["submissions"].get(progress["requests"].get(family, ""), {})
                reasons = plan["preflight_blockers"] or [sub.get("reason") or "No certified ordinary result; tier or own checkpoint prerequisites unmet"]
            cells.append(dict(family=family, task_id=name, original_tier=assignment["qualification_tier"],
                importance=assignment["importance"], ordinary_status=recorded["gate_status"] if recorded else "BLOCKED",
                ordinary_attempt=recorded["attempt_id"] if recorded else None, reason=reasons,
                diagnostic_status=diagnostic["gate_status"] if diagnostic else None,
                diagnostic_attempt=diagnostic["attempt_id"] if diagnostic else None,
                diagnostic_qualification_input=False))
        group = [c for c in cells if c["family"] == family and c["importance"] == "required"]
        standings.append(dict(family=family, tier1=dict(Counter(c["ordinary_status"] for c in group if c["original_tier"] == 1)),
            tier2=dict(Counter(c["ordinary_status"] for c in group if c["original_tier"] == 2)),
            diagnostic_tier2=dict(Counter(c["diagnostic_status"] for c in group if c["diagnostic_status"] is not None)),
            paid_seconds=sum(v["seconds"] for v in cost.values() if v["family"] == family)))
    certificate_hashes = {aid: {n: file_hash(root / f"reports/forge/attempts/{aid}/{n}.json")
                                for n in ("request", "result", "evidence")} for aid in attempts}
    result = dict(schema_version=1, scope="recipe_prior_refactor_selected_leader_measurement",
        qualification_input=False, baseline_sha256=file_hash(OUT / "baseline.json"),
        ordinary_registration_sha256=file_hash(artifacts / "ordinary-registration.json"),
        diagnostic_registration_sha256=file_hash(diagnostic_path) if diagnostic_path.exists() else None,
        source_digest=progress["source_digest"], selected_leaders=standings, task_cells=cells,
        certified_results=rows, exact_certificate_hashes=certificate_hashes, costs=cost,
        total_paid_seconds=sum(x["seconds"] for x in cost.values()), media=media,
        interpretation="Fresh ordinary grades under the explicit recipe-owned prior contract. Diagnostic values are separate measurements with no qualification credit. Historical pin/leaderboard/default selection is unchanged.")
    atomic_json(options.output / "results.json", result)
    lines = ["# Recipe-owned prior measurement", "", result["interpretation"], "",
        "| Selected family | Ordinary Tier1 | Ordinary Tier2 | Separate Tier2 diagnostics | Paid seconds |",
        "| --- | --- | --- | --- | --- |"]
    for x in standings:
        counts = lambda v: ", ".join(f"{n} {status}" for status, n in sorted(v.items())) or "none"
        lines.append(f'| {x["family"]} | {counts(x["tier1"])} | {counts(x["tier2"])} | {counts(x["diagnostic_tier2"])} | {x["paid_seconds"]:.3f} |')
    lines += ["", "[Exact task statuses, certificate/source identities and measured metrics](results.json).",
              "", "One current leaderboard for this goal. No archived qualification or selection pin was rewritten."]
    atomic_text(options.output / "README.md", "\n".join(lines) + "\n")
    print(dict(attempts=len(attempts), paid_seconds=result["total_paid_seconds"], cells=len(cells), gifs=len(media)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-diagnostic", "run", "collect"))
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010"))
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--install", action="store_true", help="Explicitly write fresh declarations into configs; never admits a worker")
    parser.add_argument("--source-commit")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUT / "publication")
    parser.add_argument("--media", action="store_true", help="Export retained scored training views; no resampling/rescoring")
    options = parser.parse_args()
    if options.registration is None:
        options.registration = options.artifacts / ("diagnostic-registration.json" if options.action == "prepare-diagnostic" else "ordinary-registration.json")
    if options.action == "run":
        require(options.source_commit and (options.submit or options.drain), "Run requires --source-commit and explicit --submit/--drain")
        require(options.workers_per_gpu > 0, "workers-per-gpu must be positive")
    dict(prepare=prepare, run=run, collect=collect, **{"prepare-diagnostic": prepare_diagnostic})[options.action](options)


if __name__ == "__main__":
    main()
