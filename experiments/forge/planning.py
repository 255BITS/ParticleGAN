"""Resolve small idea declarations to frozen scientific requests, without running."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .contracts import atomic_json, identifier, read_json, stable_hash, validate_idea
from .sampling import candidate_blockers, task_blockers
from .priors import task_prior
from .sources import compute_profile, inspect_source, runtime_manifest, snapshot_source
from .views import (load_tasks, load_view, task_evaluation_fingerprint,
                    task_execution_fingerprint, validate_view, view_fingerprint)

FORMULATION_FIELDS = ("recipe_preset", "recipe_overrides", "prior", "extensions", "requires_capabilities",
                      "api_changes", "implementation", "initializer", "claim_contract")



def candidate_revision_for(source_digest: str, candidate: dict) -> str:
    """Identity of an already resolved formulation, independent of its label.

    Keep the field selection in one place for planning and frozen-request
    validation. Callers at execution boundaries must first reconstruct and
    validate resolved_recipe from the actual public formulation declaration.
    """
    formulation = {key: candidate.get(key) for key in FORMULATION_FIELDS}
    # Preserve identities of existing declarations with no adaptation contract.
    if candidate.get("host_adaptation") is not None:
        formulation["host_adaptation"] = candidate["host_adaptation"]
    formulation.update(resolved_recipe=candidate["resolved_recipe"], prior=candidate["prior"],
                       api_version=candidate.get("api_version", "forge-api-v1"))
    return stable_hash({"source": source_digest, "formulation": formulation})


def rekey_jobs(jobs):
    """Rebind complete prerequisite identities after a protocol or lane change."""
    by_task = {member: job for job in jobs for member in job.get("task_ids", [job["task_id"]])}
    bound, visiting = set(), set()
    def bind(job):
        group = job.get("execution_group", job["task_id"])
        if group in bound:
            return job["compatibility_key"]
        if group in visiting:
            raise ValueError("cyclic execution-group prerequisites")
        visiting.add(group)
        parents = job["science"].get("prerequisites", {})
        if parents:
            job["science"]["prerequisites"] = {name: bind(by_task[name]) for name in sorted(parents)}
        job["compatibility_key"] = stable_hash(job["science"])
        visiting.remove(group)
        bound.add(group)
        return job["compatibility_key"]
    for job in jobs:
        bind(job)
    return jobs


def declaration_paths(root: Path) -> list[Path]:
    """Ordinary ideas and immutable configurations share the same planner."""
    paths = sorted(path for directory in ("ideas", "configurations")
                   for path in (Path(root) / "configs/forge" / directory).glob("*.json"))
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("candidate ids must be unique across ideas and configurations")
    return paths


def discover_candidate_ids(root: Path) -> list[str]:
    return sorted(path.stem for path in declaration_paths(root))


def load_idea(root: Path, idea_id: str) -> dict:
    identifier(idea_id, "idea")
    matches = [path for path in declaration_paths(root) if path.stem == idea_id]
    if not matches:
        raise FileNotFoundError(f"no declared candidate: {idea_id}")
    path = matches[0]
    idea = read_json(path)
    validate_idea(idea)
    if idea["id"] != idea_id:
        raise ValueError("idea filename must match id")
    if path.parent.name == "configurations":
        from .configuration_search import validate_configuration_declaration
        validate_configuration_declaration(idea, root=root)
    return idea


def new_idea(root: Path, idea_id: str, parent: str, *, goal="discriminator_stability", hypothesis=None) -> Path:
    identifier(idea_id, "idea")
    path = Path(root) / "configs/forge/ideas" / f"{idea_id}.json"
    if path.exists():
        raise ValueError(f"idea already exists: {path}")
    inherited = load_idea(root, parent)
    idea = {k: deepcopy(inherited[k]) for k in FORMULATION_FIELDS if k in inherited}
    if "host_adaptation" in inherited:
        idea["host_adaptation"] = deepcopy(inherited["host_adaptation"])
    from .decision_contracts import scaffold
    idea.update(schema_version=2, id=idea_id, parent=parent, goal=goal,
                hypothesis=hypothesis or "TODO: state why this mechanism should improve the selected goal",
                changed_factors=["TODO: describe the substantive change before enqueue"],
                mechanism_class=inherited.get("mechanism_class", "structural"),
                mechanism_rationale="TODO: explain structural, constant/floor or sampling-only change",
                api_version="forge-api-v1", lifecycle="proposed", execution_path="public_trainer",
                prior_art=[parent], guide="EXPERIMENTATION.md", decision_contract=scaffold(parent, goal))
    atomic_json(path, idea)
    return path


def resolve_idea(root: Path, idea_id: str, *, view_id: str | None = None,
                 through_tier: int = 1, queue_root: Path | None = None,
                 freeze_source: bool = False, execution_backend: str = "cuda",
                 cuda_model: str | None = None, declaration: dict | None = None) -> dict:
    """Tier placement never enters candidate/task compatibility keys."""
    root = Path(root).resolve()
    idea = load_idea(root, idea_id) if declaration is None else deepcopy(declaration)
    validate_idea(idea)
    if idea["id"] != idea_id:
        raise ValueError("declaration id must match requested candidate")
    if declaration is not None and "configuration_id" in idea:
        from .configuration_search import validate_configuration_declaration
        validate_configuration_declaration(idea, root=root)
    if through_tier not in (1, 2, 3):
        raise ValueError("through-tier must be 1, 2, or 3")
    defaults = read_json(root / "configs/forge/defaults.json")
    view = load_view(root, view_id or idea["goal"])
    all_tasks = load_tasks(root)
    validate_view(view, all_tasks)
    tasks = {a["task"]: deepcopy(all_tasks[a["task"]]) for a in view["assignments"]}
    protocol_id = defaults["protocol"]
    protocol = read_json(root / "configs/forge/protocols" / f"{protocol_id}.json")
    prior = {**defaults["prior"], **idea.get("prior", {})}
    blockers = []
    if "TODO" in idea["hypothesis"] or any("TODO" in x for x in idea["changed_factors"]):
        blockers.append("finish the scaffold's hypothesis and changed_factors before enqueue")
    if idea.get("extensions") and not idea.get("api_changes"):
        blockers.append("extensions require api_changes describing variables, provider, affected hosts and migration")
    if idea.get("implementation"):
        blockers.append("custom implementation loaders are unsupported; implement reusable changes in the public package and declare their Recipe/API bindings")
    from .api import CapabilityError, FormulationContext, task_formulation_context
    from .initialization import task_initializer
    try:
        context = FormulationContext(recipe_preset=idea.get("recipe_preset"),
            recipe_overrides=idea.get("recipe_overrides", {}), prior=prior,
            seed=protocol["seed"], requires_capabilities=idea.get("requires_capabilities", []),
            extensions=idea.get("extensions", {}), initializer=idea.get("initializer", "deterministic_orthogonal"),
            execution_path=idea.get("execution_path", "public_trainer"))
        recipe = asdict(context.recipe)
        if ("configuration_id" in idea
                and stable_hash(recipe) != stable_hash(idea.get("resolved_configuration_recipe"))):
            raise ValueError("configuration frozen Recipe differs from current public defaults or API bindings; declare a new configuration")
        capabilities = [name for name, enabled in context.capabilities().items() if enabled]
        rng = context.streams.manifest()
    except CapabilityError as exc:
        recipe, capabilities, rng = idea.get("recipe_overrides", {}), [], protocol["rng"]
        blockers.extend(exc.blockers)
    candidate = {**idea, "resolved_recipe": recipe, "prior": prior, "capabilities": capabilities,
                 "claim_contract": idea.get("claim_contract", {"schedule": "scheduled", "scoring_weights": "live"})}
    blockers.extend(candidate_blockers(candidate))
    for name, value in view.get("eligibility", {}).get("claim_contract", {}).items():
        if candidate["claim_contract"].get(name) != value:
            blockers.append(f"view requires claim_contract.{name}={value!r}")
    for name in view.get("eligibility", {}).get("requires_capabilities", []):
        if name not in capabilities:
            blockers.append(f"view requires capability {name}")
    # Referenced evaluator source outside normal code roots must travel with a job.
    extra_sources = set(idea.get("source_files", []))
    # A view selects evidence, not a different candidate implementation. Capture
    # the same catalog evaluator support set for every view so quality/stability
    # requests can share their identical task receipts.
    for task in all_tasks.values():
        extra_sources.update(relative for relative in task["evaluation"].get("sources", {})
                             if (root / relative).is_file())
        # Published architecture declarations can live outside the code roots.
        # Capture the catalog's support set regardless of selected view. The
        # selected task's preflight validates every hash and reports missing or
        # malformed declarations as BLOCKED, before any worker is allocated.
        from .hostprofiles import profile_source_paths
        try:
            profile_sources = profile_source_paths(task)
        except (KeyError, TypeError, ValueError):
            profile_sources = {}
        extra_sources.update(relative for relative in profile_sources
                             if (root / relative).is_file())
    for task in tasks.values():
        for relative, digest in task["evaluation"].get("sources", {}).items():
            extra_sources.add(relative)
            from .contracts import file_hash
            if file_hash(root / relative) != digest:
                blockers.append(f"{task['id']}: evaluator source changed; revise the task definition: {relative}")
        # The task owns its sampling law. Candidate priors describe the reference
        # formulation and cannot replace even a task's MoG width or code path.
        try:
            task_context = task_formulation_context(candidate, task, protocol, root=root)
            task["field_ownership"] = task_context.receipt()["field_ownership"]
            available = {name for name, enabled in task_context.capabilities().items() if enabled}
            missing = set(task["requires_capabilities"]) - available
            task["preflight_blockers"] = [f"missing capability {cap}" for cap in sorted(missing)]
        except ValueError as error:
            task["preflight_blockers"] = getattr(error, "blockers", [str(error)])
        from .adapters import adapter_preflight
        task["preflight_blockers"].extend(adapter_preflight(task, candidate, root=root))
        task["preflight_blockers"].extend(task_blockers(task))
        if task["adapter"] == "native100_continuation":
            from .nativeprofiles import validate_native_continuation
            try:
                validate_native_continuation(tasks[task["execution"]["continuation_of"]], task, root=root)
            except (KeyError, TypeError, ValueError, OSError) as error:
                task["preflight_blockers"].append(f"{task['id']}: {error}")
    source = inspect_source(root, sorted(extra_sources))
    candidate_revision = candidate_revision_for(source["digest"], candidate)
    runtime = runtime_manifest()
    groups = {}
    for task_id, task in tasks.items():
        group = task["execution"].get("execution_group", task_id)
        groups.setdefault(group, []).append(task_id)
    jobs = []
    compute_profiles = {}
    for group, members in groups.items():
        # Stable member order must not depend on view order/retiering. The
        # execution producer precedes its checkpoint-derived evaluation tasks.
        member_set = set(members)
        members.sort(key=lambda x: (sum(d["task"] in member_set for d in tasks[x].get("dependencies", [])), x))
        task = tasks[members[0]]
        backend = "cpu" if task["resources"].get("gpus") == 0 else execution_backend
        if backend not in compute_profiles:
            compute_profiles[backend] = compute_profile(backend, cuda_model)
        science = {"candidate_revision": candidate_revision,
                   "execution": {m: task_execution_fingerprint(tasks[m]) for m in members},
                   "evaluation": {m: task_evaluation_fingerprint(tasks[m]) for m in members},
                   "protocol": protocol, "seed": protocol["seed"], "rng": rng,
                   "initializer": idea.get("initializer", "deterministic_orthogonal"), "runtime": runtime,
                   "task_initializers": {m: task_initializer(tasks[m]) for m in members},
                   "compute": {**compute_profiles[backend], "threads": task["resources"]["cpu_threads"]}}
        if view.get("evidence_scope") == "research_diagnostic":
            # Diagnostic measurements cannot alias an ordinary qualification job.
            science["evidence_use"] = "research_diagnostic"
        prerequisites = set()
        def collect_dependencies(name):
            for dependency in tasks[name].get("dependencies", []):
                required = dependency["task"] if isinstance(dependency, dict) else dependency
                if required not in member_set and required not in prerequisites:
                    prerequisites.add(required)
                    collect_dependencies(required)
        for member in members:
            collect_dependencies(member)
        if prerequisites:
            science["prerequisites"] = {name: {
                "execution": task_execution_fingerprint(tasks[name]),
                "evaluation": task_evaluation_fingerprint(tasks[name]),
            } for name in sorted(prerequisites)}
        resources = task["resources"]
        jobs.append({"task_id": members[0], "task_ids": members,
                     "execution_group": group, "compatibility_key": stable_hash(science),
                     "science": science, "budget_seconds": max(tasks[m]["resources"]["timeout_seconds"] for m in members),
                     "resources": {"memory_mb": resources["gpu_memory_mb"], "gpus": resources["gpus"],
                                   **({"host_memory_mb": resources["host_memory_mb"]} if "host_memory_mb" in resources else {}),
                                   "cpu_threads": resources["cpu_threads"], "allow_cpu": backend == "cpu",
                                   "backend": backend, "gpu_model": compute_profiles[backend].get("model") if backend == "cuda" else None}})
    rekey_jobs(jobs)
    request = {"schema_version": 1, "candidate": candidate, "candidate_revision": candidate_revision,
               "requires_independent_grading": True,
               "source": source, "runtime": runtime, "protocol": protocol, "rng": rng,
               "execution_backend": execution_backend, "compute_profiles": compute_profiles,
               "view": view, "policy_fingerprint": view_fingerprint(view), "tasks": tasks,
               "jobs": jobs, "through_tier": through_tier, "preflight_blockers": blockers}
    if "decision_contract" in candidate:
        from .decision_contracts import inspect_contract
        review = inspect_contract(root, request)
        request["decision_review"] = review
        request["decision_admission"] = review["receipt"]
        blockers.extend(review["blockers"])
    elif "configuration_id" not in candidate and (root / "configs/forge/legacy-ideas-v1.json").is_file():
        from .decision_contracts import validate_legacy_admission
        try:
            validate_legacy_admission(request, root=root)
        except ValueError as error:
            blockers.append(str(error))
    if freeze_source:
        if blockers:
            raise ValueError("submission blocked: " + "; ".join(blockers))
        if queue_root is None:
            raise ValueError("queue_root is required to freeze source")
        source["snapshot_path"] = str(snapshot_source(root, queue_root, source))
    return request


def plan_summary(request: dict, queue_state: dict | None = None, *, include_ownership=False) -> dict:
    existing = (queue_state or {}).get("jobs", {})
    by_task = {m: job for job in request["jobs"] for m in job.get("task_ids", [job["task_id"]])}
    tasks = []
    seen_groups = set()
    total = 0
    for assignment in sorted(request["view"]["assignments"], key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"])):
        task_id = assignment["task"]
        job = by_task[task_id]
        saved = existing.get(job["compatibility_key"])
        reusable = bool(saved and saved.get("status") == "terminal")
        allowed = assignment["qualification_tier"] <= request["through_tier"]
        if allowed and not reusable and job["compatibility_key"] not in seen_groups:
            total += job["budget_seconds"]
            seen_groups.add(job["compatibility_key"])
        tasks.append({**assignment, "reusable": reusable, "shared_pending": bool(saved and not reusable),
                      "permitted_by_tier_cap": allowed, "budget_seconds": job["budget_seconds"],
                      "execution_group": job["execution_group"],
                      "blockers": request["tasks"][task_id].get("preflight_blockers", []),
                      **({"field_ownership": request["tasks"][task_id].get("field_ownership")}
                         if include_ownership else {})})
    return {"candidate": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
            "view": request["view"]["id"], "policy_fingerprint": request["policy_fingerprint"],
            "through_tier": request["through_tier"], "tasks": tasks, "worst_case_seconds": total,
            **({"decision_contract": request["decision_review"]} if "decision_review" in request else {}),
            "preflight_blockers": request["preflight_blockers"], "guide": "EXPERIMENTATION.md"}
