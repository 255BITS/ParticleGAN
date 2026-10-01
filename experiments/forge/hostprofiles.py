"""Revalidate explicit host semantics at frozen-request execution boundaries.

Only snapshots containing this module opt into this boundary. Older frozen
registrations retain their original execution rules. Cached preflight results
are never an authority for current host architecture, prior or resource locks.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import canonical, stable_hash

MODULE = "experiments/forge/hostprofiles.py"
PROFILE_ADAPTERS = {"image_profile": {"transfer_image"},
                    "vector_profile": {"transfer_vector"},
                    "native_profile": {"native100", "native100_continuation"}}
SUPPORTED_ADAPTERS = frozenset().union(*PROFILE_ADAPTERS.values())



def profile_source_paths(task: dict) -> list[str]:
    """One catalog support-file inventory; resolvers own exact source pins."""
    execution = task.get("execution", {})
    paths = set()
    if "image_profile" in execution:
        from .imageprofiles import profile_declaration
        if canonical(execution["image_profile"]) != canonical(profile_declaration()):
            raise ValueError("unsupported image profile source declaration")
        paths.add(profile_declaration()["source"]["path"])
    if "vector_profile" in execution:
        from .vectorprofiles import profile_source_files as vector_sources
        paths.update(vector_sources(task))
    if "native_profile" in execution:
        from .nativeprofiles import native_profile_source_files
        paths.update(native_profile_source_files(task))
    return sorted(paths)


def _members(request, task_ids):
    if task_ids is not None:
        return set(task_ids)
    authorized = {row["task"] for row in request["view"]["assignments"]
                  if row["qualification_tier"] <= request["through_tier"]}
    members = set(authorized)
    for job in request["jobs"]:
        group = set(job.get("task_ids", [job["task_id"]]))
        if group & authorized:
            members.update(group)
    return members


def _validate_task(task, candidate, root):
    execution, adapter = task["execution"], task["adapter"]
    for marker, adapters in PROFILE_ADAPTERS.items():
        if marker in execution and adapter not in adapters:
            raise ValueError(f"{marker} is unsupported by adapter {adapter}")
    if adapter not in SUPPORTED_ADAPTERS:
        return False
    if "host_initialization" in execution:
        raise ValueError("host initialization must be declared inside a supported native profile")
    if adapter.startswith("native100"):
        from .nativeprofiles import native_profile_blockers
        blockers = native_profile_blockers(task, candidate, root=root)
        if blockers:
            raise ValueError("; ".join(blockers))
        # Legacy native tasks still use their old resource/init path. Their
        # generic construction validation belongs to the ordinary adapter.
        if "native_profile" not in execution:
            return False
        from .nativeprofiles import resolve_native_spec
        resources = resolve_native_spec(task, root=root)["resources"]
    else:
        if adapter == "transfer_image":
            from .imageprofiles import resolve_image_spec
            spec = resolve_image_spec(task, root=root)
        else:
            from .vectorprofiles import resolve_vector_spec
            spec = resolve_vector_spec(task, root=root)
        if execution["steps"] != spec["steps"]:
            raise ValueError("execution budget differs from the explicit host card")
        resources = {"num_particles": spec["particles"], "z_dim": spec["z_dim"],
                     "batch_size": spec["batch_size"] if adapter == "transfer_image" else spec["batch"]}
    prior = execution.get("prior")
    if not isinstance(prior, dict) or not isinstance(candidate.get("prior"), dict):
        raise ValueError("host request needs an explicit resolved task and candidate prior")
    # Only a task's declared finite-cloud exception may override the candidate's
    # sampler. Other tasks must execute the resolved candidate prior literally.
    if prior.get("kind") != "particle_cloud" and canonical(prior) != canonical(candidate["prior"]):
        raise ValueError("task prior differs from the resolved candidate prior")
    from .api import host_recipe_overrides, task_policy_blockers
    blockers = task_policy_blockers(task, candidate)
    if blockers:
        raise ValueError("; ".join(blockers))
    overrides = host_recipe_overrides(candidate, execution, resources)
    # Shared public validation without model construction or GPU allocation.
    # Native policy syntax/applicability was already resolved by its helper.
    from .api import FormulationContext
    FormulationContext(recipe_preset=candidate.get("recipe_preset"), recipe_overrides=overrides, prior=prior,
        device="cpu", requires_capabilities=tuple(candidate.get("requires_capabilities", ()))
            + tuple(task["requires_capabilities"]), extensions=candidate.get("extensions", {}),
        initializer=candidate.get("initializer", "deterministic_orthogonal"))
    return True



def _validate_candidate_identity(request):
    from dataclasses import asdict
    from .api import FormulationContext
    from .planning import candidate_revision_for

    candidate = request["candidate"]
    # Rebuild from declarations, not candidate-provided resolved_recipe echoes.
    # Recipe resolution has no model construction or training draw and does not
    # depend on promotion seed; the registered RNG protocol is checked separately.
    context = FormulationContext(recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}),
        prior=candidate.get("prior"), device="cpu",
        requires_capabilities=candidate.get("requires_capabilities", ()),
        extensions=candidate.get("extensions", {}),
        initializer=candidate.get("initializer", "deterministic_orthogonal"),
        execution_path=candidate.get("execution_path", "public_trainer"))
    resolved_recipe = asdict(context.recipe)
    if canonical(candidate.get("resolved_recipe")) != canonical(resolved_recipe):
        raise ValueError("candidate resolved_recipe differs from its actual public formulation")
    expected = candidate_revision_for(request["source"]["digest"],
                                     {**candidate, "resolved_recipe": resolved_recipe})
    if request.get("candidate_revision") != expected:
        raise ValueError("candidate scientific identity differs from its actual formulation/source")


def _validate_job_identity(request, checked):
    from .views import task_execution_fingerprint, task_evaluation_fingerprint
    covered = set()
    for job in request["jobs"]:
        members = set(job.get("task_ids", [job["task_id"]]))
        relevant = members & checked
        if not relevant:
            continue
        if covered & relevant:
            raise ValueError("host task appears in multiple execution jobs")
        covered.update(relevant)
        science = job["science"]
        leader_resources = request["tasks"][job["task_id"]]["resources"]
        locked_resources = {"memory_mb": leader_resources["gpu_memory_mb"],
                            "gpus": leader_resources["gpus"], "cpu_threads": leader_resources["cpu_threads"]}
        if "host_memory_mb" in leader_resources:
            locked_resources["host_memory_mb"] = leader_resources["host_memory_mb"]
        if (any(job["resources"].get(key) != value for key, value in locked_resources.items())
                or job["budget_seconds"] != max(request["tasks"][name]["resources"]["timeout_seconds"] for name in members)
                or science.get("compute", {}).get("threads") != locked_resources["cpu_threads"]):
            raise ValueError("host job resources differ from its frozen task/compute budget")
        for member in relevant:
            task = request["tasks"][member]
            if task.get("adapter") == "native100_continuation" and "native_profile" in task["execution"]:
                parent_id = task["execution"]["continuation_of"]
                parents = [row for row in request["jobs"] if parent_id in row.get("task_ids", [row["task_id"]])]
                if (len(parents) != 1 or science.get("prerequisites", {}).get(parent_id)
                        != parents[0]["compatibility_key"]):
                    raise ValueError("native continuation scientific identity lacks its exact own-parent job")
        expected_execution = {name: task_execution_fingerprint(request["tasks"][name]) for name in members}
        expected_evaluation = {name: task_evaluation_fingerprint(request["tasks"][name]) for name in members}
        if (science.get("execution") != expected_execution or science.get("evaluation") != expected_evaluation
                or science.get("candidate_revision") != request["candidate_revision"]
                or science.get("initializer") != request["candidate"].get("initializer", "deterministic_orthogonal")
                or job["compatibility_key"] != stable_hash(science)):
            raise ValueError("host job scientific identity differs from its resolved task/formulation")
    if covered != checked:
        raise ValueError("authorized host task lacks its execution job")


def validate_request_host_profiles(request: dict, *, task_ids=None) -> None:
    """Check all authorized grouped members against the actual frozen root.

    Call before queue mutation/reservation, and again immediately before dispatch.
    Valid registered diagnostics and promotions retain their own namespaced job
    keys; those identities still bind the same measured host and effective prior.
    """
    source = request.get("source", {})
    snapshot = source.get("snapshot_path")
    present = isinstance(snapshot, str) and bool(snapshot) and (Path(snapshot) / MODULE).is_file()
    if MODULE not in source.get("files", {}) and not present:
        return
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("host profile validation requires a frozen source snapshot")
    from .sources import verify_snapshot
    root = Path(snapshot)
    verify_snapshot(root, source)
    blockers, checked = [], set()
    tasks = request.get("tasks", {})
    for member in sorted(_members(request, task_ids)):
        if member not in tasks:
            blockers.append(f"{member}: authorized task lacks its frozen definition")
            continue
        task = tasks[member]
        try:
            if _validate_task(task, request["candidate"], root):
                checked.add(member)
            if task.get("adapter") == "native100_continuation" and "native_profile" in task["execution"]:
                from .nativeprofiles import validate_native_continuation
                parent_id = task["execution"]["continuation_of"]
                if parent_id not in tasks:
                    raise ValueError("native continuation lacks its declared parent task")
                parent = tasks[parent_id]
                _validate_task(parent, request["candidate"], root)
                checked.add(parent_id)
                validate_native_continuation(parent, task, root=root)
        except (KeyError, TypeError, ValueError, OSError) as error:
            blockers.append(f"{member}: {error}")
    if not blockers:
        try:
            _validate_candidate_identity(request)
            _validate_job_identity(request, checked)
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(str(error))
    if blockers:
        raise ValueError("host profile blocked: " + "; ".join(blockers))
