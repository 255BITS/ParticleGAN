"""Revalidate experiment ownership at frozen-request execution boundaries.

Only snapshots containing this module opt into this boundary. Older frozen
registrations retain their original execution rules. Cached preflight results
are never an authority for current host architecture, prior or resource locks.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import canonical, stable_hash
from .initialization import MODULE as INITIALIZATION_MODULE, task_initializer
from .priors import task_prior

MODULE = "experiments/forge/hostprofiles.py"
PROFILE_ADAPTERS = {"image_profile": {"transfer_image"},
                    "vector_profile": {"transfer_vector"},
                    "native_profile": {"native100", "native100_continuation"}}
SUPPORTED_ADAPTERS = frozenset().union(*PROFILE_ADAPTERS.values())
BOUNDARY_ADAPTERS = SUPPORTED_ADAPTERS | {"transfer_behavior", "ring_endurance", "clockfree_audit", "paired_adaptation", "word_joint"}



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


def _validate_task(task, candidate, root, *, explicit_initializer=True):
    from .taskrecipes import bind_task_candidate
    initializer = task_initializer(task, candidate, explicit=explicit_initializer)
    reference_candidate = candidate
    candidate = bind_task_candidate(candidate, task)
    execution, adapter = task["execution"], task["adapter"]
    for marker, adapters in PROFILE_ADAPTERS.items():
        if marker in execution and adapter not in adapters:
            raise ValueError(f"{marker} is unsupported by adapter {adapter}")
    if "host_initialization" in execution and (adapter in SUPPORTED_ADAPTERS or explicit_initializer):
        raise ValueError("host initialization must be declared inside a supported native profile")
    if adapter not in SUPPORTED_ADAPTERS:
        if explicit_initializer and adapter in BOUNDARY_ADAPTERS:
            if adapter == "word_joint":
                from .word_adapter import word_context
                word_context({"candidate": reference_candidate, "protocol": {"seed": 0}}, task, "cpu", root=root)
            else:
                from .api import task_formulation_context
                task_formulation_context(reference_candidate, task, device="cpu", root=root)
            return True
        return False
    if adapter.startswith("native100"):
        from .nativeprofiles import native_profile_blockers
        blockers = native_profile_blockers(task, candidate, root=root, explicit_initializer=explicit_initializer)
        if blockers:
            raise ValueError("; ".join(blockers))
        # Legacy native tasks retain their old resource/init path. Current MLP
        # tasks own their resources and participate in the same frozen checks.
        if "native_profile" not in execution:
            if "resources" not in execution:
                return False
            resources = execution["resources"]
        else:
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
    prior = task_prior(task)
    from .api import host_recipe_overrides, task_policy_blockers
    blockers = task_policy_blockers(task, candidate)
    if blockers:
        raise ValueError("; ".join(blockers))
    overrides = host_recipe_overrides(candidate, execution, resources)
    # Shared public validation without model construction or GPU allocation.
    # Native policy syntax/applicability was already resolved by its helper.
    from .api import FormulationContext
    if explicit_initializer:
        from .api import task_formulation_context
        task_formulation_context(reference_candidate, task, device="cpu", root=root)
    else:
        FormulationContext(recipe_preset=candidate.get("recipe_preset"), recipe_overrides=overrides, prior=prior,
            device="cpu", requires_capabilities=tuple(candidate.get("requires_capabilities", ()))
                + tuple(task["requires_capabilities"]), extensions=candidate.get("extensions", {}),
            initializer=initializer)
    return True



def _validate_candidate_identity(request):
    from dataclasses import asdict
    from .api import FormulationContext
    from .planning import candidate_revision_for

    candidate = request["candidate"]
    from .taskrecipes import validate_host_adaptation
    validate_host_adaptation(candidate)
    # Rebuild from declarations, not candidate-provided resolved_recipe echoes.
    # Recipe resolution has no model construction or training draw and does not
    # depend on promotion seed; the registered RNG protocol is checked separately.
    # Carry the already validated exact Track B metadata identity. The public
    # context retains all prior, Recipe, ownership and other-family guards.
    from .atlas_existing_mog import CANDIDATE_ID, VIEW_ID, supports_candidate
    existing_mog_id = (CANDIDATE_ID if supports_candidate(candidate)
                       and request.get("view", {}).get("id") == VIEW_ID else None)
    if candidate.get('id') == 'atlas-existing-mog-radius-observer844-v1' and request.get('view', {}).get('id') == 'atlas_existing_mog_radius_observer844_v1':
        from .atlas844_radius_owner import supports_candidate as supports844
        if supports844(candidate):
            existing_mog_id = candidate['id']
    if candidate.get('id') == 'atlas-existing-mog-longer871-v1' and request.get('view', {}).get('id') == 'atlas_existing_mog_longer871_v1':
        from .atlas871_longer_owner import supports_candidate as supports871
        if supports871(candidate):
            existing_mog_id = candidate['id']
    context = FormulationContext(recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}),
        prior=candidate.get("prior"), device="cpu",
        requires_capabilities=candidate.get("requires_capabilities", ()),
        extensions=candidate.get("extensions", {}),
        initializer=candidate.get("initializer", "deterministic_orthogonal"),
        execution_path=candidate.get("execution_path", "public_trainer"),
        candidate_id=existing_mog_id)
    resolved_recipe = asdict(context.recipe)
    if "configuration_id" in candidate:
        from .configuration_search import recipe_identity_fields, validate_configuration_declaration
        validate_configuration_declaration(candidate)
        # Older immutable cards recorded the one paired-logistic objective
        # implicitly. Normalize only that default for this legacy comparison;
        # the actual request, alternative objectives and source identity stay
        # explicit and continue through the strict checks below.
        if canonical(recipe_identity_fields(candidate["resolved_configuration_recipe"])) != canonical(recipe_identity_fields(resolved_recipe)):
            raise ValueError("configuration frozen Recipe differs from its actual public formulation")
    if canonical(candidate.get("resolved_recipe")) != canonical(resolved_recipe):
        raise ValueError("candidate resolved_recipe differs from its actual public formulation")
    expected = candidate_revision_for(request["source"]["digest"],
                                     {**candidate, "resolved_recipe": resolved_recipe})
    if request.get("candidate_revision") != expected:
        raise ValueError("candidate scientific identity differs from its actual formulation/source")


def _validate_job_identity(request, checked, *, explicit_initializer=True):
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
            if task.get("adapter") == "native100_continuation":
                parent_id = task["execution"]["continuation_of"]
                parents = [row for row in request["jobs"] if parent_id in row.get("task_ids", [row["task_id"]])]
                if (len(parents) != 1 or science.get("prerequisites", {}).get(parent_id)
                        != parents[0]["compatibility_key"]):
                    raise ValueError("native continuation scientific identity lacks its exact own-parent job")
        expected_execution = {name: task_execution_fingerprint(request["tasks"][name]) for name in members}
        expected_evaluation = {name: task_evaluation_fingerprint(request["tasks"][name]) for name in members}
        if explicit_initializer:
            expected_initializers = {name: task_initializer(request["tasks"][name], request["candidate"])
                                     for name in members}
            if science.get("task_initializers") != expected_initializers:
                raise ValueError("host job initialization differs from its experiment-owned task policies")
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
    # The new source contract owns initialization in tasks. Existing frozen
    # requests without the helper retain their original candidate fallback and
    # scientific keys; their old source executes its original binding rules.
    explicit_initializer = INITIALIZATION_MODULE in source.get("files", {})
    blockers, checked = [], set()
    tasks = request.get("tasks", {})
    for member in sorted(_members(request, task_ids)):
        if member not in tasks:
            blockers.append(f"{member}: authorized task lacks its frozen definition")
            continue
        task = tasks[member]
        try:
            if _validate_task(task, request["candidate"], root, explicit_initializer=explicit_initializer):
                checked.add(member)
            if task.get("adapter") == "native100_continuation":
                from .nativeprofiles import validate_native_continuation
                parent_id = task["execution"]["continuation_of"]
                if parent_id not in tasks:
                    raise ValueError("native continuation lacks its declared parent task")
                parent = tasks[parent_id]
                if _validate_task(parent, request["candidate"], root, explicit_initializer=explicit_initializer):
                    checked.add(parent_id)
                validate_native_continuation(parent, task, root=root)
        except (KeyError, TypeError, ValueError, OSError) as error:
            blockers.append(f"{member}: {error}")
    if not blockers:
        try:
            _validate_candidate_identity(request)
            _validate_job_identity(request, checked, explicit_initializer=explicit_initializer)
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(str(error))
    if blockers:
        raise ValueError("host profile blocked: " + "; ".join(blockers))
