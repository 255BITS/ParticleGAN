"""Explicit named reforms retain the main view's complete required roster.

Only the selected family's declared questions are replaced. Other questions
remain present and must earn their own valid evidence; a subset is never a
fully qualified family. Metadata resolution constructs no model and reserves
no compute.
"""
from copy import deepcopy
import json
from pathlib import Path

from .contracts import stable_hash
from . import policy_contracts as independent


NAMED_PARENTS = {
    "conditional_policy_selected_cloud_v1": (
        "trajectory", "residual_student", "unipolar", "mid_scale_identity"),
    "routed_policy_selected_cloud_v1": ("unused_token_hold",),
    "multibank_policy_v1": ("cover_leftover",),
    "ae_routed_policy_v1": ("ae_gan_hold",),
    "word_joint_policy_min11_v1": ("five_word_joint_acquisition",),
}


def project_view(view, selected, cohort, family):
    """Pure projection after caller validation; usable with frozen JSON bytes."""
    mapping = {parent: f"{parent}_{cohort}" for parent in NAMED_PARENTS[cohort]}
    resolved = deepcopy(view)
    resolved.update(revision=independent.PARENT_REVISION + 1, task_cohort=cohort,
                    policy_family=family,
                    policy_adapted_parent_ids=list(NAMED_PARENTS[cohort]),
                    parent_view_fingerprint=stable_hash(view), cohort_fingerprint=stable_hash(selected),
                    policy_scope="Named conditional/encoder family reforms; all 26 original required slots remain. "
                        "Unadapted questions need their own evidence. CPU controls, other families and historical "
                        "results confer no learned or whole-family qualification.")
    for assignment in resolved["assignments"]:
        assignment["task"] = mapping.get(assignment["task"], assignment["task"])
    return resolved


def load_task_variants(root, parent_tasks, cohort):
    """Read exactly a known cohort, checking original and implementation bytes."""
    if cohort == independent.COHORT:
        return independent.load_policy_variants(root, parent_tasks)
    if not isinstance(cohort, str) or cohort not in NAMED_PARENTS:
        raise ValueError(f"unknown explicit policy task_cohort {cohort!r}")
    from .policy_cohorts import validate_policy_task
    root = Path(root).resolve()
    expected = {f"{parent}_{cohort}.json" for parent in NAMED_PARENTS[cohort]}
    directory = root / "configs/forge/task-variants" / cohort
    paths = {p.name: p for p in directory.glob("*.json")}
    if set(paths) != expected:
        raise ValueError("named policy cohort needs exactly its declared variant roster")
    variants = {}
    for name in sorted(paths):
        path = paths[name]
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("named policy variant must be a source-owned regular file")
        task = json.loads(path.read_bytes())
        validate_policy_task(task, root=root, allow_compiler_annotations=False)
        parent = task["policy_parent"]["id"]
        if (parent not in NAMED_PARENTS[cohort] or parent not in parent_tasks
                or task["task_cohort"] != cohort or task["id"] + ".json" != name
                or task["id"] in variants):
            raise ValueError("named policy filename, parent or cohort identity differs")
        actual = json.loads((root / "configs/forge/tasks" / f"{parent}.json").read_bytes())
        if stable_hash(actual) != stable_hash(parent_tasks[parent]):
            raise ValueError("supplied named policy parent differs from its bound source")
        variants[task["id"]] = task
    return variants


def resolve_task_view(view, tasks, candidate):
    """Keep all 26 required slots; no historical or cross-family substitution."""
    cohort = candidate.get("task_cohort")
    if cohort == independent.COHORT or cohort is None:
        return independent.resolve_policy_view(view, tasks, candidate)
    if not isinstance(cohort, str) or cohort not in NAMED_PARENTS:
        raise ValueError(f"unknown explicit policy task_cohort {cohort!r}")
    from .policy_cohorts import validate_policy_task
    assignments = view.get("assignments", [])
    if (view.get("id") != independent.PARENT_VIEW_ID
            or view.get("goal") != independent.PARENT_VIEW_ID
            or type(view.get("revision")) is not int
            or view["revision"] != independent.PARENT_REVISION
            or [a.get("task") for a in assignments] != list(independent.PARENT_TASK_IDS)
            or [sum(a.get("qualification_tier") == tier for a in assignments)
                for tier in (1, 2, 3)] != [5, 19, 2]
            or any(a.get("importance") != "required"
                   or a.get("qualification_tier") != (1 if i < 5 else 2 if i < 24 else 3)
                   or a.get("order") != i - (0 if i < 5 else 5 if i < 24 else 24)
                   for i, a in enumerate(assignments))):
        raise ValueError("named policy resolves only the unchanged main 5/19/2 view")
    mapping = {parent: f"{parent}_{cohort}" for parent in NAMED_PARENTS[cohort]}
    selected, families = {}, set()
    for parent in independent.PARENT_TASK_IDS:
        name = mapping.get(parent, parent)
        if parent not in tasks or name not in tasks:
            raise ValueError(f"named policy lacks its full parent/variant slot {parent}")
        task = tasks[name]
        if parent in mapping:
            validate_policy_task(task)
            if task["policy_parent"]["id"] != parent:
                raise ValueError("named policy variant belongs to a different required slot")
            families.add(task["policy_family"])
        selected[name] = deepcopy(task)
    if len(families) != 1:
        raise ValueError("named policy view cannot pool distinct families")
    return project_view(view, selected, cohort, next(iter(families))), selected
