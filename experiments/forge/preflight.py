"""Task-local readiness shared by planning and frozen-request admission."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from .contracts import file_hash

MODULE = "experiments/forge/preflight.py"


def task_preflight(task: dict, candidate: dict, protocol: dict, *, root: Path, tasks: dict) -> list[str]:
    from .api import task_formulation_context
    from .adapters import adapter_preflight
    from .sampling import task_blockers

    blockers = []
    for relative, digest in task["evaluation"].get("sources", {}).items():
        try:
            observed = file_hash(root / relative)
        except OSError:
            blockers.append(f"{task['id']}: evaluator source unavailable: {relative}")
            continue
        if observed != digest:
            blockers.append(f"{task['id']}: evaluator source changed; revise the task definition: {relative}")
    try:
        context = task_formulation_context(candidate, task, protocol, root=root)
        task["field_ownership"] = context.receipt()["field_ownership"]
        available = {name for name, enabled in context.capabilities().items() if enabled}
        blockers.extend(f"missing capability {cap}"
                        for cap in sorted(set(task["requires_capabilities"]) - available))
    except ValueError as error:
        blockers.extend(getattr(error, "blockers", [str(error)]))
    blockers.extend(adapter_preflight(task, candidate, root=root))
    blockers.extend(task_blockers(task))
    if task["adapter"] == "native100_continuation":
        from .nativeprofiles import validate_native_continuation
        try:
            validate_native_continuation(tasks[task["execution"]["continuation_of"]], task, root=root)
        except (KeyError, TypeError, ValueError, OSError) as error:
            blockers.append(f"{task['id']}: {error}")
    return list(dict.fromkeys(blockers))


def recheck_request(request: dict) -> None:
    """Recompute cached blockers against verified source, without executing models.

    Minimal fixtures and older sources lacking this boundary retain their
    original cached checks; newly planned real snapshots contain the helper.
    """
    source = request.get("source", {})
    snapshot = source.get("snapshot_path")
    present = isinstance(snapshot, str) and (Path(snapshot) / MODULE).is_file()
    if MODULE not in source.get("files", {}) and not present:
        return
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("task preflight requires a frozen source snapshot")
    from .sources import verify_snapshot
    root = Path(snapshot)
    verify_snapshot(root, source)
    if (request.get("candidate", {}).get("id") != "atlas-noisy025-tier1-717-v1"
            or request.get("view", {}).get("id") != "atlas_noisy025_tier1_717_v1"):
        for task in request["tasks"].values():
            task["preflight_blockers"] = task_preflight(task, request["candidate"], request["protocol"],
                                                      root=root, tasks=request["tasks"])
        return
    from .atlas_noisy025_tier1 import validate_request_scope
    validate_request_scope(request)
    checked = deepcopy(request["tasks"])
    for task in checked.values():
        cached = task.pop("preflight_blockers", [])
        if not isinstance(cached, list) or any(not isinstance(reason, str) for reason in cached):
            raise ValueError("malformed cached task preflight blockers")
        if "field_ownership" in task and not isinstance(task.pop("field_ownership"), dict):
            raise ValueError("malformed cached task field ownership")
    for task in checked.values():
        task["preflight_blockers"] = task_preflight(task, request["candidate"], request["protocol"],
                                                  root=root, tasks=checked)
    # Only regenerated administrative annotations are committed. A source or
    # namespace failure remains a blocker; no scientific field is replaced.
    for name, task in checked.items():
        request["tasks"][name]["preflight_blockers"] = task["preflight_blockers"]
        request["tasks"][name].pop("field_ownership", None)
        if "field_ownership" in task:
            request["tasks"][name]["field_ownership"] = task["field_ownership"]
