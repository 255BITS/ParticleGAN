"""Explicit intake for remaining declared cells, preserving original sources.

RUN_DIRECTORY/remaining-sources.json selects archived policy requests and
unbound current task declarations. This never ports a formulation or retries
completed work. Source differences and checkpoint requirements remain blockers.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

from .contracts import atomic_json, file_hash, read_json, stable_hash
from .views import task_execution_fingerprint, task_evaluation_fingerprint


def policy_scope(family, source, revision):
    return stable_hash([family, source, revision, "selected-cloud-after-investigation"])


def frozen_source(root, manifest, directory):
    from .sources import verify_snapshot
    target = directory / "remaining-source" / manifest["digest"]
    if target.exists():
        verify_snapshot(target, manifest)
        return target
    if stable_hash(manifest["files"]) != manifest["digest"]:
        raise ValueError("invalid original source manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "source"
        stage.mkdir()
        for relative, digest in manifest["files"].items():
            member = stage / relative
            if not member.resolve().is_relative_to(stage.resolve()):
                raise ValueError("source member escapes snapshot")
            original = root / relative
            data = original.read_bytes() if original.is_file() else b""
            if hashlib.sha256(data).hexdigest() != digest:
                data = subprocess.check_output(["git", "show", manifest["origin_commit"] + ":" + relative], cwd=root)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("original source bytes unavailable: " + relative)
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_bytes(data)
        atomic_json(stage / "forge-source.json", manifest)
        verify_snapshot(stage, manifest)
        stage.rename(target)
    return target


def _freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("remaining attempt request changed: " + str(path))
    else:
        atomic_json(path, value)


def _job(path, request, card, family, **extra):
    from .attempt_coverage import identity
    execution, evaluation = task_execution_fingerprint(card), task_evaluation_fingerprint(card)
    key = identity(request["source"]["digest"], request["candidate_revision"], card["id"], execution, evaluation)
    return key, dict(key=key, family=family, candidate_id=request["candidate"]["id"],
        candidate_revision=request["candidate_revision"], source_digest=request["source"]["digest"],
        task=card["id"], execution_sha256=execution, evaluation_sha256=evaluation,
        request_path=str(path), request_sha256=file_hash(path), source_root=request["source"]["snapshot_path"],
        timeout_seconds=card["resources"]["timeout_seconds"], resources=card["resources"],
        dependencies=card.get("dependencies", []), status="PENDING", diagnostic=True, **extra)


def prepare(root, directory):
    config = directory / "remaining-sources.json"
    if not config.is_file():
        return {}
    from .attempt_coverage import WORKER, selected_tasks
    jobs = {}
    settings = read_json(config)
    for entry in settings.get("policy_requests", []):
        original = Path(entry["request"])
        if file_hash(original) != entry["sha256"]:
            raise ValueError("original policy request bytes changed")
        request = read_json(original)
        if request["protocol"]["seed"] != 0:
            raise ValueError("remaining attempts exclude seed-only studies")
        source = frozen_source(Path(entry["source_root"]), request["source"], directory)
        scope = policy_scope(entry["family"], request["source"]["digest"], request["candidate_revision"])
        path = directory / "remaining-requests" / (scope + ".json")
        if not path.is_file():
            spec = path.with_suffix(".prepare.json")
            atomic_json(spec, dict(source_root=str(source), original_request=str(original),
                request_sha256=entry["sha256"], parents=entry["parents"], output=str(path)))
            subprocess.run([sys.executable, str(WORKER), "prepare_policy", str(spec)], check=True, timeout=120)
        request = read_json(path)
        if (request["source"]["digest"], request["candidate_revision"]) != (
                read_json(original)["source"]["digest"], read_json(original)["candidate_revision"]):
            raise ValueError("remaining policy source or recipe changed")
        for parent in entry["parents"]:
            card = next(t for t in request["tasks"].values() if t.get("policy_parent", {}).get("id") == parent)
            key, job = _job(path, request, card, "policy-" + entry["family"],
                policy_scope=scope, parent_task=parent, label=entry["family"] + " selected-cloud after investigation")
            jobs[key] = job
    selected = {family: row for family, row, tasks in selected_tasks(read_json(root / "reports/forge/technique-inventory.json"))}
    for entry in settings.get("unbound_current_tasks", []):
        family, row = entry["family"], selected[entry["family"]]
        original = read_json(directory / "requests" / (family + ".json"))
        if (original["source"]["digest"], original["candidate_revision"]) != (
                row["bindings"]["source_digest"], row["candidate_revision"]):
            raise ValueError("remaining tasks differ from selected source/recipe")
        for name in entry["tasks"]:
            if name in row["bindings"]["task_contracts"]:
                raise ValueError("cannot replace a recorded task binding")
            card = read_json(root / "configs/forge/tasks" / (name + ".json"))
            request = deepcopy(original)
            request["tasks"] = {name: card}
            path = directory / "remaining-requests" / (family + "-" + name + "-" + stable_hash(card)[:16] + ".json")
            _freeze(path, request)
            key, job = _job(path, request, card, family, label=family + " current unbound task")
            jobs[key] = job
    return jobs
