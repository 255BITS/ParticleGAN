"""Intake for the scoped policy-cloud coverage cohort only (Atlas and E22).

``attempt_coverage --intake policy-cloud`` uses this module instead of the
selected, research, remaining and checkpoint-repair intakes. Each family's
request is planned once by the ordinary planner from the committed
``policy_cloud_coverage`` view in the frozen execution checkout, then frozen
with its own source snapshot. Records keep the ordinary attempt format plus an
explicit cohort scope, label, owning family, reporting tier and parent task.
Nothing here changes a recipe, task, seed or qualification.
"""
from pathlib import Path
import subprocess
import sys

from .contracts import atomic_json, read_json, stable_hash
from .policy_coverage import (BLOCKED_HOSTS, COHORT, DIAGNOSTIC_CONTINUATIONS, LABEL, PARENTS, VIEW,
                              reporting_tier)
from .remaining_attempts import _job

FAMILIES = ("atlas", "e22")
SETTINGS = "policy-cloud.json"
# Fixed task wall-time limits are part of each question. These short-limit
# tasks run alone on one GPU so co-tenancy cannot spend their budget.
TIMEOUT_SENSITIVE = frozenset({"gaussian1d_smoke", "gaussian1d_stability", "ring16_acquisition"})


def cohort_scope(family, source, revision):
    """One displayed row: one family, frozen source, recipe and cohort."""
    return stable_hash([family, source, revision, COHORT])


def settings(directory, source=None):
    """Persist the intake's families and frozen source on first preparation."""
    path = Path(directory) / SETTINGS
    if path.is_file():
        value = read_json(path)
        if source is not None and Path(value["source_root"]) != Path(source):
            raise ValueError("policy-cloud state belongs to another frozen source checkout")
        return value
    if source is None:
        raise ValueError("policy-cloud intake needs --source-root on first preparation")
    value = {"schema_version": 1, "cohort": COHORT, "view": VIEW, "families": list(FAMILIES),
             "source_root": str(Path(source).resolve()), "label": LABEL, "timeout_sensitive_gpu": "1"}
    atomic_json(path, value)
    return value


def _request(directory, value, family):
    from .attempt_coverage import WORKER
    path = Path(directory) / "requests" / ("policy-cloud-" + family + ".json")
    if not path.is_file():
        spec = Path(directory) / "requests" / ("policy-cloud-" + family + "-prepare.json")
        atomic_json(spec, {"source_root": value["source_root"], "requests": [
            {"candidate_id": family, "view_id": VIEW, "tasks": [], "output": str(path)}]})
        subprocess.run([sys.executable, str(WORKER), "prepare", str(spec)], check=True,
                       stdout=subprocess.DEVNULL, timeout=900)
    request = read_json(path)
    if (request["view"]["id"] != VIEW or request["candidate"]["id"] != family
            or request["protocol"]["seed"] != 0 or not request["source"].get("snapshot_path")):
        raise ValueError(f"{family}: frozen policy-cloud request has another view, family, seed or no snapshot")
    return path, request


def prepare(root, source, directory):
    """Return only this cohort's jobs; selected/research/repair work is never planned."""
    value = settings(directory, source)
    if value["cohort"] != COHORT or value["view"] != VIEW:
        raise ValueError("policy-cloud settings name another cohort")
    jobs = {}
    for family in value["families"]:
        if family not in FAMILIES:
            raise ValueError("policy-cloud intake covers only the selected Atlas and E22 recipes")
        path, request = _request(directory, value, family)
        scope = cohort_scope(family, request["source"]["digest"], request["candidate_revision"])
        assigned = {row["task"]: row for row in request["view"]["assignments"]}
        parents = {}
        for name, assignment in assigned.items():
            card = request["tasks"][name]
            if card.get("task_cohort") != COHORT:
                raise ValueError(f"{name}: policy-cloud view assigns a task outside its cohort")
            parent = card["policy_parent"]["id"]
            if assignment["qualification_tier"] != reporting_tier(parent):
                raise ValueError(f"{name}: tier differs from the selected-view tier")
            parents[parent] = name
            extra = dict(cohort_scope=scope, label=value["label"], owning_family=family,
                         reporting_tier=assignment["qualification_tier"], parent_task=parent,
                         diagnostic_continuation=parent in DIAGNOSTIC_CONTINUATIONS)
            if parent in TIMEOUT_SENSITIVE:
                extra.update(exclusive_gpu=True, required_gpu=value["timeout_sensitive_gpu"])
            key, job = _job(path, request, card, "policy-cloud-" + family, **extra)
            if parent in BLOCKED_HOSTS and job["dependencies"]:
                # The test's own ownership boundary refuses it before training;
                # record that reason instead of waiting on a blocked producer.
                job["dependency_note"] = ("Explicit ownership blocker precedes the declared dependencies: " +
                                          ", ".join(dep["task"] for dep in job["dependencies"]))
                job["dependencies"] = []
            jobs[key] = job
        if set(parents) != set(PARENTS):
            raise ValueError(f"{family}: policy-cloud request must assign all {len(PARENTS)} selected tests")
    return jobs


def plan_summary(state):
    """Compact read-only listing used to prove which jobs would execute."""
    rows = sorted(state["jobs"].values(), key=lambda j: (j.get("owning_family", ""), j.get("reporting_tier", 0), j["task"]))
    return [{"family": job["family"], "owning_family": job.get("owning_family"), "tier": job.get("reporting_tier"),
             "parent_task": job.get("parent_task"), "task": job["task"], "status": job["status"],
             "blocked_host": job.get("parent_task") in BLOCKED_HOSTS,
             "diagnostic_continuation": job.get("diagnostic_continuation", False),
             "dependencies": [dep["task"] for dep in job.get("dependencies", [])],
             "timeout_seconds": job["timeout_seconds"], "required_gpu": job.get("required_gpu"),
             "exclusive_gpu": job.get("exclusive_gpu", False)} for job in rows]
