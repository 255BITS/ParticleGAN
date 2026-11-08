"""Intake for the scoped policy-cloud coverage cohorts only (Atlas and E22).

``attempt_coverage --intake policy-cloud`` uses this module instead of the
selected, research, remaining and checkpoint-repair intakes. It schedules two
scoring laws for each family, each planned once by the ordinary planner from
its committed view in the frozen execution checkout, then frozen with its own
source snapshot:

* clean (``policy_cloud_coverage``): all 35 selected-view tests, measured
  without the learned output noise;
* as served (``policy_cloud_coverage_as_served``): the 23 tests that measure
  output samples, identical except that the measured samples keep the learned
  output noise exactly as GANTrainer serves it.

Law-independent tests (no measured output samples, or an ownership blocker)
run once in the clean request; their records name the as-served scope too, so
publication shows one verdict in both rows. Checkpoint dependents consume
their own same-law producer. Records keep the ordinary attempt format plus an
explicit cohort scope, law, label, owning family, reporting tier and parent
task. Nothing here changes a recipe, task, seed or qualification.
"""
from pathlib import Path
import subprocess
import sys

from .contracts import atomic_json, read_json, stable_hash
from .policy_coverage import (BLOCKED_HOSTS, COHORT, DIAGNOSTIC_CONTINUATIONS, FAMILIES, LABEL, LAW_INDEPENDENT,
                              PARENTS, SERVED_COHORT, SERVED_LABEL, SERVED_PARENTS, SERVED_VIEW, VIEW, reporting_tier)
from .remaining_attempts import _job

SETTINGS = "policy-cloud.json"
# One frozen request per law and family; each law plans its own committed view.
LAWS = {"clean": dict(cohort=COHORT, view=VIEW, label=LABEL, prefix="policy-cloud-"),
        "as_served": dict(cohort=SERVED_COHORT, view=SERVED_VIEW, label=SERVED_LABEL, prefix="policy-cloud-as-served-")}
# Fixed task wall-time limits are part of each question. These short-limit
# tasks run alone on one GPU so co-tenancy cannot spend their budget.
TIMEOUT_SENSITIVE = frozenset({"gaussian1d_smoke", "gaussian1d_stability", "ring16_acquisition"})


def cohort_scope(family, source, revision, cohort=COHORT):
    """One displayed row: one family, frozen source, recipe and cohort (law)."""
    return stable_hash([family, source, revision, cohort])


def settings(directory, source=None):
    """Persist the intake's families, laws and frozen source on first preparation."""
    path = Path(directory) / SETTINGS
    if path.is_file():
        value = read_json(path)
        if value.get("schema_version") != 2:
            raise ValueError("policy-cloud state was prepared by the clean-only intake; use a new state directory")
        if source is not None and Path(value["source_root"]) != Path(source):
            raise ValueError("policy-cloud state belongs to another frozen source checkout")
        return value
    if source is None:
        raise ValueError("policy-cloud intake needs --source-root on first preparation")
    value = {"schema_version": 2, "families": list(FAMILIES), "source_root": str(Path(source).resolve()),
             "laws": {law: {key: spec[key] for key in ("cohort", "view", "label")} for law, spec in LAWS.items()},
             "shared_tests": sorted(LAW_INDEPENDENT), "timeout_sensitive_gpu": "1"}
    atomic_json(path, value)
    return value


def _requests(directory, value):
    """Plan each missing (law, family) request once, in one worker call."""
    from .attempt_coverage import WORKER
    requests = Path(directory) / "requests"
    paths = {(law, family): requests / (LAWS[law]["prefix"] + family + ".json")
             for law in LAWS for family in value["families"]}
    missing = [{"candidate_id": family, "view_id": LAWS[law]["view"], "tasks": [], "output": str(path)}
               for (law, family), path in paths.items() if not path.is_file()]
    if missing:
        spec = requests / "policy-cloud-prepare.json"
        atomic_json(spec, {"source_root": value["source_root"], "requests": missing})
        subprocess.run([sys.executable, str(WORKER), "prepare", str(spec)], check=True,
                       stdout=subprocess.DEVNULL, timeout=1800)
    result = {}
    for (law, family), path in paths.items():
        request = read_json(path)
        if (request["view"]["id"] != LAWS[law]["view"] or request["candidate"]["id"] != family
                or request["protocol"]["seed"] != 0 or not request["source"].get("snapshot_path")):
            raise ValueError(f"{family}: frozen {law} policy-cloud request has another view, family, seed or no snapshot")
        result[law, family] = path, request
    return result


def _extra(law, family, scope, assignment, parent, timeout_gpu):
    extra = dict(cohort_scope=scope, cohort_id=LAWS[law]["cohort"], measurement_law=law, label=LAWS[law]["label"],
                 owning_family=family, reporting_tier=assignment["qualification_tier"], parent_task=parent,
                 diagnostic_continuation=parent in DIAGNOSTIC_CONTINUATIONS)
    if parent in TIMEOUT_SENSITIVE:
        extra.update(exclusive_gpu=True, required_gpu=timeout_gpu)
    return extra


def prepare(root, source, directory):
    """Return only these cohorts' jobs; selected/research/repair work is never planned."""
    from .views import task_evaluation_fingerprint, task_execution_fingerprint
    value = settings(directory, source)
    if value["laws"] != {law: {key: spec[key] for key in ("cohort", "view", "label")} for law, spec in LAWS.items()}:
        raise ValueError("policy-cloud settings name other cohorts, views or labels")
    planned = _requests(directory, value)
    jobs = {}
    for family in value["families"]:
        if family not in FAMILIES:
            raise ValueError("policy-cloud intake covers only the selected Atlas and E22 recipes")
        (clean_path, clean), (served_path, served) = planned["clean", family], planned["as_served", family]
        identity = clean["source"]["digest"], clean["candidate_revision"]
        if (served["source"]["digest"], served["candidate_revision"]) != identity:
            raise ValueError(f"{family}: both laws must use one frozen source and recipe")
        scopes = {law: cohort_scope(family, *identity, LAWS[law]["cohort"]) for law in LAWS}
        parents = {}
        for name, assignment in ((row["task"], row) for row in clean["view"]["assignments"]):
            card = clean["tasks"][name]
            if card.get("task_cohort") != COHORT:
                raise ValueError(f"{name}: policy-cloud view assigns a task outside its cohort")
            parent = card["policy_parent"]["id"]
            if assignment["qualification_tier"] != reporting_tier(parent):
                raise ValueError(f"{name}: tier differs from the selected-view tier")
            parents[parent] = name
            extra = _extra("clean", family, scopes["clean"], assignment, parent, value["timeout_sensitive_gpu"])
            if parent in LAW_INDEPENDENT:
                # One run (or refusal) supplies the verdict of both laws.
                extra.update(measurement_law="shared", shared_scopes=[scopes["as_served"]],
                             shared_reason=LAW_INDEPENDENT[parent])
            key, job = _job(clean_path, clean, card, LAWS["clean"]["prefix"] + family, **extra)
            if parent in BLOCKED_HOSTS and job["dependencies"]:
                # The test's own ownership boundary refuses it before training;
                # record that reason instead of waiting on a blocked producer.
                job["dependency_note"] = ("Explicit ownership blocker precedes the declared dependencies: " +
                                          ", ".join(dep["task"] for dep in job["dependencies"]))
                job["dependencies"] = []
            jobs[key] = job
        if set(parents) != set(PARENTS):
            raise ValueError(f"{family}: clean policy-cloud request must assign all {len(PARENTS)} selected tests")
        own, shared = set(), set()
        for name, assignment in ((row["task"], row) for row in served["view"]["assignments"]):
            card = served["tasks"][name]
            parent = card["policy_parent"]["id"]
            if assignment["qualification_tier"] != reporting_tier(parent):
                raise ValueError(f"{name}: tier differs from the selected-view tier")
            if card.get("task_cohort") == COHORT:
                # A shared declaration must be the clean request's own job.
                twin = clean["tasks"].get(name)
                if (parent not in LAW_INDEPENDENT or twin is None
                        or task_execution_fingerprint(card) != task_execution_fingerprint(twin)
                        or task_evaluation_fingerprint(card) != task_evaluation_fingerprint(twin)):
                    raise ValueError(f"{name}: as-served view shares a task that is not its clean law-independent twin")
                shared.add(parent)
                continue
            if card.get("task_cohort") != SERVED_COHORT:
                raise ValueError(f"{name}: as-served view assigns a task outside its cohort")
            own.add(parent)
            extra = _extra("as_served", family, scopes["as_served"], assignment, parent, value["timeout_sensitive_gpu"])
            key, job = _job(served_path, served, card, LAWS["as_served"]["prefix"] + family, **extra)
            jobs[key] = job
        if own != set(SERVED_PARENTS) or shared != set(LAW_INDEPENDENT):
            raise ValueError(f"{family}: as-served request must run {len(SERVED_PARENTS)} tests and share "
                             f"{len(LAW_INDEPENDENT)} law-independent ones")
    return jobs


def plan_summary(state):
    """Compact read-only listing used to prove which jobs would execute."""
    order = {"clean": 0, "shared": 0, "as_served": 1}
    rows = sorted(state["jobs"].values(), key=lambda j: (j.get("owning_family", ""), order.get(j.get("measurement_law"), 2),
                                                         j.get("reporting_tier", 0), j["task"]))
    return [{"family": job["family"], "owning_family": job.get("owning_family"), "law": job.get("measurement_law"),
             "tier": job.get("reporting_tier"), "parent_task": job.get("parent_task"), "task": job["task"],
             "status": job["status"], "blocked_host": job.get("parent_task") in BLOCKED_HOSTS,
             "diagnostic_continuation": job.get("diagnostic_continuation", False),
             "dependencies": [dep["task"] for dep in job.get("dependencies", [])],
             "timeout_seconds": job["timeout_seconds"], "required_gpu": job.get("required_gpu"),
             "exclusive_gpu": job.get("exclusive_gpu", False)} for job in rows]
