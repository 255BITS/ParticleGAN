"""Explicit checkpoint repairs with unchanged recipes and frozen task laws."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

from .contracts import atomic_json, file_hash, read_json, stable_hash
from .planning import candidate_revision_for, rekey_jobs
from .remaining_attempts import _freeze, _job, frozen_source
from .sources import verify_snapshot
from .views import task_execution_fingerprint


def align_prior_contracts(request, children):
    """Repair only equivalent learned-prior ownership metadata, retaining law and recipe."""
    for name in children:
        child = request["tasks"][name]
        parent = request["tasks"][child["execution"]["continuation_of"]]
        prior = deepcopy(child["execution"]["prior"])
        if (request["candidate"]["resolved_recipe"].get("prior_update") != "learned"
                or parent["execution"].get("prior_contract") != "recipe_owned_v1"
                or prior.pop("learnable", None) is not True or prior != parent["execution"]["prior"]):
            raise ValueError("prior ownership repair must preserve the actual learned prior and its complete initial law")
        before = task_execution_fingerprint(child)
        child["execution"]["prior"] = deepcopy(parent["execution"]["prior"])
        child["execution"]["prior_contract"] = parent["execution"]["prior_contract"]
        request.setdefault("task_metadata_repairs", {})[name] = dict(original_execution=before,
            repaired_execution=task_execution_fingerprint(child), parent=parent["id"],
            change="Align the continuation's ownership schema with its own learned-prior producer; distribution and resolved recipe unchanged.")


def repaired_source(root, original, directory, patches):
    allowed = {"experiments/forge/state.py", "experiments/forge/adapters.py",
               "experiments/forge/continuation.py", "experiments/forge/views.py"}
    if set(patches) - allowed:
        raise ValueError("checkpoint repair cannot modify formulation sources")
    manifest = deepcopy(original)
    if set(patches) - set(manifest["files"]):
        raise ValueError("repair cannot add undeclared source members")
    manifest["files"].update(patches)
    manifest["digest"] = stable_hash(manifest["files"])
    manifest["repair"] = dict(original_source=original["digest"], files=patches,
                              formulation_changed=False)
    target = directory / "repair-source" / manifest["digest"]
    if target.exists():
        verify_snapshot(target, manifest)
        return target, manifest
    baseline = frozen_source(root, original, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "source"
        shutil.copytree(baseline, stage, ignore=shutil.ignore_patterns("__pycache__"))
        for relative, digest in patches.items():
            current = Path(__file__).resolve().parents[2] / relative
            if file_hash(current) != digest:
                raise ValueError("repair bytes differ from the explicitly frozen plan: " + relative)
            shutil.copyfile(current, stage / relative)
        atomic_json(stage / "forge-source.json", manifest)
        verify_snapshot(stage, manifest)
        stage.rename(target)
    return target, manifest


def prepare(root, directory):
    path = directory / "checkpoint-repairs.json"
    if not path.is_file():
        return {}
    jobs = {}
    for entry in read_json(path)["repairs"]:
        originals = []
        for pin in entry["requests"]:
            original = Path(pin["path"])
            if file_hash(original) != pin["sha256"]:
                raise ValueError("checkpoint recovery original request changed")
            originals.append(read_json(original))
        request = deepcopy(originals[0])
        for other in originals:
            if (other["candidate_revision"], other["source"]["digest"], other["protocol"]) != (
                    request["candidate_revision"], request["source"]["digest"], request["protocol"]):
                raise ValueError("checkpoint recovery cannot pool original cohorts")
            request["tasks"].update(other["tasks"])
        if request["protocol"]["seed"] != 0:
            raise ValueError("checkpoint repair retains protocol seed zero")
        source, manifest = repaired_source(Path(entry["source_root"]), request["source"], directory, entry["patches"])
        request["source"] = {**manifest, "snapshot_path": str(source)}
        request["candidate_revision"] = candidate_revision_for(manifest["digest"], request["candidate"])
        align_prior_contracts(request, entry.get("align_prior_contracts", []))
        for job in request["jobs"]:
            job["science"]["candidate_revision"] = request["candidate_revision"]
            for name in entry.get("align_prior_contracts", []):
                if name in job["science"]["execution"]:
                    job["science"]["execution"][name] = task_execution_fingerprint(request["tasks"][name])
        rekey_jobs(request["jobs"])
        scope = stable_hash([entry["id"], manifest["digest"], request["candidate_revision"]])
        target = directory / "repair-requests" / (scope + ".json")
        _freeze(target, request)
        for name, tier in entry["tasks"].items():
            card = request["tasks"][name]
            if not any(name in j.get("task_ids", [j["task_id"]]) for j in request["jobs"]):
                raise ValueError("repair has no original execution job for " + name)
            key, job = _job(target, request, card, "repair-" + scope[:12],
                label=entry["label"], owning_family=entry["family"], repair_scope=scope,
                repair_superseded=bool(entry.get("superseded_by")),
                reporting_tier=tier, diagnostic_continuation=name in entry.get("diagnostic_continuations", []),
                repair_origin_source=manifest["repair"]["original_source"])
            jobs[key] = job
    return jobs
