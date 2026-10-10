"""Read-only source/declaration reproduction; never submit or start a worker."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from particlegan import get_recipe

spec = importlib.util.spec_from_file_location(
    "projection_baseline_existing_workflow", ROOT / "reports/forge/bcap-three-phase/phase3.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
report = Path(__file__).resolve().parent
registration_path = report / "registration.json"
registration = json.loads(registration_path.read_text())
proposal = json.loads((report / "spec.json").read_text())
original = json.loads((report / "preparation.json").read_text())
require = workflow.require
require(proposal["review_status"] == "approved_conditional_admission", "Review state changed")
requests = workflow.resolved(ROOT, registration)

def recipe(request):
    candidate = request["candidate"]
    return asdict(get_recipe(candidate["recipe_preset"], **candidate["recipe_overrides"]))

baseline, candidate = requests["baseline"], requests["candidate"]
left, right = recipe(baseline), recipe(candidate)
delta = {key: dict(baseline=left[key], candidate=right[key])
         for key in left if left[key] != right[key]}
require(delta == {"constraint_geometry_mode": dict(baseline="none", candidate="direction_blend")},
        "The complete recipe must differ only in direction blend")
require(registration["source_digest"] == original["frozen_scientific_digest"],
        "Original scientific digest changed")
require(baseline["source"]["files"] == candidate["source"]["files"], "Paired source differs")
for name, digest in baseline["source"]["files"].items():
    require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
            f"Scientific source changed: {name}")
for name, task in baseline["tasks"].items():
    a, b = deepcopy(task), deepcopy(candidate["tasks"][name])
    ao, bo = a.pop("field_ownership"), b.pop("field_ownership")
    require(a == b, f"Original task conditions changed: {name}")
    av = ao["recipe_fields"].pop("constraint_geometry_mode")
    bv = bo["recipe_fields"].pop("constraint_geometry_mode")
    require(av["value"] == "none" and bv["value"] == "direction_blend" and ao == bo,
            f"Unexpected ownership delta: {name}")
print(json.dumps(dict(scope="read_only_frozen_preparation", head=workflow.head(ROOT),
    registration_sha256=hashlib.sha256(registration_path.read_bytes()).hexdigest(),
    scientific_digest=registration["source_digest"], complete_recipe_delta=delta,
    tasks=len(registration["task_ids"]), source_files=len(baseline["source"]["files"]),
    paired_full_reservation_seconds=registration["full_reservation_seconds"],
    paid_ceiling_seconds=registration["campaign_ceiling_seconds"],
    software_allowance_seconds=registration["software_allowance_seconds"],
    computed_preflight={role: plan["admission"] for role, plan in registration["arms"].items()},
    admission="Pending explicit root predicate; this check submits nothing", workers_launched=0)))
