"""One newly authorized five-track campaign, retaining prior queue evidence.

Reuse the validated collector and public Forge execution from the closed search.
This campaign has its own immutable admission receipt and one finite round;
its shared queue retains every earlier paid charge and exact cached baseline.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRIOR_ARCHIVE = Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012")
ARCHIVE = Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012")
CAMPAIGN = "r1r2-five-track-repair-v1"

spec = importlib.util.spec_from_file_location(
    "r1r2_closed_search_workflow", ROOT / "reports/forge/r1r2-tier1-search/workflow.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def validate_manifest(value):
    shared.require(value["schema_version"] == 1 and value["round"] == 1
                   and value["id"] == CAMPAIGN, "Use this separately registered single-round campaign")
    shared.require(value["max_candidates"] == 6 and len(value["candidates"]) == 6
                   and not value.get("searches"), "Exactly five new recipes and one cached baseline")
    shared.require(value["paid_cap_seconds"] == 5400,
                   "Retain the declared shared-queue paid ceiling, including earlier charges")
    shared.require(value["campaign"]["candidate_budget_seconds"] == shared.ALLOWANCE
                   and value["campaign"]["budget_seconds"] == 6 * shared.ALLOWANCE,
                   "Reserve the full six-role Tier1 comparison")
    shared.require(value["baseline_role"] == "baseline"
                   and value["candidates"][0]["candidate"] == "recipe-prior-refactor-r1r2-v1",
                   "Retain the exact cached scheduled baseline")
    shared.require(len({c["role"] for c in value["candidates"]}) == 6,
                   "Distinct global recipe roles are required")
    return value


def storage(archive):
    shared.require(archive == ARCHIVE, "Use the declared independent admission archive")
    shared.require((archive / "queue").is_symlink()
                   and (archive / "queue").resolve() == PRIOR_ARCHIVE / "queue",
                   "Reuse the original queue and charges; never rerun or copy the baseline")
    current = original_storage(archive)
    prior = original_storage(PRIOR_ARCHIVE)
    shared.require(current["bytes"] + prior["bytes"] < shared.ARCHIVE_CAP,
                   "Shared artifacts exceed the combined archive cap")
    return dict(current=current, prior=prior)


def requests(root, queue_root, manifest):
    resolved, searches = original_requests(root, queue_root, manifest)
    prior = shared.read_json(root / "reports/forge/r1r2-tier1-search/round2-registration.json")
    saved = {k: v for k, v in prior["candidates"]["baseline"].items() if k != "card"}
    shared.require(resolved["baseline"]["projection"] == saved,
                   "Cached baseline source/runtime/task/job bindings must remain exact")
    state = shared.read_json(queue_root / "queue/state.json")
    shared.require(all(state["jobs"].get(key, {}).get("result") is not None
                       and len(state["jobs"][key]["attempts"]) == 1
                       for key in saved["task_keys"].values()),
                   "All seven cached baseline jobs must already have one retained result")
    return resolved, searches


original_storage = shared.storage
original_requests = shared.requests
shared.validate_manifest = validate_manifest
shared.storage = storage
shared.requests = requests
shared.OUT = OUT

if __name__ == "__main__":
    shared.main()
