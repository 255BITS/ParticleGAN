"""Saved group reuse must verify evidence and never authorize another training run."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from experiments.forge import attempt_coverage as runner
from experiments.forge import grouped_evidence as reuse
from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sampling import executed_receipt, PUBLIC_PRIOR_CLEAN
from experiments.forge.views import grade_result, task_execution_fingerprint, task_evaluation_fingerprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_group(tmp_path):
    cards = {name: read_json(ROOT / "configs/forge/tasks" / (name + ".json"))
             for name in ("ring_hold", "ring_extension")}
    for card in cards.values():
        card["execution"].pop("prior_contract")  # Explicit legacy numerical fixture.
        card["execution"]["prior"]["learnable"] = True
    source = tmp_path / "frozen"
    files = {}
    for relative in ("experiments/forge/views.py",
                     "reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        files[relative] = file_hash(target)
    manifest = dict(files=files, digest=stable_hash(files))
    execution = {name: task_execution_fingerprint(card) for name, card in cards.items()}
    evaluation = {name: task_evaluation_fingerprint(card) for name, card in cards.items()}
    science = dict(execution=execution, evaluation=evaluation)
    request = dict(source=manifest, candidate_revision="recipe", tasks=cards, jobs=[
        dict(task_ids=list(cards), compatibility_key=stable_hash(science), science=science)])
    request_path = tmp_path / "request.json"
    atomic_json(request_path, request)
    directory = tmp_path / "producer"
    artifact = directory / "provenance/state.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"retained fixture state, never loaded or trained")
    checkpoint = dict(path="state.pt", sha256=file_hash(artifact), completed_steps=2900,
                      artifact_root=str(artifact.parent), artifact_manifest=manifest_artifacts(artifact.parent),
                      purpose="provenance_only", prerequisite_eligible=False)
    raw = dict(cost=dict(completed_steps=2900), evidence={
        **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean"),
        "dense": [dict(step=s, modes=8, hq=.95) for s in range(1201, 2901)],
        "continuity": dict(mode="uninterrupted", run_id="one-run", resume_count=0,
                           original_schedule_horizon=1200, max_total_steps=7500),
        "provenance_checkpoint": checkpoint})
    atomic_json(directory / "raw.json", raw)
    atomic_json(directory / "result.json", dict(status="PASS", runtime={"fixture": True},
                raw_sha256=file_hash(directory / "raw.json")))
    jobs = []
    for name in cards:
        job = dict(family="fixture", candidate_id="candidate", source_digest=manifest["digest"],
                   candidate_revision="recipe", task=name, execution_sha256=execution[name],
                   evaluation_sha256=evaluation[name], request_path=str(request_path),
                   request_sha256=file_hash(request_path), source_root=str(source),
                   dependencies=cards[name]["dependencies"], attempted=True,
                   resources=dict(gpus=1, gpu_memory_mb=1), timeout_seconds=3600)
        job["key"] = stable_hash(reuse._identity(job))
        jobs.append(job)
    parent, child = jobs
    parent.update(status="PASS", origin="new_attempt", directory=str(directory), attempt_id="producer-id",
                  result_sha256=file_hash(directory / "result.json"), evidence=str(directory / "result.json"))
    child.update(status="BLOCKED", block_kind="prerequisite", attempt_id="refused-id",
                 reason="certified artifact binding unavailable", result_sha256="old-result",
                 evidence="original-refusal.json")
    atomic_json(directory / "request.json", parent)
    return parent, child


def rewrite_raw(parent, change):
    directory = Path(parent["directory"])
    raw = read_json(directory / "raw.json")
    change(raw)
    atomic_json(directory / "raw.json", raw)
    result = read_json(directory / "result.json")
    result["raw_sha256"] = file_hash(directory / "raw.json")
    atomic_json(directory / "result.json", result)
    parent["result_sha256"] = file_hash(directory / "result.json")


def test_regrades_complete_group_without_training_or_checkpoint_promotion(saved_group, monkeypatch):
    from particlegan import GANTrainer
    def forbidden(*args, **kwargs):
        pytest.fail("saved evidence reuse cannot train or sample")
    monkeypatch.setattr(GANTrainer, "step", forbidden)
    monkeypatch.setattr(GANTrainer, "sample", forbidden)
    parent, child = saved_group
    before = deepcopy(saved_group)
    certificate = reuse.grade_saved_group(parent, child)
    assert certificate["grade"]["status"] == "PASS"
    assert certificate["grade"]["metrics"]["extension_passing"] == 300
    assert certificate["optimizer_updates_added"] == certificate["sampling_draws_added"] == 0
    assert certificate["qualification_input"] is False
    assert saved_group == before
    # A provenance-only checkpoint still grants no permission to resume training.
    assert runner.prerequisite_bindings(child, dict(jobs={"parent": parent})) == {}


def test_extension_failure_is_preserved_even_when_parent_passes(saved_group):
    parent, child = saved_group
    rewrite_raw(parent, lambda raw: raw["evidence"]["dense"][-1].update(hq=.8))
    certificate = reuse.grade_saved_group(parent, child)
    assert certificate["parent_grade"]["status"] == "PASS"
    assert certificate["grade"]["status"] == "FAIL"
    assert certificate["grade"]["metrics"]["extension_passing"] == 299


@pytest.mark.parametrize("change", [
    lambda raw: raw["evidence"]["dense"].pop(),
    lambda raw: raw["evidence"]["dense"].pop(10),
    lambda raw: raw["evidence"]["dense"][250].update(hq=.8),
    lambda raw: raw["evidence"]["continuity"].update(resume_count=1),
    lambda raw: raw["evidence"]["continuity"].update(run_id=""),
])
def test_missing_checks_failed_hold_or_restarted_state_cannot_gain_credit(saved_group, change):
    parent, child = saved_group
    rewrite_raw(parent, change)
    with pytest.raises(ValueError):
        reuse.grade_saved_group(parent, child)


@pytest.mark.parametrize("field", ["source_digest", "candidate_revision", "family"])
def test_different_source_recipe_or_family_cannot_supply_a_parent(saved_group, field):
    parent, child = saved_group
    child[field] = "different"
    child["key"] = stable_hash(reuse._identity(child))
    with pytest.raises(ValueError, match="same passing producer"):
        reuse.grade_saved_group(parent, child)


@pytest.mark.parametrize("filename", ["raw.json", "result.json", "provenance/state.pt", "request.json"])
def test_tampered_saved_inputs_are_rejected(saved_group, filename):
    parent, child = saved_group
    path = Path(parent["directory"]) / filename
    if filename == "request.json":
        spec = read_json(path)
        spec["candidate_revision"] = "another recipe"
        atomic_json(path, spec)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        reuse.grade_saved_group(parent, child)


def test_reconciliation_preserves_refusal_is_idempotent_and_never_launches_training(saved_group, tmp_path, monkeypatch):
    parent, child = saved_group
    state = dict(jobs={parent["key"]: parent, child["key"]: child})
    old_child = deepcopy(child)
    calls = []
    def grade_only(command, **kwargs):
        assert command[-2] == "reuse_group"
        path = Path(command[-1])
        spec = read_json(path)
        atomic_json(path.parent / "result.json", reuse.grade_saved_group(spec["parent"], spec["child"]))
        calls.append(command)
    monkeypatch.setattr(runner.subprocess, "run", grade_only)
    assert runner.reconcile_grouped(tmp_path / "repo", tmp_path / "runs", state) == [child["key"]]
    assert child["status"] == "PASS" and child["origin"] == "reused_execution"
    assert child["updates"] == child["seconds"] == 0
    assert child["reused_execution"]["previous_attempt"]["attempt_id"] == old_child["attempt_id"]
    assert child["reused_execution"]["previous_attempt"]["status"] == "BLOCKED"
    assert runner.reconcile_grouped(tmp_path / "repo", tmp_path / "runs", state) == []
    assert len(calls) == 1
    assert parent["status"] == "PASS"
    runner.publish(tmp_path / "repo", state)
    from experiments.forge.research_tiers import tier_rows
    rows = tier_rows(tmp_path / "repo", tmp_path / "repo/reports/forge/technique-inventory.md")
    assert len(rows["fixture"]) == 1 and "1/1" in rows["fixture"][0]
    assert "Verified extension from the existing uninterrupted run" in rows["fixture"][0]
    report = read_json(tmp_path / "repo" / runner.REPORT)
    assert report["summary"]["total"] == 2 and report["summary"]["counts"] == {"PASS": 2}
    assert report["qualification_input"] is False


def test_failed_parent_is_never_reconciled(saved_group, tmp_path, monkeypatch):
    parent, child = saved_group
    parent["status"] = "FAIL"
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must stay blocked"))
    assert runner.reconcile_grouped(tmp_path, tmp_path, dict(jobs={"p": parent, "c": child})) == []
    assert child["status"] == "BLOCKED"
