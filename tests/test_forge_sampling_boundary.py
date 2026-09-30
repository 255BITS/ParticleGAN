"""Hand-built new-source requests must satisfy sampling contracts before updates."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.sampling import (ENUMERATED_PRIOR_CLEAN, PUBLIC_PRIOR_CLEAN,
    executed_receipt, validate_request_sampling)
from experiments.forge.sources import inspect_source, snapshot_source
from test_forge_queue import request, campaign, grade

ROOT = Path(__file__).resolve().parents[1]


def prospective(tmp_path, *, grouped=False):
    req = request(tmp_path, cap=1)
    checkout = tmp_path / "worktree"
    module = checkout / "experiments/forge/sampling.py"
    module.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "experiments/forge/sampling.py", module)
    source = inspect_source(checkout)
    source["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "queue", source))
    req["source"] = source
    req["candidate"]["claim_contract"] = {"sampling_law": "task_declared"}
    for spec in req["tasks"].values():
        spec.update(adapter="transfer_vector", execution={"steps": 24},
                    evaluation={**executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")})
        spec["preflight_blockers"] = []  # A forged empty cache cannot bypass recomputation.
    req["preflight_blockers"] = []
    if grouped:
        req["jobs"][0]["task_ids"] = ["t1", "t2"]
        req["view"]["assignments"][1]["qualification_tier"] = 1
        req["jobs"].pop(1)
    return req


def mutate(req, name):
    if name == "candidate_omitted":
        req["candidate"].pop("claim_contract")
    elif name == "candidate_law":
        req["candidate"]["claim_contract"]["sampling_law"] = "public_noisy"
    elif name == "task_omitted":
        req["tasks"].pop("t1")
    elif name == "task_version":
        req["tasks"]["t1"]["evaluation"].pop("sampling_contract_version")
    elif name == "task_law":
        req["tasks"]["t1"]["evaluation"]["sampling_law"] = ENUMERATED_PRIOR_CLEAN
    elif name == "task_noise":
        req["tasks"]["t1"]["evaluation"]["eval_output_noise"] = "public_recipe_schedule"
    else:
        raise AssertionError(name)


@pytest.mark.parametrize("mutation", ["candidate_omitted", "candidate_law", "task_omitted",
                                     "task_version", "task_law", "task_noise"])
def test_queue_rejects_handbuilt_missing_or_mismatching_contract_before_mutation(tmp_path, mutation):
    req = prospective(tmp_path)
    mutate(req, mutation)
    q = Queue(tmp_path / "queue", grader=grade)
    with pytest.raises(ValueError, match="sampling contract blocked"):
        q.submit(req, campaign())
    assert not q.inspect()["jobs"]
    assert not (q.root / "queue/state.json").exists()


def test_queue_checks_grouped_child_not_only_execution_producer(tmp_path):
    req = prospective(tmp_path, grouped=True)
    req["tasks"]["t2"]["evaluation"].pop("sampling_contract_version")
    with pytest.raises(ValueError, match="t2: new planning requires"):
        Queue(tmp_path / "queue", grader=grade).submit(req, campaign())


def test_queue_accepts_valid_prospective_contract_without_executing(tmp_path):
    req = prospective(tmp_path)
    q = Queue(tmp_path / "queue", grader=grade)
    entry = q.submit(req, campaign())
    assert entry["status"] == "queued"
    assert all(not job["attempts"] for job in q.inspect()["jobs"].values())


def test_legacy_frozen_source_remains_executable_without_new_contract(tmp_path):
    req = request(tmp_path, cap=1)
    before = deepcopy(req)
    validate_request_sampling(req)
    q = Queue(tmp_path / "queue", grader=grade)
    assert q.submit(req, campaign())["status"] == "queued"
    assert req == before


def test_manifest_omission_does_not_downgrade_existing_new_source(tmp_path):
    req = prospective(tmp_path)
    req["source"]["files"].pop("experiments/forge/sampling.py")
    req["source"]["digest"] = stable_hash(req["source"]["files"])
    req["candidate"].pop("claim_contract")
    with pytest.raises(ValueError, match="undeclared files"):
        Queue(tmp_path / "queue", grader=grade).submit(req, campaign())


def test_declared_new_source_requires_a_real_snapshot(tmp_path):
    req = prospective(tmp_path)
    req["source"].pop("snapshot_path")
    with pytest.raises(ValueError, match="requires a frozen source snapshot"):
        Queue(tmp_path / "queue", grader=grade).submit(req, campaign())


@pytest.mark.parametrize("mutation", ["candidate_omitted", "candidate_law", "task_version", "task_law"])
def test_runtime_rechecks_new_source_contract_before_adapter(tmp_path, monkeypatch, mutation):
    from experiments.forge import runtime
    req = prospective(tmp_path, grouped=True)
    if mutation.startswith("task_"):
        # Mutate only the grouped child, preserving the leader and cached checks.
        field = "sampling_contract_version" if mutation == "task_version" else "sampling_law"
        if mutation == "task_version":
            req["tasks"]["t2"]["evaluation"].pop(field)
        else:
            req["tasks"]["t2"]["evaluation"][field] = ENUMERATED_PRIOR_CLEAN
    else:
        mutate(req, mutation)
    req["campaign_id"] = "boundary-fixture"
    calls = []
    monkeypatch.setattr("experiments.forge.adapters.run_task", lambda *args: calls.append(args))
    # This tests dispatch only. It does not construct models, allocate CUDA or
    # alter process-wide PyTorch execution settings.
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda _: None, use_deterministic_algorithms=lambda _: None,
        backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace()))))
    monkeypatch.setattr(runtime, "MemoryProbe", lambda **kwargs: SimpleNamespace(snapshot=lambda: {}))
    path = tmp_path / "attempt/request.json"
    atomic_json(path, {"request": req, "job": req["jobs"][0],
                       "worker": {"device": "cpu", "attempt": "fixture"}})
    assert runtime.execute(path) == 1
    assert calls == []
    raw = read_json(path.parent / "raw-result.json")
    assert raw["applicability"]["status"] == "unsupported"
    assert "sampling contract blocked" in raw["error"]["message"]
