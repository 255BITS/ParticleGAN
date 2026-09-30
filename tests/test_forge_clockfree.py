"""Short deterministic API probes; these are not candidate qualifications."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.clockfree import run_clockfree, source_audit
from experiments.forge.state import state_digest
from experiments.forge.sampling import FIELDS, expected_policy
from experiments.forge.views import grade_result


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    request = {"candidate": {"recipe_overrides": {"lr_floor": 1, "network_lr_floor": 1,
        "input_noise_std": 0, "output_noise_warmup": 0, "d_guard_min_steps": 0}},
        "protocol": {"seed": 0}}
    yield task, request
    torch.set_num_threads(previous)


def test_public_state_only_settings_survive_all_clock_perturbations(probe, tmp_path):
    task, request = probe
    original_rng = torch.get_rng_state().clone()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    assert {field: raw["evidence"][field] for field in FIELDS} == expected_policy(task)
    assert torch.equal(original_rng, torch.get_rng_state())
    assert raw["cost"]["completed_updates"] == 24
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", grade
    assert grade["metrics"]["parity_comparisons"] == 4


def test_annealed_candidate_cannot_pass_external_clock_probe(probe, tmp_path):
    task, request = probe
    request["candidate"]["recipe_overrides"] = {}
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "FAIL"


def test_delayed_guard_is_blocked_even_when_short_prefix_agrees(probe, tmp_path):
    task, request = probe
    request["candidate"]["recipe_overrides"]["d_guard_min_steps"] = 200
    raw = run_clockfree(request, task, tmp_path, "cpu")
    assert all(row["reference_sha256"] == row["changed_sha256"] for row in raw["evidence"]["comparisons"])
    assert grade_result(task, raw)["status"] == "BLOCKED"


def test_clock_grader_rejects_claim_only_hashes_or_changed_artifacts(probe, tmp_path):
    task, request = probe
    raw = run_clockfree(request, task, tmp_path, "cpu")
    fake = deepcopy(raw)
    fake["evidence"]["comparisons"][0]["changed_sha256"] = "0" * 64
    assert grade_result(task, fake)["status"] == "INVALID"
    path = Path(raw["evidence"]["artifact_root"]) / "comparisons.pt"
    proof = torch.load(path, weights_only=True)
    proof["trajectories"]["restart"][0]["trainer"]["completed_steps"] += 1
    torch.save(proof, path)
    assert grade_result(task, raw)["status"] == "INVALID"


def test_source_audit_keeps_unknown_extensions_and_periodic_releases_visible():
    from particlegan import Recipe
    recipe = Recipe(lr_floor=1, network_lr_floor=1, input_noise_std=0,
                    output_noise_warmup=0, d_guard_min_steps=0, reg_every=2).to_dict()
    audit = source_audit(recipe, {"unreviewed": True})
    assert len(audit["unexplained_clock_dependencies"]) == 2


def test_clock_audit_recipe_cannot_hide_a_measured_delayed_guard(probe, tmp_path):
    from experiments.forge.artifacts import manifest_artifacts
    task, request = probe
    request["candidate"]["recipe_overrides"]["d_guard_min_steps"] = 200
    raw = run_clockfree(request, task, tmp_path, "cpu")
    root = Path(raw["evidence"]["artifact_root"])
    path = root / "comparisons.pt"
    proof = torch.load(path, weights_only=True)
    proof["recipe"]["d_guard_min_steps"] = 0
    torch.save(proof, path)
    raw["evidence"]["artifact_manifest"] = manifest_artifacts(root)
    raw["evidence"]["source_audit"] = source_audit(proof["recipe"], proof["extensions"])
    assert grade_result(task, raw)["status"] == "INVALID"


def test_state_identity_covers_tensor_values_shapes_and_scalar_types():
    state = {"tensor": torch.arange(4, dtype=torch.bfloat16), "age": 1}
    assert state_digest(state) == state_digest(deepcopy(state))
    assert state_digest(state) != state_digest({**state, "age": True})
    assert state_digest(state) != state_digest({**state, "tensor": state["tensor"].reshape(2, 2)})
    state["tensor"][0] = 9
    assert state_digest(state) != state_digest({"tensor": torch.arange(4, dtype=torch.bfloat16), "age": 1})
    payload = json.dumps(["int", 7]).encode()
    tensor = torch.tensor(list(payload), dtype=torch.uint8)
    assert state_digest([tensor]) != state_digest([("tensor", "torch.uint8", (len(payload),)), 7])


def test_clock_proof_requires_actual_optimizer_updates(probe, tmp_path):
    from experiments.forge.artifacts import manifest_artifacts
    from experiments.forge.clockfree import _comparisons
    task, request = probe
    raw = run_clockfree(request, task, tmp_path, "cpu")
    root = Path(raw["evidence"]["artifact_root"])
    path = root / "comparisons.pt"
    proof = torch.load(path, weights_only=True)
    for name, trajectory in proof["trajectories"].items():
        for state in trajectory:
            state["trainer"]["optimizers"] = deepcopy(proof["initial"]["trainer"]["optimizers"])
    torch.save(proof, path)
    raw["evidence"]["artifact_manifest"] = manifest_artifacts(root)
    raw["evidence"]["comparisons"] = _comparisons(proof)
    assert grade_result(task, raw)["status"] == "INVALID"
