"""Original direct-table purpose, real public-policy lifecycle and strict restore."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_policy_blockers
from experiments.forge.policy_behavior_adapters import TwoPolePolicyFixture, behavior_preflight
from experiments.forge.policy_contracts import (_parent_record, _prospective_variant, COHORT,
                                              REQUIRED_POLICY_SOURCES)
from experiments.forge.policy_adapters import evaluation_state, typed_state_digest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_one_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def definition():
    path = ROOT / "configs/forge/tasks/two_pole.json"
    parent = json.loads(path.read_text())
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
        REQUIRED_POLICY_SOURCES | {"benchmarks/locked_shared/two_pole.py", "benchmarks/legacy/locked_shared.py"}}
    task = _prospective_variant(parent, _parent_record(parent, hashlib.sha256(path.read_bytes()).hexdigest()), sources)
    candidate = {"recipe_preset": "atlas", "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5},
                 "task_cohort": COHORT}
    return {"candidate": candidate, "protocol": {"seed": 0}}, task


def test_two_pole_has_real_public_owner_original_constants_and_observer_parity():
    request, task = definition()
    plain = TwoPolePolicyFixture(request, task)
    observed = TwoPolePolicyFixture(request, task)
    assert plain.recipe.total_steps is None and plain.max_steps == 80
    assert tuple(plain.prior.z.shape) == (12, 1) and torch.count_nonzero(plain.prior.z) == 0
    assert plain.particle_l2 == .02
    assert plain.context.policy is plain.policy and plain.policy.row_semantics == "independent"
    for _ in range(3):
        plain.step(); observed.step()
        before = typed_state_digest(evaluation_state(observed.state_dict()))
        row = observed.observe()
        assert set(row) == {"mean_abs", "grad_med"}
        assert before == typed_state_digest(evaluation_state(observed.state_dict()))
    assert typed_state_digest(plain.state_dict()) == typed_state_digest(observed.state_dict())
    assert observed.guards()["hooks_exercised"]
    assert observed.guards()["optimizer_updates"] == {"prior": 3, "discriminator": 3}
    assert observed.context.receipt()["policy_lifecycle"]["controls"]["row_evidence_observations"] == 3
    assert observed.policy_observations[-1]["latent_policy"] == "not_applied_to_parameter_measurement"


def test_public_checkpoint_restores_exact_next_update_and_input_is_unchanged():
    request, task = definition()
    original = TwoPolePolicyFixture(request, task); original.step(); original.step()
    saved = original.state_dict(); digest = typed_state_digest(saved)
    restored = TwoPolePolicyFixture(request, task); restored.load_state_dict(saved)
    assert typed_state_digest(restored.state_dict()) == digest
    original.step(); restored.step()
    assert typed_state_digest(original.state_dict()) == typed_state_digest(restored.state_dict())
    assert typed_state_digest(saved) == digest


@pytest.mark.parametrize("field", ["recipe", "cursor", "streams", "parameters", "audit"])
def test_changed_state_rejected_without_mutating_public_owner(field):
    request, task = definition()
    source = TwoPolePolicyFixture(request, task); source.step()
    target = TwoPolePolicyFixture(request, task)
    state = source.state_dict(); before = typed_state_digest(target.state_dict())
    if field == "recipe":
        state["recipe"]["lr"] *= 2
    elif field == "cursor":
        state["caller_cursor"] = 0
    elif field == "streams":
        state["policy"]["streams"]["noise_generator"] = state["policy"]["streams"]["eval_generator"].clone()
    elif field == "parameters":
        with torch.no_grad():
            state["policy"]["table"][0, 0] = float("nan")
    else:
        state["lifecycle_audit"]["calls"]["finish_step"] = 0
    with pytest.raises(ValueError):
        target.load_state_dict(state)
    assert typed_state_digest(target.state_dict()) == before


def test_original_task_remains_blocked_and_changed_measurement_is_not_admitted():
    request, task = definition()
    original = json.loads((ROOT / "configs/forge/tasks/two_pole.json").read_text())
    assert task_policy_blockers(original, request["candidate"])
    assert not behavior_preflight(task, request["candidate"])
    changed = deepcopy(task); changed["evaluation"]["policy_observation"]["latent_policy"] = "actual_selected_public_policy"
    assert behavior_preflight(changed, request["candidate"])
    changed = deepcopy(task); changed["execution"]["resources"]["num_particles"] = 16
    assert behavior_preflight(changed, request["candidate"])


def test_short_software_adapter_exports_actual_measurement_arrays_and_exact_state(tmp_path):
    import numpy as np
    from experiments.forge.policy_behavior_adapters import run_behavior
    from experiments.forge.artifacts import verify_artifacts
    request, task = definition()
    # Explicitly tiny execution fixture: it has three actual updates and never
    # supplies the original 80-update/24-observation scientific qualification.
    task["execution"]["steps"] = 3
    result = run_behavior(request, task, tmp_path)
    assert result["cost"]["completed_steps"] == 3
    assert [row["step"] for row in result["evidence"]["observations"]] == [1, 2, 3]
    assert result["evidence"]["scoring_weights"] == "state_selected"
    assert result["evidence"]["policy_observation"]["sampler"] == "served_snapshot"
    assert result["evidence"]["guards"]["optimizer_updates"] == {"prior": 3, "discriminator": 3}
    assert all(row["pure"] for row in result["evidence"]["policy_purity"])
    root = Path(result["evidence"]["artifact_root"])
    verify_artifacts(root, result["evidence"]["artifact_manifest"])
    for row in result["evidence"]["observations"]:
        with np.load(root / "observations" / f"step_{row['step']:06d}.npz", allow_pickle=False) as arrays:
            assert set(arrays.files) == {"real", "particles", "critic_inputs", "critic_gradient"}
            assert arrays["particles"].shape == (12, 1)
            assert arrays["critic_gradient"].shape == (24, 1)
            assert all(np.isfinite(arrays[name]).all() for name in arrays.files)
            assert float(torch.from_numpy(arrays["particles"]).abs().mean()) == row["mean_abs"]
            assert float(torch.from_numpy(arrays["critic_gradient"]).abs().median()) == row["grad_med"]
    state = torch.load(root / "state.pt", weights_only=True)
    assert typed_state_digest(state) == result["evidence"]["checkpoint"]["state_sha256"]
