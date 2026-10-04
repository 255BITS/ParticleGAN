"""Small public-owner controls; these software prefixes confer no toy PASS."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.adapters import _Run
from experiments.forge.policy_adapters import (PolicyLifecycleAudit, controls_receipt,
    evaluation_state, finite_policy_state, typed_state_digest)


ROOT = Path(__file__).resolve().parents[1]
CLOUD = {"kind": "particle_cloud", "sigma": 0., "standardize": False,
         "learnable": True, "exception_reason": "tiny isolated public-policy software fixture"}


@pytest.fixture(autouse=True)
def cpu_one_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def tiny(preset):
    context = FormulationContext(recipe_preset=preset, prior=CLOUD, seed=19,
        recipe_overrides={"num_particles": 12, "z_dim": 2, "batch_size": 8})
    generator = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)), component="generator")
    critic = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)), component="discriminator")
    trainer = context.build_trainer(generator, critic, max_steps=3)
    task = {"id": "isolated_policy_control", "execution": {"steps": 3, "policy_contract": {"schema_version": 1}},
            "evaluation": {"policy_observation": {"schema_version": 1, "sampler": "GANTrainer.sample",
                "weight_selector": "state_selected", "output_noise": False,
                "latent_policy": "actual_selected_public_policy", "diagnostic_credit": False}}}
    return context, trainer, task


def test_typed_digest_keeps_types_nonfinite_sentinels_and_tensor_payloads():
    state = {"x": [float("nan"), -math.inf], "z": torch.tensor([0., 1.])}
    assert typed_state_digest(state) == typed_state_digest(deepcopy(state))
    for changed in ({"x": tuple(state["x"]), "z": state["z"]},
                    {"x": state["x"], "z": state["z"].double()},
                    {"x": state["x"], "z": torch.tensor([0., 2.])}):
        assert typed_state_digest(changed) != typed_state_digest(state)


@pytest.mark.parametrize("root", ["trainer", "policy"])
def test_health_retains_only_source_defined_lr_sentinels(root):
    diagnostic = {root: {"lr_settle": [[{"r_b": [torch.tensor([float("nan"), 0.])],
        "last": {"t_b": math.inf, "mean_r_b": float("nan"), "log_bf_b": -math.inf},
        "last_look": {"log_bf_2b": -math.inf}}]]}}
    assert finite_policy_state(diagnostic)
    for family in ("models", "optimizers", "controller"):
        assert not finite_policy_state({root: {family: {"bad": float("nan")}}})
    bad = deepcopy(diagnostic); bad[root]["lr_settle"][0][0]["last"]["log_bf_b"] = math.inf
    assert not finite_policy_state(bad)
    bad = deepcopy(diagnostic); bad[root]["lr_settle"][0][0]["unknown"] = float("nan")
    assert not finite_policy_state(bad)
    bad = deepcopy(diagnostic); bad[root]["lr_settle"][0][0]["r_b"][0][0] = math.inf
    assert not finite_policy_state(bad)


@pytest.mark.parametrize("preset", ["atlas", "e22"])
def test_observer_sampling_and_hook_audit_preserve_exact_next_trajectory(preset, tmp_path):
    plain, baseline, _ = tiny(preset)
    observed, trainer, task = tiny(preset)
    run = _Run(observed, trainer, tmp_path, task)
    real = torch.arange(16, dtype=torch.float32).reshape(8, 2).sin()
    baseline.step(real); run.step(real)
    before = typed_state_digest(evaluation_state(observed.state_dict()))
    samples = run.evaluate(lambda: run.sample(32))
    assert samples.shape == (32, 2)
    assert before == typed_state_digest(evaluation_state(observed.state_dict()))
    baseline.step(real); run.step(real)
    assert typed_state_digest(evaluation_state(plain.state_dict())) == typed_state_digest(evaluation_state(observed.state_dict()))
    controls = controls_receipt(trainer.policy, trainer.completed_steps)
    assert controls["implementation_observed"]
    assert controls["lifecycle"]["calls"] == {name: 2 for name in run.policy_audit.calls}
    assert all(controls["enabled"][name] for name, requested in controls["requested"].items() if requested)
    assert finite_policy_state(observed.state_dict())
    assert run.policy_purity[-1]["pure"]
    assert controls["quality_qualification"] is False


def test_unobserved_owner_and_forged_cursor_cannot_claim_complete_controls():
    _, trainer, _ = tiny("atlas")
    controls = controls_receipt(trainer.policy, 0)
    assert controls["implementation_observed"] is False
    with pytest.raises(ValueError, match="cursor"):
        controls_receipt(trainer.policy, 1)
    audit = PolicyLifecycleAudit(trainer.policy)
    audit.calls["finish_step"] = 1
    assert controls_receipt(trainer.policy, 0)["implementation_observed"] is False


def test_policy_read_that_mutates_training_is_rejected(tmp_path):
    context, trainer, task = tiny("atlas")
    run = _Run(context, trainer, tmp_path, task)
    def bad_observer():
        with torch.no_grad():
            trainer.prior.z.add_(.01)
        return {"value": 0.}
    with pytest.raises(RuntimeError, match="observation changed"):
        run.evaluate(bad_observer)


def test_policy_read_cannot_consume_training_rng(tmp_path):
    context, trainer, task = tiny("atlas")
    run = _Run(context, trainer, tmp_path, task)
    with pytest.raises(RuntimeError, match="observation changed"):
        run.evaluate(lambda: torch.randn(2, generator=trainer.latent_generator))


def test_original_policy_isolation_population_boundary_is_not_disabled():
    context = FormulationContext(recipe_preset="atlas", prior=CLOUD, seed=19,
        recipe_overrides={"num_particles": 8, "z_dim": 2, "batch_size": 8})
    generator = context.construct(lambda: nn.Linear(2, 2), component="generator")
    critic = context.construct(lambda: nn.Linear(2, 1), component="discriminator")
    with pytest.raises(ValueError, match="isolation needs enough particles"):
        context.build_trainer(generator, critic, max_steps=3)


def test_synthetic_critic_guard_probe_retains_requested_amsgrad_memory():
    from experiments.forge.mechanisms import _probe
    from particlegan import get_recipe
    row = _probe(get_recipe("atlas"), "critic_guard")
    assert row["status"] == "PASS"
    assert row["training_evidence"] is False
    assert row["measurements"]["clipped_tensors"] > 0


def test_fake_lifecycle_method_cannot_be_certified_as_public_execution():
    _, trainer, _ = tiny("atlas")
    trainer.policy.finish_step = lambda: None
    with pytest.raises(ValueError, match="unmodified public"):
        PolicyLifecycleAudit(trainer.policy)


@pytest.mark.parametrize("parent_id", ["img_intensity2", "vector_two_broad"])
def test_short_actual_scalar_host_exports_same_scored_selected_cloud(parent_id, tmp_path, monkeypatch):
    import numpy as np
    from experiments.forge.adapters import run_task, adapter_preflight
    from experiments.forge.artifacts import verify_artifacts
    from experiments.forge.policy_contracts import (COHORT, REQUIRED_POLICY_SOURCES,
                                                  _prospective_variant, _parent_record)
    from experiments.forge import policy_cohorts
    path = ROOT / "configs/forge/tasks" / f"{parent_id}.json"
    parent = json.loads(path.read_text())
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in REQUIRED_POLICY_SOURCES}
    task = _prospective_variant(parent, _parent_record(parent, hashlib.sha256(path.read_bytes()).hexdigest()), sources)
    original = deepcopy(task)
    validate = policy_cohorts.validate_policy_task
    validate(original, root=ROOT)
    task["execution"]["steps"] = 2  # software prefix, never the original 600/1200 budget
    with pytest.raises(ValueError):
        validate(task, root=ROOT)
    short_digest = typed_state_digest(task)

    def structural_prefix_declaration(value, root=None):
        # Only this exact two-update software fixture substitutes declaration
        # checking. The original full task must validate; actual policy owners,
        # arrays, observers, RNG and complete checkpoint checks remain real.
        if typed_state_digest(value) == short_digest:
            return validate(original, root=root)
        return validate(value, root=root)

    monkeypatch.setattr(policy_cohorts, "validate_policy_task", structural_prefix_declaration)
    candidate = {"recipe_preset": "atlas", "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5}, "task_cohort": COHORT}
    assert adapter_preflight(task, candidate, root=ROOT) == []
    request = {"candidate": candidate, "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    result = run_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert result["cost"]["completed_steps"] == 2
    evidence = result["evidence"]
    assert evidence["guards"]["all_finite"] and evidence["guards"]["hooks_exercised"]
    assert evidence["scoring_weights"] == "state_selected"
    assert evidence["policy_controls"]["lifecycle"]["observed_updates"] == 2
    assert len(evidence["retained_draws"]) == 2
    verify_artifacts(evidence["artifact_root"], evidence["artifact_manifest"])
    for draw in evidence["retained_draws"]:
        with np.load(Path(evidence["artifact_root"]) / draw["path"], allow_pickle=False) as archive:
            assert len(archive["samples"]) == (32 if parent_id.startswith("img_") else 4096)
            assert np.isfinite(archive["samples"]).all()
    state = torch.load(Path(evidence["artifact_root"]) / "state.pt", weights_only=True)
    assert state["trainer"]["max_steps"] == 2 and state["recipe"]["total_steps"] is None
    assert typed_state_digest(state) == evidence["checkpoint"]["state_sha256"]


def test_policy_word_cannot_reach_legacy_live_components_at_preflight_or_dispatch(tmp_path):
    from experiments.forge.adapters import adapter_preflight, _dispatch_task
    from experiments.forge.api import CapabilityError
    from experiments.forge.policy_contracts import (COHORT, REQUIRED_POLICY_SOURCES,
                                                  _prospective_variant, _parent_record)
    path = ROOT / "configs/forge/tasks/five_word_joint_acquisition.json"
    parent = json.loads(path.read_text())
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in REQUIRED_POLICY_SOURCES}
    task = _prospective_variant(parent, _parent_record(parent, hashlib.sha256(path.read_bytes()).hexdigest()), sources)
    candidate = {"recipe_preset": "atlas", "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5}, "task_cohort": COHORT}
    blockers = adapter_preflight(task, candidate, root=ROOT)
    assert blockers and "joint" in blockers[0] and "ordered policy ownership" in blockers[0]
    request = {"candidate": candidate, "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    with pytest.raises(CapabilityError, match="ordered policy ownership"):
        _dispatch_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert not list(tmp_path.iterdir())
