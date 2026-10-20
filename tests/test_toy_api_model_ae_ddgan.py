"""Paired/independent cohort controls and public composed execution contracts."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from experiments.forge.rng import NamedStreams
from experiments.forge.state import state_digest
from benchmarks.toy_audit.api_contract import validate_case, validate_observation
from benchmarks.toy_audit.api_model_ddgan import ALPHA_BAR, BASELINE, MixtureTarget, ROOT
from benchmarks.toy_audit.api_model_ae_ddgan import (
    build_case, list_cases, paired_chain, paired_metrics,
)


@pytest.fixture(autouse=True)
def one_torch_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def rng(component):
    return NamedStreams(0).generator("eval", component=component)


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_pair_scorer_passes_correct_pairs_rejects_correct_marginal_wrong_pairs(kind):
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    shuffled = clean.roll(1, 0)
    metrics, failures = paired_metrics(clean, clean, shuffled, target)
    assert not failures and metrics["paired_nmse"] == 0
    assert "paired_nmse" in paired_metrics(clean, shuffled, shuffled, target)[1]
    collapsed = clean.mean(0).expand_as(clean)
    assert "paired_shuffled_gain" in paired_metrics(clean, collapsed, collapsed, target)[1]
    assert "paired_nonfinite" in paired_metrics(clean, clean * float("nan"), shuffled, target)[1]
    with pytest.raises(ValueError, match="matching"):
        paired_metrics(clean, clean[:-1], shuffled, target)


def test_pair_chain_uses_four_real_transitions_and_only_code_carries_identity():
    fixture = build_case("api-model-ae-ddgan-gaussian-chain", max_steps=1)
    clean = fixture.target.sample(256, rng("clean"))
    class Oracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.steps, self.inputs = [], []
        def forward(self, xt, codes, t):
            self.steps.append(int(t[0])); self.inputs.append(xt.clone())
            return codes
    oracle = Oracle()
    result, states = paired_chain(oracle, fixture.schedule, clean,
        start_rng=rng("start"), reverse_rng=rng("reverse"), return_states=True)
    assert oracle.steps == [4, 3, 2, 1] and len(states) == 5
    assert not torch.equal(states[0], clean)
    torch.testing.assert_close(result, clean, rtol=0, atol=0)


@pytest.mark.parametrize("case", list_cases(), ids=lambda c: c["id"])
def test_task_owns_conditions_and_preserves_complete_selected_recipe(case):
    validate_case(case)
    task = json.loads((ROOT / f"configs/forge/tasks/model_ae_ddgan_{case['target']}_chain.json").read_text())
    fixture = build_case(case["id"], max_steps=1)
    assert task["evaluation"]["thresholds"] == case["forge_thresholds"]
    assert task["execution"]["steps"] == case["default_steps"]
    assert task["execution"]["prior"] == dict(kind="mog", sigma=.25, standardize=False)
    assert task["execution"]["prior_contract"] == "recipe_owned_v1"
    assert task["evaluation"]["guards"]["exact_optimizer_updates"]
    assert tuple(fixture.recipe.alpha_bar) == ALPHA_BAR
    candidate = json.loads((ROOT / f"configs/forge/ideas/{BASELINE}.json").read_text())
    adapted = {"model", "conditioning", "encoder_mode", "num_classes", "alpha_bar"}
    for name, value in candidate["recipe_overrides"].items():
        if name not in adapted:
            assert json.loads(json.dumps(getattr(fixture.recipe, name))) == value, name
    assert fixture.recipe.encoder_mode == "ae" and fixture.recipe.model == "ddgan"
    assert fixture.recipe.prior_regularizer == "sbnp" and fixture.recipe.prior_update == "learned"
    assert build_case(case["id"], max_steps=1, recipe=fixture.recipe).recipe == fixture.recipe
    if case["target"] == "bimodal":
        assert task["dependencies"] == [{"task": "model_ae_ddgan_gaussian_chain", "kind": "gate"}]


def test_update_reaches_encoder_and_union_of_prior_rows_and_exact_resume():
    fixture = build_case("api-model-ae-ddgan-gaussian-chain", max_steps=3)
    encoder_before = {name: p.detach().clone() for name, p in fixture.E.named_parameters()}
    prior_before, global_before = fixture.prior.z.detach().clone(), torch.get_rng_state().clone()
    bound_rows = []
    original_bind = fixture.policy.observe_sampled_rows
    def record_rows(indices):
        bound_rows.append(indices.detach().flatten().clone())
        original_bind(indices)
    fixture.policy.observe_sampled_rows = record_rows
    fixture.step()
    assert len(bound_rows) == 2  # independent GAN draw and selected AE centers
    assert fixture.batch_sequence_sha256 == fixture.batch_digest
    assert not torch.equal(prior_before, fixture.prior.z)
    assert any(not torch.equal(encoder_before[name], p) for name, p in fixture.E.named_parameters())
    assert torch.equal(global_before, torch.get_rng_state())
    state = fixture.state_dict()
    expected = fixture.step()
    restored = build_case("api-model-ae-ddgan-gaussian-chain", max_steps=3)
    restored.load_state_dict(state)
    assert restored.step() == expected
    assert state_digest(fixture.state_dict()) == state_digest(restored.state_dict())
    assert torch.equal(global_before, torch.get_rng_state())


def test_shuffled_chain_shares_exact_noise_and_evaluation_preserves_all_state():
    fixture = build_case("api-model-ae-ddgan-gaussian-chain", max_steps=1)
    fixture.step()
    fixture.E.train(); fixture.E.net[1].eval()
    before = state_digest(fixture.state_dict())
    codes, starts, noises, steps = [], [], [], []
    original_forward, original_reverse = fixture.G.forward, fixture.schedule.reverse
    def recorded_forward(xt, z, t):
        codes.append(z.detach().clone())
        starts.append(xt.detach().clone())
        return original_forward(xt, z, t)
    def recorded_reverse(clean, xt, t, eta):
        noises.append(eta.detach().clone()); steps.append(int(t[0]))
        return original_reverse(clean, xt, t, eta)
    fixture.G.forward, fixture.schedule.reverse = recorded_forward, recorded_reverse
    record = validate_observation(fixture.observe(n=128))
    assert state_digest(fixture.state_dict()) == before
    assert not record["passed"] and "samples_per_panel" in record["failed_bounds"]
    assert record["metrics"]["paired_reverse_steps"] == 4
    # Last eight calls are paired and shuffled chains, after independent and
    # conditional distribution checks. Starts/noise are exactly pair matched.
    assert steps[-8:] == [4, 3, 2, 1] * 2
    assert torch.equal(starts[-8], starts[-4])
    for i in range(4):
        assert torch.equal(noises[-8 + i], noises[-4 + i])
        assert torch.equal(codes[-8 + i].roll(1, 0), codes[-4 + i])
    assert all(torch.equal(codes[-8], codes[-8 + i]) for i in range(4))
    # Independent generation at the start uses fresh latent draws every step.
    assert not torch.equal(codes[0], codes[1])


def test_frozen_prior_is_respected_and_mismatched_checkpoints_rejected():
    baseline = build_case("api-model-ae-ddgan-gaussian-chain", max_steps=1)
    fixture = build_case("api-model-ae-ddgan-gaussian-chain", max_steps=1,
        recipe=baseline.recipe.replace(prior_update="frozen"))
    prior = fixture.prior.z.clone()
    fixture.step()
    torch.testing.assert_close(fixture.prior.z, prior, rtol=0, atol=0)
    assert fixture.last_losses["prior_regularization"] == 0
    state = deepcopy(fixture.state_dict()); state["case_id"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        fixture.load_state_dict(state)
    with pytest.raises(ValueError, match="seed 0"):
        build_case("api-model-ae-ddgan-gaussian-chain", seed=1)


@pytest.mark.parametrize("changed", [dict(input_noise_std=.1), dict(output_noise_std=.1),
    dict(serve_average=.9), dict(particle_birth_death=True),
    dict(kinetic_transport_weight=.1), dict(constraint_geometry_mode="strict_progress")])
def test_unsupported_composed_mechanisms_are_rejected_before_reservation(changed):
    from experiments.forge.model_toy_adapter import preflight
    candidate = json.loads((ROOT / f"configs/forge/ideas/{BASELINE}.json").read_text())
    candidate["recipe_overrides"].update(changed)
    for kind in ("gaussian", "bimodal"):
        task = json.loads((ROOT / f"configs/forge/tasks/model_ae_ddgan_{kind}_chain.json").read_text())
        assert preflight(task, candidate, root=ROOT)
