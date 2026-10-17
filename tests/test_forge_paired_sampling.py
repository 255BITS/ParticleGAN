"""Paired sampling stays separate from required clean scoring and training RNG."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.mechanisms import MechanismAudit, mechanism_blockers
from experiments.forge.paired_sampling import DECLARATION, FIELD, add_output_noise, paired_sampling_blockers
from experiments.forge.nativeprofiles import (RELEASE_PROFILE_ID, build_native_models,
    native_host_initialization, native_profile_blockers, resolve_native_spec)
from experiments.forge.sampling import task_blockers
from experiments.forge.state import state_digest
from particlegan.training import output_noise_std


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_paired_addition_matches_public_sample_law_and_zero_amplitude_consumes_no_draws():
    context = FormulationContext(recipe_overrides={"num_particles": 12, "z_dim": 2,
        "total_steps": 2, "output_noise_std": .029, "output_noise_warmup": 0})
    trainer = context.build_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    before = context.state_dict()
    stream = torch.Generator().manual_seed(731)
    same = torch.Generator().set_state(stream.get_state())
    clean = trainer.sample(200, generator=stream)
    paired = add_output_noise(clean, output_noise_std(context.recipe, 0), stream)
    public_noisy = trainer.sample(200, generator=same, output_noise=True)
    assert torch.equal(paired, public_noisy)
    assert state_digest(context.state_dict()) == state_digest(before)
    old = stream.get_state().clone()
    assert add_output_noise(clean, 0., stream) is clean
    assert torch.equal(old, stream.get_state())


def test_paired_contract_is_native_diagnostic_only_and_required_policy_stays_clean():
    task = json.loads((ROOT / "configs/forge/tasks/grid100_affine_paired_laws_v1.json").read_text())
    assert task_blockers(task) == []
    assert task["evaluation"]["eval_output_noise"] == "clean"
    assert task["evaluation"][FIELD] == DECLARATION
    task["evaluation"][FIELD]["qualification_reuse"] = True
    assert paired_sampling_blockers(task)
    assert task_blockers(task)
    task["evaluation"][FIELD] = None
    assert paired_sampling_blockers(task)
    assert task_blockers(task)
    task["evaluation"][FIELD] = deepcopy(DECLARATION)
    task["adapter"] = "transfer_vector"
    assert paired_sampling_blockers(task)


def test_release_host_pins_full_resources_gaussian_cloud_and_actual_initial_tensors():
    task = json.loads((ROOT / "configs/forge/tasks/grid100_release07_cloud_named_v1.json").read_text())
    spec = resolve_native_spec(task)
    assert task["execution"]["native_profile"]["id"] == RELEASE_PROFILE_ID
    assert spec["resources"] == {"z_dim": 4, "batch_size": 256, "num_particles": 20000}
    prior = task["execution"]["prior"]
    assert native_profile_blockers(task, {"prior": prior}) == []
    assert native_profile_blockers(task, {"prior": prior, "recipe_overrides": {"z_dim": 2}})
    from test_release07_parity import NEUTRAL_K3P_FIELDS
    context = FormulationContext(prior=prior, host_initialization=native_host_initialization(task),
        recipe_overrides={**spec["resources"], **NEUTRAL_K3P_FIELDS, "reg_arm": "b_cap"})
    g, d = build_native_models(context, spec)
    before = torch.get_rng_state().clone()
    trainer = context.build_trainer(g, d)
    assert torch.equal(before, torch.get_rng_state())
    assert trainer.prior.z.shape == (20000, 4)
    expected = torch.empty_like(trainer.prior.z)
    stream = torch.Generator().manual_seed(context.streams.seed_for("init", component="prior", purpose="z"))
    nn.init.normal_(expected, 0., 1., generator=stream)
    assert torch.equal(trainer.prior.z, expected)
    assert not trainer.prior_mechanisms["a2"]["requested"]
    assert context.receipt()["initialization"]["prior"]["parameters"]["z"]["tensor_sha256"] == state_digest(expected)
    changed = deepcopy(task)
    changed["execution"]["prior"]["sigma"] = .025
    assert native_profile_blockers(changed, {"prior": prior})


@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
def test_fixed_penalty_mechanisms_never_claim_a_k3p_anchor(arm):
    context = FormulationContext(recipe_overrides={"reg_arm": arm, "num_particles": 12,
        "batch_size": 4, "total_steps": 1})
    trainer = context.build_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    audit = MechanismAudit(context.recipe, trainer.opt_d, [trainer.opt_g])
    result = trainer.step(torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]), collect_stats=True)
    audit.observe_penalty(result["penalty_stats"])
    receipt = audit.receipt()
    anchor = receipt["mechanisms"]["critic_anchor"]
    assert anchor["requested"] is False and anchor["enabled"] is False and anchor["applied"] == 0
    assert "probe" not in anchor
    assert receipt["mechanisms"]["critic_penalty"]["applied"] == 1
    assert mechanism_blockers(receipt) == []
