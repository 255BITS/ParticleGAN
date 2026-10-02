"""Bounded two-update integration checks; these runs cannot earn qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.behavior_adapters import BehaviorComponents, FROZEN_HOST_RECIPE_FIELDS, HOSTS, run_behavior
from experiments.forge.views import grade_result, load_tasks
from experiments.forge.sampling import FIELDS, expected_policy


ROOT = Path(__file__).resolve().parents[1]


def request():
    return dict(protocol=dict(seed=0), candidate=dict(recipe_overrides={
        "input_noise_std": 0., "output_noise_std": 0.,
    }))


@pytest.mark.parametrize("host", HOSTS)
def test_configured_paper_training_recipe_reaches_all_behavior_components(tmp_path, host):
    """Exercise the real factory/callback ordering, including prior bundles."""
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2
    candidate = json.loads((ROOT / "configs/forge/ideas/r3gan-stacked-training-toy-v1.json").read_text())
    result = run_behavior(dict(protocol=dict(seed=0), candidate=candidate), task, tmp_path / host)
    applied = result["applied"]
    assert set(applied["public_optimizers"]) == {"Adam"}
    schedule = applied["training_schedules"]
    assert schedule["horizon"] == 2
    gamma = schedule["penalty_coefficient"]
    # Conditional hosts sum one penalty per context into the same D update.
    contexts = {"unipolar": 2, "mid_scale_identity": 4}.get(host, 1)
    assert gamma["observations"] == 2 * contexts
    assert gamma["first"] == pytest.approx(1.)
    assert gamma["last"] == pytest.approx(.1)
    for role in schedule["optimizer_groups"].values():
        assert role["lr"]["minimum"] == role["lr"]["maximum"] == .0002
        assert role["beta2"]["first"] == pytest.approx(.9)
        assert role["beta2"]["last"] == pytest.approx(.99)
    mechanisms = result["evidence"]["guards"]["mechanism_audit"]["mechanisms"]
    assert mechanisms["critic_penalty"]["applied"] == 2 * contexts
    assert all(not row["requested"] and not row["enabled"] and row["applied"] == 0
               for name, row in mechanisms.items() if name != "critic_penalty")
    assert applied["recipe"]["input_noise_std"] == 0
    assert applied["recipe"]["output_noise_std"] == 0
    assert result["evidence"]["guards"]["all_finite"]
    assert result["evidence"]["guards"]["hooks_exercised"]
    assert grade_result(load_tasks(ROOT)[host], result)["status"] == "INCOMPLETE"


def test_penalty_schedule_starting_at_zero_remains_requested(tmp_path):
    task = deepcopy(load_tasks(ROOT)["two_pole"])
    task["execution"]["steps"] = 2
    candidate = json.loads((ROOT / "configs/forge/ideas/r3gan-stacked-training-toy-v1.json").read_text())
    candidate["recipe_overrides"].update(reg_coeff=0., reg_coeff_end=1.)
    result = run_behavior(dict(protocol=dict(seed=0), candidate=candidate), task, tmp_path)
    row = result["evidence"]["guards"]["mechanism_audit"]["mechanisms"]["critic_penalty"]
    assert row["requested"] and row["enabled"]
    # The kernel runs on both calls; the observed coefficient records zero pressure first.
    assert row["applied"] == 2
    coefficient = result["applied"]["training_schedules"]["penalty_coefficient"]
    assert coefficient["first"] == 0 and coefficient["last"] == 1
    assert result["evidence"]["guards"]["hooks_exercised"]


@pytest.mark.parametrize("noise_enabled", [False, True])
def test_named_noise_receipt_reports_actual_streams_without_consuming_rng(noise_enabled):
    candidate = dict(protocol=dict(seed=0), candidate={}) if noise_enabled else request()
    components = BehaviorComponents(candidate, load_tasks(ROOT)["two_pole"])
    noise = components.noise
    streams = components.context.streams
    # Exercise the ordinary sampling methods without constructing a model or
    # optimizer. Labels must remain accurate after the generators have advanced.
    noise.set_step(0)
    noise.input(torch.zeros(12, 1))
    noise.set_step(16)
    noise.output(torch.zeros(12, 1), generator_step=True)
    before = streams.audit()
    global_before = torch.get_rng_state().clone()

    applied = components.receipt()
    assert components.receipt() == applied
    assert streams.audit() == before
    assert torch.equal(torch.get_rng_state(), global_before)
    bindings = applied["rng"]["bindings"].values()
    input_binding = next(row for row in bindings
                         if (row["family"], row["component"], row["purpose"]) == ("noise", "critic", "input"))
    output_binding = next(row for row in bindings
                          if (row["family"], row["component"], row["purpose"]) == ("noise", "generator", "output"))
    receipt = applied["noise"]
    assert receipt["seed"] == applied["rng"]["seed"] == 0
    assert receipt["rng_derivation"] == applied["rng"]["version"] == "forge-rng-v1"
    assert receipt["d_noise_seed"] == input_binding["seed"] == noise.input_stream.initial_seed()
    assert receipt["output_noise_seed"] == output_binding["seed"] == noise.output_stream.initial_seed()
    assert receipt["output_noise_train_state_initial_sha256"] == output_binding["initial_state_sha256"]
    assert receipt["output_noise_seed_offset"] is None
    assert receipt["d_noise_seed"] != 901 and receipt["output_noise_seed"] != 1901
    assert applied["public_optimizers"] == []


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("formulation", ["ka2", "k3p"])
def test_original_objectives_bind_public_components_and_record_real_updates(tmp_path, host, formulation):
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2
    before = torch.get_rng_state().clone()
    declaration = request()
    declaration["candidate"]["recipe_overrides"]["critic_formulation"] = formulation
    result = run_behavior(declaration, task, tmp_path / host, "cpu")
    assert {field: result["evidence"][field] for field in FIELDS} == expected_policy(task)
    assert torch.equal(before, torch.get_rng_state())
    assert result["execution_path"] == "public_components"
    assert result["device"] == "cpu"
    assert result["applied"]["public_optimizers"]
    assert all(name in {"K3PGeneratorAdam", "KA2CriticAdam" if formulation == "ka2" else "K3PCriticAdam"}
               for name in result["applied"]["public_optimizers"])
    assert len(result["evidence"]["observations"]) == 2
    assert result["evidence"]["guards"]["all_finite"]
    assert result["evidence"]["guards"]["hooks_exercised"]
    assert result["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert all(n == 2 for n in result["evidence"]["guards"]["optimizer_updates"].values())
    # A shortened unit fixture is never sufficient evidence for the actual task.
    assert grade_result(load_tasks(ROOT)[host], result)["status"] == "INCOMPLETE"
    assert (tmp_path / host / "result.json").is_file()


def test_unused_control_does_not_invent_a_prior_optimizer(tmp_path):
    task = deepcopy(load_tasks(ROOT)["unused_token_hold"])
    task["execution"]["steps"] = 2
    result = run_behavior(request(), task, tmp_path)
    assert set(result["evidence"]["guards"]["optimizer_updates"]) == {"generator", "discriminator"}
    assert "not applicable" in result["applied"]["mechanism_applicability"]["a2"]


def test_scheduled_components_reject_clockfree_claim_before_updates(tmp_path):
    task = load_tasks(ROOT)["two_pole"]
    candidate = request()
    candidate["candidate"]["claim_contract"] = {"learning": "clockfree"}
    with pytest.raises(ValueError, match="clock-free"):
        run_behavior(candidate, task, tmp_path)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("host", ["two_pole", "ae_gan_hold"])
def test_default_input_output_noise_stays_isolated_during_evaluation(tmp_path, host):
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2
    result = run_behavior(dict(protocol=dict(seed=0), candidate={}), task, tmp_path)
    assert result["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert result["evidence"]["guards"]["all_finite"]
    assert result["applied"]["recipe"]["input_noise_std"] > 0
    assert result["applied"]["recipe"]["output_noise_std"] > 0
    if host == "ae_gan_hold":
        assert result["applied"]["prior"]["kind"] == "mog"
        assert result["applied"]["prior"]["sigma"] > 0


def test_ae_scorer_preserves_callers_noise_schedule(tmp_path, monkeypatch):
    from benchmarks.locked_shared.hosts import ae_gan_hold
    original = ae_gan_hold.evaluate
    sigmas = []

    def observe(encoder, decoder, prior, recipe):
        sigmas.append(decoder.policy.output_sigma)
        return original(encoder, decoder, prior, recipe)

    monkeypatch.setattr(ae_gan_hold, "evaluate", observe)
    task = deepcopy(load_tasks(ROOT)["ae_gan_hold"])
    task["execution"]["steps"] = 2
    candidate = dict(protocol=dict(seed=0), candidate=dict(recipe_overrides={
        "output_noise_std": 0.1, "output_noise_warmup": 1.0,
    }))
    run_behavior(candidate, task, tmp_path)
    assert sigmas[0] == 0.0  # The host's initial measurement uses step zero.
    assert 0.05 in sigmas  # Intermediate checks retain their current step.
    assert sigmas[-1] == 0.1


@pytest.mark.parametrize("field", sorted(FROZEN_HOST_RECIPE_FIELDS))
def test_explicit_host_resource_or_objective_overrides_block_before_run(tmp_path, field):
    candidate = request()
    candidate["candidate"]["recipe_overrides"][field] = 17
    with pytest.raises(ValueError, match="owned by the frozen host"):
        run_behavior(candidate, load_tasks(ROOT)["two_pole"], tmp_path)
    assert not (tmp_path / "result.json").exists()


def test_lazy_penalty_invocations_without_applications_block_activation(tmp_path):
    task = deepcopy(load_tasks(ROOT)["two_pole"])
    task["execution"]["steps"] = 2
    candidate = request()
    candidate["candidate"]["recipe_overrides"]["reg_every"] = 100
    result = run_behavior(candidate, task, tmp_path)
    guard = result["evidence"]["guards"]
    penalty = guard["mechanism_audit"]["mechanisms"]["critic_penalty"]
    assert penalty["calls"] == 2 and penalty["eligible"] == penalty["applied"] == 0
    assert not guard["hooks_exercised"]
    grade = grade_result(load_tasks(ROOT)["two_pole"], result)
    assert grade["status"] == "BLOCKED"
    assert "critic_penalty" in str(grade["reasons"])


def test_disabled_mechanisms_are_explicit_ablations_without_fake_activation(tmp_path):
    task = deepcopy(load_tasks(ROOT)["two_pole"])
    task["execution"]["steps"] = 2
    candidate = request()
    candidate["candidate"]["recipe_overrides"].update(reg_coeff=0., reg_anchor_weight=0.,
        d_guard_ratio=0., latent_damping_max_rate=0., direct_particle_gain=False)
    result = run_behavior(candidate, task, tmp_path)
    guard = result["evidence"]["guards"]
    assert guard["hooks_exercised"]
    assert all(not r["requested"] and not r["enabled"] and r["applied"] == 0
               for r in guard["mechanism_audit"]["mechanisms"].values())
    assert "disabled" in result["applied"]["mechanism_applicability"]["a2"]


def test_delayed_guard_has_no_host_activation_credit_but_real_bounded_probe(tmp_path):
    task = deepcopy(load_tasks(ROOT)["two_pole"])
    task["execution"]["steps"] = 2
    candidate = request()
    candidate["candidate"]["recipe_overrides"]["d_guard_min_steps"] = 10_000
    result = run_behavior(candidate, task, tmp_path)
    guards = result["evidence"]["guards"]
    row = guards["mechanism_audit"]["mechanisms"]["critic_guard"]
    assert row["enabled"] and row["calls"] == 2
    assert row["eligible"] == row["applied"] == 0
    assert row["probe"]["measurements"]["initial_adam_step"] == 10_000
    assert row["probe"]["measurements"]["clipped_tensors"] > 0
    assert row["probe"]["measurements"]["optimizer_updates"] == 1
    assert row["probe"]["training_evidence"] is False
    assert guards["hooks_exercised"]
    assert guards["optimizer_updates"]["discriminator"] == 2
