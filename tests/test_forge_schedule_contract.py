"""Operational schedule controls on the real public trainer, not qualifications."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.clockfree import run_clockfree, source_audit, schedule_blockers
from experiments.forge.views import grade_result, load_tasks


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scheduled():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    task = json.loads((ROOT / "configs/forge/tasks/schedule_contract_audit.json").read_text())
    request = {"candidate": {"recipe_overrides": {"critic_formulation": "k3p", "reg_arm": "b_cap"}},
               "protocol": {"seed": 0}}
    yield task, request
    torch.set_num_threads(previous)


def test_declared_bcap_schedules_have_independent_numeric_and_replay_proof(scheduled, tmp_path):
    task, request = scheduled
    original_rng = torch.get_rng_state().clone()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    assert torch.equal(original_rng, torch.get_rng_state())
    assert raw["cost"]["completed_updates"] == task["execution"]["steps"] == 231
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", grade
    assert grade["metrics"]["maximum_schedule_error"] == 0
    assert grade["metrics"]["schedule_replay_state_mismatches"] == 0
    assert grade["metrics"]["guard_released_parameter_checks"] > 0
    assert grade["metrics"]["clockfree_claim"] is False
    # Clock labels and horizon affect this scheduled recipe as predicted.
    assert all(row["changed_sha256"] != row["reference_sha256"]
               for row in raw["evidence"]["comparisons"] if row["condition"] in {"step_label", "horizon"})
    from experiments.forge.tier1_media import render
    media = render(task, {**raw, "gate_status": grade["status"]}, tmp_path, tmp_path / "schedule.gif")
    assert media["observation_count"] == 4
    assert media["optimizer_updates_added"] == media["sampling_draws_added"] == 0
    assert (tmp_path / "schedule.gif").stat().st_size > 0


def test_delayed_beta2_schedule_fails_even_with_expected_branch_differences(scheduled, tmp_path, monkeypatch):
    import particlegan.recipe_schedules as schedules
    task, request = scheduled
    request["candidate"]["recipe_overrides"].update(optimizer_family="adam", d_guard_ratio=0,
        reg_anchor_weight=0, latent_damping_max_rate=0, direct_particle_gain=False,
        beta2_end=.95, beta2_anneal_end=1)
    original = schedules.apply_optimizer_schedule
    monkeypatch.setattr(schedules, "apply_optimizer_schedule",
                        lambda step, recipe, optimizer: original(max(0, step - 100), recipe, optimizer))
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "FAIL", grade
    assert grade["metrics"]["maximum_schedule_error"] > 1e-12
    assert grade["metrics"]["schedule_replay_state_mismatches"] > 0


def test_delayed_lr_cosine_fails_in_its_active_interior(scheduled, tmp_path, monkeypatch):
    import particlegan.training as training
    task, request = scheduled
    original = training.learning_rate_scales
    monkeypatch.setattr(training, "learning_rate_scales",
                        lambda step, recipe: original(max(0, step - 100), recipe))
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "FAIL", grade
    assert grade["metrics"]["maximum_schedule_error"] > 1e-12
    assert grade["metrics"]["schedule_replay_state_mismatches"] > 0


def test_optional_beta2_and_coefficient_schedules_follow_their_actual_clocks(scheduled, tmp_path):
    task, request = scheduled
    request["candidate"]["recipe_overrides"].update(optimizer_family="adam", d_guard_ratio=0,
        reg_anchor_weight=0, latent_damping_max_rate=0, direct_particle_gain=False,
        beta2_end=.95, beta2_anneal_end=1, prior_betas=(0, .99), reg_coeff_end=4, reg_coeff_anneal_end=1)
    raw = run_clockfree(request, task, tmp_path, "cpu")
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", grade
    proof = torch.load(Path(raw["evidence"]["artifact_root"]) / "comparisons.pt", weights_only=True)
    controls = proof["schedule_observations"]
    assert controls["step_label"][0]["betas"][1][0][1] == .95
    assert controls["step_label"][0]["coefficient"] == controls["reference"][0]["coefficient"]
    assert controls["horizon"][0]["coefficient"] != controls["reference"][0]["coefficient"]


def test_delayed_critic_coefficient_schedule_fails(scheduled, tmp_path, monkeypatch):
    import particlegan.recipe_schedules as schedules
    task, request = scheduled
    request["candidate"]["recipe_overrides"].update(reg_coeff_end=4, reg_coeff_anneal_end=1)
    original = schedules.apply_penalty_schedule
    monkeypatch.setattr(schedules, "apply_penalty_schedule",
                        lambda step, recipe, penalty: original(max(0, step - 100), recipe, penalty))
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "FAIL", grade
    assert grade["metrics"]["maximum_schedule_error"] > 1e-12


def test_undeclared_clock_side_effect_fails_when_all_consumed_schedules_match(scheduled, tmp_path, monkeypatch):
    from particlegan.policy import UpdatePolicy
    task, request = scheduled
    original = UpdatePolicy.finish_step

    def hidden_change(policy):
        original(policy)
        if policy.completed_steps >= task["execution"]["step_label_offset"]:
            with torch.no_grad():
                next(policy.G.parameters()).add_(.001)

    monkeypatch.setattr(UpdatePolicy, "finish_step", hidden_change)
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "FAIL", grade
    assert grade["metrics"]["maximum_schedule_error"] == 0
    assert grade["metrics"]["schedule_replay_state_mismatches"] > 0


def test_broken_checkpoint_restoration_is_rejected(scheduled, tmp_path, monkeypatch):
    from experiments.forge.api import FormulationContext
    task, request = scheduled
    original = FormulationContext.load_state_dict

    def broken(context, state):
        original(context, state)
        with torch.no_grad():
            next(context._trainer.G.parameters()).add_(.001)

    monkeypatch.setattr(FormulationContext, "load_state_dict", broken)
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "INVALID", grade


def test_guard_release_uses_real_adam_history_and_rejects_a_delayed_threshold(scheduled, tmp_path, monkeypatch):
    from particlegan.k3p import CriticSpikeGuard
    task, request = scheduled
    request["candidate"]["recipe_overrides"]["d_guard_ratio"] = 1e-5
    original = CriticSpikeGuard.apply_

    def delayed(guard, optimizer):
        guard.min_steps += 1
        try:
            return original(guard, optimizer)
        finally:
            guard.min_steps -= 1

    monkeypatch.setattr(CriticSpikeGuard, "apply_", delayed)
    grade = grade_result(task, run_clockfree(request, task, tmp_path, "cpu"))
    assert grade["status"] == "FAIL", grade
    assert grade["metrics"]["maximum_guard_relative_error"] > 1e-6


def test_source_audit_does_not_hide_delayed_optional_schedules():
    from particlegan import Recipe
    recipe = Recipe(critic_formulation="k3p", reg_arm="b_cap", lr_floor=1, network_lr_floor=1,
                    input_noise_std=0, output_noise_warmup=0, d_guard_min_steps=0,
                    optimizer_family="adam", d_guard_ratio=0, reg_anchor_weight=0,
                    latent_damping_max_rate=0, direct_particle_gain=False,
                    beta2_end=.95, reg_coeff_end=4).to_dict()
    dependencies = source_audit(recipe, {})["unexplained_clock_dependencies"]
    assert len(dependencies) == 2
    assert any("beta2" in value for value in dependencies)
    assert any("coefficient" in value for value in dependencies)


def test_schedule_claims_cannot_be_stamped_or_tolerances_relaxed(scheduled, tmp_path):
    task, request = scheduled
    raw = run_clockfree(request, task, tmp_path, "cpu")
    forged = deepcopy(raw)
    forged["evidence"]["schedule_contract"]["maximum_schedule_error"] = 5
    assert grade_result(task, forged)["status"] == "INVALID"
    relaxed = deepcopy(task)
    relaxed["evaluation"]["schedule_tolerance"] = 1
    assert grade_result(relaxed, raw)["status"] == "INVALID"


def test_schedule_task_and_strict_clockfree_task_keep_separate_evaluators():
    tasks = load_tasks(ROOT)
    assert tasks["schedule_contract_audit"]["evaluation"]["kind"] == "schedule_contract"
    assert tasks["clockfree_audit"]["evaluation"]["kind"] == "clockfree_parity"


def test_unsupported_periodic_control_blocks_before_training(scheduled):
    from experiments.forge.adapters import adapter_preflight
    task, request = scheduled
    request["candidate"]["recipe_overrides"]["reg_every"] = 2
    assert any("periodic" in reason for reason in adapter_preflight(task, request["candidate"]))


def test_scheduled_activation_cannot_hide_unsupported_penalty_counters():
    from particlegan import Recipe
    recipe = Recipe(critic_formulation="k3p", reg_arm="b_cap", reg_coeff=0,
                    reg_coeff_end=2, reg_every=2).to_dict()
    assert any("periodic" in reason for reason in schedule_blockers(recipe, {}))
    # The conservative source audit also retains an implicit KA2 switch if an
    # activation is represented by a future or unvalidated declaration.
    recipe.update(reg_arm=None, critic_formulation="ka2", reg_every=1)
    assert any("KA2 switches" in reason for reason in schedule_blockers(recipe, {}))
