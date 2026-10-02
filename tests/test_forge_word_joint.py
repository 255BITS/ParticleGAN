"""The inverse-learning host keeps paired semantics and explicit evidence scope."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture, word_bank, score_words, word_oracle_controls
from experiments.forge.adapters import adapter_preflight
from experiments.forge.contracts import read_json
from experiments.forge.views import grade_result, load_tasks, load_view
from experiments.forge.word_adapter import run_word


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def request():
    task = load_tasks(ROOT)["five_word_joint_acquisition"]
    candidate = read_json(ROOT / "configs/forge/ideas/five-word-joint-ka2-v1.json")
    return {"candidate": candidate, "candidate_revision": "software-validation-only",
            "protocol": {"seed": 0}, "tasks": {task["id"]: task}}, task


def test_word_scorer_detects_wrong_pairs_despite_perfect_generated_marginal():
    controls = word_oracle_controls()
    assert all(control["passed"] == control["expected_pass"] for control in controls.values())
    assert controls["swapped_reconstruction"]["failed_bounds"] == [
        "reconstruction_exact", "minimum_reconstruction_token_probability"]
    assert controls["diffuse_correct_argmax"]["metrics"]["quality_fraction"] == 0
    assert controls["wrong_padding"]["metrics"]["quality_fraction"] == 0


@pytest.mark.parametrize("mutate, reason", [
    (lambda c: c["recipe_overrides"].update(row_evidence_gate=True), "row_evidence_gate"),
    (lambda c: c["recipe_overrides"].update(serve_average=.5), "policy-control"),
    (lambda c: c["recipe_overrides"].update(continuous_policy="dv12", total_steps=None), "policy-control"),
    (lambda c: c["recipe_overrides"].update(encoder_mode="ae"), "particle encoders"),
    (lambda c: c["recipe_overrides"].update(batch_size=128), "frozen host resource"),
    (lambda c: c["recipe_overrides"].update(prior_kind="mog"), "explicit prior"),
    (lambda c: c.update(initializer="supplied"), "named deterministic"),
])
def test_incompatible_candidates_block_before_training(mutate, reason):
    frozen, task = request()
    mutate(frozen["candidate"])
    assert any(reason in blocker for blocker in adapter_preflight(task, frozen["candidate"], root=ROOT))


def test_task_bound_geometry_prior_and_source_are_checked():
    frozen, task = request()
    assert adapter_preflight(task, frozen["candidate"], root=ROOT) == []
    for mutation in (
            lambda t: t["execution"]["host_definition"].update(words=["wrong"]),
            lambda t: t["execution"]["prior"].update(sigma=.01),
            lambda t: t["evaluation"]["thresholds"].append(["modes", ">=", 1]),
            lambda t: t["evaluation"]["sources"].update({"benchmarks/toy_audit/api_images.py": "0" * 64})):
        altered = deepcopy(task)
        mutation(altered)
        assert adapter_preflight(altered, frozen["candidate"], root=ROOT)


def test_joint_api_updates_encoder_and_all_roles_with_isolated_clean_sampling(tmp_path):
    frozen, task = request()
    global_rng = torch.get_rng_state().clone()
    raw, records = run_word(frozen, task, tmp_path, "cpu", execution_limit=2, capture_media=True)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert raw["recipe"]["prior_kind"] == "particles"
    assert raw["prior"]["kind"] == "particle_cloud" and raw["prior"]["sigma"] == 0
    assert raw["recipe"]["total_steps"] == 20000
    assert raw["execution_path"] == "public_components"
    guards = raw["evidence"]["guards"]
    assert guards["optimizer_updates"] == {"generator": 2, "encoder": 2, "prior": 2, "discriminator": 2}
    assert guards["all_finite"] and guards["hooks_exercised"]
    assert guards["unintended_rng_deviations"] == 0
    assert [record["step"] for record in records] == [0, 1, 2]
    assert all(record["metrics"]["served_averaged"] == 0 and
               record["metrics"]["policy_latent_perturbation"] == 0 and
               record["metrics"]["output_noise_added"] == 0 for record in records)
    # Incomplete budget never qualifies, regardless of a trainer status string.
    raw["status"] = "PASS"
    assert grade_result(task, raw)["gate_status"] == "INCOMPLETE"
    # A forged complete curve cannot conceal only two actual optimizer updates.
    oracle = {"sample_count": 1024, "quality_fraction": 1., "modes": 5, "mass_tv": 0.,
              "reconstruction_exact": 1, "minimum_reconstruction_token_probability": 1.}
    raw["evidence"]["observations"] = [{"step": math.ceil(i * 20001 / 24), **oracle} for i in range(1, 25)]
    raw["evidence"]["live"] = oracle
    assert grade_result(task, raw)["gate_status"] == "INCOMPLETE"
    state = torch.load(tmp_path / "state.pt", weights_only=False)
    assert state["fixture"]["api_state"]["completed_steps"] == 2
    assert set(raw["evidence"]["host"]["models"]) == {"generator", "discriminator", "encoder"}


def test_standalone_word_fixture_retains_its_public_recipe_and_seed_offsets():
    # Existing published cases remain on the original constructor path.
    fixture = WordFixture(device="cpu", seed=24002, recipe_name="ka2", max_steps=1)
    published = next(row for row in read_json(ROOT / "reports/toy_audit/api_contract/runs.json")["cases"]
                     if row["id"] == "image-five-words-joint-ae")
    assert json.loads(json.dumps(fixture.recipe.to_dict())) == published["recipe"]
    assert torch.equal(fixture.data_generator.get_state(), torch.Generator().manual_seed(24103).get_state())
    assert torch.equal(fixture.policy.latent_generator.get_state(), torch.Generator().manual_seed(24004).get_state())


def test_new_view_preserves_existing_meaningful_smoke_prerequisites():
    view = load_view(ROOT, "five_word_joint")
    tier1 = [row["task"] for row in view["assignments"] if row["qualification_tier"] == 1]
    assert tier1 == ["two_pole", "unused_token_hold", "ae_gan_hold"]
    assert view["assignments"][-1] == {"task": "five_word_joint_acquisition", "qualification_tier": 2,
                                       "importance": "required", "order": 0}


def test_report_joins_word_example_through_its_explicit_retained_question():
    from experiments.forge.tier_report import build_report
    guide = next(row for row in build_report(ROOT)["experiment_guides"] if row["id"] == "five_word_joint")
    assert "source-family-15" in guide["original_question_ids"]
    variant = next(row for row in guide["api_variants"] if row["id"] == "image-five-words-joint-ae")
    assert variant["verdict"] == "PASS" and variant["media_available"]
    assert variant["qualification_input"] is False
