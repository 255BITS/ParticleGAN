"""The inverse-learning host keeps paired semantics and explicit evidence scope."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture, word_bank, score_words, word_oracle_controls
from experiments.forge.adapters import adapter_preflight
from experiments.forge.contracts import read_json
from experiments.forge.views import grade_result, load_tasks, load_view
from experiments.forge.word_adapter import run_word, word_context


ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="word numerical fixtures require CUDA")


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
    (lambda c: c.update(initializer="supplied"), "candidate initializer conflicts"),
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


def test_word_context_uses_the_same_task_owned_resources_initializer_and_receipt_as_planning():
    from experiments.forge.api import task_formulation_context, task_recipe_resources
    frozen, task = request()
    assert task["execution"]["initializer"] == "deterministic_orthogonal"
    assert task["execution"]["execution_path"] == "public_components"
    assert task_recipe_resources(task) == {"num_particles": 5, "z_dim": 2, "batch_size": 256}
    planned = task_formulation_context(frozen["candidate"], task, frozen["protocol"], device="cpu")
    bound = word_context(frozen, task, "cpu", root=ROOT)
    assert bound.receipt() == planned.receipt()
    receipt = bound.receipt()["field_ownership"]
    assert receipt["task_contract"]["initialization"]["owner"] == "task"
    assert receipt["task_contract"]["initialization"]["value"] == "deterministic_orthogonal"
    for field in ("num_particles", "z_dim", "batch_size"):
        assert receipt["recipe_fields"][field]["owner"] == "task"
        assert receipt["recipe_fields"][field]["value"] == task["execution"]["host_definition"]["resources"][field]


def test_frozen_word_boundary_checks_sources_and_candidate_overrides_with_cached_preflight(tmp_path):
    import shutil
    from experiments.forge.hostprofiles import validate_request_host_profiles
    from experiments.forge.sources import inspect_source, snapshot_source
    from test_forge_hostprofiles import bind_candidate, rebind
    from test_forge_initialization import current_request

    frozen, task = request()
    current = current_request(tmp_path, [task])
    checkout = tmp_path / "worktree"
    for name in task["evaluation"]["sources"]:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    manifest = inspect_source(checkout, set(current["source"]["files"]) | set(task["evaluation"]["sources"]))
    manifest["snapshot_path"] = str(snapshot_source(checkout, tmp_path / "word-source-queue", manifest))
    current["source"] = manifest
    current["candidate"] = {**deepcopy(frozen["candidate"]), "prior": current["candidate"]["prior"]}
    bind_candidate(current)
    rebind(current)
    validate_request_host_profiles(current)
    assert not current["tasks"][task["id"]]["preflight_blockers"]
    current["candidate"]["recipe_overrides"]["batch_size"] = 128
    bind_candidate(current)
    rebind(current)
    with pytest.raises(ValueError, match="frozen host resource batch_size"):
        validate_request_host_profiles(current)


@CUDA_ONLY
def test_joint_api_updates_encoder_and_all_roles_with_isolated_clean_sampling(tmp_path):
    frozen, task = request()
    global_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(DEVICE).clone()
    raw, records = run_word(frozen, task, tmp_path, DEVICE, execution_limit=2, capture_media=True)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(DEVICE))
    assert raw["recipe"]["prior_kind"] == "particles"
    assert raw["prior"]["kind"] == "particle_cloud" and raw["prior"]["sigma"] == 0
    assert raw["recipe"]["total_steps"] == 20000
    assert raw["execution_path"] == "public_components"
    assert raw["initializer"] == task["execution"]["initializer"]
    assert raw["field_ownership"]["recipe_fields"]["num_particles"]["value"] == 5
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


@CUDA_ONLY
@pytest.mark.parametrize("capture_media", [False, True])
def test_saved_word_records_preserve_scored_views_update_state_and_rng(tmp_path, monkeypatch, capture_media):
    from experiments.forge.contracts import file_hash
    from experiments.forge.state import state_digest
    original, scored = WordFixture.observe, []
    def observe(fixture, *args, **kwargs):
        observed = original(fixture, *args, **kwargs)
        scored.append(deepcopy({"step": fixture.completed_steps, **observed}))
        return observed
    monkeypatch.setattr(WordFixture, "observe", observe)
    frozen, task = request()
    global_rng = torch.get_rng_state().clone()
    receipts, states = [], []
    for retain in (False, True):
        output = tmp_path / str(retain)
        options = {} if retain else {"retain_scored_outputs": False}
        result = run_word(frozen, task, output, DEVICE, execution_limit=2,
                          capture_media=capture_media, **options)
        if capture_media:
            raw, records = result
            assert [record["step"] for record in records] == [0, 1, 2]
        else:
            raw = result
        receipts.append(raw)
        states.append(torch.load(output / "state.pt", weights_only=False))
        assert torch.equal(global_rng, torch.get_rng_state())
    assert state_digest(states[0]) == state_digest(states[1])
    checks_per_run = 3 if capture_media else 2
    assert len(scored) == 2 * checks_per_run
    expected = [record for record in scored[checks_per_run:] if record["step"]]
    archived_path = tmp_path / "True/observed-records.pt"
    archived = torch.load(archived_path, weights_only=False)
    assert state_digest(archived) == state_digest(expected)
    assert [record["step"] for record in archived] == [1, 2]
    assert state_digest(scored[:checks_per_run]) == state_digest(scored[checks_per_run:])
    descriptor = receipts[1]["evidence"].pop("saved_observer_outputs")
    assert descriptor == {"path": "observed-records.pt", "sha256": file_hash(archived_path),
        "bytes": archived_path.stat().st_size, "observation_count": 2,
        "kind": "scored_word_records_v1", "optimizer_updates_added": 0, "sampling_draws_added": 0}
    compared = [deepcopy(receipt["evidence"]) for receipt in receipts]
    # Separate output directories have different certificate roots; the
    # certified state, byte identities and every actual observation must match.
    for evidence in compared:
        evidence["provenance_checkpoint"].pop("artifact_root")
    assert compared[0] == compared[1]
    assert not (tmp_path / "False/observed-records.pt").exists()


@CUDA_ONLY
def test_standalone_word_fixture_retains_its_public_recipe_and_seed_offsets():
    # Existing published cases remain on the original constructor path.
    fixture = WordFixture(device=DEVICE, seed=24002, recipe_name="ka2", max_steps=1)
    published = next(row for row in read_json(ROOT / "reports/toy_audit/api_contract/runs.json")["cases"]
                     if row["id"] == "image-five-words-joint-ae")
    assert json.loads(json.dumps(fixture.recipe.to_dict())) == published["recipe"]
    assert all(parameter.is_cuda for model in (fixture.G, fixture.D, fixture.E) for parameter in model.parameters())
    assert torch.equal(fixture.data_generator.get_state(), torch.Generator(device=DEVICE).manual_seed(24103).get_state())
    assert torch.equal(fixture.policy.latent_generator.get_state(), torch.Generator(device=DEVICE).manual_seed(24004).get_state())


@CUDA_ONLY
@pytest.mark.parametrize("optimizer_family", ["formulation", "adam"])
def test_joint_prior_optimizer_honors_distinct_prior_betas(optimizer_family):
    frozen, task = request()
    if optimizer_family == "adam":
        frozen["candidate"] = read_json(next((ROOT / "configs/forge/configurations").glob("r1r2--302b*.json")))
    frozen["candidate"]["recipe_overrides"].update(
        optimizer_family=optimizer_family, betas=[0., .999], prior_betas=[0., .9])
    components = word_context(frozen, task, DEVICE)
    fixture = WordFixture(device=DEVICE, seed=0, recipe_name=None, max_steps=1, components=components)
    prior = fixture.opt_g.param_groups[2]
    assert tuple(prior["betas"]) == (0., .9)
    assert prior["params"] == list(fixture.prior.parameters())
    assert all(tuple(group["betas"]) == (0., .999) for group in fixture.opt_g.param_groups[:2])


def test_joint_acquisition_is_required_smoke_after_existing_prerequisites():
    view = load_view(ROOT, "discriminator_stability")
    tier1 = [row["task"] for row in view["assignments"] if row["qualification_tier"] == 1]
    assert tier1[:4] == ["gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold"]
    assert next(row for row in view["assignments"] if row["task"] == "five_word_joint_smoke") == {
        "task": "five_word_joint_smoke", "qualification_tier": 1, "importance": "required", "order": 5}
    task = load_tasks(ROOT)["five_word_joint_acquisition"]
    assert task["execution"]["steps"] == 20001
    assert task["execution"]["original_schedule_horizon"] == 20000
    assert task["resources"]["timeout_seconds"] == 900


def test_report_joins_word_example_through_its_explicit_retained_question():
    from experiments.forge.tier_report import build_report
    guide = next(row for row in build_report(ROOT)["experiment_guides"] if row["id"] == "five_word_joint")
    assert "source-family-15" in guide["original_question_ids"]
    variant = next(row for row in guide["api_variants"] if row["id"] == "image-five-words-joint-ae")
    assert variant["verdict"] == "PASS" and variant["media_available"]
    assert variant["qualification_input"] is False


def test_failed_renderer_preserves_source_observations_and_state_without_training(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    import numpy as np
    import signal
    import sys

    path = ROOT / "reports/forge/five-word-joint/reproduce_demo.py"
    spec = importlib.util.spec_from_file_location("word_demo_export_failure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "failed-media"
    values = torch.zeros(1, 28, 6)
    record = {"step": 32, "passed": False, "failed_bounds": ["quality_fraction"],
              "views": [{"target": values, "samples": values}]}
    # A synthetic completed adapter return isolates the media/error path.
    # It performs no model construction, training or additional sampling.
    def completed_adapter(request, task, destination, device, **kwargs):
        assert (destination / "source-manifest.json").is_file()
        torch.save({"completed_steps": 32, "synthetic_export_test": True}, destination / "state.pt")
        (destination / "adapter-receipt.json").write_text('{"scope": "synthetic_export_test"}\n')
        return {"evidence": {"sampling_law": "generated_and_paired_reconstructed_prior_without_output_noise"}}, [record]
    def failed_renderer(*args, **kwargs):
        assert (output / "observations.npz").is_file()
        raise RuntimeError("injected renderer failure")
    monkeypatch.setattr(module, "run_word", completed_adapter)
    monkeypatch.setattr(module, "grade_result", lambda *args: {"gate_status": "INCOMPLETE"})
    monkeypatch.setattr(module, "render_gif", failed_renderer)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(path), "--output", str(output)])
    previous_handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(RuntimeError, match="injected renderer failure"):
        module.main()
    source = read_json(output / "source-manifest.json")
    assert source["files"]["reports/forge/five-word-joint/reproduce_demo.py"]
    assert torch.load(output / "state.pt", weights_only=False)["completed_steps"] == 32
    assert read_json(output / "adapter-receipt.json")["scope"] == "synthetic_export_test"
    with np.load(output / "observations.npz") as arrays:
        assert np.array_equal(arrays["step32_view0_target"], values.numpy())
    assert not (output / "goal.gif").exists()
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) == previous_handler
