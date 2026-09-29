"""Small CPU adapter contracts. Budgets here cannot qualify a real candidate."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.forge import adapters
from experiments.forge.api import CapabilityError
from experiments.forge.views import grade_result


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def task(name, steps=2):
    value = json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())
    value["execution"]["steps"] = steps
    return value


def request(value):
    return {"candidate": {"recipe_overrides": {"input_noise_std": .01, "output_noise_std": .02,
                         "output_noise_warmup": 0, "lr": .0001},
             "prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}},
            "candidate_revision": "unit-test", "protocol": {"seed": 0}, "tasks": {value["id"]: value}}


def execute(value, directory):
    return adapters.run_task(request(value), {"task_id": value["id"]}, directory, "cpu")


def test_vector_uses_public_mog_and_real_evaluator_with_isolated_rng(tmp_path):
    value = task("vector_two_broad")
    value["execution"]["host_definition"].update(hidden=8, layers=1, batch=4, particles=12)
    global_rng = torch.get_rng_state().clone()
    raw = execute(value, tmp_path)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert raw["execution_path"] == "public_trainer"
    assert raw["prior_mechanisms"]["a2"]["enabled"]
    expected_guards = {"all_finite": True, "hooks_exercised": True,
        "unintended_rng_deviations": 0,
        "optimizer_updates": {"generator": 2, "discriminator": 2, "prior": 2}}
    assert {key: raw["evidence"]["guards"][key] for key in expected_guards} == expected_guards
    from experiments.forge.mechanisms import mechanism_blockers
    assert mechanism_blockers(raw["evidence"]["guards"]["mechanism_audit"]) == []
    assert [p["step"] for p in raw["evidence"]["observations"]] == [1, 2]
    assert raw["evidence"]["live"]["sample_count"] == 4096
    assert raw["recipe"]["lr"] == .0001
    assert raw["recipe"]["lr"] != value["execution"]["host_definition"]["lr"]
    # A tiny unit run never fabricates the 24 frozen protocol observations.
    assert grade_result(value, raw)["gate_status"] == "INCOMPLETE"
    assert not (tmp_path / "state.pt").exists()


def test_image_explicit_cloud_keeps_public_output_noise_and_real_metric(tmp_path):
    value = task("img_intensity2")
    raw = execute(value, tmp_path)
    evidence = raw["evidence"]
    assert raw["prior"]["kind"] == "particle_cloud" and raw["prior"]["sigma"] == 0
    assert raw["prior"]["exception_reason"]
    assert evidence["sampling_law"] == "enumerated_centers_with_public_output_noise"
    assert evidence["guards"]["optimizer_updates"]["prior"] == 2
    assert evidence["guards"]["unintended_rng_deviations"] == 0
    assert "hq" in evidence["live"] and "modes" in evidence["live"]


def test_ring_first_window_and_extension_are_one_uninterrupted_state(tmp_path, monkeypatch):
    value = task("ring_hold", 10)
    value["execution"]["original_schedule_horizon"] = 2
    value["evaluation"].update(start_step=2, confirmation_checks=1, settling_budget=4,
                               hold_budget=2, extension_steps=2, max_total_steps=10)
    # Synthetic observation law isolates the lifecycle protocol from stochastic quality.
    import benchmarks.locked_shared.mode_hold as host
    monkeypatch.setattr(host, "diversity", lambda *args, **kwargs: {"modes": 8, "hq": 1.})
    frozen = request(value)
    raw = adapters.run_task(frozen, {"task_id": value["id"], "task_ids": ["ring_hold", "ring_extension"]}, tmp_path, "cpu")
    hold, extension = raw["task_results"].values()
    assert hold == extension
    assert hold["cost"]["completed_steps"] == 7
    assert hold["recipe"]["total_steps"] == 2
    assert [p["step"] for p in hold["evidence"]["dense"]] == [3, 4, 5, 6, 7]
    assert hold["evidence"]["continuity"]["mode"] == "uninterrupted"
    assert grade_result(value, hold)["gate_status"] == "PASS"
    extension_task = deepcopy(value)
    extension_task["evaluation"]["kind"] = "ring_extension"
    assert grade_result(extension_task, extension)["gate_status"] == "PASS"
    saved = torch.load(tmp_path / "state.pt", weights_only=False)
    assert saved["trainer"]["completed_steps"] == 7
    assert saved["trainer"]["max_steps"] == 10


def test_ring_failure_does_not_search_for_later_good_window(tmp_path, monkeypatch):
    value = task("ring_hold", 10)
    value["execution"]["original_schedule_horizon"] = 2
    value["evaluation"].update(start_step=2, confirmation_checks=1, settling_budget=4,
                               hold_budget=2, extension_steps=2, max_total_steps=10)
    import benchmarks.locked_shared.mode_hold as host
    points = iter(({"modes": 8, "hq": 1.}, {"modes": 2, "hq": .2}))
    monkeypatch.setattr(host, "diversity", lambda *args, **kwargs: next(points))
    raw = execute(value, tmp_path)
    assert raw["cost"]["completed_steps"] == 4
    assert grade_result(value, raw)["gate_status"] == "FAIL"


def test_native_writes_auditable_draws_without_short_run_certification(tmp_path):
    value = task("grid100", 2)
    value["evaluation"].update(eval_interval=1, early_eval_steps=[0, 1, 2], eval_samples=64)
    frozen = request(value)
    frozen["candidate"]["recipe_overrides"].update(num_particles=12, batch_size=4)
    raw = adapters.run_task(frozen, {"task_id": value["id"]}, tmp_path, "cpu")
    directory = Path(raw["evidence"]["artifact_root"]) / "grid100"
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert config == summary["config"]
    assert summary["eval_steps"] == [0, 1, 2]
    with np.load(directory / "final_samples.npz") as arrays:
        assert arrays["live"].shape == (64, 2)
    with np.load(directory / "holdout_samples.npz") as arrays:
        assert arrays["live"].shape == (100000, 2)
    assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert grade_result(value, raw)["gate_status"] != "PASS"


def test_unknown_adapter_blocks_before_constructing_or_updating(tmp_path):
    value = task("clockfree_audit")
    value["adapter"] = "not_implemented"
    with pytest.raises(CapabilityError, match="no public adapter"):
        execute(value, tmp_path)
    assert not list(tmp_path.iterdir())


def test_native_continuation_restores_own_prefix_and_matches_uninterrupted_state(tmp_path):
    from experiments.forge.continuation import verify_native_prefix
    from experiments.forge.state import state_digest
    prefix_task = task("grid100", 2)
    prefix_task["evaluation"].update(eval_interval=1, early_eval_steps=[0, 1, 2], eval_samples=64)
    frozen = request(prefix_task)
    frozen["candidate"]["recipe_overrides"].update(num_particles=12, batch_size=4)
    frozen["jobs"] = [{"task_id": "grid100", "compatibility_key": "fixture-prefix"}]
    prefix_raw = adapters.run_task(frozen, {"task_id": "grid100"}, tmp_path / "prefix", "cpu")
    continued = deepcopy(prefix_task)
    continued.update(id="grid100_14k", adapter="native100_continuation")
    continued["execution"].update(steps=4, original_schedule_horizon=2, preserve_prefix_steps=2,
                                 continuation_of="grid100", incremental_steps=2)
    frozen["tasks"][continued["id"]] = continued
    # The synthetic prerequisite lets this short test reach the adapter. Its
    # 64-sample/two-step run never claims to pass the production quality gate.
    dependency = {"attempt_id": "unit-prefix", "candidate_revision": frozen["candidate_revision"],
        "compatibility_key": "fixture-prefix", "result_hash": "fixture-only", "result": {
            **prefix_raw, "task_id": "grid100", "compatibility_key": "fixture-prefix", "gate_status": "PASS"}}
    result = adapters.run_task(frozen, {"task_id": continued["id"], "prerequisites": {"grid100": dependency}},
                               tmp_path / "continued", "cpu")
    verify_native_prefix(continued, result["evidence"])
    assert result["recipe"]["total_steps"] == 2
    assert result["cost"]["completed_steps"] == 4
    events_path = Path(result["evidence"]["artifact_root"]) / "grid100/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [row["elapsed"] for row in events] == sorted(row["elapsed"] for row in events)
    assert grade_result(continued, result)["gate_status"] != "PASS"
    full = deepcopy(continued)
    full["id"], full["adapter"] = "grid100", "native100"
    for field in ("preserve_prefix_steps", "continuation_of", "incremental_steps"):
        full["execution"].pop(field)
    frozen["tasks"]["grid100"] = full
    uninterrupted = adapters.run_task(frozen, {"task_id": "grid100"}, tmp_path / "full", "cpu")
    assert result["evidence"]["checkpoint"]["state_sha256"] == uninterrupted["evidence"]["checkpoint"]["state_sha256"]
    from experiments.forge.artifacts import manifest_artifacts
    from experiments.forge.contracts import file_hash
    artifact_root = Path(result["evidence"]["artifact_root"])
    final_path = artifact_root / result["evidence"]["checkpoint"]["path"]
    original_bytes = final_path.read_bytes()
    final_state = torch.load(final_path, weights_only=True)
    for field in ("trainer_recipe", "extensions", "rng_seed", "duplicated_rng", "missing_rng"):
        changed = deepcopy(final_state)
        if field == "trainer_recipe":
            changed["trainer"]["recipe"]["lr"] *= 2
        elif field == "extensions":
            changed["extensions"]["undeclared"] = True
        elif field == "rng_seed":
            changed["streams"]["manifest"]["seed"] = 1
        elif field == "missing_rng":
            changed["trainer"]["streams"] = {}
        else:
            changed["trainer"]["streams"]["latent_generator"] = changed["trainer"]["streams"]["latent_generator"].roll(1)
        torch.save(changed, final_path)
        forged = deepcopy(result["evidence"])
        forged["checkpoint"].update(sha256=file_hash(final_path), state_sha256=state_digest(changed))
        forged["artifact_manifest"] = manifest_artifacts(artifact_root)
        with pytest.raises(ValueError, match="static|RNG"):
            verify_native_prefix(continued, forged)
    final_path.write_bytes(original_bytes)
    source = Path(result["evidence"]["artifact_root"]) / "resume-restored.pt"
    changed = torch.load(source, weights_only=True)
    changed["trainer"]["completed_steps"] = 1
    torch.save(changed, source)
    with pytest.raises(ValueError, match="exact prefix"):
        verify_native_prefix(continued, result["evidence"])


def test_behavior_dispatch_uses_the_shared_component_adapter(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    behavior = ModuleType("experiments.forge.behavior_adapters")
    value = task("two_pole")
    sentinel = {"evidence": {"from": "components"}}
    behavior.run_behavior = lambda req, selected, output, device: sentinel
    monkeypatch.setitem(sys.modules, behavior.__name__, behavior)
    assert execute(value, tmp_path) is sentinel
