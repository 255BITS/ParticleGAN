"""Six-update public-trainer fixtures prove plumbing, never full adaptation quality."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.adaptation import run_adaptation, verify_pair_artifacts, _summary
from experiments.forge.api import CapabilityError
from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import atomic_json
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup():
    task = json.loads((ROOT / "configs/forge/tasks/target_shift_recovery.json").read_text())
    task["execution"].update(steps=6, control_steps=6, shift_step=4, diagnostic_every=2)
    task["execution"]["host_definition"].update(hidden=8, layers=1, batch=4, eval_samples=64)
    request = {"candidate": {"recipe_overrides": {"lr": .0001, "input_noise_std": .01, "output_noise_std": .02},
                             "prior": task["execution"]["prior"]},
               "candidate_revision": "explicit-short-unit-fixture", "protocol": {"seed": 0}}
    return request, task


@pytest.fixture
def pair(tmp_path):
    request, task = setup()
    return task, run_adaptation(request, task, tmp_path, "cpu")


def test_public_checkpoint_pairs_prefix_and_freezes_every_training_state(pair):
    task, raw = pair
    evidence = raw["evidence"]
    proof = verify_pair_artifacts(evidence)
    assert proof["prefix_step"] == proof["frozen_final_step"] == 4
    assert proof["active_final_step"] == 6
    active, frozen = evidence["active"], evidence["frozen"]
    assert active["diagnostic"][:2] == frozen["diagnostic"][:2]
    assert [p["step"] for p in active["diagnostic"]] == [2, 4, 6]
    assert active["shift_pair"] == frozen["shift_pair"]
    assert [r["updates"] for r in active["optimizer_final"]] == [6, 6]
    assert [r["updates"] for r in frozen["optimizer_final"]] == [4, 4]
    assert active["noise_horizon"] == frozen["noise_horizon"] == 1200
    assert raw["execution_path"] == "public_trainer"
    assert evidence["guards"]["optimizer_updates"] == {"generator": 6, "discriminator": 6, "prior": 6}
    assert evidence["guards"]["unintended_rng_deviations"] == 0
    assert grade_result(task, raw)["gate_status"] == "INVALID"


def test_sampling_consumes_matched_eval_streams_without_changing_global_rng(tmp_path):
    request, task = setup()
    global_before = torch.get_rng_state().clone()
    raw = run_adaptation(request, task, tmp_path, "cpu")
    assert torch.equal(torch.get_rng_state(), global_before)
    root = Path(raw["evidence"]["artifact_root"])
    active = torch.load(root / "active-final.pt", weights_only=True)
    frozen = torch.load(root / "frozen-final.pt", weights_only=True)
    bindings = active["streams"]["manifest"]["bindings"]
    for key, binding in bindings.items():
        if binding["family"] == "eval":
            assert torch.equal(active["streams"]["states"][key], frozen["streams"]["states"][key])
    assert state_digest(active["streams"]) != state_digest(frozen["streams"])


def test_mutated_frozen_model_is_detected_even_after_manifest_refresh(pair):
    _, raw = pair
    evidence = raw["evidence"]
    root = Path(evidence["artifact_root"])
    state = torch.load(root / "frozen-final.pt", weights_only=True)
    parameter = next(iter(state["trainer"]["models"]["G"].values()))
    parameter.add_(.1)
    torch.save(state, root / "frozen-final.pt")
    evidence["artifact_manifest"] = manifest_artifacts(root)
    with pytest.raises(ValueError, match="frozen training state changed"):
        verify_pair_artifacts(evidence)


def test_forged_optimizer_counter_cannot_replace_measured_checkpoint(pair):
    _, raw = pair
    evidence = raw["evidence"]
    root = Path(evidence["artifact_root"])
    evidence["frozen"]["optimizer_final"][0]["updates"] = 6
    atomic_json(root / "frozen.json", evidence["frozen"])
    evidence["artifact_manifest"] = manifest_artifacts(root)
    with pytest.raises(ValueError, match="measured final optimizer counters"):
        verify_pair_artifacts(evidence)


def test_saved_shift_samples_bind_before_after_target_metrics(pair):
    _, raw = pair
    evidence = raw["evidence"]
    root = Path(evidence["artifact_root"])
    for arm in ("active", "frozen"):
        evidence[arm]["shift_pair"]["after"]["hq"] = 1.
        atomic_json(root / (arm + ".json"), evidence[arm])
    evidence["artifact_manifest"] = manifest_artifacts(root)
    with pytest.raises(ValueError, match="saved live samples"):
        verify_pair_artifacts(evidence)


@pytest.mark.parametrize("field,value", [("original_schedule_horizon", 3600), ("shift", [0., 0.]),
                                          ("control_steps", 5), ("frozen_control", False)])
def test_incompatible_protocol_blocks_before_training(tmp_path, field, value):
    request, task = setup()
    task["execution"][field] = value
    with pytest.raises(CapabilityError):
        run_adaptation(request, task, tmp_path, "cpu")
    assert not list(tmp_path.iterdir())


def test_source_mismatch_blocks_before_training(tmp_path):
    request, task = setup()
    request["source"] = {"files": {"experiments/forge/adaptation.py": "0" * 64}}
    with pytest.raises(CapabilityError, match="executed source differs"):
        run_adaptation(request, task, tmp_path, "cpu")
    assert not list(tmp_path.iterdir())


def test_frozen_pass_is_not_a_valid_negative_control():
    # Explicit evaluator fixture, with no training and no scientific card.
    from benchmarks.toy100.continuous_probe import match_frozen_control
    execution = {"original_schedule_horizon": 1200, "shift_step": 2400}
    points = [{"step": s, "modes": 8, "hq": 1.} for s in range(10, 3601, 10)]
    common = {"mode": "scheduled", "config_sha256": "fixture", "source_sha256": {"fixture": "fixture"},
              "runtime": "fixture", "steps": 3600, "noise_horizon": 1200, "diagnostic_every": 10,
              "dense_after": None, "dense_until": None, "shift_step": 2400, "shift": [1., 0.],
              "shift_pair": {}, "diagnostic": points, **_summary(points, execution)}
    active = {**deepcopy(common), "freeze_after_shift": False, "optimizer_final": [{"updates": 3600}]}
    frozen = {**deepcopy(common), "freeze_after_shift": True, "optimizer_final": [{"updates": 2400}]}
    result = match_frozen_control(active, frozen)
    assert result["status"] == "FAIL"
    assert result["matched_control"]["sensitivity_pass"] is False
    for point in frozen["diagnostic"]:
        if point["step"] > 2400:
            point.update(modes=0, hq=0.)
    frozen.update(_summary(frozen["diagnostic"], execution))
    assert match_frozen_control(active, frozen)["status"] == "PASS"
