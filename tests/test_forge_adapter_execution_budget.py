"""Separate task execution allowances from original recipe schedule horizons."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters
from experiments.forge.api import FormulationContext
from experiments.forge.contracts import read_json


ROOT = Path(__file__).resolve().parents[1]


def request(task):
    return {"candidate": {"recipe_overrides": {"lr": .0001, "input_noise_std": 0.,
            "output_noise_std": 0.}}, "candidate_revision": "software-budget-fixture",
            "protocol": {"seed": 0}, "tasks": {task["id"]: task}}


@pytest.mark.parametrize("name,adapter,module", [
    ("ring16_acquisition", adapters._vector, "experiments.forge.vectorprofiles.build_vector_models"),
    ("img_bars4", adapters._image, "experiments.forge.imageprofiles.build_image_models"),
])
def test_metadata_passes_task_budget_1600_without_changing_schedule_400(tmp_path, monkeypatch, name, adapter, module):
    task = read_json(ROOT / "configs/forge/tasks" / f"{name}.json")
    task["execution"].update(steps=1600, original_schedule_horizon=400)
    task["execution"]["host_definition"]["steps"] = 1600
    initial = deepcopy(task)
    monkeypatch.setattr(module, lambda *args: (object(), object()))
    class ConstructionBoundary(Exception):
        pass
    def capture(context, generator, discriminator, *, max_steps=None):
        assert context.recipe.total_steps == 400
        assert max_steps == 1600
        raise ConstructionBoundary
    monkeypatch.setattr(FormulationContext, "build_trainer", capture)
    with pytest.raises(ConstructionBoundary):
        adapter(request(task), task, tmp_path, "cpu")
    assert task == initial
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="neural execution-budget fixtures require CUDA")
@pytest.mark.parametrize("name", ["vector_two_broad", "img_bars4"])
def test_cuda_adapter_finishes_four_updates_beyond_two_step_schedule(tmp_path, name):
    device = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
    assert torch.device(device).type == "cuda"
    task = read_json(ROOT / "configs/forge/tasks" / f"{name}.json")
    task["execution"].update(steps=4, original_schedule_horizon=2, produces_state=True)
    task["execution"]["host_definition"].update(steps=4, particles=8)
    if name == "vector_two_broad":
        task["execution"]["host_definition"].update(hidden=4, layers=1, batch=4, fourier=0)
    else:
        task["execution"]["host_definition"].update(width=2, batch_size=4)
    task["evaluation"]["observations"] = 2
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = adapters.run_task(request(task), {"task_id": task["id"]}, tmp_path, device)
    finally:
        torch.set_num_threads(old_threads)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    assert raw["recipe"]["total_steps"] == 2
    assert raw["cost"]["completed_steps"] == 4
    assert raw["evidence"]["guards"]["optimizer_updates"] == {
        "generator": 4, "discriminator": 4, "prior": 4}
    assert raw["evidence"]["guards"]["all_finite"] is True
    assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
    state = torch.load(tmp_path / "state.pt", weights_only=True, map_location=device)
    assert state["trainer"]["max_steps"] == 4
    assert state["trainer"]["recipe"]["total_steps"] == 2
    assert state["trainer"]["completed_steps"] == 4
    assert all(value.device.type == "cuda" for value in state["trainer"]["models"]["G"].values())
    assert [row["step"] for row in raw["evidence"]["observations"]] == [2, 4]
