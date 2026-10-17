"""Image data-law parity with the reference host, without any optimizer update."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.transfer_suite import image_tasks, toy100_compatibility
from experiments.forge import adapters
from experiments.forge.rng import NamedStreams
from particlegan import GANTrainer


ROOT = Path(__file__).resolve().parents[1]


class BatchCaptured(Exception):
    pass


@pytest.mark.parametrize("task_id", ["img_intensity2", "img_bars4"])
def test_image_training_data_matches_reference_clamped_noise_law(task_id, tmp_path, monkeypatch):
    task = json.loads((ROOT / "configs/forge/tasks" / f"{task_id}.json").read_text())
    spec = next(value for value in image_tasks.TASKS if value["name"] == task_id)
    for key in ("pattern", "modes", "batch_size", "noise_std"):
        assert task["execution"]["host_definition"][key] == spec[key]
    centers = image_tasks.templates(spec)
    stream = NamedStreams(0).generator("data", component="target", purpose="training")
    initial_state = stream.get_state().clone()
    captured = {}

    def capture_reference(real, **kwargs):
        captured["reference"] = real.clone()
        captured["reference_rng"] = torch.get_rng_state().clone()
        raise BatchCaptured

    # Execute the real compatibility host's acquisition code. Its training
    # context is a stub solely so the test cannot update a model or optimizer.
    reference_trainer = SimpleNamespace(completed_steps=0, step=capture_reference)
    monkeypatch.setattr(toy100_compatibility, "setup_image",
                        lambda *args: {"trainer": reference_trainer, "centers": centers})
    noise = {"input_noise_std": 0., "input_noise_anneal_end": 1.,
             "output_noise_std": 0., "output_noise_warmup": 0.}
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(initial_state)
        with pytest.raises(BatchCaptured):
            toy100_compatibility.run_image(spec, {}, noise)

    def forbid_update(*args, **kwargs):
        pytest.fail("data-law parity must stop before any training update")

    monkeypatch.setattr(GANTrainer, "step", forbid_update)
    monkeypatch.setattr(torch.optim.Adam, "step", forbid_update)

    def capture_forge(run, real):
        captured["forge"] = real.clone()
        data = run.context.streams.generator("data", component="target", purpose="training")
        captured["forge_rng"] = data.get_state().clone()
        assert run.trainer.completed_steps == 0
        raise BatchCaptured

    monkeypatch.setattr(adapters._Run, "step", capture_forge)
    request = {"candidate": {}, "protocol": {"seed": 0}, "tasks": {task_id: task}}
    old_threads = torch.get_num_threads()
    global_rng = torch.get_rng_state().clone()
    try:
        torch.set_num_threads(1)
        with pytest.raises(BatchCaptured):
            adapters.run_task(request, {"task_id": task_id}, tmp_path, "cpu")
    finally:
        torch.set_num_threads(old_threads)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(captured["forge_rng"], captured["reference_rng"])
    assert torch.equal(captured["forge"], captured["reference"])
    assert captured["forge"].min() == 0
    assert captured["forge"].max() <= 1
    if task_id == "img_bars4":
        assert captured["forge"].max() == 1
    # Ensure this fixture exposes the omitted-clamp regression, rather than
    # coincidentally drawing only noise within the legal image range.
    indices = torch.randint(len(centers), (spec["batch_size"],), generator=stream)
    unclamped = centers[indices] + spec["noise_std"] * torch.randn(
        centers[indices].shape, generator=stream)
    assert unclamped.min() < 0
    if task_id == "img_bars4":
        assert unclamped.max() > 1
    assert not (tmp_path / "adapter-receipt.json").exists()
