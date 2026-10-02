"""Verify real public updates, isolation and honest incomplete-run grading."""
import json
import random

import numpy as np
from PIL import Image
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from benchmarks.toy_audit import api_run


class PublicSoftwareFixture:
    """API software control, with no claim of learned distribution quality."""
    api_components = ("particlegan.GANTrainer", "particlegan.Recipe")

    def __init__(self):
        self.recipe = get_recipe("k3p", z_dim=2, num_particles=16, batch_size=16)
        torch.manual_seed(5)
        self.trainer = GANTrainer(self.recipe, nn.Linear(2, 2), nn.Linear(2, 1),
                                  seed=7, max_steps=8)
        self.data = torch.Generator().manual_seed(9)

    def step(self):
        self.trainer.step(torch.randn(16, 2, generator=self.data))

    def observe(self, n, seed):
        samples = self.trainer.sample(n, generator=torch.Generator().manual_seed(seed))
        fraction = float(torch.isfinite(samples).float().mean())
        return {"metrics": {"finite_fraction": fraction}, "passed": fraction == 1.,
                "failed_bounds": [] if fraction == 1. else ["finite_fraction < 1"],
                "views": [{"kind": "scatter", "title": "API software control: finite output",
                           "target": torch.zeros(n, 2), "samples": samples}]}

    def state_dict(self):
        return self.trainer.state_dict()


def test_real_api_prefix_cannot_qualify_a_larger_default_budget(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    fixture = PublicSoftwareFixture()
    before = fixture.trainer.G.weight.detach().clone()
    monkeypatch.setattr(api_run.contract, "build", lambda *args, **kwargs: fixture)
    case = {"id": "software-control", "goal": "Execute finite public GAN updates",
            "default_steps": 8, "eval_samples": 32, "legacy_ids": ["software-control"]}
    result = api_run.run_case(case, tmp_path / "run", steps=6, eval_samples=32, frames=7)
    assert result["status"] == "COMPLETE"
    assert result["completed_updates"] == fixture.trainer.completed_steps == 6
    assert not torch.equal(before, fixture.trainer.G.weight)
    assert result["metric_passed"] and result["sustained_metric_passed"]
    assert result["verdict"] == "FAIL" and not result["default_protocol_complete"]
    assert "default budget" in result["failed_bounds"][0]
    with Image.open(tmp_path / "run/goal.gif") as gif:
        assert gif.n_frames == 7
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["artifacts"]["goal.gif"]["sha256"] == api_run.file_hash(tmp_path / "run/goal.gif")


def test_eval_isolation_preserves_all_ambient_rngs():
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state().clone()
    with api_run.isolated_evaluation():
        random.random()
        np.random.randn(4)
        torch.randn(4)
    assert random.getstate() == python
    restored = np.random.get_state()
    assert numpy[0] == restored[0] and np.array_equal(numpy[1], restored[1])
    assert numpy[2:] == restored[2:]
    assert torch.equal(cpu, torch.get_rng_state())


def test_nonfinite_metrics_are_preserved_as_explicit_json_failures(tmp_path):
    path = tmp_path / "failure.json"
    api_run.write_json(path, {"passed": False, "rmse": float("nan")})
    assert json.loads(path.read_text()) == {"passed": False, "rmse": "nan"}
