"""CUDA evidence checks for the standalone prior study, without training."""
from copy import deepcopy
import json

import pytest
import torch

from benchmarks.toy_audit import tier1_prior_smoke as study
from experiments.forge.api import task_formulation_context
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="prior study requires CUDA")


def test_oracles_and_shape_impostors_separate_full_and_smoke_questions():
    result = study.controls(device="cuda:0")
    assert result["passed"]
    assert result["training_updates"] == 0
    assert all(row["input_device"] == "cuda:0" for row in result["controls"])
    impostors = [row for row in result["controls"] if row["control"] in ("all_mode_atoms", "same_moment_uniform")]
    assert len(impostors) == 2
    assert all(row["smoke_pass"] and not row["full_pass"] for row in impostors)


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_prior_grid_preserves_networks_and_changes_actual_prior(task_id):
    protocol = study.declaration()
    candidate = json.loads((study.ROOT / protocol["candidate_path"]).read_text())
    original = json.loads((study.ROOT / protocol["tasks"][task_id]["path"]).read_text())
    network_hashes, locations = set(), {}
    for condition in protocol["arms"].values():
        task = deepcopy(original)
        task["execution"]["prior"] = condition["prior"]
        task["execution"]["host_definition"]["particles"] = condition["particles"]
        task["requires_capabilities"] = ["checkpoint", "named_rng", "live_sampling", "learned_locations"]
        context = task_formulation_context(candidate, task, {"seed": 0}, device="cuda:0", root=study.ROOT)
        g, d = build_vector_models(context, task["execution"]["host_definition"])
        trainer = context.build_trainer(g, d, max_steps=task["execution"]["steps"])
        assert trainer.prior.z.device.type == "cuda"
        assert trainer.prior.z.shape == (condition["particles"], original["execution"]["host_definition"]["z_dim"])
        expected = "ParticlePrior" if condition["prior"]["kind"] == "particle_cloud" else "MoGParticlePrior"
        assert type(trainer.prior).__name__ == expected
        if expected == "MoGParticlePrior":
            assert float(trainer.prior.sigma) == pytest.approx(condition["prior"]["sigma"])
        assert trainer.recipe.batch_size == 128
        assert trainer.recipe.lr == .012
        assert trainer.recipe.optimizer_family == "dualnorm"
        network_hashes.add((state_digest(g.state_dict()), state_digest(d.state_dict())))
        locations.setdefault(condition["particles"], set()).add(state_digest(trainer.prior.z))
    assert len(network_hashes) == 1
    assert all(len(values) == 1 for values in locations.values())


def test_changed_task_is_rejected_before_spend(tmp_path, monkeypatch):
    protocol = study.declaration()
    protocol["tasks"]["ring16_acquisition"]["sha256"] = "0" * 64
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(study, "PROTOCOL", path)
    with pytest.raises(ValueError, match="task changed"):
        study.declaration()
