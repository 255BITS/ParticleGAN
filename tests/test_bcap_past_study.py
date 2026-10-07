"""The diagnostic preserves matched conditions and rejects incomplete holds."""
from copy import deepcopy

import pytest
import torch

from benchmarks.toy_audit import bcap_past_extrapolation as study
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.api import CapabilityError, task_formulation_context
from experiments.forge.state import state_digest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no CPU fallback")


@reproducible_execution
def matched_probe(task_id, *, device):
    protocol = study.declaration()
    states = []
    for arm in protocol["candidates"]:
        context, trainer, _ = study.build(arm, task_id, device)
        assert study.initial_proof(context, task_id, protocol)["matched"]
        assert trainer.prior.z.device.type == "cuda"
        assert trainer.recipe.batch_size == 128
        assert trainer.recipe.lr_floor == trainer.recipe.network_lr_floor == 1.
        states.append(context.state_dict()["trainer"])
    for key in ("models", "streams", "initial_lrs", "optimizers"):
        assert len({state_digest(state[key]) for state in states}) == 1


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_only_game_update_changes_initial_conditions(task_id):
    matched_probe(task_id, device="cuda:0")


def test_strict_hold_and_shift_budget_cannot_be_shortened():
    protocol = study.declaration()
    context, _, task = study.build("simultaneous", "gaussian1d_acquisition", "cuda:0")
    target, score = study.scorer("gaussian1d_acquisition")
    spec = task["execution"]["host_definition"]
    rng = context.streams.generator("eval", component="oracle", purpose="controls", device="cpu")
    points = target(spec, 4096, rng, 0).to("cuda:0")
    metrics = score(points, spec, 0)
    rows = [dict(step=s, full_pass=True, metrics=metrics) for s in study.checkpoints("gaussian1d_acquisition", "stationary", protocol)]
    assert study.summarize(rows, "gaussian1d_acquisition", "stationary", protocol)["combined_verdict"] == "PASS"
    bad = deepcopy(rows)
    bad[30]["full_pass"] = False
    assert study.summarize(bad, "gaussian1d_acquisition", "stationary", protocol)["hold_verdict"] == "FAIL"
    with pytest.raises(ValueError, match="missing"):
        study.summarize(rows[:-1], "gaussian1d_acquisition", "stationary", protocol)
    assert len(study.checkpoints("gaussian1d_acquisition", "shift", protocol)) == 48


def test_cpu_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        study.execute(tmp_path / "absent", device="cpu")
    assert not (tmp_path / "absent").exists()


def test_component_host_cannot_silently_ignore_joint_updates():
    import json
    protocol = study.declaration()
    candidate = json.loads((study.ROOT / protocol["candidates"]["extrapolation_from_past"]).read_text())
    task = json.loads((study.ROOT / protocol["tasks"]["gaussian1d_acquisition"]["path"]).read_text())
    task["adapter"] = "transfer_behavior"
    with pytest.raises(CapabilityError, match="GANTrainer"):
        task_formulation_context(candidate, task, {"seed": 0}, device="cuda:0", root=study.ROOT)
