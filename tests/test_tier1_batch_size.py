"""CUDA checks of batch-only binding, real-example coupling and retention gates."""
from copy import deepcopy

import pytest
import torch

from benchmarks.toy_audit import tier1_batch_size as study

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="batch study requires CUDA")


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_batch_change_matches_archived_initial_models_and_streams(task_id):
    protocol = study.declaration()
    for batch in (128, 512):
        context, trainer, _ = study.build(task_id, batch, "cuda:0")
        assert study.verify_initial(context, protocol["tasks"][task_id])["matched"]
        assert trainer.recipe.batch_size == batch
        assert trainer.recipe.total_steps == protocol["tasks"][task_id]["original_schedule_horizon"]
        assert trainer.recipe.lr_floor == trainer.recipe.network_lr_floor == 1.
        assert trainer.prior.z.device.type == "cuda"


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_larger_batch_preserves_flat_real_example_stream(task_id):
    context, _, task = study.build(task_id, 512, "cuda:0")
    target, _ = study.scorer(task_id)
    a = context.streams.generator("data", component="target", purpose="training", device="cpu")
    b = torch.Generator(device="cpu"); b.set_state(a.get_state())
    large, _ = study.grouped_real(target, task["execution"]["host_definition"], 512, a, 0)
    small = torch.cat([study.grouped_real(target, task["execution"]["host_definition"], 128, b, i)[0]
                       for i in range(4)])
    assert torch.equal(large.to("cuda:0"), small.to("cuda:0"))
    assert torch.equal(a.get_state(), b.get_state())


def test_a_failed_hold_check_cannot_be_replaced_by_a_good_endpoint():
    protocol = study.declaration()
    context, _, task = study.build("gaussian1d_acquisition", 512, "cuda:0")
    target, score = study.scorer("gaussian1d_acquisition")
    stream = context.streams.generator("data", component="target", purpose="oracle", device="cpu")
    metrics = score(target(task["execution"]["host_definition"], 4096, stream, 0).to("cuda:0"),
                    task["execution"]["host_definition"], 0)
    rows = [dict(step=s, full_pass=True, metrics=metrics) for s in study.checkpoints("gaussian1d_acquisition", protocol)]
    assert study.summarize(rows, "gaussian1d_acquisition", protocol)["combined_verdict"] == "PASS"
    bad = deepcopy(rows)
    bad[30]["full_pass"] = False
    summary = study.summarize(bad, "gaussian1d_acquisition", protocol)
    assert summary["acquisition_verdict"] == "PASS"
    assert summary["final_terminal_suffix"] >= 5
    assert summary["combined_verdict"] == summary["hold_verdict"] == "FAIL"
    with pytest.raises(ValueError, match="missing"):
        study.summarize(rows[:-1], "gaussian1d_acquisition", protocol)


def test_cpu_fallback_rejected_before_output_creation(tmp_path):
    path = tmp_path / "not-created"
    with pytest.raises(ValueError, match="requires CUDA"):
        study.execute(path, device="cpu")
    assert not path.exists()
