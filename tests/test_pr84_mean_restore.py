"""Post-step G output-bias restore: the loss forward stays uncentered."""

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator
from benchmarks.locked_shared.mode_hold import BATCH, ModeHoldRecipe
from reports.toy100.pr84_mean_restore_candidate import (
    MeanRestoreRecorder, pr84_mean_restore_candidate, restore_output_mean,
)
from reports.toy100.pr84_reach_candidate import ReachRecorder


def test_bias_restore_matches_pre_step_mean_on_the_same_batch():
    generator = SimpleMLPGenerator(4, 16, 1, 2)
    batch = torch.randn(32, 4)
    pre = generator(batch).mean(0).detach()
    with torch.no_grad():
        generator.net[-1].weight.add_(0.15)
        generator.net[-1].bias.add_(torch.tensor([0.4, -0.25]))
    post = generator(batch).mean(0)
    assert not torch.allclose(pre, post, atol=1e-3)
    bias = generator.net[-1].bias
    delta, fixed = restore_output_mean(generator, bias, (batch.detach(),), pre)
    assert torch.allclose(fixed, pre, atol=1e-5)
    assert torch.allclose(delta, post - pre, atol=1e-5)
    assert torch.allclose(generator(batch).mean(0), pre, atol=1e-5)


def test_hook_records_the_raw_forward_and_does_not_center_it():
    generator = SimpleMLPGenerator(4, 16, 1, 2)
    with torch.no_grad():
        generator.net[-1].bias.copy_(torch.tensor([1.5, -0.7]))
    recorder = MeanRestoreRecorder(start_step=0)
    recorder.phase = 1
    recorder.passthrough = False
    recorder._ensure_hook(generator)
    batch = torch.randn(64, 4)
    output = generator(batch)
    assert recorder._last is not None
    assert torch.allclose(recorder._last[1], output.mean(0))
    assert recorder._last[1][0] > 0.5
    assert recorder._last[1][1] < -0.2


def test_factory_is_stall_reach_and_a_short_run_restores_the_training_batch():
    recipe = ModeHoldRecipe(steps=2, n_particles=12)
    with pr84_mean_restore_candidate(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, MeanRestoreRecorder)
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall" and recorder.reach == 0.5 and not recorder.game_bound
        mode_hold.train_mode_hold(recipe)
    assert recorder.outer_steps == 2
    assert len(recorder.records) == 2
    assert all(row["mean_restore_batch"] == BATCH for row in recorder.records)
    assert all(row["mean_restore_residual"] < 1e-4 for row in recorder.records)
    assert max(row["mean_restore_delta"] for row in recorder.records) > 1e-8
    stats = recorder.receipt()
    assert stats["mean_restore_n"] == 2 and stats["mean_restore_fires"] == 2
    assert stats["ramp"] == "stall"
