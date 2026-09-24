import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import ReachRecorder
from reports.toy100.pr84_zero_mean_outputs import ZeroMeanReachRecorder, zero_mean_g_outputs
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mode_hold import ModeHoldRecipe


def test_zero_mean_removes_batch_mean_and_passthrough_is_identity():
    recorder = ZeroMeanReachRecorder(start_step=0)
    batch = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    centered = recorder.zero_mean(batch)
    assert torch.allclose(centered.mean(0), torch.zeros(2))
    assert torch.allclose(centered - centered.mean(0), batch - batch.mean(0))
    recorder.passthrough = True
    assert recorder.zero_mean(batch) is batch
    recorder.passthrough = False
    recorder.enabled = False
    assert recorder.zero_mean(batch) is batch


def test_injected_train_forward_centers_and_stall_reach_stays():
    with zero_mean_g_outputs(task="mode_hold") as (recorder, source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall"
        assert recorder.reach == .5
        assert source.count("_extra_state.zero_mean(fake)") == 2
        seen = []
        original = recorder.zero_mean

        def watch(fake):
            out = original(fake)
            seen.append(float(out.detach().mean(0).abs().max()))
            return out

        recorder.zero_mean = watch
        mode_hold.train_mode_hold(ModeHoldRecipe(steps=1), seed=0)
        assert recorder.centered_batches == 6
        assert seen and max(seen) < 1e-5
    with zero_mean_g_outputs(task="trajectory") as (_recorder, source):
        assert source.count("_extra_state.zero_mean(fake)") == 2
    assert base.SmoothedBothBoundRecorder is not ZeroMeanReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ZeroMeanReachRecorder)
