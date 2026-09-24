"""Host timing and accounting checks for the sampled-real Chamfer adapter."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def _mode_host(context, *, steps=3):
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.pr84_smoothed_parity import _state_sha

    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200, output_noise_rng="isolated")
    with context as (recorder, _):
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps),
                                           noise_policy=policy, diagnostics=True)
    return result, _state_sha(recorder), policy.receipt(), recorder


def test_disabled_chamfer_correction_matches_pr84_full_state_and_noise():
    from reports.toy100.chamfer_smoothed_candidate import chamfer_smoothed_candidate
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate

    baseline = _mode_host(pr84_smoothed_candidate())
    observed = _mode_host(chamfer_smoothed_candidate(correction=False))
    assert baseline[:3] == observed[:3]
    assert observed[-1].batch_replays_verified == 6
    assert observed[-1].chamfer_records == []


def test_conditional_trajectory_is_exact_pr84_and_skips_marginal_chamfer():
    from benchmarks.locked_shared import trajectory
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.chamfer_smoothed_candidate import chamfer_smoothed_candidate
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.pr84_smoothed_parity import _state_sha

    def run(context):
        policy = NoisePolicy(.029, .5, .1, 400, output_noise_rng="isolated")
        with patch.dict(trajectory.PROTOCOL, {"steps": 3}):
            with context as (recorder, _):
                result = trajectory.train(noise_policy=policy, diagnostics=True)
        return result, _state_sha(recorder), policy.receipt(), recorder

    baseline = run(pr84_smoothed_candidate(task="trajectory"))
    candidate = run(chamfer_smoothed_candidate(task="trajectory"))
    assert baseline[:3] == candidate[:3]
    assert candidate[-1].chamfer_records == []
    assert candidate[-1].batch_replays_verified == 0


def test_active_chamfer_uses_current_first_real_batch_after_g_before_ema():
    from reports.toy100 import chamfer_smoothed_candidate as module
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate

    baseline = _mode_host(pr84_smoothed_candidate(), steps=1)
    seen = []
    original = module.ChamferSmoothedRecorder._correct

    def audited(self, real, opt_g):
        assert self.phase == 2 and self.outer_steps == 0
        assert self.rows[opt_g]["calls"] == 3
        assert all(int(opt_g.state[p]["step"]) == 1
                   for group in opt_g.param_groups for p in group["params"])
        assert real is self.real_samples[0]
        assert torch.equal(real, self.phase_zero_samples[0])
        assert not torch.equal(real, self.real_samples[1])
        ema_before = self._local["ema_z"].clone()
        original(self, real, opt_g)
        prior = self._local["prior"]
        expected_ema = ema_before.mul(self._local["recipe"].ema).add(prior.z.detach(),
                                                                      alpha=1 - self._local["recipe"].ema)
        seen.append(expected_ema)

    with patch.object(module.ChamferSmoothedRecorder, "_correct", audited):
        active = _mode_host(module.chamfer_smoothed_candidate(), steps=1)
    recorder = active[-1]
    assert len(seen) == len(recorder.chamfer_records) == 1
    assert torch.equal(recorder._local["ema_z"], seen[0])
    assert baseline[2] == active[2]
    assert recorder.rng_replay_verified == recorder.batch_replays_verified == 2
    info = recorder.receipt()
    assert info["method"] == module.METHOD
    assert info["total_objective"] == "S=C+Q, both coefficients exactly one"
    assert info["helper_sha256"] == hashlib.sha256(
        Path(module.__file__).with_name("chamfer_pullback.py").read_bytes()).hexdigest()
    for optimizer, row in recorder.rows.items():
        assert row["calls"] == 3
        assert all(int(optimizer.state[p]["step"]) == 1
                   for group in optimizer.param_groups for p in group["params"])
    receipt = recorder.chamfer_records[0]
    assert receipt["objective_before"] == pytest.approx(
        receipt["coverage_before"] + receipt["backward_before"])
    assert receipt["objective_after"] == pytest.approx(
        receipt["coverage_after"] + receipt["backward_after"])
    assert receipt["outer_step"] == 1
