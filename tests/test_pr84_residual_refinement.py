"""Short, fixed-source checks for the residual_student refinement binding."""

import hashlib
import sys
from unittest.mock import patch

import pytest
import torch

from benchmarks.locked_shared.hosts import residual_student
from benchmarks.locked_shared.observation import recording
from benchmarks.toy100.warm_equilibrium_probe import _feed_hash
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100 import pr84_residual_refinement as binding
from reports.toy100 import pr84_smoothed_candidate as frozen


def _state(local, policy):
    digest = hashlib.sha256()
    _feed_hash(digest, dict(
        head=local["head"].state_dict(), critic=local["critic"].state_dict(),
        prior=local["prior"].state_dict(), opt_g=local["opt_g"].state_dict(),
        opt_d=local["opt_d"].state_dict(), torch_rng=torch.get_rng_state(),
        input_rng=policy.input_stream.get_state(),
        output_rng=None if policy.output_stream is None else policy.output_stream.get_state(),
        input_sigma=policy.input_sigma, output_sigma=policy.output_sigma))
    return digest.hexdigest()


def _short_host(context=None, *, isolated=False, observer=False):
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 400, output_noise_warmup=.2,
                         output_noise_rng="isolated" if isolated else None)
    captured = {}

    def profile(frame, event, arg):
        if (event == "return" and frame.f_code.co_name == "train"
                and frame.f_globals.get("__name__") == residual_student.__name__):
            captured.update(frame.f_locals)

    with patch.dict(residual_student.PROTOCOL, {
            "steps": 2, "lr": .00425, "particle_l2": 0.,
            "vicreg_weight": 0., "cover_weight": 1.5,
    }), recording(2, schedule="cosine", start=0., floor=1.):
        if observer:
            sys.setprofile(profile)
        try:
            if context is None:
                result = residual_student.train(noise_policy=policy)
                recorder = None
            else:
                with context as (recorder, _):
                    result = residual_student.train(noise_policy=policy)
                captured = recorder._local
        finally:
            if observer:
                sys.setprofile(None)
    return result, _state(captured, policy), policy.receipt(), recorder


def test_identity_adapter_is_bitwise_original_host_including_native_g_loss():
    original = _short_host(observer=True)
    with binding.pr84_residual_refinement() as (rec, _):
        rec.enabled = False
        # The context itself must stay open while the host runs.
        disabled = _short_host_context_body(rec)
    assert disabled[0] == original[0]
    assert disabled[1] == original[1]
    assert disabled[2] == original[2]
    assert len(rec.refinement_records) == 0


def _short_host_context_body(rec, *, isolated=False):
    """Run after a caller has already entered the scratch context."""
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 400, output_noise_warmup=.2,
                         output_noise_rng="isolated" if isolated else None)
    with patch.dict(residual_student.PROTOCOL, {
            "steps": 2, "lr": .00425, "particle_l2": 0.,
            "vicreg_weight": 0., "cover_weight": 1.5,
    }), recording(2, schedule="cosine", start=0., floor=1.):
        result = residual_student.train(noise_policy=policy)
    return result, _state(rec._local, policy), policy.receipt(), rec


class _OriginalResidualPR84(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, refinement=False):
        super().__init__(start_step=start_step)

    def phases(self, step, opt_d, opt_g, local):
        yield from super().phases(step, opt_d, opt_g, binding._alias(local))


@pytest.mark.parametrize("isolated", [False, True])
def test_disabled_refinement_is_bitwise_original_pr84_residual_binding(isolated):
    with patch.object(binding, "ResidualRefinementRecorder", _OriginalResidualPR84):
        original = _short_host(binding.pr84_residual_refinement(refinement=False), isolated=isolated)
    disabled = _short_host(binding.pr84_residual_refinement(refinement=False), isolated=isolated)
    assert original[:3] == disabled[:3]
    assert original[-1].records == disabled[-1].records
    assert disabled[-1].fit_gradient_evaluations == disabled[-1].parity_gradient_evaluations == 0


@pytest.mark.parametrize("isolated", [False, True])
def test_active_bank_matches_first_d_gradient_and_preserves_noise_and_moments(isolated):
    original = _short_host(binding.pr84_residual_refinement(refinement=False), isolated=isolated)
    active = _short_host(binding.pr84_residual_refinement(), isolated=isolated)
    rec = active[-1]
    assert active[2] == original[2]
    assert rec.bank_rng_verified == rec.fit_rng_verified == rec.parity_gradient_evaluations == 2
    assert rec.rng_replay_verified == 4
    assert [row["host_update"] for row in rec.refinement_records] == [1, 2]
    assert all(row["first_bank_gradient_bitwise_equal"] for row in rec.refinement_records)
    assert all(row["bank_pairs"] == 8 * 12 and row["bank_native_batch_size"] == 12
               and row["input_sigma"] > 0 and row["frozen_g_stencil_width"] == 0
               and row["conditioning"] == "unchanged native slow coordinates"
               for row in rec.refinement_records)
    assert torch.equal(rec._bank["slow"], rec._local["slow"].repeat(8, 1))
    assert torch.equal(rec._bank["real"], rec._local["paired"].repeat(8, 1))
    # The production residual_student call uses its default shared pairing.
    assert torch.equal(rec._bank["real"][:12], rec._local["fast"])
    for optimizer, record in rec.rows.items():
        assert record["calls"] == 6
        assert {float(group["lr"]) for group in optimizer.param_groups} == {.00425}
        assert all(int(optimizer.state[p]["step"]) == 2
                   for group in optimizer.param_groups for p in group["params"])
    receipt = rec.receipt()
    assert receipt["fit_sample_pairs_evaluated"] == sum(
        row["bank_pairs"] * row["closure_calls"] for row in rec.refinement_records)
    assert receipt["shared_gate_eligible"] is False
    assert receipt["native_g_objective"].endswith("supervised residual")


def test_host_source_guard_rejects_drift_before_training():
    with patch.object(binding, "HOST_FUNCTION_SHA256", "0" * 64):
        with pytest.raises(RuntimeError, match="training function source changed"):
            with binding.pr84_residual_refinement():
                pass
