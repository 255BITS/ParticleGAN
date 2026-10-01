"""Persistent noise is explicit; the observer must preserve ordinary Adam."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from benchmarks.transfer_suite import legacy_noise_adapters as legacy
from reports.toy100.persistent_noise_scratch import (
    constant_input_noise, persistent_noise, persistent_noise_regrade,
)
from reports.toy100.persistent_noise_probe import verify_receipt


def small_game(observed):
    parameters = [torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
                  for value in (.7, -.4, .2)]
    d, g, prior = parameters
    opt_d = torch.optim.Adam([d], lr=.001, betas=(0., .999))
    opt_g = torch.optim.Adam([dict(params=[g]), dict(params=[prior], lr=.002,
                                                  _comparison_prior=True)],
                             lr=.001, betas=(0., .999))
    context = persistent_noise() if observed else patch.object(
        legacy, "linear_input_noise", constant_input_noise)
    before_rng = torch.random.get_rng_state()
    with context as recorder:
        policy = legacy.NoisePolicy(.029, .5, .1, 5, seed=0,
                                    output_noise_warmup=.2)
        for step in range(5):
            policy.set_step(step)
            opt_d.zero_grad()
            (d * policy.input(torch.ones(32, dtype=torch.float64)).mean()).backward()
            opt_d.step()
            opt_g.zero_grad()
            ((g + prior) * policy.input(torch.ones(32, dtype=torch.float64)).mean()).backward()
            opt_g.step()
        receipt = recorder.receipt() if observed else None
    assert torch.equal(before_rng, torch.random.get_rng_state())
    applied = [dict(role=role, parameters=1, lr=lr, betas=[0., .999])
               for role, lr in (("d", .001), ("g", .001), ("prior", .002))]
    record = dict(noise_receipt=policy.receipt(), applied=applied,
                  noise=dict(input_noise_std=.5, output_noise_std=.029,
                             output_noise_warmup=.2))
    return parameters, [opt_d.state_dict(), opt_g.state_dict()], receipt, record


def test_observer_is_bitwise_ordinary_adam_and_does_not_consume_global_rng():
    observed, states, receipt, record = small_game(True)
    expected, expected_states, _, _ = small_game(False)
    assert all(torch.equal(a, b) for a, b in zip(observed, expected))
    for actual, baseline in zip(states, expected_states):
        assert actual["param_groups"] == baseline["param_groups"]
        for position, state in actual["state"].items():
            assert all(torch.equal(value, baseline["state"][position][key])
                       for key, value in state.items())
    verify_receipt(receipt, record, dict(lr=.001, input_sigma=.5), 5)


def test_noise_remains_constant_past_original_burnin_and_context_restores():
    original = legacy.linear_input_noise
    original_step = torch.optim.Adam.step
    assert original(.5, 4, 5, .1) == 0
    with pytest.raises(RuntimeError, match="sentinel"):
        with persistent_noise():
            assert legacy.linear_input_noise(.5, 4, 5, .1) == .5
            raise RuntimeError("sentinel")
    assert legacy.linear_input_noise is original
    assert torch.optim.Adam.step is original_step


@pytest.mark.parametrize("tamper", ["clock", "rate", "application", "moments", "coverage"])
def test_receipt_rejects_noise_rate_state_and_coverage_tampering(tamper):
    _, _, receipt, record = small_game(True)
    receipt = deepcopy(receipt)
    if tamper == "clock":
        receipt["clock"][-1]["input_sigma"] = 0
    elif tamper == "rate":
        receipt["updates"][-1]["groups"][0]["lr"] = .0001
    elif tamper == "application":
        receipt["inputs"][-1]["sigma"] = 0
    elif tamper == "moments":
        receipt["updates"][-1]["groups"][0]["moment_steps"] = [1]
    else:
        receipt["updates"][-1]["input_calls"] = 0
    with pytest.raises(AssertionError):
        verify_receipt(receipt, record, dict(lr=.001, input_sigma=.5), 5)


def test_policy_specific_regrade_is_scoped_and_does_not_change_production_gate():
    from benchmarks import toy_suite
    original = toy_suite.linear_input_noise
    with persistent_noise_regrade():
        assert toy_suite.linear_input_noise(.1, 99, 100, .1) == .1
        with pytest.raises(ValueError, match="common gate"):
            toy_suite._reject_scratch_optimizer(dict(shared_gate_eligible=False))
    assert toy_suite.linear_input_noise is original
