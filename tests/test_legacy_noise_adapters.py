"""Optional custom-host noise keeps zero-noise behavior and isolates evaluation."""

import math

import pytest
import torch

from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy


def test_evaluation_uses_separate_rng_streams():
    policy = NoisePolicy(0.029, 0.5, 0.5, 20, seed=17)
    policy.set_step(0)
    torch.manual_seed(888)
    before_global = torch.random.get_rng_state().clone()
    before_d = policy.input_stream.get_state().clone()
    with policy.evaluation(3):
        first = policy.output(torch.zeros(32, 2))
        policy.input(torch.zeros(32, 2))
    assert torch.equal(before_global, torch.random.get_rng_state())
    assert torch.equal(before_d, policy.input_stream.get_state())
    with policy.evaluation(3):
        repeated = policy.output(torch.zeros(32, 2))
    assert torch.equal(first, repeated)
    assert math.isclose(float(first.std()), 0.029, rel_tol=0.3)


def test_output_warmup_starts_without_rng_then_reaches_one_shared_peak():
    policy = NoisePolicy(0.029, 0.5, 0.5, 10, seed=17,
                         output_noise_warmup=0.5)
    base = torch.zeros(64, 2)
    torch.manual_seed(777)
    policy.set_step(0)
    before = torch.random.get_rng_state().clone()
    first = policy.output(base)
    assert first is base
    assert torch.equal(before, torch.random.get_rng_state())
    assert policy.output_sigma == 0.0

    policy.set_step(1)
    assert math.isclose(policy.output_sigma, 0.029 / 5)
    policy.set_step(5)
    assert policy.output_sigma == 0.029
    policy.set_step(9)
    assert policy.output_sigma == 0.029
    training_sigma = policy.output_sigma
    d_state = policy.input_stream.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    with policy.evaluation(0):
        assert policy.output_sigma == 0.0
        assert policy.output(base) is base
    with policy.evaluation(3):
        assert math.isclose(policy.output_sigma, 0.029 * 3 / 5)
        evaluated = policy.output(base)
        policy.input(base)
    assert evaluated.abs().sum() > 0
    assert policy.output_sigma == training_sigma
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.equal(d_state, policy.input_stream.get_state())
    receipt = policy.receipt()
    assert receipt["output_sigma_first"] == 0.0
    assert receipt["output_sigma_last"] == 0.029
    assert receipt["output_nonzero_steps"] == 3
    assert receipt["output_train_calls"] == 1
    assert receipt["output_train_elements"] == 0
    assert receipt["output_eval_elements"] == base.numel()


def test_invalid_output_warmup_is_rejected():
    for fraction in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="output_noise_warmup"):
            NoisePolicy(0.029, 0.5, 0.5, 10,
                        output_noise_warmup=fraction)
