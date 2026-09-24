"""Focused checks for the scratch D allocation score and RNG isolation."""

import math

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, wrap_input, wrap_output
from particlegan import ParticlePrior
from reports.toy100.adaptive_d_allocation_scratch import AdaptiveDAllocation


def test_difference_score_needs_separation_before_switching():
    policy = AdaptiveDAllocation(eval_batch=16)
    # With no separation, the e-process has no evidence to stop D early.
    assert policy._log_e_increment(torch.zeros(16)) < 0
    # Clear real > fake evidence crosses the same declared log(10) boundary.
    assert policy._log_e_increment(torch.full((16,), 8.0)) > math.log(10)


def test_fresh_score_does_not_advance_training_random_streams():
    torch.manual_seed(0)
    noise = NoisePolicy(.029, .5, .1, 1200, seed=0,
                        output_noise_warmup=.2, output_noise_rng="isolated")
    noise.set_step(1000)
    generator = wrap_output(SimpleMLPGenerator(4, 8, 2, 2), noise)
    critic = wrap_input(SimpleMLPDiscriminator(2, 8, 2, 3), noise)
    prior = ParticlePrior(12, 4, init_std=.5,
                          generator=torch.Generator().manual_seed(0))
    train = torch.Generator().manual_seed(123)
    controller = AdaptiveDAllocation(eval_batch=16)
    global_before = torch.get_rng_state().clone()
    streams_before = [stream.get_state().clone() for stream in
                      (train, noise.input_stream, noise.output_stream)]
    eval_before = controller.eval_stream.get_state().clone()
    score = controller._fresh_log_evalue(dict(means=mode_hold.ring_means(),
                                               prior=prior, generator=generator,
                                               critic=critic, noise_policy=noise))
    assert math.isfinite(score)
    assert controller.eval_queries == 1
    assert torch.equal(global_before, torch.get_rng_state())
    assert all(torch.equal(before, stream.get_state()) for before, stream in
               zip(streams_before, (train, noise.input_stream, noise.output_stream)))
    assert not torch.equal(eval_before, controller.eval_stream.get_state())
