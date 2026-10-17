"""Tiny public spatial host matches the native handoff's critical dataflow."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from benchmarks.routed_conditioning import film_damping as common
from benchmarks.routed_conditioning.spatial_damping import make_loop, time_features
from particlegan import init


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_native_spatial_dataflow_has_no_outer_source_identity():
    loop = make_loop()
    G = loop.policy.G
    context, codes = loop.fit_context[:2], loop.policy.table[:2]
    source = G.project(G.host.encode(context[:, :2]))
    condition = torch.cat((F.layer_norm(codes, (4,), eps=1e-3), time_features(context)), 1)
    gain, shift = (.25 * G.condition(condition).tanh()).chunk(2, 1)
    hidden = G.input(source) * (1 + gain[:, :, None, None]) + shift[:, :, None, None]
    expected = G.host.decode(G.skip(source) + G.output(F.silu(G.blocks(hidden))))
    torch.testing.assert_close(G(context, codes), expected, rtol=0, atol=0)
    assert G.host.prefix.weight.dtype == G.host.suffix.weight.dtype == torch.bfloat16
    assert G.host.encode(context[:, :2]).dtype == torch.float32
    assert expected.shape == (2, 2, 8, 8) and expected.dtype == torch.float32
    assert G.host.training is False


def test_spatial_profile_changes_only_initial_shift_and_dense_native_update():
    rng = torch.get_rng_state().clone()
    original, neutral = make_loop(), make_loop("shift_zero_native")
    assert torch.equal(rng, torch.get_rng_state())
    p, q = original.policy, neutral.policy
    assert len(original.fit_context) == 245 and len(original.guard_context) == 64 and len(original.test_context) == 256
    assert original.metadata["data_hashes"] == neutral.metadata["data_hashes"]
    for role in ("critic", "encoder", "router", "table"):
        assert original.metadata["initial_hashes"][role] == neutral.metadata["initial_hashes"][role]
    for name, parameter in p.G.named_parameters():
        other = dict(q.G.named_parameters())[name]
        if name == "condition.weight":
            assert torch.equal(parameter[:4], other[:4])
            assert parameter[4:].count_nonzero() > 0 and other[4:].count_nonzero() == 0
        else:
            assert torch.equal(parameter, other)
    assert p.D.quadratic.weight.count_nonzero() == 0
    for model in p._training_modules().values():
        assert all(spec is not None for spec in init.declarations(model).values())
    frozen = deepcopy(q.G.host.state_dict())
    row = common.update(neutral)
    assert row["dense_gradient_rows"] == 128
    assert row["gradient_energy"]["encoder"] > 0 and row["gradient_energy"]["table"] > 0
    assert set(row["generator_block"]["components"]) == {"project", "input", "condition_gain", "condition_shift", "blocks", "output", "skip"}
    assert q.penalty.last_stats["applied"] and q.opt_d.record.observed_steps == 1
    assert q.controller is q.opt_d.continuous_controller and q.reopen_guard is not None
    assert q.row_evidence is not None and q.birth_death is q.routed_control
    for name, value in q.G.host.state_dict().items():
        assert torch.equal(frozen[name], value)
    assert all(parameter.grad is None for parameter in q.G.host.parameters())


def test_spatial_raw_energy_head_and_saturation_diagnostics():
    loop = make_loop()
    error = torch.arange(256, dtype=torch.float32).reshape(2, 2, 8, 8) / 100
    expected = error.square().mean((2, 3)) * (32 ** .5)
    torch.testing.assert_close(loop.policy.D.energy(error), expected, rtol=0, atol=0)
    assert loop.metadata["penalty_dimension"] == 128
    assert loop.metadata["initial_code_jacobian_frobenius"] > 0
    for stats in loop.metadata["initial_condition_stats"].values():
        assert stats["mean_tanh_derivative"] > .95
        assert stats["saturated_fraction_derivative_lt_05"] == 0


@pytest.mark.parametrize("shift_rule", ["original", "whole_branch_zero", "additive_code_zero", "preserve_additive_code"])
def test_zero_hidden_film_geometry_and_public_routed_gan_path(shift_rule):
    """Geometry contract only: no optimizer update or stationarity evidence.

    h*(1+gain(code,time))+shift(code,time) loses gain's code derivative at h=0.
    Zeroing the entire shift branch then closes both paths. Zeroing its time
    columns/bias while preserving additive code columns retains a live path.
    """
    loop = make_loop()
    p, G = loop.policy, loop.policy.G
    width = G.condition.out_features // 2
    with torch.no_grad():
        if shift_rule == "whole_branch_zero":
            G.condition.weight[width:].zero_()
            G.condition.bias[width:].zero_()
        elif shift_rule == "additive_code_zero":
            G.condition.weight[width:, :4].zero_()
        elif shift_rule == "preserve_additive_code":
            G.condition.weight[width:, 4:].zero_()
            G.condition.bias[width:].zero_()
    assert G.condition.weight.requires_grad and G.condition.weight[:width, :4].norm() > 0
    context = loop.fit_context[:1]
    code = torch.tensor([[.2, -.4, .1, .7]], requires_grad=True)
    captured = []
    # Execute the actual prefix/project/input path, then control just the
    # pre-FiLM hidden value. This intervention is never an experiment arm.
    zero_hidden = G.input.register_forward_hook(lambda _module, _args, value: torch.zeros_like(value))
    capture_hidden = G.blocks.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        G(context, code)
        conditioned = captured[-1][0, :, 0, 0]
        actual = torch.stack([torch.autograd.grad(value, code, retain_graph=True)[0][0]
                              for value in conditioned])
        normalized = F.layer_norm(code, (4,), eps=1e-3)
        pre_shift = G.condition(torch.cat((normalized, time_features(context)), 1))[0, width:]
        code_normalization = torch.autograd.functional.jacobian(
            lambda value: F.layer_norm(value, (4,), eps=1e-3), code).reshape(4, 4)
        derivative = 1 - pre_shift.tanh().square()
        expected = .25 * derivative[:, None] * (G.condition.weight[width:, :4] @ code_normalization)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-9)
        assert derivative.min() > .95
        path_closed = shift_rule in ("whole_branch_zero", "additive_code_zero")
        assert (float(actual.square().sum()) == 0) == path_closed

        # The same geometry is visible through the complete public bank route
        # and pure GAN loss, without replacing the routing callback or critic.
        prediction = p.routed_generate(context, sigma=0, perturb=False)
        real = torch.zeros_like(loop.fit_targets[:1])
        loss = p.recipe.make_loss().g_loss(p.D(real + prediction - loop.fit_targets[:1]),
                                          p.D(real).detach())
        gradients = torch.autograd.grad(loss, tuple(p.encoder.parameters()))
        energy = sum(float(value.square().sum()) for value in gradients)
        assert (energy == 0) == path_closed
        assert p.completed_steps == 0 and G.condition.weight.requires_grad
    finally:
        capture_hidden.remove()
        zero_hidden.remove()


def test_initial_spatial_sensitivity_and_actual_adam_motion_are_distinct():
    original, neutral = make_loop(), make_loop("shift_zero_native")
    original_jacobian = original.metadata["initial_code_jacobian_frobenius"]
    neutral_jacobian = neutral.metadata["initial_code_jacobian_frobenius"]
    assert original_jacobian > 10 * neutral_jacobian > 0
    assert original.metadata["data_hashes"] == neutral.metadata["data_hashes"]
    for role in ("encoder", "critic", "router", "table"):
        assert original.metadata["initial_hashes"][role] == neutral.metadata["initial_hashes"][role]
    a, b = common.update(original), common.update(neutral)
    # These are squared norms. Adam preconditioning means a 47.8x reduction in
    # E gradient energy is not a 47.8x reduction in actual E displacement.
    grad_ratio = a["gradient_energy"]["encoder"] / b["gradient_energy"]["encoder"]
    motion_ratio = a["displacement_energy"]["encoder"] / b["displacement_energy"]["encoder"]
    assert grad_ratio > 40 and 2 < motion_ratio < 3 and grad_ratio > 10 * motion_ratio
    assert min(a["gradient_energy"]["encoder"], b["gradient_energy"]["encoder"],
               a["displacement_energy"]["encoder"], b["displacement_energy"]["encoder"]) > 0
    for stream in ("batch_rng", "paired_noise_rng"):
        assert torch.equal(getattr(original, stream).get_state(), getattr(neutral, stream).get_state())
