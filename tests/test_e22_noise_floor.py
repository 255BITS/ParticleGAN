"""Learned noise must escape its floor after the model and table settle."""
from copy import deepcopy
import io
import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch.set_num_threads(previous)


def make_trainer(device, *, dtype=torch.float32, output_noise_std=.125):
    recipe = get_recipe("e22", num_particles=16, z_dim=2, batch_size=8,
                        output_noise_std=output_noise_std, particle_birth_death=False,
                        row_evidence_gate=False, birth_death_isolation=False,
                        birth_death_feature_scale="none")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        G = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)).to(device=device, dtype=dtype)
        D = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)).to(device=device, dtype=dtype)
        prior = recipe.make_prior(dtype=dtype).to(device)
    return GANTrainer(recipe, G, D, prior=prior, seed=31, serial_backward=True)


def real_batch(policy, index=0):
    return torch.arange(16, device=policy.device, dtype=policy.dtype).reshape(8, 2) / 16 + index / 32


@pytest.mark.parametrize("device", DEVICES)
def test_initial_noise_floor_keeps_half_tie_gradient_and_exact_numeric_bound(device):
    trainer = make_trainer(device)
    policy = trainer.policy
    raw = policy.log_output_sigma.exp()
    floor = raw.new_tensor(.125)
    if raw < floor:
        # The concrete regression on affected CUDA exponential kernels.
        old = torch.maximum(raw, floor)
        assert torch.autograd.grad(old, policy.log_output_sigma)[0] == 0
    noise = policy.begin_step(real_batch(policy))
    assert noise.output_sigma.requires_grad
    assert_tree_equal(noise.output_sigma.detach(), torch.maximum(raw.detach(), floor))
    assert noise.output_sigma >= floor
    noise.output_sigma.backward()
    expected_gradient = raw.detach() / 2 if raw <= floor else raw.detach()
    assert_tree_equal(policy.log_output_sigma.grad, expected_gradient)
    assert policy.log_output_sigma.grad > 0
    assert type(policy.output_sigma()) is float
    assert policy.output_sigma() == float(noise.output_sigma.detach())
    assert policy._output_sigma(.125, detach=False).requires_grad
    policy.abort_step()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [-.1, .1])
def test_genuinely_below_and_above_floor_keep_existing_value_and_derivative(device, offset):
    policy = make_trainer(device).policy
    with torch.no_grad():
        policy.log_output_sigma.add_(offset)
    raw = policy.log_output_sigma.exp()
    reference = torch.maximum(raw, raw.new_tensor(.125))
    expected_grad = torch.autograd.grad(reference, policy.log_output_sigma)[0]
    sigma = policy._output_sigma(.125, detach=False)
    actual_grad = torch.autograd.grad(sigma, policy.log_output_sigma)[0]
    assert_tree_equal(sigma, reference)
    assert_tree_equal(actual_grad, expected_grad)
    if offset < 0:
        assert actual_grad == 0 and sigma == .125
    else:
        assert_tree_equal(actual_grad, raw)
        assert sigma > .125


@pytest.mark.parametrize("device", DEVICES)
def test_float64_initialization_keeps_native_maximum_gradient(device):
    policy = make_trainer(device, dtype=torch.float64, output_noise_std=.04).policy
    raw = policy.log_output_sigma.exp()
    floor = raw.new_tensor(.04)
    sigma = policy._output_sigma(.04, detach=False)
    gradient = torch.autograd.grad(sigma, policy.log_output_sigma)[0]
    assert_tree_equal(sigma, torch.maximum(raw, floor))
    # The native FP64 fixture rounds above .04 and must retain the full
    # exponential derivative. Kernels that tie use maximum's half derivative.
    assert_tree_equal(gradient, raw if raw > floor else raw / 2)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_controller_floor_keeps_unclamped_learned_noise(device):
    policy = make_trainer(device).policy
    policy.controller.mobility = 0.
    for index, testers in enumerate(policy.lr_settle.testers):
        if index != 1:
            for tester in testers:
                tester.s = 1. / 64.
    raw = policy.log_output_sigma.exp()
    sigma = policy._output_sigma(.125, detach=False)
    gradient = torch.autograd.grad(sigma, policy.log_output_sigma)[0]
    assert_tree_equal(sigma, raw)
    assert_tree_equal(gradient, raw)


@pytest.mark.parametrize("device", DEVICES)
def test_paired_game_learns_noise_on_first_update_and_resumes_with_served_sigma(device):
    from examples.e22_routed_sites import checkpoint, make_loop, restore, update

    loop = make_loop(mode="no_rows", device=device, tokens=3, batch_size=4)
    policy = loop.policy
    initial = policy.log_output_sigma.detach().clone()
    update(loop)
    # Both halves reuse StepNoise.output_sigma. Critic pairs run without grad;
    # the generator's shared-noise fake path retains its differentiable sigma.
    assert policy.log_output_sigma.grad is not None
    assert bool(torch.isfinite(policy.log_output_sigma.grad))
    assert policy.log_output_sigma.grad != 0
    assert not torch.equal(initial, policy.log_output_sigma.detach())
    assert policy.output_sigma() >= .125
    if policy.device.type == "cuda":
        assert policy.output_sigma() > .125
    saved = cpu_roundtrip(checkpoint(loop))
    snapshot = policy.served_snapshot()
    served = policy.served_model()
    assert served.output_sigma == snapshot["output_sigma"] == policy.output_sigma()
    sampling = torch.Generator(device=device).manual_seed(11)
    output = served.routed_forward(loop.test_context, output_noise=True, generator=sampling)
    expected_rows = [update(loop) for _ in range(2)]
    expected = checkpoint(loop)
    restored = make_loop(mode="no_rows", device=device, tokens=3, batch_size=4)
    restore(restored, saved)
    assert_tree_equal(snapshot, restored.policy.served_snapshot())
    restored_sampling = torch.Generator(device=device).manual_seed(11)
    assert_tree_equal(output, restored.policy.served_model().routed_forward(
        restored.test_context, output_noise=True, generator=restored_sampling))
    assert_tree_equal(expected_rows, [update(restored) for _ in range(2)])
    assert_tree_equal(expected, checkpoint(restored))


@pytest.mark.parametrize("device", DEVICES)
def test_float32_native_and_external_noise_learning_have_exact_same_device_parity(device):
    from examples.e22_external_loop import make_loop, update

    native = make_trainer(device)
    external = make_loop(native.recipe, deepcopy(native.G), deepcopy(native.D),
                         deepcopy(native.prior), seed=31)
    initial = native.log_output_sigma.detach().clone()
    for index in range(3):
        real = real_batch(native.policy, index)
        assert_tree_equal(native.step(real), update(external, real))
        assert_tree_equal(native.policy.state_dict(), external.policy.state_dict())
        assert_tree_equal(native.log_output_sigma.grad, external.policy.log_output_sigma.grad)
        if index == 0:
            assert native.log_output_sigma.grad != 0
            assert not torch.equal(initial, native.log_output_sigma.detach())
    assert not torch.equal(initial, native.log_output_sigma.detach())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("formulation", ["e22", "e22_routed"])
def test_frozen_noise_cannot_block_floor_release_and_exact_resume(device, formulation):
    if formulation == "e22":
        factory = lambda: make_trainer(device)
        advance = lambda loop: loop.step(real_batch(loop.policy))
        save = lambda loop: loop.state_dict()
        recover = lambda loop, state: loop.load_state_dict(state)
    else:
        from examples.e22_routed_sites import checkpoint, make_loop, restore, update

        factory = lambda: make_loop(device=device, tokens=3, batch_size=4, probe_interval=1000)
        advance, save, recover = update, checkpoint, restore

    loop = factory()
    policy = loop.policy
    base = policy.recipe.output_noise_std
    with torch.no_grad():
        policy.log_output_sigma.copy_(policy.log_output_sigma.new_tensor(math.log(base * .75)))
    clamped_parameter = policy.log_output_sigma.detach().clone()
    noise_tester = policy.lr_settle.testers[0][-1]
    # Produce a real frozen verdict: the physical floor suppresses the noise
    # gradient while at least one model group still has its full rate.
    for _ in range(2 * noise_tester.K):
        policy.lr_settle.testers[0][0].s = 1.
        advance(loop)
        assert policy.log_output_sigma.grad == 0
        assert_tree_equal(policy.log_output_sigma.detach(), clamped_parameter)
    assert noise_tester.counts["frozen"] >= 1
    assert noise_tester.s == 1.
    assert policy.output_sigma() == base

    # Supply the model/table settlement boundary without altering the frozen
    # noise tester. The floor is allowed to fall on the next update.
    for index, testers in enumerate(policy.lr_settle.testers):
        if index != 1:
            for tester, role in zip(testers, policy.roles[index]):
                if role != "noise" and tester is not None:
                    tester.s = 1. / 64.
    policy.controller.mobility = .25
    raw = policy.log_output_sigma.exp()
    sigma = policy._output_sigma(base, detach=False)
    assert_tree_equal(sigma, raw)
    assert_tree_equal(torch.autograd.grad(sigma, policy.log_output_sigma)[0], raw)
    assert policy.output_sigma() < base
    before = cpu_roundtrip(save(loop))
    served_before = policy.served_snapshot()
    assert served_before["output_sigma"] == policy.output_sigma()

    expected_updates = []
    for _ in range(2):
        expected_updates.append(advance(loop))
        assert policy.log_output_sigma.grad != 0
        assert_tree_equal(policy.opt_g.param_groups[-1]["lr"], policy.initial_lrs[0][-1])
    assert not torch.equal(policy.log_output_sigma.detach(), clamped_parameter)
    expected_gradient = policy.log_output_sigma.grad.detach().clone()
    expected = save(loop)
    served_after = policy.served_snapshot()
    served_model = policy.served_model()
    sampling = torch.Generator(device=device).manual_seed(11)
    if formulation == "e22":
        expected_output = served_model.sample(5, output_noise=True, generator=sampling)
    else:
        expected_output = served_model.routed_forward(
            loop.test_context, output_noise=True, generator=sampling)

    restored = factory()
    recover(restored, before)
    assert_tree_equal(served_before, restored.policy.served_snapshot())
    assert_tree_equal(expected_updates, [advance(restored) for _ in range(2)])
    assert_tree_equal(expected_gradient, restored.policy.log_output_sigma.grad)
    assert_tree_equal(expected, save(restored))
    assert_tree_equal(served_after, restored.policy.served_snapshot())
    restored_model = restored.policy.served_model()
    sampling = torch.Generator(device=device).manual_seed(11)
    if formulation == "e22":
        actual_output = restored_model.sample(5, output_noise=True, generator=sampling)
    else:
        actual_output = restored_model.routed_forward(
            restored.test_context, output_noise=True, generator=sampling)
    assert_tree_equal(expected_output, actual_output)


@pytest.mark.parametrize("unsettled_role", ["generator", "encoder", "router", "table"])
def test_noise_floor_waits_for_each_model_role_with_separate_table_owner(unsettled_role):
    from tests.test_e22_ownership import _components

    policy = _components()
    base = policy.recipe.output_noise_std
    policy.controller.mobility = .25
    with torch.no_grad():
        policy.log_output_sigma.add_(math.log(.75))
    unsettled = None
    for index, testers in enumerate(policy.lr_settle.testers):
        for tester, role in zip(testers, policy.roles[index]):
            if role not in ("noise", "critic"):
                tester.s = 1. / 64.
            if role == unsettled_role:
                unsettled = tester
    assert unsettled is not None
    unsettled.s = 1. / 32.
    sigma = policy._output_sigma(base, detach=False)
    assert sigma == base
    assert torch.autograd.grad(sigma, policy.log_output_sigma)[0] == 0
    unsettled.s = 1. / 64.
    sigma = policy._output_sigma(base, detach=False)
    raw = policy.log_output_sigma.exp()
    assert_tree_equal(sigma, raw)
    assert_tree_equal(torch.autograd.grad(sigma, policy.log_output_sigma)[0], raw)
