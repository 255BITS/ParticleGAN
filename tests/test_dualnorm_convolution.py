"""CUDA algebra and public checkpoint contracts; these are software checks.

Scientific image quality gates and actual-training GIFs belong to the registered
four-host Forge study, rather than additional toy acquisition experiments.
"""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.forge.state import state_digest
from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.dualnorm import NormalizedOptimizer


@pytest.fixture(autouse=True)
def cuda_contract():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for convolution verification")
    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        yield


def recipe(**overrides):
    return get_recipe("bcap", **{"optimizer_convolution": "per_offset", "optimizer_smoothing": 0., **overrides})


def gradient_like(weight, offset=0.):
    values = torch.arange(weight.numel(), dtype=weight.dtype).reshape(weight.shape)
    return (values + offset).sin() + (values * .37 + offset).cos()


def reference_step(module, gradient, *, smoothing, rate):
    """Independent scalar-loop SVD reference in the layer's channel coordinates."""
    result = torch.zeros_like(gradient)
    inputs, outputs = module.in_channels // module.groups, module.out_channels // module.groups
    scale = math.sqrt(outputs / inputs) / math.prod(module.kernel_size)
    transposed = isinstance(module, nn.ConvTranspose2d)
    for group in range(module.groups):
        for row in range(module.kernel_size[0]):
            for column in range(module.kernel_size[1]):
                if transposed:
                    matrix = gradient[group * inputs:(group + 1) * inputs, :, row, column].T
                else:
                    matrix = gradient[group * outputs:(group + 1) * outputs, :, row, column]
                left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
                cutoff = max(inputs, outputs) * torch.finfo(matrix.dtype).eps * singular[0]
                weights = singular.gt(cutoff).to(matrix.dtype)
                if smoothing:
                    weights *= singular / torch.hypot(singular, singular.new_tensor(smoothing))
                update = -rate * scale * ((left * weights) @ right)
                if transposed:
                    result[group * inputs:(group + 1) * inputs, :, row, column] = update.T
                else:
                    result[group * outputs:(group + 1) * outputs, :, row, column] = update
    return result


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
@pytest.mark.parametrize("channels,kernel,groups", [((6, 4), (2, 3), 2), ((4, 6), (1, 1), 2),
                                                   ((3, 2), (3, 2), 1)])
@pytest.mark.parametrize("smoothing", [0., .3])
def test_kernel_step_matches_channel_svd_reference(kind, channels, kernel, groups, smoothing):
    module = kind(*channels, kernel, groups=groups, bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = recipe(lr=.2, optimizer_smoothing=smoothing).make_generator_optimizer(module)
    module.weight.grad = gradient_like(module.weight)
    expected = reference_step(module, module.weight.grad, smoothing=smoothing, rate=.2)
    optimizer.step()
    torch.testing.assert_close(module.weight, expected, rtol=1e-13, atol=1e-14)
    assert module.weight.device.type == "cuda"
    metadata = optimizer.param_groups[0]["dualnorm_convolution"]
    assert metadata["groups"] == groups and metadata["kernel_size"] == list(kernel)
    # In particular a narrowing 1x1 layer must use sqrt(out/in), without the
    # existing dense rule's max(1, out/in) clamp.
    assert metadata["in_channels"] == channels[0] and metadata["out_channels"] == channels[1]


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
@pytest.mark.parametrize("smoothing", [0., .3])
def test_grouped_kernel_direction_obeys_spatial_max_channel_rms_bound(kind, smoothing):
    module = kind(6, 4, (2, 3), groups=2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = recipe(lr=1., optimizer_smoothing=smoothing).make_generator_optimizer(module)
    module.weight.grad = gradient_like(module.weight)
    optimizer.step()
    input_values = torch.arange(2 * 6 * 6 * 7, dtype=torch.float64).reshape(2, 6, 6, 7).sin()
    if kind is nn.ConvTranspose2d:
        output = F.conv_transpose2d(input_values, -module.weight, stride=2, padding=2,
                                   output_padding=1, groups=2, dilation=2)
    else:
        output = F.conv2d(input_values, -module.weight, stride=2, padding=2, groups=2, dilation=2)
    input_norm = input_values.square().mean(dim=1).sqrt().amax()
    output_norm = output.square().mean(dim=1).sqrt().amax()
    assert output_norm <= input_norm * (1 + 1e-12)
    assert output_norm > 0


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
def test_narrowing_grouped_kernel_attains_unit_rms_bound_on_constant_input(kind):
    module = kind(6, 4, (2, 3), groups=2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = recipe(lr=1.).make_generator_optimizer(module)
    module.weight.grad = torch.ones_like(module.weight)
    optimizer.step()
    input_values = torch.ones(1, 6, 6, 7, dtype=torch.float64)
    operation = F.conv_transpose2d if kind is nn.ConvTranspose2d else F.conv2d
    output = operation(input_values, -module.weight, groups=2)
    output_norm = output.square().mean(dim=1).sqrt().amax()
    torch.testing.assert_close(output_norm, output_norm.new_tensor(1.), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
def test_rank_mask_and_zero_slice_are_local_to_channel_matrices(kind):
    module = kind(2, 2, (1, 2), bias=False)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = recipe(lr=.2, optimizer_smoothing=.3).make_generator_optimizer(module)
    module.weight.grad = torch.zeros_like(module.weight)
    module.weight.grad[:, :, 0, 0] = torch.diag(torch.tensor([1., 1e-9]))
    optimizer.step()
    expected = torch.zeros_like(module.weight)
    expected[0, 0, 0, 0] = -.1 / math.hypot(1., .3)
    torch.testing.assert_close(module.weight, expected)
    assert torch.count_nonzero(module.weight[:, :, 0, 1]) == 0


def test_momentum_preserves_storage_shape_and_per_slice_epsilon_skips():
    module = nn.ConvTranspose2d(1, 1, (1, 3), bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = recipe(lr=.3, optimizer_momentum=.5).make_generator_optimizer(module)
    module.weight.grad = module.weight.new_tensor([2., 5e-9, 0.]).reshape(module.weight.shape)
    optimizer.step()
    first = module.weight.detach().clone()
    torch.testing.assert_close(first.flatten(), module.weight.new_tensor([-.1, 0., 0.]))
    module.weight.grad = torch.zeros_like(module.weight)
    optimizer.step()
    assert torch.equal(module.weight, first)
    module.weight.grad = module.weight.new_tensor([-.5, 0., 0.]).reshape(module.weight.shape)
    optimizer.step()
    assert torch.equal(module.weight, first)  # Current gradient cancels history.
    history = optimizer.state[module.weight]["momentum_buffer"]
    assert history.shape == module.weight.shape and history[0, 0, 0, 0] == 0


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
def test_momentum_is_accumulated_before_smoothing(kind):
    module = kind(4, 6, (2, 3), groups=2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = recipe(lr=.2, optimizer_momentum=.5, optimizer_smoothing=.3).make_generator_optimizer(module)
    expected, history = torch.zeros_like(module.weight), torch.zeros_like(module.weight)
    for offset in (0., .9):
        module.weight.grad = gradient_like(module.weight, offset)
        history = .5 * history + module.weight.grad
        expected += reference_step(module, history, smoothing=.3, rate=.2)
        optimizer.step()
        torch.testing.assert_close(module.weight, expected, rtol=1e-13, atol=1e-14)
        torch.testing.assert_close(optimizer.state[module.weight]["momentum_buffer"], history)


def test_public_factories_bind_encoder_critic_and_sampled_prior_without_changing_roles():
    configuration = recipe(lr=.012, d_lr_mult=1.5, prior_lr_mult=2.5, num_particles=8, z_dim=2,
                           standardize=False, optimizer_smoothing=1e-5)
    generator = nn.Sequential(nn.ConvTranspose2d(2, 3, 3), nn.Conv2d(3, 1, 1))
    encoder = nn.Conv2d(1, 2, 3)
    critic = nn.Conv2d(1, 1, 3)
    prior = configuration.make_prior()
    opt_g, opt_d = configuration.make_optimizers(generator, critic, prior, encoder=encoder)
    assert sum("dualnorm_convolution" in group for group in opt_g.param_groups) == 3
    assert sum("dualnorm_convolution" in group for group in opt_d.param_groups) == 1
    for group in opt_g.param_groups:
        assert group["lr"] == pytest.approx(.03 if group["role"] == "prior" else .012)
    assert all(group["lr"] == pytest.approx(.018) for group in opt_d.param_groups)
    prior_group = opt_g.param_groups[-1]
    assert prior_group["algorithm"] == "rownorm" and "dualnorm_convolution" not in prior_group
    prior.z.grad = torch.ones_like(prior.z)
    before = prior.z.detach().clone()
    opt_g.set_sampled_rows(prior.z, torch.tensor([0, 2], dtype=torch.long))
    opt_g.step()
    assert torch.equal(prior.z[1], before[1]) and not torch.equal(prior.z[0], before[0])


def test_unlabelled_high_rank_and_unsupported_modes_fail_closed():
    module = nn.Conv2d(2, 2, 3)
    with pytest.raises(ValueError, match="metadata"):
        get_recipe("bcap", optimizer_convolution="none").make_optimizers(module, nn.Linear(2, 1))
    with pytest.raises(ValueError, match="metadata"):
        recipe().make_generator_optimizer(module.parameters())
    with pytest.raises(ValueError, match="metadata"):
        NormalizedOptimizer([module.weight], convolution="per_offset")
    with pytest.raises(ValueError, match="metadata"):
        recipe().make_generator_optimizer(nn.Conv1d(2, 2, 3))
    for value in (True, "flatten", None):
        with pytest.raises(ValueError, match="convolution"):
            recipe(optimizer_convolution=value)
        with pytest.raises(ValueError, match="convolution"):
            NormalizedOptimizer([nn.Parameter(torch.ones(2, 2))], convolution=value)
    for family in ("adam", "formulation", "dualnorm_D_only", "nsgda_layer"):
        with pytest.raises(ValueError, match="dualnorm"):
            recipe(optimizer_family=family)


def test_contradictory_shared_kernel_module_contracts_are_rejected():
    convolution = nn.Conv2d(4, 4, 3, groups=2, bias=False)
    transpose = nn.ConvTranspose2d(4, 4, 3, groups=2, bias=False)
    transpose.weight = convolution.weight
    with pytest.raises(ValueError, match="shared convolution"):
        recipe().make_generator_optimizer(nn.Sequential(convolution, transpose))
    with pytest.raises(ValueError, match="shared convolution"):
        recipe().make_optimizers(convolution, nn.Linear(1, 1), encoder=transpose)


@pytest.mark.parametrize("field,value", [("layout", "conv_transpose2d_in_out"), ("groups", 1),
                                        ("kernel_size", [1, 9]), ("update_version", "per_offset_polar_v2")])
def test_checkpoint_kernel_contract_rejected_before_mutating_state(field, value):
    module = nn.Conv2d(4, 4, 3, groups=2, bias=False)
    optimizer = recipe(optimizer_momentum=.5).make_generator_optimizer(module)
    module.weight.grad = gradient_like(module.weight)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    before = state_digest(saved)
    invalid = deepcopy(saved)
    invalid["param_groups"][0]["dualnorm_convolution"][field] = value
    with pytest.raises(ValueError, match="dualnorm_convolution"):
        optimizer.load_state_dict(invalid)
    assert state_digest(optimizer.state_dict()) == before
    if field == "layout":
        other = recipe(optimizer_momentum=.5).make_generator_optimizer(
            nn.ConvTranspose2d(4, 4, 3, groups=2, bias=False))
        before_other = state_digest(other.state_dict())
        with pytest.raises(ValueError, match="dualnorm_convolution"):
            other.load_state_dict(saved)
        assert state_digest(other.state_dict()) == before_other


def test_disabled_convolution_packets_and_opt_in_dense_updates_are_identical():
    base = get_recipe("bcap", optimizer_convolution="none")
    enabled = base.replace(optimizer_convolution="per_offset")
    assert "optimizer_convolution" not in base.to_dict()
    assert enabled.to_dict()["optimizer_convolution"] == "per_offset"
    left = nn.Linear(4, 2, dtype=torch.float64)
    right = deepcopy(left)
    old = base.make_generator_optimizer(left)
    new = enabled.make_generator_optimizer(right)
    assert "convolution" not in old.state_dict()["dualnorm"]
    for group in old.param_groups + new.param_groups:
        assert "dualnorm_convolution" not in group
    for a, b in zip(left.parameters(), right.parameters()):
        a.grad = gradient_like(a)
        b.grad = a.grad.clone()
    old.step()
    new.step()
    for a, b in zip(left.parameters(), right.parameters()):
        assert torch.equal(a, b)
    assert old.state_dict()["param_groups"] == new.state_dict()["param_groups"]
    with pytest.raises(ValueError):
        old.load_state_dict(new.state_dict())


def test_public_trainer_restores_exact_next_convolution_update_and_rejects_bad_layout_atomically():
    def build():
        configuration = recipe(num_particles=8, z_dim=2, batch_size=4, total_steps=8,
                               optimizer_smoothing=1e-5, standardize=False,
                               prior_kind="mog", sigma_rel=.1)
        generator = nn.Sequential(nn.Unflatten(1, (2, 1, 1)), nn.ConvTranspose2d(2, 3, 3),
                                  nn.Tanh(), nn.Conv2d(3, 1, 1))
        critic = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Tanh(), nn.Flatten(), nn.Linear(2, 1))
        prior = configuration.make_prior()
        for module in (generator, critic, prior):
            deterministic_orthogonal_(module, seed=0)
        return GANTrainer(configuration, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device="cuda:0").manual_seed(0))
    trainer = build()
    batch = torch.arange(36, dtype=torch.float32).reshape(4, 1, 3, 3).sin()
    trainer.step(batch)
    saved = deepcopy(trainer.state_dict())
    trainer.step(batch)
    expected = state_digest(trainer.state_dict())
    restored = build()
    restored.load_state_dict(saved)
    restored.step(batch)
    assert state_digest(restored.state_dict()) == expected
    invalid = deepcopy(restored.state_dict())
    conv_group = next(group for group in invalid["optimizers"][0]["param_groups"]
                      if "dualnorm_convolution" in group)
    conv_group["dualnorm_convolution"]["layout"] = "conv2d_out_in"
    # Also alter a compatible model tensor: optimizer preflight must reject
    # the packet before loading any model, optimizer, counters or RNG state.
    next(iter(invalid["models"]["G"].values())).add_(1.)
    before = state_digest(restored.state_dict())
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)
    assert state_digest(restored.state_dict()) == before
