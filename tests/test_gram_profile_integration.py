"""CPU integration contracts, not GAN acquisition or training experiments.

Private seed0 fixtures contain three small FP32 matrices and a bias. Numerical
gates are FP64-oracle agreement (2e-6 relative/2e-7 absolute), the existing public
DirectionBlend law, and byte-exact same-profile checkpoint continuation. Trainer
tests exercise synthetic optimizer updates and public checkpoint APIs only;
GANTrainer.step, sample generation, CUDA queries and GIFs are deliberately absent.
"""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim import GramGuardError
from particlegan.optim import dualnorm
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.optim.strict_progress import common_descent


PROFILE = "gram_fp64_guarded_v1"


def same_tree(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                           right.detach().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_tree(a, b)
    else:
        assert left == right


def settings():
    return (torch.get_default_dtype(), torch.get_default_device(),
            torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
            torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"),
            torch.autograd.is_multithreading_enabled())


@pytest.fixture(autouse=True)
def caller_runtime_is_preserved():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng, flags = torch.get_rng_state().clone(), settings()
    try:
        yield
        same_tree(torch.get_rng_state(), rng)
        assert settings() == flags
    finally:
        torch.set_num_threads(threads)


def fixture_values():
    stream = torch.Generator(device="cpu").manual_seed(0)
    shapes = ((2, 2), (3, 2), (2, 3), (2,))
    values = [torch.randn(shape, generator=stream, device="cpu") * .2 for shape in shapes]
    gradients = [torch.randn(shape, generator=stream, device="cpu") for shape in shapes]
    return values, gradients


def build(values, *, protected=True, backend="native"):
    parameters = [nn.Parameter(value.clone()) for value in values]
    cls = DirectionBlendOptimizer if protected else NormalizedOptimizer
    optimizer = cls(parameters, family="dualnorm", momentum=0., smoothing=.001,
                    lr=.03, dense_profile=PROFILE, svd_backend=backend,
                    convolution="per_offset" if backend == "batched" else "none")
    return parameters, optimizer


def flatten(values):
    return torch.cat([value.detach().flatten() for value in values])


def linear(parameters, coefficients):
    return sum((parameter * coefficient).sum() for parameter, coefficient in zip(parameters, coefficients))


def normal_rows(gradients, branch):
    gradient = flatten(gradients)
    if branch == "inactive":
        return gradient[None]
    if branch == "R1":
        return -gradient[None]
    if branch == "pareto":
        return torch.stack((-gradient, gradient))
    tangent = torch.arange(1, gradient.numel() + 1, dtype=torch.float32, device="cpu")
    tangent -= (tangent @ gradient) / (gradient @ gradient) * gradient
    tangent *= .1 * gradient.norm() / tangent.norm()
    return torch.stack((-gradient, -gradient + tangent))


def bind(parameters, optimizer, gradients, normals):
    optimizer.zero_grad(set_to_none=True)
    flat = torch.cat([parameter.flatten() for parameter in parameters])
    protected = tuple((flat * row).sum() for row in normals)
    def forbidden():
        raise AssertionError("DirectionBlend cannot probe finite losses")
    constraint_geometry_backward(linear(parameters, gradients), optimizer, protected,
                                 protected_evaluator=forbidden)


def oracle_base(values, gradients, rate=.03):
    expected = []
    for value, gradient in zip(values, gradients):
        if value.ndim == 2:
            u, singular, vh = torch.linalg.svd(gradient.double(), full_matrices=False)
            cutoff = max(gradient.shape) * torch.finfo(torch.float32).eps * singular[0]
            gains = (singular > cutoff) * singular / torch.hypot(singular, singular.new_tensor(.001))
            direction = ((u * gains) @ vh).float()
            factor = math.sqrt(max(1., value.shape[0] / value.shape[1]))
        else:
            direction = gradient / torch.hypot(gradient.norm(), gradient.new_tensor(.001))
            factor = 1.
        target = value.clone()
        target.add_(direction, alpha=-rate * factor)
        expected.append(target)
    return expected


@pytest.mark.parametrize("backend", ["native", "batched"])
@pytest.mark.parametrize("branch", ["inactive", "R1", "R2", "pareto"])
def test_public_protected_branches_keep_selected_dense_math_and_one_clock(backend, branch):
    values, gradients = fixture_values()
    plain, base = build(values, protected=False, backend=backend)
    for parameter, gradient in zip(plain, gradients):
        parameter.grad = gradient.clone()
    base.step()
    for actual, expected in zip(plain, oracle_base(values, gradients)):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    parameters, optimizer = build(values, backend=backend)
    normals = normal_rows(gradients, branch)
    bind(parameters, optimizer, gradients, normals)
    same_tree(optimizer.state_dict()["constraint_geometry"]["pending"], normals)
    grad_refs = [(p.grad, p.grad._version, p.grad.clone()) for p in parameters]
    flags = [p.requires_grad for p in parameters]
    displacement = flatten([p.detach() - old for p, old in zip(plain, values)])
    conflicting = bool((normals.double() @ displacement.double() > 0).any())
    assert conflicting == (branch != "inactive")
    direction, possible = common_descent(displacement, normals) if conflicting else (displacement, True)
    optimizer.step()
    if branch == "inactive":
        for actual, expected in zip(parameters, plain):
            same_tree(actual.detach(), expected.detach())
    else:
        expected_flat = flatten(values) + direction if possible else flatten(values)
        same_tree(flatten(parameters), expected_flat)
    for parameter, (reference, version, gradient) in zip(parameters, grad_refs):
        assert parameter.grad is reference and parameter.grad._version == version
        same_tree(parameter.grad, gradient)
        assert optimizer.state[parameter]["step"] == 1
    assert [p.requires_grad for p in parameters] == flags
    assert optimizer._protected is None
    assert optimizer.state_dict()["constraint_geometry"]["pending"] is None
    assert optimizer.constraint_geometry_stats["steps"] == 1
    assert optimizer.direction_blend_stats["conflict_steps"] == int(conflicting)
    assert optimizer.direction_blend_stats["pareto_stalls"] == int(branch == "pareto")
    assert optimizer.direction_blend_stats["blended_steps"] == int(branch in ("R1", "R2"))
    if branch in ("R1", "R2"):
        actual = flatten([p.detach() - old for p, old in zip(parameters, values)])
        assert float((normals.double() @ actual.double()).max()) < 0


def test_inactive_fp32_boundary_preserves_applied_bytes_without_recomposition():
    values = [torch.zeros(2, 2, device="cpu") for _ in range(3)] + [torch.tensor([.1234567], device="cpu")]
    gradients = [torch.diag(torch.tensor([1., .2], device="cpu")) for _ in range(3)] + [torch.ones(1, device="cpu")]
    plain, base = build(values, protected=False)
    parameters, optimizer = build(values)
    for opt in (base, optimizer):
        opt.param_groups[0]["lr"] = .12345676
    for parameter, gradient in zip(plain, gradients):
        parameter.grad = gradient.clone()
    base.step()
    assert not torch.equal(values[-1] + (plain[-1].detach() - values[-1]), plain[-1].detach())
    bind(parameters, optimizer, gradients, normal_rows(gradients, "inactive"))
    optimizer.step()
    for actual, expected in zip(parameters, plain):
        same_tree(actual.detach(), expected.detach())
    assert optimizer.constraint_geometry_stats["projected_steps"] == 0


@pytest.mark.parametrize("backend", ["native", "batched"])
def test_public_pending_restore_then_fixed_next_update_is_byte_exact(backend):
    values, gradients = fixture_values()
    parameters, optimizer = build(values, backend=backend)
    bind(parameters, optimizer, gradients, normal_rows(gradients, "R2"))
    pending = deepcopy(optimizer.state_dict())
    restored_parameters, restored = build(values, backend=backend)
    restored.load_state_dict(pending)
    for parameter, source in zip(restored_parameters, parameters):
        parameter.grad = source.grad.clone()
    same_tree(restored.state_dict(), pending)
    optimizer.step()
    restored.step()
    same_tree(flatten(parameters), flatten(restored_parameters))
    same_tree(optimizer.state_dict(), restored.state_dict())
    fresh_parameters, fresh = build([p.detach() for p in parameters], backend=backend)
    fresh.load_state_dict(deepcopy(optimizer.state_dict()))
    next_gradients = [gradient.flip(-1).clone() for gradient in gradients]
    for ps, opt in ((parameters, optimizer), (restored_parameters, restored), (fresh_parameters, fresh)):
        bind(ps, opt, next_gradients, normal_rows(next_gradients, "R1"))
        opt.step()
        assert opt._protected is None
        assert all(opt.state[p]["step"] == 2 for p in ps)
    same_tree(flatten(parameters), flatten(fresh_parameters))
    same_tree(flatten(parameters), flatten(restored_parameters))
    same_tree(optimizer.state_dict(), fresh.state_dict())
    same_tree(optimizer.state_dict(), restored.state_dict())


def test_later_gram_hold_is_terminal_no_svd_fallback_or_rollback_claim(monkeypatch):
    values = [torch.zeros(2, 2, device="cpu") for _ in range(3)] + [torch.zeros(2, device="cpu")]
    gradients = [torch.diag(torch.tensor([1., .2], device="cpu")),
                 torch.diag(torch.tensor([1., 2 * torch.finfo(torch.float32).eps], device="cpu")),
                 torch.eye(2, device="cpu"), torch.ones(2, device="cpu")]
    parameters, optimizer = build(values)
    bind(parameters, optimizer, gradients, normal_rows(gradients, "inactive"))
    def forbidden(*args, **kwargs):
        raise AssertionError("A terminal Gram HOLD must not run native SVD")
    monkeypatch.setattr(dualnorm, "polar_factor", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    with pytest.raises(GramGuardError):
        optimizer.step()
    assert torch.count_nonzero(parameters[0]) > 0
    same_tree(parameters[1].detach(), values[1])
    same_tree(parameters[2].detach(), values[2])
    assert [optimizer.state[p].get("step", 0) for p in parameters] == [1, 1, 0, 0]
    assert optimizer.constraint_geometry_stats["steps"] == 0
    # A failed base update is not rolled back (partial tensors/clocks above),
    # but its pending normals are aborted so the next bind/step can proceed.
    assert optimizer._protected is None
    assert optimizer.direction_blend_stats["conflict_steps"] == 0


@pytest.mark.parametrize("change", ["appended_group", "changed_parameter_dtype"])
def test_selected_ineligible_dense_is_rejected_before_any_step_mutation(change):
    values, gradients = fixture_values()
    parameters, optimizer = build(values, protected=False)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    if change == "appended_group":
        extra = nn.Parameter(torch.ones(2, 2, dtype=torch.float64, device="cpu"))
        optimizer.add_param_group(dict(params=[extra], role="generator"))
        extra.grad = torch.ones_like(extra)
        parameters.append(extra)
    else:
        parameters[-2].data = parameters[-2].data.double()
        parameters[-2].grad = gradients[-2].double()
    before = [p.detach().clone() for p in parameters]
    state = deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="FP32"):
        optimizer.step()
    same_tree(optimizer.state_dict(), state)
    for parameter, expected in zip(parameters, before):
        same_tree(parameter.detach(), expected)


def trainer(profile):
    # Construction and public deterministic initialization cannot consume the
    # caller stream. Frozen prior keeps this fixture about dense integration.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe("bcap", z_dim=2, num_particles=4, batch_size=2,
            total_steps=3, standardize=False, prior_update="frozen",
            optimizer_dense_profile=profile)
        generator = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2))
        critic = nn.Linear(2, 1)
        prior = recipe.make_prior()
        for model in (generator, critic, prior):
            deterministic_orthogonal_(model, seed=0)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


def synthetic_trainer_optimizer_update(value):
    # No trainer.step or real/fake sampling: only a synthetic scalar API fixture.
    stream = torch.Generator(device="cpu").manual_seed(0)
    parameters = [p for p in value.G.parameters() if p.requires_grad]
    gradients = [torch.randn(p.shape, generator=stream, device="cpu") for p in parameters]
    bind(parameters, value.opt_g, gradients, normal_rows(gradients, "R2"))
    value.opt_g.step()
    value.opt_d.zero_grad(set_to_none=True)
    for parameter in value.D.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=stream, device="cpu")
    value.opt_d.step()
    assert value.completed_steps == 0  # Explicitly not a GAN training update.


def test_public_trainer_selected_profile_checkpoint_and_fixed_optimizer_continuation():
    source = trainer(PROFILE)
    synthetic_trainer_optimizer_update(source)
    saved = source.state_dict()
    assert saved["recipe"]["optimizer_dense_profile"] == PROFILE
    assert all(opt["dualnorm"]["dense_profile"] == PROFILE for opt in saved["optimizers"])
    target = trainer(PROFILE)
    target.load_state_dict(deepcopy(saved))
    same_tree(target.state_dict(), saved)
    synthetic_trainer_optimizer_update(source)
    synthetic_trainer_optimizer_update(target)
    same_tree(source.state_dict(), target.state_dict())
    assert source.opt_g._protected is target.opt_g._protected is None


def test_public_trainer_default_omitted_and_explicit_recipe_defaults_still_restore():
    source = trainer("default")
    saved = source.state_dict()
    assert "optimizer_dense_profile" not in saved["recipe"]
    assert all("dense_profile" not in opt["dualnorm"] for opt in saved["optimizers"])
    historical = deepcopy(saved)
    historical["recipe"]["optimizer_dense_profile"] = "default"
    target = trainer("default")
    target.load_state_dict(historical)
    same_tree(source.state_dict(), target.state_dict())
    synthetic_trainer_optimizer_update(source)
    synthetic_trainer_optimizer_update(target)
    same_tree(source.state_dict(), target.state_dict())


@pytest.mark.parametrize("source_profile,target_profile", [("default", PROFILE), (PROFILE, "default")])
def test_public_trainer_cross_profile_restore_rejects_before_live_mutation(source_profile, target_profile):
    source, target = trainer(source_profile), trainer(target_profile)
    before = target.state_dict()
    with pytest.raises(ValueError, match="recipe"):
        target.load_state_dict(source.state_dict())
    same_tree(target.state_dict(), before)
