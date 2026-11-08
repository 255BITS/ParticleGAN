"""Tiny opt-in CUDA numerical integration; no throughput or GAN quality claim.

Root owns admission and CUDA execution. Collection never queries a CUDA device;
the runtime fixture skips when unavailable. All random fixtures originate from
private seed0 CPU generators and use explicit logical cuda:0 after admission.
Bounds: at most four parameters, largest matrix4x3, fifteen test instances.
"""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan.optim import GramGuardError
from particlegan.optim import dualnorm
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.optim.strict_progress import common_descent
from test_gram_profile_integration import (
    PROFILE, bind, fixture_values, flatten, normal_rows, same_tree, settings,
)


pytestmark = pytest.mark.cuda


@pytest.fixture
def cuda_device():
    # Deliberately a runtime fixture: CPU collection initializes/queries no GPU.
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU profile coverage is separate")
    device = torch.device("cuda:0")
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    flags = settings()
    yield device
    same_tree(torch.get_rng_state(), cpu_rng)
    same_tree(torch.cuda.get_rng_state(device), cuda_rng)
    assert settings() == flags


def optimizer(parameters, *, protected=False, backend="native", rate=.03):
    cls = DirectionBlendOptimizer if protected else NormalizedOptimizer
    return cls(parameters, family="dualnorm", momentum=0., smoothing=.001,
               lr=rate, dense_profile=PROFILE, svd_backend=backend,
               convolution="per_offset" if backend == "batched" else "none")


def build(values, device, *, protected=True, backend="native"):
    parameters = [nn.Parameter(value.detach().to(device).clone()) for value in values]
    return parameters, optimizer(parameters, protected=protected, backend=backend)


def cpu_packet(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_packet(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_packet(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_packet(child) for child in value)
    return deepcopy(value)


def matrix_fixture(shape, rank_deficient):
    stream = torch.Generator(device="cpu").manual_seed(0)
    size = min(shape)
    left, _ = torch.linalg.qr(torch.randn(shape[0], size, generator=stream,
                                         dtype=torch.float64, device="cpu"))
    right, _ = torch.linalg.qr(torch.randn(shape[1], size, generator=stream,
                                          dtype=torch.float64, device="cpu"))
    singular = torch.tensor([1.7, .2, 0. if rank_deficient else .03], dtype=torch.float64, device="cpu")
    gradient = ((left * singular) @ right.T).float()
    initial = torch.randn(shape, generator=stream, dtype=torch.float32, device="cpu") * .2
    return initial, gradient


def cpu64_oracle(gradient):
    u, singular, vh = torch.linalg.svd(gradient.double(), full_matrices=False)
    cutoff = max(gradient.shape) * torch.finfo(torch.float32).eps * singular[0]
    gains = (singular > cutoff) * singular / torch.hypot(singular, singular.new_tensor(.001))
    return ((u * gains) @ vh).float()


@pytest.mark.parametrize("shape", [(3, 3), (4, 3), (3, 4)], ids=["square", "tall", "wide"])
@pytest.mark.parametrize("rank_deficient", [False, True], ids=["full_rank", "rank_deficient"])
def test_cuda_selected_public_step_matches_cpu64_and_actual_fp32_aspect_delta(cuda_device, shape, rank_deficient):
    initial, gradient = matrix_fixture(shape, rank_deficient)
    direction = cpu64_oracle(gradient)
    factor = math.sqrt(max(1., shape[0] / shape[1]))
    expected = initial.clone()
    expected.add_(direction, alpha=-.03 * factor)
    expected_delta = expected - initial  # Actual stored FP32 subtraction first.
    parameter = nn.Parameter(initial.to(cuda_device))
    parameter.grad = gradient.to(cuda_device)
    reference, version = parameter.grad, parameter.grad._version
    before = parameter.detach().clone()
    opt = optimizer([parameter])
    opt.step()
    actual_delta = (parameter.detach() - before).cpu()
    torch.testing.assert_close(parameter.detach().cpu(), expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual_delta.double(), expected_delta.double(), rtol=2e-6, atol=2e-7)
    assert parameter.grad is reference and parameter.grad._version == version
    same_tree(parameter.grad.cpu(), gradient)
    assert opt.state[parameter]["step"] == 1


@pytest.mark.parametrize("branch", ["inactive", "R1", "R2", "pareto"])
def test_cuda_selected_directionblend_uses_actual_public_rounded_branch(cuda_device, branch):
    values, cpu_gradients = fixture_values()
    gradients = [gradient.to(cuda_device) for gradient in cpu_gradients]
    normals = normal_rows(cpu_gradients, branch).to(cuda_device)
    plain, base = build(values, cuda_device, protected=False)
    parameters, opt = build(values, cuda_device)
    for parameter, gradient in zip(plain, gradients):
        parameter.grad = gradient.clone()
    base.step()
    originals = [value.to(cuda_device) for value in values]
    displacement = flatten([parameter.detach() - old for parameter, old in zip(plain, originals)])
    conflict = bool((normals.double() @ displacement.double() > 0).any())
    assert conflict == (branch != "inactive")
    direction, possible = common_descent(displacement, normals) if conflict else (displacement, True)
    bind(parameters, opt, gradients, normals)
    grad_refs = [(p.grad, p.grad._version, p.grad.clone()) for p in parameters]
    flags = [p.requires_grad for p in parameters]
    opt.step()
    if branch == "inactive":
        for actual, expected in zip(parameters, plain):
            same_tree(actual.detach(), expected.detach())
    else:
        expected = flatten(originals) + direction if possible else flatten(originals)
        same_tree(flatten(parameters), expected)
    for parameter, (reference, version, gradient) in zip(parameters, grad_refs):
        assert parameter.grad is reference and parameter.grad._version == version
        same_tree(parameter.grad, gradient)
        assert opt.state[parameter]["step"] == 1
    assert [p.requires_grad for p in parameters] == flags
    assert opt._protected is None and opt.constraint_geometry_stats["steps"] == 1
    assert opt.direction_blend_stats["pareto_stalls"] == int(branch == "pareto")
    assert opt.direction_blend_stats["blended_steps"] == int(branch in ("R1", "R2"))


@pytest.mark.parametrize("backend", ["native", "batched"])
def test_cuda_selected_public_cpu_pending_restore_and_fixed_next_are_exact(cuda_device, backend):
    values, cpu_gradients = fixture_values()
    gradients = [value.to(cuda_device) for value in cpu_gradients]
    normals = normal_rows(cpu_gradients, "R2").to(cuda_device)
    parameters, opt = build(values, cuda_device, backend=backend)
    bind(parameters, opt, gradients, normals)
    saved_cpu = cpu_packet(opt.state_dict())
    restored_parameters, restored = build(values, cuda_device, backend=backend)
    restored.load_state_dict(saved_cpu)
    for parameter, source in zip(restored_parameters, parameters):
        parameter.grad = source.grad.clone()
    same_tree(restored.state_dict(), opt.state_dict())
    opt.step()
    restored.step()
    same_tree(flatten(parameters), flatten(restored_parameters))
    same_tree(opt.state_dict(), restored.state_dict())
    fresh_parameters, fresh = build([parameter.detach().cpu() for parameter in parameters], cuda_device, backend=backend)
    fresh.load_state_dict(cpu_packet(opt.state_dict()))
    next_cpu = [value.flip(-1).clone() for value in cpu_gradients]
    next_gradients = [value.to(cuda_device) for value in next_cpu]
    next_normals = normal_rows(next_cpu, "R1").to(cuda_device)
    for ps, candidate in ((parameters, opt), (restored_parameters, restored), (fresh_parameters, fresh)):
        bind(ps, candidate, next_gradients, next_normals)
        candidate.step()
        assert candidate._protected is None
        assert all(candidate.state[p]["step"] == 2 for p in ps)
    same_tree(flatten(parameters), flatten(fresh_parameters))
    same_tree(flatten(parameters), flatten(restored_parameters))
    same_tree(opt.state_dict(), fresh.state_dict())
    same_tree(opt.state_dict(), restored.state_dict())


@pytest.mark.parametrize("adjacent", ["below", "tie", "above"])
def test_cuda_exact_and_adjacent_cutoff_is_terminal_without_svd_fallback(cuda_device, adjacent, monkeypatch):
    cutoff = torch.tensor(2 * torch.finfo(torch.float32).eps, device="cpu")
    tail = cutoff
    if adjacent == "below":
        tail = torch.nextafter(cutoff, cutoff.new_tensor(0.))
    elif adjacent == "above":
        tail = torch.nextafter(cutoff, cutoff.new_tensor(1.))
    gradient = torch.diag(torch.stack((cutoff.new_tensor(1.), tail))).to(cuda_device)
    parameter = nn.Parameter(torch.zeros(2, 2, device=cuda_device))
    parameter.grad = gradient
    before = parameter.detach().clone()
    opt = optimizer([parameter])
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA Gram HOLD must not retry native SVD")
    monkeypatch.setattr(dualnorm, "polar_factor", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    with pytest.raises(GramGuardError):
        opt.step()
    same_tree(parameter.detach(), before)
    assert opt.state[parameter]["step"] == 1  # Terminal failure is not rollback.
