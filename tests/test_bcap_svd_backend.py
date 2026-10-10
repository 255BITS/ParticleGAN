"""Bounded software algebra/resume checks; no trained-gate qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.api_images import WordFixture
from experiments.forge.api import task_formulation_context
from particlegan import get_recipe
from particlegan.optim.dualnorm import NormalizedOptimizer, polar_factor


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("backend", [None, False, "gesvd", "approximate"])
def test_invalid_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="backend"):
        get_recipe("bcap", optimizer_svd_backend=backend)
    with pytest.raises(ValueError, match="backend"):
        NormalizedOptimizer([nn.Parameter(torch.zeros(2, 2))], svd_backend=backend)


@pytest.mark.parametrize("family", ["formulation", "adam", "sgda", "dualnorm_D_only"])
def test_cpu_backend_requires_full_dualnorm(family):
    with pytest.raises(ValueError, match="dualnorm"):
        get_recipe("bcap", optimizer_family=family, optimizer_smoothing=0.,
                   optimizer_convolution="none", optimizer_svd_backend="cpu")


def test_native_defaults_preserve_old_recipe_and_optimizer_packet_fields():
    recipe = get_recipe("bcap")
    assert "optimizer_svd_backend" not in recipe.to_dict()
    weight = nn.Parameter(torch.zeros(2, 3))
    native = NormalizedOptimizer([weight])
    assert native.state_dict()["dualnorm"].keys() == {"schema", "family", "momentum", "sampled_rows"}
    assert all("svd_backend" not in group for group in native.state_dict()["param_groups"])
    native.load_state_dict(deepcopy(native.state_dict()))
    assert recipe.replace(optimizer_svd_backend="cpu").to_dict()["optimizer_svd_backend"] == "cpu"
    from experiments.forge.techniques import validate_same_technique
    with pytest.raises(ValueError, match="optimizer_svd_backend"):
        validate_same_technique(recipe, recipe.replace(optimizer_svd_backend="cpu"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("smoothing", [0., .001, .3])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (0, 2)])
def test_cpu_input_polar_is_bitwise_native(dtype, smoothing, shape):
    matrix = torch.arange(shape[0] * shape[1], dtype=torch.float64).reshape(shape).sin().to(dtype)
    expected = polar_factor(matrix, smoothing=smoothing)
    actual = polar_factor(matrix, smoothing=smoothing, backend="cpu")
    assert actual.device == matrix.device and actual.dtype == dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_roles_history_sampled_rows_and_resume(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    def build(backend):
        parameters = [nn.Parameter(torch.zeros(3, 2, device=device)) for _ in range(4)]
        groups = [dict(params=[p], role=role) for p, role in zip(parameters,
                  ("generator", "encoder", "critic", "prior"))]
        return parameters, NormalizedOptimizer(groups, momentum=.5, smoothing=.001,
                                               svd_backend=backend, lr=.03)
    parameters, optimizer = build("cpu")
    def update(parameters, optimizer, offset):
        for p in parameters:
            p.grad = (torch.arange(p.numel(), device=device).reshape(p.shape) + offset).float().sin()
        optimizer.set_sampled_rows(parameters[-1], torch.tensor([0, 2], device=device))
        optimizer.step()
    update(parameters, optimizer, .2)
    assert torch.equal(parameters[-1][1], torch.zeros_like(parameters[-1][1]))
    saved = deepcopy(optimizer.state_dict())
    assert saved["dualnorm"]["svd_backend"] == "cpu"
    originals = [p.detach().clone() for p in parameters]
    update(parameters, optimizer, .7)
    restored, resumed = build("cpu")
    with torch.no_grad():
        for p, value in zip(restored, originals):
            p.copy_(value)
    resumed.load_state_dict(saved)
    update(restored, resumed, .7)
    equal(parameters, restored)
    equal(optimizer.state_dict(), resumed.state_dict())
    assert all(optimizer.state[p]["step"] == 2 for p in parameters)
    assert optimizer._sampled_rows == {}
    _, native = build("native")
    native_before = deepcopy(native.state_dict())
    with pytest.raises(ValueError, match="checkpoint"):
        native.load_state_dict(saved)
    equal(native_before, native.state_dict())
    active_before = deepcopy(resumed.state_dict())
    with pytest.raises(ValueError, match="checkpoint"):
        resumed.load_state_dict(native_before)
    equal(active_before, resumed.state_dict())
    # A CPU-native input takes exactly the same update operations in both modes.
    if device == "cpu":
        native_parameters, native = build("native")
        update(native_parameters, native, .2)
        update(native_parameters, native, .7)
        equal(parameters, native_parameters)
        native_state = native.state_dict()
        active_state = optimizer.state_dict()
        active_state["dualnorm"].pop("svd_backend")
        equal(native_state, active_state)


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_grouped_convolution_cpu_reference_and_resume(kind, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    def build(target):
        module = kind(6, 4, (2, 3), groups=2, bias=True, device=target)
        with torch.no_grad():
            for p in module.parameters():
                p.zero_()
        recipe = get_recipe("bcap", optimizer_svd_backend="cpu", optimizer_convolution="per_offset",
                            optimizer_smoothing=.001, lr=.03)
        return module, recipe.make_generator_optimizer(module)
    module, optimizer = build(device)
    reference, cpu = build("cpu")
    def update(module, optimizer):
        for p in module.parameters():
            p.grad = (torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape) * .3).sin().to(p.device)
        optimizer.step()
    update(module, optimizer)
    update(reference, cpu)
    for p, expected in zip(module.parameters(), reference.parameters()):
        # Kernel polar is exactly the CPU reference; bias normalization remains native.
        if p.ndim == 4:
            assert torch.equal(p.cpu(), expected)
        else:
            torch.testing.assert_close(p.cpu(), expected, atol=2e-8, rtol=1e-6)
        assert torch.isfinite(p).all()
    saved = deepcopy(optimizer.state_dict())
    model_saved = deepcopy(module.state_dict())
    update(module, optimizer)
    restored, resumed = build(device)
    restored.load_state_dict(model_saved)
    resumed.load_state_dict(saved)
    update(restored, resumed)
    equal(module.state_dict(), restored.state_dict())
    equal(optimizer.state_dict(), resumed.state_dict())


def words(backend, device):
    candidate = json.loads((ROOT / "configs/forge/ideas/bcap-develop-integration-combined-v1.json").read_text())
    candidate["recipe_overrides"]["optimizer_svd_backend"] = backend
    task = json.loads((ROOT / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    return context, WordFixture(device=device, seed=0, recipe_name=None, max_steps=20001, components=context)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_public_word_active_backend_exact_resume_and_named_streams(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    context, fixture = words("cpu", device)
    for _ in range(3):
        fixture.step()
    saved, named = deepcopy(fixture.state_dict()), deepcopy(context.streams.state_dict())
    suffix = [fixture.step() for _ in range(2)]
    observation = fixture.observe()
    restored_context, restored = words("cpu", device)
    restored.policy.load_state_dict(saved["api_state"])
    restored.data_generator.set_state(saved["data_generator"])
    restored.restore_component_transport(saved.get("component_transport"))
    restored_context.streams.load_state_dict(named)
    equal(suffix, [restored.step() for _ in range(2)])
    equal(observation, restored.observe())
    equal(fixture.state_dict(), restored.state_dict())
    equal(context.streams.state_dict(), restored_context.streams.state_dict())
    assert fixture.transport.active_calls == 5
    assert fixture.opt_g.constraint_geometry_stats["steps"] == 5


def test_rank_deficient_cuda_cpu_polar_uses_full_cpu_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    matrix = torch.tensor([[3., 0., 0.], [0., 1e-9, 0.]])
    expected = polar_factor(matrix, smoothing=.001)
    actual = polar_factor(matrix.cuda(), smoothing=.001, backend="cpu")
    assert torch.equal(actual.cpu(), expected)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 1
    assert torch.linalg.matrix_norm(actual.cpu(), ord=2) <= 1
    with pytest.raises(torch.linalg.LinAlgError):
        polar_factor(torch.full((2, 2), float("nan"), device="cuda"), backend="cpu")
