"""CUDA software contracts; no distribution-quality or qualification claim."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init
from benchmarks.toy_audit.reproducibility import construction_rng, reproducible_execution
from experiments.forge import adapters
from experiments.forge.api import CapabilityError
from experiments.forge.state import state_digest


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA policy check requires GPU")
DEVICE = "cuda:0"


def trainer():
    recipe = get_recipe("bcap", z_dim=2, num_particles=16, batch_size=8,
                        total_steps=4, prior_kind="mog", sigma_rel=.1, standardize=False)
    with construction_rng(0, DEVICE):
        G = nn.Sequential(nn.Linear(2, 4, device=DEVICE), nn.LeakyReLU(.2),
                          nn.Linear(4, 2, device=DEVICE))
        D = nn.Sequential(nn.Linear(2, 4, device=DEVICE), nn.LeakyReLU(.2),
                          nn.Linear(4, 1, device=DEVICE))
        prior = recipe.make_prior(sigma=.1).to(DEVICE)
        for offset, model in enumerate((G, D, prior)):
            init.deterministic_orthogonal_(model, seed=offset)
        return GANTrainer(recipe, G, D, prior=prior, seed=0)


def test_import_default_changes_only_autograd_scheduler():
    # Fresh interpreter: imports must enforce policy without changing compute
    # thread pools, CUDA flags, or replacing Torch's context-manager function.
    source = """
import torch
torch.set_num_threads(2)
torch.autograd.set_multithreading_enabled(True)
before = (torch.get_num_threads(), torch.get_num_interop_threads(),
          torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
original = torch.autograd.set_multithreading_enabled
import particlegan
assert not torch.autograd.is_multithreading_enabled()
assert torch.autograd.set_multithreading_enabled is original
assert before == (torch.get_num_threads(), torch.get_num_interop_threads(),
                  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
assert torch.ones(1, device='cuda:0').is_cuda
"""
    subprocess.run([sys.executable, "-c", source], check=True, timeout=30)


def test_public_update_scopes_forward_and_backward_and_resumes_exactly():
    owner = trainer()
    modes = []
    handles = []
    for model in (owner.G, owner.D):
        handles.append(model.register_forward_pre_hook(
            lambda *args: modes.append(torch.autograd.is_multithreading_enabled())))
        for parameter in model.parameters():
            handles.append(parameter.register_hook(
                lambda gradient: modes.append(torch.autograd.is_multithreading_enabled())))
    real = torch.linspace(-1., 1., 16, device=DEVICE).reshape(8, 2)
    initial = owner.G[0].weight.detach().clone()
    with torch.autograd.set_multithreading_enabled(True):
        owner.step(real)
        assert torch.autograd.is_multithreading_enabled()
        checkpoint = owner.state_dict()
        owner.step(real)
        expected = owner.state_dict()
    assert owner.serial_backward is True
    assert checkpoint["serial_backward"] is True
    assert modes and not any(modes)
    assert not torch.equal(initial, owner.G[0].weight)
    assert all(p.is_cuda and bool(torch.isfinite(p).all()) for p in owner.G.parameters())
    restored = trainer()
    restored.load_state_dict(checkpoint)
    with torch.autograd.set_multithreading_enabled(True):
        restored.step(real)
        assert torch.autograd.is_multithreading_enabled()
    assert state_digest(restored.state_dict()) == state_digest(expected)
    for handle in handles:
        handle.remove()


def test_disabled_mode_cannot_be_overridden_or_silently_resume_legacy_state():
    owner = trainer()
    with pytest.raises(ValueError, match="requires serial_backward=True"):
        GANTrainer(owner.recipe, owner.G, owner.D, prior=owner.prior, serial_backward=False)
    with pytest.raises(TypeError, match="boolean"):
        GANTrainer(owner.recipe, owner.G, owner.D, prior=owner.prior, serial_backward=1)
    with pytest.raises(AttributeError):
        owner.serial_backward = False
    original = owner.state_dict()
    for marker in (False, None):
        legacy = deepcopy(original)
        if marker is None:
            legacy.pop("serial_backward")
        else:
            legacy["serial_backward"] = marker
        with pytest.raises(ValueError, match="pinned original source"):
            owner.load_state_dict(legacy)
        assert state_digest(owner.state_dict()) == state_digest(original)
    with torch.autograd.set_multithreading_enabled(True):
        def fail(*args):
            assert not torch.autograd.is_multithreading_enabled()
            raise RuntimeError("software failure")
        owner.G.register_forward_pre_hook(fail)
        with pytest.raises(RuntimeError, match="software failure"):
            owner.step(torch.zeros(8, 2, device=DEVICE))
        assert torch.autograd.is_multithreading_enabled()
    assert owner.completed_steps == 0


def test_benchmark_and_forge_scopes_enforce_policy_in_new_worker(tmp_path, monkeypatch):
    modes = []

    def gradient_probe():
        modes.append(torch.autograd.is_multithreading_enabled())
        x = torch.tensor([2.], device=DEVICE, requires_grad=True)
        derivative = torch.autograd.grad(x.pow(3).sum(), x, create_graph=True)[0]
        curvature = torch.autograd.grad(derivative.sum(), x)[0]
        assert torch.equal(curvature, torch.tensor([12.], device=DEVICE))

    def dispatch(*args, **kwargs):
        gradient_probe()
        return {"cost": {}}

    monkeypatch.setattr(adapters, "_dispatch_task", dispatch)

    @reproducible_execution
    def benchmark(*, device):
        gradient_probe()

    def worker():
        with torch.autograd.set_multithreading_enabled(True):
            result = adapters.run_task({}, {}, tmp_path, DEVICE)
            assert result["autograd_multithreading_enabled"] is False
            assert torch.autograd.is_multithreading_enabled()
            benchmark(device=DEVICE)
            assert torch.autograd.is_multithreading_enabled()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(worker).result(timeout=30)
    assert modes == [False, False]


def test_pinned_legacy_owner_factories_block_before_models(tmp_path, monkeypatch):
    from experiments.forge import canonical_image_adapter, canonical_mog_adapter
    assert torch.ones(1, device=DEVICE).is_cuda

    def forbidden(*args, **kwargs):
        pytest.fail("legacy compatibility check constructed a model or optimizer")

    monkeypatch.setattr(nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    factories = (
        canonical_mog_adapter._trainer_class,
        lambda: canonical_mog_adapter.construct_owner(
            tmp_path, {}, device=DEVICE, source_guard=lambda: None),
        lambda: canonical_image_adapter.construct_owner(
            tmp_path, {}, device=DEVICE, source_guard=lambda: None),
    )
    for factory in factories:
        with pytest.raises(CapabilityError, match="pinned original GANTrainer source") as error:
            factory()
        assert error.value.status == "BLOCKED"
