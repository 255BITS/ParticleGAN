"""Many-site routing keeps synchronization bounded without changing recovery."""
import importlib
import io
import math
from pathlib import Path

import pytest
import torch

from particlegan.continuous import DataDriftController


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.fixture(autouse=True)
def serial_backward():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def api():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
        return importlib.import_module("e22_routed_readbacks")


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("device", DEVICES)
def test_71_site_lazy_and_eager_observation_match_gradients_resume_and_serving(api, monkeypatch, device):
    # Observe diagnostics and validate values after every site on one owner,
    # reproducing the former synchronization boundaries with identical math.
    loops = [api.make_loop(device=device, sites=71, tokens=2, particles=8, batch_size=4)
             for _ in range(2)]
    lazy, eager = loops
    eager_controller = eager.policy.controller
    original = DataDriftController.perturb_latent

    def observe(controller, *args, **kwargs):
        result = original(controller, *args, **kwargs)
        if controller is eager_controller and kwargs.get("record", False):
            assert len(controller.latent_applications) <= 2
        return result

    monkeypatch.setattr(DataDriftController, "perturb_latent", observe)
    forward = eager.policy.routed_control.spec.model_forward

    def eager_forward(models, context, candidate, execution):
        mix = execution.mix

        def checked(site, logits):
            assert bool(torch.isfinite(logits).all())
            assert bool(torch.isfinite((logits.to(candidate.table.dtype) + candidate.log_mass).softmax(-1)).all())
            return mix(site, logits)

        execution.mix = checked
        return forward(models, context, candidate, execution)

    eager.policy.routed_control.spec.model_forward = eager_forward
    for _ in range(2):
        metrics = [api.sites_example.update(loop) for loop in loops]
        same(metrics[0], metrics[1])
        same(api.sites_example.checkpoint(lazy), api.sites_example.checkpoint(eager))
        for owner in ("generator", "encoder", "router", "critic"):
            parameters = [list(loop.policy._training_modules()[owner].parameters()) for loop in loops]
            for a, b in zip(*parameters):
                same(a.grad, b.grad)
        same(lazy.policy.table.grad, eager.policy.table.grad)
        same(lazy.policy.log_output_sigma.grad, eager.policy.log_output_sigma.grad)
    assert lazy.policy.table.grad.norm(dim=-1).gt(0).all()
    assert lazy.policy.routed_control.evidence.counters["updates"] == 2
    assert lazy.policy.G.first_host.weight.dtype == torch.bfloat16
    saved = cpu_roundtrip(api.sites_example.checkpoint(lazy))
    restored = api.make_loop(device=device, sites=71, tokens=2, particles=8, batch_size=4)
    api.sites_example.restore(restored, saved)
    same(api.sites_example.update(restored), api.sites_example.update(lazy))
    same(api.sites_example.checkpoint(restored), api.sites_example.checkpoint(lazy))
    same(restored.policy.served_snapshot(), lazy.policy.served_snapshot())
    context = lazy.test_context[:4]
    for averaged in (False, True):
        same(restored.policy.routed_generate(context, sigma=0, perturb=False, averaged=averaged),
             lazy.policy.routed_generate(context, sigma=0, perturb=False, averaged=averaged))
    same(restored.policy.served_model().routed_forward(context),
         lazy.policy.served_model().routed_forward(context))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scalar readbacks required")
def test_cuda_scalar_reads_do_not_grow_with_routing_site_count(api):
    counts = []
    for site_count in (1, 71):
        loop = api.make_loop(device="cuda", sites=site_count, tokens=2, particles=8, batch_size=4)
        with torch.no_grad():
            api.forward_pair(loop)  # warm up first-use kernels outside profiling
            count, outputs = api.scalar_extractions(lambda: api.forward_pair(loop))
        del outputs
        counts.append(count)
        assert len(loop.policy.controller.latent_applications) == 2
    # Geometry validates once per prior and finish once per full forward;
    # the old implementation's two-forward slope was 14 reads per extra site.
    assert counts[1] == counts[0]
    assert counts[1] <= 24


def test_readback_benchmark_restores_warmup_and_profiles_outside_training(api):
    result = api.benchmark(sites=2, tokens=2, particles=8, batch_size=4, steps=1, warmup_steps=1)
    assert result["structural_evaluations"] == 0
    assert result["steps"] == 1
    assert result["probe_interval"] == 1000
    assert len(result["last_dv12_applications"]) == 2
    assert result["milliseconds_per_update"] > 0
