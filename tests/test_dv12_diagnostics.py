"""Lazy DV12 diagnostics preserve native values without per-site host reads."""
from copy import deepcopy
import io
from types import SimpleNamespace

import pytest
import torch

from particlegan.continuous import DataDriftController


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]
METRICS = ("radius_min", "radius_mean", "radius_max", "perturbation_rms", "clipped_fraction")
LEGACY_KEYS = ("variant", "latent_bandwidth", "latent_applications", "mobility", "game_trust",
               "game_ratio", "payoff_error", "data_memory", "fast_reference", "fast_mean_variance",
               "slow_mean_variance", "mean_covariance", "alignment", "data_score", "data_drive",
               "last_cosine", "previous_gradient", "location", "scale", "projection", "reference",
               "variance", "updates", "reopens", "closed")


@pytest.fixture(autouse=True)
def single_threaded():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def setup(device="cpu", dtype=torch.float32):
    controller = DataDriftController("dv12")
    prior = SimpleNamespace(z=torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 0.], [0., 2.]],
                                          device=device, dtype=dtype))
    controller.observe_prior(prior)
    controller.latent_bandwidth.copy_(prior.z.new_tensor([.9, 1.25]))
    latent = prior.z.new_tensor([[-.5, .2], [.1, .5], [.9, 1.3]]).requires_grad_()
    return controller, prior, latent


def native_float_values(application):
    # The native recorder converted each original reduction independently.
    # Keeping clipped_fraction's float32 reduction is essential with FP64 codes.
    radius, displacement, fraction, dtype = application
    with torch.autocast(device_type=radius.device.type, dtype=dtype, enabled=dtype is not None):
        return {"radius_min": float(radius.min()), "radius_mean": float(radius.mean()),
                "radius_max": float(radius.max()),
                "perturbation_rms": float(displacement.square().mean().sqrt()),
                "clipped_fraction": float((fraction < 1.).float().mean())}


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize("device", DEVICES)
def test_many_recorded_sites_do_not_read_scalars_or_compute_discarded_metrics(device):
    controller, prior, latent = setup(device)
    recorded = torch.Generator(device=device).manual_seed(37)
    clean_recording, _, _ = setup(device)
    unrecorded = torch.Generator(device=device).manual_seed(37)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for index in range(71):
            output = controller.perturb_latent(latent + index / 64, recorded, prior, record=True)
    counts = {event.key: event.count for event in profile.key_averages()}
    assert counts.get("aten::_local_scalar_dense", 0) == 0
    assert counts.get("aten::item", 0) == 0
    assert counts.get("aten::stack", 0) == 0
    assert counts.get("aten::mean", 0) == 0
    assert counts.get("aten::max", 0) == 0
    for index in range(71):
        expected = clean_recording.perturb_latent(latent + index / 64, unrecorded, prior)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(recorded.get_state(), unrecorded.get_state(), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(latent.grad, torch.ones_like(latent), rtol=0, atol=0)
    records = controller._latent_application_records
    assert len(records) == 2
    assert sum(value.numel() for record in records for value in record[:3]) == 2 * len(latent) * (latent.shape[1] + 2)
    assert all(not value.requires_grad and value.grad_fn is None for record in records for value in record[:3])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_public_float_metrics_exactly_match_native_conversions_outside_forward_context(device, dtype, autocast):
    controller, prior, latent = setup(device, dtype)
    stream = torch.Generator(device=device).manual_seed(37)
    with torch.autocast(device_type=latent.device.type, dtype=torch.bfloat16, enabled=autocast):
        for index in range(3):
            controller.perturb_latent(latent + index / 64, stream, prior, record=True)
    expected = [native_float_values(value) for value in controller._latent_application_records]
    # Reading under the opposite autocast setting must preserve the recorded law.
    with torch.autocast(device_type=latent.device.type, dtype=torch.bfloat16, enabled=not autocast):
        actual = controller.latent_applications
    assert actual == expected
    assert len(actual) == 2
    assert all(tuple(row) == METRICS and all(type(value) is float for value in row.values()) for row in actual)
    assert controller.diagnostics()["latent_applications"] == expected


@pytest.mark.parametrize("device", DEVICES)
def test_metrics_materialize_in_one_transfer_and_public_list_assignment_stays_supported(device, monkeypatch):
    controller, prior, latent = setup(device, torch.float64)
    stream = torch.Generator(device=device).manual_seed(37)
    for _ in range(3):
        controller.perturb_latent(latent, stream, prior, record=True)
    transfers = []
    cpu = torch.Tensor.cpu

    def observed_cpu(tensor, *args, **kwargs):
        transfers.append((tuple(tensor.shape), tensor.dtype))
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", observed_cpu)
    first = controller.latent_applications
    assert transfers == [((2, 5), torch.float64)]
    assert controller.latent_applications is first
    controller.state_dict()
    assert len(transfers) == 1
    controller.perturb_latent(latent, stream, prior, record=False)
    assert controller.latent_applications is first and len(transfers) == 1
    controller.perturb_latent(latent, stream, prior, record=True)
    assert len(transfers) == 1
    assert len(controller.latent_applications) == 2
    assert transfers[-1] == ((1, 5), torch.float64)
    controller.latent_applications.clear()
    controller.perturb_latent(latent, stream, prior, record=True)
    controller.latent_applications = []
    assert controller.latent_applications == [] and len(transfers) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_checkpoint_schema_roundtrip_remains_exact_and_independent(device):
    controller, prior, latent = setup(device)
    stream = torch.Generator(device=device).manual_seed(37)
    for _ in range(3):
        controller.perturb_latent(latent, stream, prior, record=True)
    before = controller.state_dict()
    assert tuple(before) == LEGACY_KEYS
    assert all(not key.startswith("_") for key in before)
    restored, _, _ = setup(device)
    restored.load_state_dict(before)
    assert_state_equal(restored.state_dict(), before)
    buffer = io.BytesIO()
    torch.save(before, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    saved["latent_bandwidth"] = saved["latent_bandwidth"].to(device)
    restored.load_state_dict(saved)
    assert_state_equal(restored.state_dict(), before)
    restored.latent_applications[0]["radius_min"] = -1.
    assert controller.latent_applications == before["latent_applications"]
    bad = deepcopy(before)
    bad["_latent_application_records"] = []
    with pytest.raises(ValueError, match="incompatible continuous controller state"):
        controller.load_state_dict(bad)
    assert_state_equal(controller.state_dict(), before)


@pytest.mark.parametrize("device", DEVICES)
def test_deepcopy_preserves_pending_values_and_never_reads_them(device, monkeypatch):
    controller, prior, latent = setup(device)
    controller.perturb_latent(latent, torch.Generator(device=device).manual_seed(37), prior, record=True)
    cpu = torch.Tensor.cpu

    def forbidden(*args, **kwargs):
        raise AssertionError("deepcopy must not materialize diagnostic values")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    copied = deepcopy(controller)
    assert len(copied._latent_application_records) == 1
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    assert_state_equal(copied.state_dict(), controller.state_dict())
