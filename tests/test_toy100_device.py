"""Device selection stays on CPU unless a CUDA run is actually selected."""

import torch
import pytest

from benchmarks.toy100.device import (
    _DefaultDeviceGenerator,
    apply_device_policy,
    configure_cuda_determinism,
    experiment_generator,
    host_device,
    rng_fork_devices,
    select_device,
)


@pytest.fixture(autouse=True)
def _fresh_selection(monkeypatch):
    import benchmarks.toy100.device as device

    monkeypatch.setattr(device, "_SELECTED", None)
    monkeypatch.setattr(device, "_RECEIPT", None)
    monkeypatch.setattr(device, "_GENERATOR_PATCHED", False)
    monkeypatch.delenv("TOY100_DEVICE", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)


def test_cpu_policy_does_not_change_runtime_state():
    original = torch.Generator
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    receipt = apply_device_policy("cpu")
    assert receipt["device"] == "cpu"
    assert receipt["deterministic"] is False
    assert torch.Generator is original
    assert host_device() == torch.device("cpu")
    assert rng_fork_devices() == []
    assert experiment_generator() is torch.default_generator
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() is deterministic


def test_auto_uses_cpu_when_cuda_is_absent():
    assert torch.cuda.is_available() is False
    original = torch.Generator
    receipt = apply_device_policy("auto")
    assert receipt["device"] == "cpu"
    assert receipt["deterministic"] is False
    assert torch.Generator is original


def test_cuda_selection_refuses_when_cuda_is_absent():
    with pytest.raises(RuntimeError, match="not available"):
        select_device("cuda")
    with pytest.raises(RuntimeError, match="not available"):
        configure_cuda_determinism(torch.device("cuda"))
    assert torch.are_deterministic_algorithms_enabled() is False


def test_routed_generator_matches_cpu_generator_when_no_device_is_selected():
    routed = _DefaultDeviceGenerator().manual_seed(5)
    plain = torch.Generator().manual_seed(5)
    explicit = _DefaultDeviceGenerator(device="cpu").manual_seed(5)
    reference = torch.Generator(device="cpu").manual_seed(5)
    assert torch.equal(torch.randn(6, generator=routed), torch.randn(6, generator=plain))
    assert torch.equal(
        torch.randn(6, generator=explicit),
        torch.randn(6, generator=reference),
    )


def test_toy100_cli_defaults_to_auto():
    from benchmarks.toy100.__main__ import _parser

    args = _parser().parse_args(["run", "--output", "/tmp/toy100-device-example"])
    assert args.device == "auto"
    chosen = _parser().parse_args(["run", "--output", "/tmp/toy100-device-example", "--device", "cpu"])
    assert chosen.device == "cpu"


def test_harness_modules_import():
    import benchmarks.locked_shared.observation
    import benchmarks.toy100.continuous_probe
    import benchmarks.transfer_suite.legacy_noise_adapters
    import benchmarks.transfer_suite.toy100_compatibility
    import benchmarks.transfer_suite.vector_tasks

    assert benchmarks.locked_shared.observation.rng_fork_devices() == []
    assert benchmarks.transfer_suite.legacy_noise_adapters.host_device().type == "cpu"
