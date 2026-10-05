"""Suite-wide isolation for process-global torch state."""

import pytest

from benchmarks.toy100.device import device_policy_scope


@pytest.fixture(autouse=True)
def _isolate_renderer_clock(request, monkeypatch):
    """Start the renderer's unit-test budget at each test, after collection.

    The archived protocol binds its test-file bytes. Keep this software-only
    clock isolation outside those frozen sources and preserve the CLI budget.
    """
    if request.module.__name__ == "test_e22_routed_caption_late_phase_renderer":
        render = request.module.render
        monkeypatch.setattr(render, "STARTED", render.time.monotonic())


@pytest.fixture(autouse=True)
def _restore_device_policy():
    """Tests that run a benchmark CLI ``main()`` in-process must not leave its
    CUDA default device, routed ``torch.Generator`` or determinism flags behind."""
    with device_policy_scope():
        yield
