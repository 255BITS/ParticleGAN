"""Suite-wide isolation for process-global torch state."""

import pytest

from benchmarks.toy100.device import device_policy_scope


@pytest.fixture(autouse=True)
def _restore_device_policy():
    """Tests that run a benchmark CLI ``main()`` in-process must not leave its
    CUDA default device, routed ``torch.Generator`` or determinism flags behind."""
    with device_policy_scope():
        yield
