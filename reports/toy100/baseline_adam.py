"""Unmodified host Adam for the canonical re-baseline.

No curvature bound, stencil, output cap, or fence. The context manager does
not patch the host. Identity versus the method differs only by the warm
probe's constant-rate switch.
"""

from contextlib import contextmanager


class BaselineRecorder:
    def __init__(self):
        self.enabled = True
        self.optimizers = None
        self.records = []

    def receipt(self):
        return dict(method="unmodified_host_adam", shared_gate_eligible=False,
                    scratch_optimizer_policy="host_adam")


@contextmanager
def baseline_adam(*, task="mode_hold", start_step=0):
    yield BaselineRecorder(), ""
