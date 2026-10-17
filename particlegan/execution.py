"""Project policy for autograd's host-side backward scheduling.

This does not change Torch's CPU operation thread pools or CUDA parallelism.
The import default applies to the importing thread; owned execution scopes
also enforce the policy when a caller or a new worker thread has enabled it.
"""
from contextlib import contextmanager

import torch


def disable_autograd_multithreading():
    """Disable multithreaded autograd scheduling on the current thread."""
    torch.autograd.set_multithreading_enabled(False)


@contextmanager
def serial_autograd():
    """Enforce the project policy for a whole update, restoring caller state."""
    with torch.autograd.set_multithreading_enabled(False):
        yield
