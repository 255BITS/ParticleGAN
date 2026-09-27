"""Host-only execution scopes and checkpoint envelope; imports no Torch."""
from contextlib import contextmanager

EXECUTION = {"serial_backward": True, "scope": "full GANTrainer.step"}


@contextmanager
def host_cuda(torch):
    """Route only frozen host tensor/implicit-generator factories to CUDA."""
    previous_device = str(torch.get_default_device())
    original_generator = torch.Generator

    class CudaDefaultGenerator(original_generator):
        def __new__(cls, device=None):
            return original_generator.__new__(cls, "cuda:0" if device is None else device)

        def __init__(self, device=None):
            pass

    try:
        with torch.device("cuda:0"):
            torch.Generator = CudaDefaultGenerator
            yield
    finally:
        torch.Generator = original_generator
        if str(torch.get_default_device()) != previous_device:
            raise RuntimeError("host CUDA scope did not restore caller device")


@contextmanager
def serial_step(torch):
    """Cover graph construction and both backwards, restoring caller state."""
    previous = torch.autograd.is_multithreading_enabled()
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        if torch.autograd.is_multithreading_enabled() != previous:
            raise RuntimeError("serial step did not restore caller autograd context")


def validate_checkpoint(envelope, identity):
    """Validate host execution/source contract before any trainer restoration."""
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema", "execution", "identity", "trainer", "data_rng"
    }:
        raise ValueError("invalid reference checkpoint envelope")
    if envelope["schema"] != 1 or envelope["execution"] != EXECUTION:
        raise ValueError("reference checkpoint execution mode differs")
    if envelope["identity"] != identity:
        raise ValueError("reference checkpoint source/fixture/protocol differs")
    if not isinstance(envelope["trainer"], dict) or envelope["trainer"].get("schema") != 3:
        raise ValueError("reference requires unchanged released schema3 trainer state")
