"""External execution mode and checkpoint envelope; deliberately no Torch import."""
from contextlib import contextmanager

EXECUTION = {"serial_backward": True, "scope": "full GANTrainer.step"}


@contextmanager
def serial_step(torch):
    previous = torch.autograd.is_multithreading_enabled()
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        if torch.autograd.is_multithreading_enabled() != previous:
            raise RuntimeError("full-step serial context did not restore its caller")


def validate_checkpoint(envelope, identity):
    expected = {"schema", "execution", "identity", "trainer", "data_rng", "means"}
    if not isinstance(envelope, dict) or set(envelope) != expected:
        raise ValueError("invalid public ring checkpoint envelope")
    if envelope["schema"] != 1 or envelope["execution"] != EXECUTION:
        raise ValueError("public ring checkpoint execution mode differs")
    if envelope["identity"] != identity:
        raise ValueError("public ring checkpoint source/recipe/protocol differs")
    if not isinstance(envelope["trainer"], dict) or envelope["trainer"].get("schema") != 3:
        raise ValueError("requires unchanged released trainer schema3")


def validate_bundle(bundle, expected):
    """Content pinning before imports; artifact mutation is an error, not a repair."""
    import hashlib
    for name, wanted in expected.items():
        actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError("bundle content differs: " + name)
