"""Device selection for toy100 and continuous-learning hosts.

``--device auto`` uses CUDA when it is available and CPU otherwise.
``--device cpu`` does not change threads, determinism, the default device, or
``torch.Generator``. CUDA runs keep PR #139's policy: FP32, deterministic
algorithms, TF32 off, and ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` set before the
first CUDA context.

``torch.Generator()`` ignores the default device and stays on CPU. After a
CUDA selection, device-less constructors are routed to that device. Explicit
``device="cpu"`` arguments are left on CPU on purpose in the calibration
sites noted below; those are not training steps.

Left on CPU, with no algorithm change:

- ``benchmarks/toy100/accuracy.py`` ``oracle_reference`` uses the published
  CPU target sampler. A CUDA generator would draw a different sequence.
- ``benchmarks/toy_suite.py`` hashes that same CPU calibration against a
  recorded init-data receipt.
- ``benchmarks/locked_shared/shared_checks.py`` scores a named CPU reference
  backend (``load_backend("cpu")``), not the experiment device.
- ``particlegan/training.py`` restores the CPU RNG slot separately from the
  CUDA RNG slot.
- CPU-built weights copied onto CUDA (``cuda_cpu_init``) stay as they are.

PR #148/#149 ``gan_followup_probe.py`` and ``canonical_env.py`` are not on
this branch. That env check pins ``CUDA_VISIBLE_DEVICES`` empty, so it is not
imported here.
"""

from __future__ import annotations

import json
import os

import torch

DEVICE_CHOICES = ("auto", "cuda", "cpu")

_SELECTED: torch.device | None = None
_RECEIPT: dict | None = None
_ORIGINAL_GENERATOR = torch.Generator
_GENERATOR_PATCHED = False


class _DefaultDeviceGenerator(_ORIGINAL_GENERATOR):
    """``torch.Generator()`` on the selected host device; explicit devices pass through."""

    def __new__(cls, device=None):
        chosen = host_device() if device is None else device
        return _ORIGINAL_GENERATOR.__new__(cls, chosen)

    def __init__(self, device=None):
        return None


def host_device() -> torch.device:
    """Selected experiment device, or CPU when no selection has been applied."""
    if _SELECTED is None:
        return torch.device("cpu")
    return _SELECTED


def rng_fork_devices() -> list[int]:
    """Devices ``fork_rng`` must save. CPU stays ``[]``, matching the old call."""
    device = host_device()
    if device.type != "cuda":
        return []
    index = device.index if device.index is not None else torch.cuda.current_device()
    return [index]


def experiment_generator() -> torch.Generator:
    """Process generator training should advance.

    CPU hosts keep ``torch.default_generator``. CUDA hosts use that device's
    default CUDA generator, which ``torch.manual_seed`` also seeds.
    """
    device = host_device()
    if device.type == "cpu":
        return torch.default_generator
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[index]


def prepare_cublas_workspace(requested: str | None) -> None:
    """Set the deterministic cuBLAS workspace before any CUDA context exists."""
    if requested is None:
        return
    text = str(requested)
    if text in ("auto", "cuda") or text.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def select_device(requested: str | None = None) -> torch.device:
    """Resolve ``auto``, ``cuda``, ``cpu``, or ``cuda:N``. ``auto`` prefers CUDA."""
    if requested is None:
        requested = os.environ.get("TOY100_DEVICE") or "auto"
    requested = str(requested).strip()
    if requested == "auto":
        concrete = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        concrete = "cuda:0"
    elif requested == "cpu" or requested.startswith("cuda"):
        concrete = requested
    else:
        raise ValueError("device must be auto, cuda, or cpu")
    device = torch.device(concrete)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cuda, or cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def configure_cuda_determinism(device: torch.device) -> None:
    """Apply PR #139's CUDA math policy. CPU selections skip this entirely."""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _route_generators() -> None:
    global _GENERATOR_PATCHED
    if _GENERATOR_PATCHED:
        return
    torch.Generator = _DefaultDeviceGenerator
    _GENERATOR_PATCHED = True


def apply_device_policy(requested: str | None = None, *, log: bool = False) -> dict:
    """Select the device and, for CUDA only, make new tensors and generators follow it."""
    global _SELECTED, _RECEIPT
    if requested is None:
        requested = os.environ.get("TOY100_DEVICE") or "auto"
    prepare_cublas_workspace(requested)
    device = select_device(requested)
    if _SELECTED is not None:
        if _SELECTED != device:
            raise RuntimeError(f"device already selected as {_SELECTED}; refusing {device}")
        if log and _RECEIPT is not None:
            print(json.dumps({"event": "DEVICE", **_RECEIPT}, sort_keys=True), flush=True)
        return dict(_RECEIPT or {})
    os.environ["TOY100_DEVICE"] = str(device)
    receipt = dict(
        device=str(device),
        requested=str(requested),
        torch=torch.__version__,
        deterministic=False,
        tf32=False,
        dtype="float32",
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    if device.type == "cuda":
        configure_cuda_determinism(device)
        torch.set_num_threads(1)
        torch.set_default_device(device)
        _route_generators()
        props = torch.cuda.get_device_properties(device)
        receipt.update(
            deterministic=True,
            threads=1,
            gpu=props.name,
            torch_cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            rng="device-less torch.Generator() uses the selected device",
        )
    _SELECTED = device
    _RECEIPT = receipt
    if log:
        print(json.dumps({"event": "DEVICE", **receipt}, sort_keys=True), flush=True)
    return dict(receipt)


def add_device_argument(parser, *, default: str = "auto"):
    """Add ``--device {auto,cuda,cpu}``. ``auto`` means CUDA when available."""
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=default,
        help="auto uses cuda when available, else cpu (default: auto)",
    )
