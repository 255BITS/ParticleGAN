"""Frozen input verification and the owned physical-GPU-0 runtime policy."""
import os
import sys

# These settings precede every NumPy, Torch and selected-package import.
sys.dont_write_bytecode = True
os.environ.update(
    CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
    CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
    NUMEXPR_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
)

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PREV = Path("/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation")
OLD = Path("/ml2/hypergan/gan-attempts/feature-cells-config-20260929/validation")
SEED = 314159
STEPS = 2000
CHECKPOINTS = (0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
DEVICE = "cuda:0"
GPU_UUID = "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
PROBLEMS = ("toy", "mnist")
VARIANTS = ("CB64-RA8",)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_sources(root):
    package = Path(root) / "particlegan"
    return {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}


def package_digest(root):
    h = hashlib.sha256()
    package = Path(root) / "particlegan"
    for p in sorted(package.rglob("*.py")):
        h.update(str(p.relative_to(package)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    tmp.replace(path)


def log(event, **kw):
    print(json.dumps(dict(event=event, **kw), allow_nan=True), flush=True)


def gpu_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
         "--format=csv,noheader"], check=True, text=True, capture_output=True,
    )
    records = []
    for line in result.stdout.strip().splitlines():
        index, uuid, pci, name, memory = [x.strip() for x in line.split(",")]
        records.append(dict(physical_index=int(index), uuid=uuid, pci_bus_id=pci,
                            name=name, memory_total=memory))
    return records


def verify_inputs(verify_local=True):
    inputs = json.loads((ROOT / "INPUTS.json").read_text())
    if verify_local:
        freeze = json.loads((ROOT / "SOURCE-FREEZE.json").read_text())
        for name, expected in freeze["local_source_sha256"].items():
            require(sha(ROOT / name) == expected, f"learned lane source changed: {name}")
    for path, expected in inputs["read_only_file_sha256"].items():
        require(sha(path) == expected, f"frozen read-only input changed: {path}")
    for name, selected in inputs["variants"].items():
        require(package_digest(selected["package_root"]) == selected["package_sha256"],
                f"{name} package digest changed")
        require(package_sources(selected["package_root"]) == selected["source_sha256"],
                f"{name} package source map changed")
        require(sha(selected["config_path"]) == selected["config_sha256"],
                f"{name} config changed")
        require(json.loads(Path(selected["config_path"]).read_text()) == selected["config"],
                f"{name} config contents changed")
    return inputs


@contextmanager
def exclusive_learned_gpu():
    # The coordinator serializes this lane with original-harness screens.
    # This lock additionally rejects overlapping learned training/replay jobs.
    with (ROOT / ".gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def configure_cuda(torch):
    inventory = gpu_inventory()
    physical = next(r for r in inventory if r["physical_index"] == 0)
    require(physical["uuid"] == GPU_UUID, "physical GPU 0 identity changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden")
    require(torch.cuda.device_count() == 1, "only physical GPU 0 may be visible")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    cuda_uuid_raw = str(properties.uuid)
    cuda_uuid_normalized = "GPU-" + cuda_uuid_raw.removeprefix("GPU-").lower()
    require(cuda_uuid_normalized == GPU_UUID.lower().replace("gpu-", "GPU-", 1),
            "visible cuda:0 is not frozen physical GPU 0")
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(0)
    return dict(device=DEVICE, physical_gpu=physical, visible_devices="0",
                cuda_uuid_raw=cuda_uuid_raw, cuda_uuid_normalized=cuda_uuid_normalized,
                cuda_memory_fraction=.2, cpu_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                cuda_device_order=os.environ["CUDA_DEVICE_ORDER"],
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version())


def select_package(inputs, variant):
    selected = inputs["variants"][variant]
    sys.path.insert(0, selected["package_root"])
    return selected


def execution_receipt(inputs, runtime):
    return dict(started_at=utc_now(), source_freeze_sha256=sha(ROOT / "SOURCE-FREEZE.json"),
                inputs_sha256=sha(ROOT / "INPUTS.json"),
                protocol_sha256=sha(ROOT / "PROTOCOL.md"),
                preparation_receipt_sha256=sha(ROOT / "preparation-receipt.json"),
                local_source_sha256=json.loads((ROOT / "SOURCE-FREEZE.json").read_text())["local_source_sha256"],
                read_only_file_sha256=inputs["read_only_file_sha256"],
                command=[sys.executable, *sys.argv], runtime=runtime)


def rng_cpu_buffers(torch, state):
    required = {"cpu_rng": state["cpu_rng"], "cuda_rng": state["cuda_rng"],
                **{f"streams.{key}": value for key, value in state["streams"].items()}}
    if "birth_death" in state:
        required["birth_death.stream"] = state["birth_death"]["stream"]
    for name, value in required.items():
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.uint8, f"RNG buffer must remain CPU uint8: {name}")
    return {name: dict(device=str(value.device), dtype=str(value.dtype), numel=value.numel())
            for name, value in required.items()}
