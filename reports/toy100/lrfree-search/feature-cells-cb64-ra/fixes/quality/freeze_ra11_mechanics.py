"""Seal the root-owned CUDA mechanics phase after CPU and source qualification."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "quality/ra11-mechanics-phase"
OWNER = ROOT / "integration/review/training-regression/post-ra10-quality/linear-output-production"
OUTPUT = ROOT / "integration/review/ra11-mechanics-gpu"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(value, files):
    if isinstance(value, dict):
        for name, item in value.items():
            if (isinstance(name, str) and name.startswith("/")
                    and isinstance(item, str) and len(item) == 64):
                assert sha(name) == item, name
                assert name not in files or files[name] == item, name
                files[name] = item
            collect(item, files)
    elif isinstance(value, list):
        for item in value:
            collect(item, files)


def main():
    assert not (AREA / "READY.json").exists() and not OUTPUT.exists()
    owner_path = OWNER / "READY.json"
    package_path = ROOT / "quality/ra11/READY.json"
    owner = json.loads(owner_path.read_text())
    package = json.loads(package_path.read_text())
    assert owner["status"] == "FROZEN_CPU_QUALIFIED"
    assert package["status"] == "CPU_VALID_GPU_PENDING"
    assert owner["backend_schema"] == package["backend_schema"] == 10
    assert owner["trainer_schema"] == package["trainer_schema"] == 5
    assert owner["package_source_sha256"] == package["package_source_sha256"]
    assert owner["package_sha256"] == package["package_sha256"]
    assert owner["config_sha256"] == package["config_sha256"]
    command = owner["gpu_command"]
    assert isinstance(command, list) and command
    assert command[-1] == str(OUTPUT / "result.json")
    files = {}
    for path in (owner_path, OWNER / "FROZEN.json", OWNER / "SOURCE-FROZEN.json",
                 package_path, ROOT / "quality/ra11/COMPOSITION.json"):
        collect(json.loads(path.read_text()), files)
        files[str(path)] = sha(path)
    for path in (Path(__file__), ROOT / "quality/run_ra11_mechanics.py",
                 ROOT / "quality/nested_slot.py", ROOT / "gpu_slot.py"):
        files[str(path)] = sha(path)
    AREA.mkdir(exist_ok=True)
    ready = dict(status="FROZEN_ROOT_GPU_PENDING", phase="ra11_cuda_mechanics",
        frozen_utc=datetime.now(timezone.utc).isoformat(), command=command,
        source_sha256=dict(sorted(files.items())),
        package_sha256=package["package_sha256"], config_sha256=package["config_sha256"],
        gpu_uuid="GPU-72c1b506-891d-b8bc-b353-e020585e1c47",
        numerical_parallelism=1, quality_verdict=None,
        scope="Fresh backend10 mechanics only; unchanged CUDA toy and full grid remain required.")
    (AREA / "READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(dict(status=ready["status"], guarded_files=len(files),
        ready_sha256=sha(AREA / "READY.json"))))


if __name__ == "__main__":
    main()
