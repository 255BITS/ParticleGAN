"""Run repository tests against the integrated frozen candidate under the serial lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from freeze import ROOT, verify
from run_screen import ENV, OLD, parked_owner

REPOSITORY = Path("/ml2/hypergan/ParticleGAN-ra11-pr155")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before = verify()
    from freeze import PACKAGE
    sources = {}
    for source in sorted((PACKAGE / "particlegan").rglob("*.py")):
        local = REPOSITORY / "particlegan" / source.relative_to(PACKAGE / "particlegan")
        assert source.read_bytes() == local.read_bytes(), f"repository differs: {local}"
        sources[str(local)] = sha(local)
    tests = {str(path): sha(path) for path in sorted((REPOSITORY / "tests").rglob("*.py"))}
    for name in ("pyproject.toml", "pytest.ini", "setup.cfg", "conftest.py"):
        path = REPOSITORY / name
        if path.exists():
            tests[str(path)] = sha(path)
    log = ROOT / "full-pytest.log"
    receipt_path = ROOT / "FULL-TESTS.json"
    assert not log.exists() and not receipt_path.exists(), "retain every test attempt"
    command = [sys.executable, "-u", "-B", "-c",
               "import torch, pytest; torch.cuda.set_device(0); "
               "torch.cuda.set_per_process_memory_fraction(.2, 0); "
               "raise SystemExit(pytest.main([\"-q\", \"-ra\"]))"]
    receipt = dict(status="WAITING_FOR_SERIAL_LANE", command=command, repository=str(REPOSITORY),
                   started=time.time(), source_integrity_before=before, sources=sources, tests=tests,
                   log=str(log), resources=dict(physical_gpu=0, memory_fraction=.2),
                   cpu_initialization_contract="35 focused checks separately passed in fresh CUDA-visible process with zero initialization attempts")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(event="full_suite_waiting", log=str(log))), flush=True)
    with (OLD / "quality/.serial-phase.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        parked_owner()
        verify()
        env = dict(os.environ, **ENV)
        receipt.update(status="RUNNING", numerical_started=time.time())
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        with log.open("x") as output:
            code = subprocess.run(command, cwd=REPOSITORY, env=env,
                                  stdout=output, stderr=subprocess.STDOUT).returncode
        for path, expected in {**sources, **tests}.items():
            assert sha(path) == expected, f"source changed during suite: {path}"
        receipt.update(status="PASS" if code == 0 else "FAIL", returncode=code, completed=time.time(),
                       source_integrity_after=verify(), log_sha256=sha(log))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(dict(event="full_suite_complete", status=receipt["status"], returncode=code,
                              wall_seconds=receipt["completed"]-receipt["started"], log=str(log))), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
