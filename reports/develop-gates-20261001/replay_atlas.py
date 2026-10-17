"""Replay the original Atlas 19 CUDA gates against a frozen develop package.

Uses the retained, unchanged RA15 host/scorer sources. Their local research
archive must be present; missing sources are errors, never substituted hosts.
Bulk logs, checkpoints and draw clouds go only to the requested artifact path.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930"
ADAPTER = STUDY / "validation-ra15"
HARNESS = Path("/ml2/hypergan/lrfree-20260926/harness")
INITIALIZER = Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA11/particlegan")
ROTATE = Path("/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py")
LOCK = Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/quality/.serial-phase.lock")
NATIVE = ("grid100", "rotated100", "staggered100")
PORTS = ("mode_hold", "img_intensity2", "img_blobs4", "img_bars4", "img_stripes2",
         "vector_two_broad", "vector_unequal_mass", "vector_unequal_width",
         "vector_anisotropic", "vector_overlap", "vector_spiral", "stationary", "ring_shift")
OPTIONS = dict(eval_output_noise=True, save_final_state=True, strict_streams=True, diagnostics=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    package = output / "source"
    shutil.copytree(REPO / "particlegan", package / "particlegan",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = package / "atlas.json"
    shutil.copyfile(REPO / "configs/100gaussians/atlas.json", config)
    assert sha(config) == "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"
    sources = list(HARNESS.rglob("*.py")) + list(HARNESS.rglob("*.json"))
    sources += list(INITIALIZER.glob("*.py"))
    sources += [ADAPTER / "screen_current.py", ADAPTER / "current_api_fixtures.py", ROTATE]
    shutil.copyfile(Path(__file__), output / "driver.py")
    sources += list((package / "particlegan").glob("*.py")) + [config, output / "driver.py"]
    frozen = {str(path): sha(path) for path in sorted(set(sources))}
    write(output / "source-freeze.json", {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "config_sha256": sha(config), "files": frozen, "new_training_updates": 0,
        "recipe": "Atlas", "sampling": "original noisy primary; clean diagnostics retained",
        "gates": [{"group": group, "task": task} for group, task in plan()],
        "budgets_changed": False, "seeds_changed": False, "thresholds_changed": False,
        "device": "original physical GPU0, RTX A6000, float32, serialized backward",
        "native_timeout_seconds": 2400, "other_timeout_seconds": 1800,
        "total_timeout_seconds": 10800})


def verify(output):
    frozen = json.loads((output / "source-freeze.json").read_text())
    for filename, expected in frozen["files"].items():
        assert sha(Path(filename)) == expected, filename
    return {"status": "VALID", "files": len(frozen["files"]),
            "source_freeze_sha256": sha(output / "source-freeze.json")}


def plan():
    return [("native", task) for task in NATIVE] + [("portability", task) for task in PORTS] + [("moving", task) for task in NATIVE]


def moving_source(output):
    source = ROTATE.read_text()
    source = replace_once(source, "REPO = '/ml2/hypergan/ParticleGAN-pr155-merge'", f"REPO = {str(output / 'source')!r}")
    source = replace_once(source, "options = json.load(open(f'{REPO}/configs/100gaussians/e22-noout.json'))",
                          f"options = json.load(open({str(output / 'source/atlas.json')!r}))")
    source = replace_once(source, "            latent, _ = table.sample(n, generator=latent_stream)",
                          "            latent, indices = table.sample(n, generator=latent_stream)")
    source = replace_once(source, "            clean = trainer._generate(model, latent, 0., latent_stream)",
                          "            clean = trainer._generate(model, latent, 0., latent_stream, indices=indices)")
    source = replace_once(source, "torch.cuda.set_device(0); torch.set_num_threads(1); torch.set_num_interop_threads(1)",
                          "torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2, 0); torch.set_num_threads(1); torch.set_num_interop_threads(1)")
    source = replace_once(source, "print('GATE ' + json.dumps(row), flush=True)",
                          "print('GATE ' + json.dumps(row), flush=True)\n"
                          "        torch.save(trainer.state_dict(), args.out + f'.checkpoint-{step:06d}.pt')")
    source = replace_once(source, "    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)",
                          "    assert len(gate_rows) == 3 and len(ok) == 2\n"
                          "    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)")
    return source


def execute(output, group, task):
    target = output / group / task
    target.mkdir(parents=True, exist_ok=False)
    before = verify(output)
    if group == "moving":
        script = target / "runner.py"
        script.write_text(moving_source(output))
        command = [sys.executable, "-u", "-B", str(script), task, str(target / "frames.npz"),
                   "--every", "500", "--points", "4096", "--steps", "1500", "--gate",
                   "--rotate-every", "500", "--rotate-deg", "30"]
    else:
        script = target / "runner.py"
        script.write_text(
            "import runpy, sys, torch\n"
            "torch.cuda.set_device(0)\n"
            "torch.cuda.set_per_process_memory_fraction(.2, 0)\n"
            f"sys.path.insert(0, {str(HARNESS)!r})\n"
            f"sys.path.insert(0, {str(ADAPTER)!r})\n"
            f"sys.argv[0] = {str(ADAPTER / 'screen_current.py')!r}\n"
            f"runpy.run_path({str(ADAPTER / 'screen_current.py')!r}, run_name='__main__')\n")
        command = [sys.executable, "-u", "-B", str(script), "--package-root", str(output / "source"),
                   "--overrides", str(output / "source/atlas.json"), "--task", task,
                   "--output", str(target), "--device", "cuda:0", "--candidate-options",
                   json.dumps(OPTIONS), "--cand", "develop-atlas-20261001"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1")
    started = time.time()
    receipt = {"group": group, "task": task, "command": command, "status": "RUNNING",
               "source_integrity_before": before, "started": started,
               "runner_sha256": sha(script)}
    write(target / "execution.json", receipt)
    print(json.dumps({"event": "start", "group": group, "task": task,
                      "log": str(target / "run.log")}), flush=True)
    timeout = 2400 if group == "native" else 1800
    with LOCK.open("r") as lock, (target / "run.log").open("x") as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(output)
        completed = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    result_path = target / ("frames.npz.verdict.json" if group == "moving" else "result.json")
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    integrity = verify(output)
    status = result.get("status", "ERROR") if completed.returncode == 0 else "ERROR"
    if group == "moving":
        assert result.get("turns") == 2 and len(result.get("periods", [])) == 3
    else:
        assert result.get("stream_deviations") == 0, result
        assert result.get("header", {}).get("options", {}).get("eval_output_noise") is True
    receipt.update(status=status, returncode=completed.returncode, completed=time.time(),
                   wall_seconds=time.time()-started, source_integrity_after=integrity,
                   result_sha256=sha(result_path), completed_steps=1500 if group == "moving" else result.get("completed_steps"))
    write(target / "execution.json", receipt)
    print(json.dumps({"event": "complete", "group": group, "task": task,
                      "status": status, "wall_seconds": receipt["wall_seconds"]}), flush=True)
    return {"group": group, "task": task, "status": status,
            "completed_steps": receipt["completed_steps"], "wall_seconds": receipt["wall_seconds"],
            "result_sha256": receipt["result_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "source-freeze.json").exists():
        prepare(output)
    verify(output)
    if args.prepare_only:
        print(json.dumps(verify(output)), flush=True)
        return
    board_path = output / "scoreboard.json"
    assert not board_path.exists(), "retain completed and interrupted attempts"
    board = {"status": "RUNNING", "required": 19, "results": [], "started": time.time()}
    write(board_path, board)
    for group, task in plan():
        assert time.time()-board["started"] < 10800, "registered total budget exhausted"
        board["current"] = {"group": group, "task": task}
        write(board_path, board)
        try:
            row = execute(output, group, task)
        except Exception as error:
            board.update(status="ERROR", error=repr(error), completed=time.time())
            write(board_path, board)
            raise
        board["results"].append(row)
        write(board_path, board)
        if row["status"] == "ERROR":
            board.update(status="ERROR", completed=time.time())
            write(board_path, board)
            raise RuntimeError(row)
    board.pop("current", None)
    board.update(status="PASS" if all(row["status"] == "PASS" for row in board["results"]) else "FAIL",
                 completed=time.time(), source_integrity_after=verify(output))
    write(board_path, board)
    print(json.dumps({"event": "all_complete", "status": board["status"], "required": 19,
                      "passed": sum(row["status"] == "PASS" for row in board["results"])}), flush=True)


if __name__ == "__main__":
    main()
