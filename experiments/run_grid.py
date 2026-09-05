#!/usr/bin/env python
"""Run YAML experiment configs concurrently, with verified completed-run reuse.

A successful child must write a nonempty ``summary.json`` final block and its
complete config. The runner records that summary's digest, the effective config
(including literal trainer DEFAULTS), and a fingerprint of the trainer and lib
sources in ``run_grid_complete.json``. Only an exact match is reusable. Legacy
summaries are rerun; previous attempts are preserved in a sibling history folder.
Failures do not cancel unrelated queued runs, but make the grid exit nonzero.

Example: python experiments/run_grid.py --configs 'configs/*.yaml' --gpus 0,1
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, Optional, Tuple
import uuid

import yaml

DEFAULT_PYTHON = ".venv/bin/python"
DEFAULT_TRAINER = "experiments/train_arm.py"
FAILURES_PATH = Path("results/failures.txt")
COMPLETION_FILE = "run_grid_complete.json"
MANIFEST_VERSION = 1


def canonical(value) -> str:
    """Reject non-JSON config values and keep bools distinct from numeric values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def trainer_defaults(trainer: str) -> Dict:
    """Read a literal DEFAULTS mapping without importing Torch or running code.

    Custom trainers without DEFAULTS must receive and report their complete
    configuration in the YAML. Dynamic DEFAULTS are rejected rather than guessed.
    """
    tree = ast.parse(Path(trainer).read_text(), filename=trainer)
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{trainer}: DEFAULTS must be a literal mapping") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{trainer}: DEFAULTS must be a mapping")
            return value
    return {}


def load_config(config_path: str, defaults: Dict) -> Dict:
    with open(config_path) as f:
        user = yaml.safe_load(f)
    if not isinstance(user, dict) or not all(isinstance(k, str) for k in user):
        raise ValueError("config must be a mapping with string keys")
    if defaults and (unknown := set(user) - set(defaults)):
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = {**defaults, **user}
    if not isinstance(cfg.get("out_dir"), str) or not cfg["out_dir"].strip():
        raise ValueError("config needs a nonempty string out_dir")
    canonical(cfg)
    return cfg


def code_provenance(trainer: str, python_bin: str) -> Dict:
    """Content hashes catch dirty code and default changes without a Git dependency.

    All local lib Python files are included conservatively, plus the trainer and
    runner. This is a source fingerprint, not a package/environment lockfile.
    """
    trainer_path = Path(trainer).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    paths = {trainer_path, Path(__file__).resolve()}
    paths.update((repo_root / "lib").rglob("*.py"))
    sources = {}
    for path in sorted(paths):
        try:
            name = str(path.relative_to(repo_root))
        except ValueError:
            name = str(path)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    executable = shutil.which(python_bin) or python_bin
    return {"trainer": str(trainer_path), "python": str(Path(executable).resolve()),
            "sources": sources}


def read_summary(out_dir: str) -> Optional[Dict]:
    try:
        summary = json.loads((Path(out_dir) / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict) or not isinstance(summary.get("final"), dict) or not summary["final"]:
        return None
    return summary


def summary_matches(out_dir: str, cfg: Dict) -> bool:
    summary = read_summary(out_dir)
    try:
        return summary is not None and canonical(summary.get("config")) == canonical(cfg)
    except (ValueError, TypeError):
        return False


def has_valid_summary(out_dir: str, cfg: Optional[Dict] = None,
                      provenance: Optional[Dict] = None) -> bool:
    """Only a verified completed attempt for this exact request is reusable."""
    if cfg is None or provenance is None or not summary_matches(out_dir, cfg):
        return False
    try:
        saved = json.loads((Path(out_dir) / COMPLETION_FILE).read_text())
        digest = hashlib.sha256((Path(out_dir) / "summary.json").read_bytes()).hexdigest()
        return (saved["version"] == MANIFEST_VERSION
                and canonical(saved["config"]) == canonical(cfg)
                and saved["provenance"] == provenance and saved["summary_sha256"] == digest)
    except (OSError, ValueError, TypeError, KeyError):
        return False


@contextmanager
def output_lock(out_dir: str):
    """An advisory lock shared by independent runner processes; no stale PID lock."""
    out = Path(out_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.run-grid.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"output directory is already running: {out}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare_attempt(out_dir: str) -> None:
    """Move an old attempt aside intact so artifacts cannot mix across variants."""
    out = Path(out_dir).resolve()
    if out.exists():
        if not out.is_dir():
            raise ValueError(f"out_dir is not a directory: {out}")
        if any(out.iterdir()):
            markers = ("summary.json", "summary.failed.json", "metrics.jsonl", "ckpt.pt",
                       COMPLETION_FILE, "requested_config.yaml")
            recognizable = any((out / name).is_file() for name in markers)
            log = out / "log.txt"
            if not recognizable and log.is_file():
                with log.open(errors="replace") as f:
                    recognizable = f.readline().startswith(("# cmd:", "# requested config:"))
            if not recognizable:
                raise ValueError(f"refusing to archive a nonempty directory without run artifacts: {out}")
            history = out.parent / ".run_grid_history" / out.name
            history.mkdir(parents=True, exist_ok=True)
            archive = history / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12])
            out.rename(archive)
            print(f"[run_grid] previous attempt preserved in {archive}", flush=True)
    out.mkdir(parents=True, exist_ok=True)


def invalidate_attempt(out_dir: str) -> None:
    """Do not let an analyzer mistake a failed child's partial summary for success."""
    out = Path(out_dir)
    (out / COMPLETION_FILE).unlink(missing_ok=True)
    summary = out / "summary.json"
    if summary.exists():
        summary.replace(out / "summary.failed.json")


def record_failure(config_path: str, out_dir: str, returncode: int, lock: threading.Lock) -> None:
    with lock:
        FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FAILURES_PATH.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tconfig={config_path}"
                    f"\tout_dir={out_dir}\treturncode={returncode}\n")


def run_one(config_path: str, cfg: Dict, gpu: str, python_bin: str, trainer: str,
            provenance: Dict, force: bool = False, echo_last_line: bool = False) -> Tuple[int, float, bool]:
    """Run under an output lock; return (returncode, seconds, reused)."""
    out_dir = cfg["out_dir"]
    t0 = time.monotonic()
    with output_lock(out_dir):
        # Recheck while locked: another invocation may have finished since preflight.
        if not force and has_valid_summary(out_dir, cfg, provenance):
            return 0, 0.0, True
        prepare_attempt(out_dir)
        log_path = Path(out_dir) / "log.txt"
        snapshot = Path(out_dir) / "requested_config.yaml"
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=True))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
        cmd = [python_bin, trainer, "--config", str(snapshot)]
        returncode = 1
        try:
            with log_path.open("w") as log:
                log.write(f"# requested config: {config_path}\n# cmd: {cmd!r}\n# CUDA_VISIBLE_DEVICES={gpu}\n")
                log.flush()
                try:
                    returncode = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                except OSError as exc:
                    log.write(f"\n[run_grid] launch failed: {exc}\n")
                    returncode = 127
                if returncode == 0 and not summary_matches(out_dir, cfg):
                    log.write("\n[run_grid] missing/incomplete summary or recorded config differs from request\n")
                    returncode = 1
                if returncode == 0 and code_provenance(trainer, python_bin) != provenance:
                    log.write("\n[run_grid] source changed during the grid; result cannot be certified\n")
                    returncode = 1
            if returncode == 0:
                manifest = {"version": MANIFEST_VERSION, "config": cfg, "provenance": provenance,
                            "summary_sha256": hashlib.sha256((Path(out_dir) / "summary.json").read_bytes()).hexdigest()}
                temporary = Path(out_dir) / (COMPLETION_FILE + ".tmp")
                temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
                temporary.replace(Path(out_dir) / COMPLETION_FILE)
        except BaseException:
            invalidate_attempt(out_dir)
            raise
        if returncode:
            invalidate_attempt(out_dir)
        if echo_last_line:
            lines = [line for line in log_path.read_text().splitlines() if line.strip()]
            if lines:
                print(f"[run_grid]   {Path(out_dir).name}: {lines[-1][:400]}", flush=True)
    return returncode, time.monotonic() - t0, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment configs across GPUs with verified reuse.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--configs", help="Glob of config YAMLs, e.g. 'configs/*.yaml' (quote it).")
    inputs.add_argument("--config_manifest", help="JSON list of config paths, as emitted by gen_sparse_configs.py.")
    parser.add_argument("--workers_per_gpu", type=int, default=5)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--force", action="store_true", help="Archive prior attempts and rerun even verified results.")
    parser.add_argument("--echo_last_line", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu <= 0:
        parser.error("--workers_per_gpu must be positive")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain distinct, nonempty GPU ids")
    try:
        config_paths = (json.loads(Path(args.config_manifest).read_text()) if args.config_manifest
                        else sorted(glob.glob(args.configs)))
        if not isinstance(config_paths, list) or not config_paths or not all(isinstance(p, str) for p in config_paths):
            raise ValueError("no configs matched, or config manifest is not a nonempty list of paths")
        defaults = trainer_defaults(args.trainer)
        provenance = code_provenance(args.trainer, args.python)
    except (OSError, ValueError, TypeError, SyntaxError) as exc:
        parser.error(str(exc))

    tasks = []
    errors = []
    outputs = {}
    protected = [Path.cwd().resolve(), Path(args.trainer).resolve(), Path(__file__).resolve()]
    repo_root = Path(__file__).resolve().parents[1]
    protected.extend((repo_root / name).resolve() for name in provenance["sources"])
    protected.append(Path(provenance["python"]))
    if args.config_manifest:
        protected.append(Path(args.config_manifest).resolve())
    protected.extend(Path(p).resolve() for p in config_paths)
    for config_path in config_paths:
        try:
            cfg = load_config(config_path, defaults)
            out = Path(cfg["out_dir"]).resolve()
            if any(out == p or out in p.parents for p in protected):
                raise ValueError(f"out_dir contains the working directory or an input/source: {out}")
            if any(out == prior or out in prior.parents or prior in out.parents for prior in outputs):
                raise ValueError(f"duplicate or nested output directory: {out}")
            outputs[out] = config_path
            tasks.append((config_path, cfg))
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            errors.append(f"{config_path}: {exc}")
    if errors:
        for error in errors:
            print(f"[run_grid] ERROR {error}", file=sys.stderr)
        return 1

    skipped = sum(not args.force and has_valid_summary(cfg["out_dir"], cfg, provenance) for _, cfg in tasks)
    n_workers = min(len(tasks), args.workers_per_gpu * len(gpus))
    print(f"[run_grid] {len(tasks)} configs matched | {skipped} already done | "
          f"{len(tasks) - skipped} to run | {n_workers} workers")
    if args.dry_run:
        for path, cfg in tasks:
            if args.force or not has_valid_summary(cfg["out_dir"], cfg, provenance):
                print(f"[run_grid] would run {path} -> {cfg['out_dir']}")
        return 0

    slots = queue.Queue()
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            slots.put(gpu)
    fail_lock = threading.Lock()

    def worker(task):
        path, cfg = task
        gpu = slots.get()
        try:
            returncode, elapsed, reused = run_one(path, cfg, gpu, args.python, args.trainer, provenance,
                                                args.force, args.echo_last_line)
        except Exception as exc:
            print(f"[run_grid] ERROR {path}: {exc}", flush=True)
            returncode, elapsed, reused = 1, 0.0, False
        finally:
            slots.put(gpu)
        if returncode:
            try:
                record_failure(path, cfg["out_dir"], returncode, fail_lock)
            except OSError as exc:
                print(f"[run_grid] could not write failure log: {exc}", file=sys.stderr)
        return path, cfg, returncode, elapsed, reused

    failed = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            _, cfg, rc, elapsed, reused = future.result()
            failed += bool(rc)
            status = "reused" if reused else ("ok" if rc == 0 else f"FAIL({rc})")
            print(f"[run_grid] {done}/{len(tasks)} {status} {Path(cfg['out_dir']).name} "
                  f"{elapsed:.0f}s | failed {failed}", flush=True)
    print(f"[run_grid] finished {len(tasks)} runs in {(time.monotonic() - start) / 60:.1f} min ({failed} failed)")
    if failed:
        print(f"[run_grid] failures logged to {FAILURES_PATH}")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
