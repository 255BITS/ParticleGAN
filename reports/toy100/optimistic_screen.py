"""Predeclare and run 24 constant-rate Optimistic Adam mechanism probes.

The previous 96-row Adam rate screen failed mode-hold. This wave changes the
update direction at that bottleneck, retaining the winner's loss and noise.
Every attempted host runs its frozen complete budget. All output is scratch
evidence: it cannot pass the production common gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DRIVER = ROOT / "reports/toy100/optimistic_transfer_episode.py"
OPTIMIZER = ROOT / "reports/toy100/optimistic_adam_scratch.py"
REGRADER = ROOT / "reports/toy100/optimistic_regrade.py"
ORDER = (
    "mode_hold", "trajectory", "residual_student", "img_stripes2",
    "img_bars4", "vector_overlap", "img_blobs4", "img_intensity2",
    "vector_unequal_mass", "vector_unequal_width",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(root: Path) -> dict:
    from benchmarks.transfer_suite import suite
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from particlegan.recipes import learning_rate_scale

    if root.exists():
        raise FileExistsError(root)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("commit source before predeclaration")
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    source = ROOT / "configs/toy100/shared_candidate.json"
    (root / "base_config.json").write_bytes(source.read_bytes())
    base = json.loads(source.read_bytes())
    rows = []
    for index, (lr, beta2, alpha) in enumerate(itertools.product(
        (.001, .0025, .00425), (.99, .999), (.125, .25, .5, 1.0),
    )):
        name = f"og{index:03d}"
        config = dict(base)
        config.pop("network_lr_horizon_cap")
        config.pop("network_lr_floor")
        config.update(name=name, lr=lr, betas=[0.0, beta2],
                      lr_anneal_start=0.0, lr_floor=1.0)
        recipe, _, _ = declared_recipe(config)
        assert recipe.lr_floor == 1.0
        assert all(learning_rate_scale(step, 7000, 0, 1) == 1
                   for step in (0, 1, 1600, 4200, 6999))
        file = f"configs/{name}.json"
        _write(root / file, config)
        rows.append(dict(id=name, base="winner_k1176_constant", alpha=alpha,
                         lr=lr, beta2=beta2, stage_order=list(ORDER),
                         config_file=file, config_sha256=_sha(root / file)))
    with tempfile.TemporaryDirectory(prefix="constant-game-source-") as temporary:
        hashes = suite.snapshot(Path(temporary))["source_sha256"]
    hashes["benchmarks/toy100/models.py"] = _sha(ROOT / "benchmarks/toy100/models.py")
    manifest = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        optimizer_source_sha256=_sha(OPTIMIZER), driver_source_sha256=_sha(DRIVER),
        regrader_source_sha256=_sha(REGRADER), screen_source_sha256=_sha(Path(__file__)),
        source_sha256=hashes, base_config_sha256=_sha(source),
        task_order=list(ORDER), rows=rows, scratch_common_gate_eligible=False,
        fixed_legacy_seed=0, fixed_native_seed=1234,
        all_attempted_hosts_use_complete_frozen_budgets=True,
        constant_rate_definition="Every actual G/D/prior schedule multiplier is 1; no network cap/floor",
        actual_update="theta -= lr * ((1+alpha)*u_current - alpha*u_previous); bias-corrected Adam directions",
        additional_gradient_evaluations_per_update=0,
        selection_rule="Stop at first strict host failure; ten passes require a fresh full19 before native3/common22",
    )
    _write(root / "predeclared_manifest.json", manifest)
    (root / "predeclared_manifest.sha256").write_text(_sha(root / "predeclared_manifest.json") + "\n")
    return manifest


def validate_manifest(root: Path) -> dict:
    manifest = json.loads((root / "predeclared_manifest.json").read_text())
    assert _sha(root / "predeclared_manifest.json") == (root / "predeclared_manifest.sha256").read_text().strip()
    assert manifest["source_commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert manifest["optimizer_source_sha256"] == _sha(OPTIMIZER)
    assert manifest["driver_source_sha256"] == _sha(DRIVER)
    assert manifest["regrader_source_sha256"] == _sha(REGRADER)
    assert manifest["screen_source_sha256"] == _sha(Path(__file__))
    assert manifest["base_config_sha256"] == _sha(root / "base_config.json")
    assert manifest["task_order"] == list(ORDER)
    assert manifest["scratch_common_gate_eligible"] is False
    assert len(manifest["rows"]) == 24
    assert {(row["lr"], row["beta2"], row["alpha"]) for row in manifest["rows"]} == set(itertools.product(
        (.001, .0025, .00425), (.99, .999), (.125, .25, .5, 1.0)))
    for file, sha in manifest["source_sha256"].items():
        assert _sha(ROOT / file) == sha, file
    for row in manifest["rows"]:
        assert row["stage_order"] == list(ORDER)
        assert _sha(root / row["config_file"]) == row["config_sha256"]
    return manifest


def _candidate(root_text: str, row: dict) -> dict:
    import torch
    from reports.toy100.optimistic_transfer_episode import run as episode
    from reports.toy100.optimistic_regrade import regrade_episode
    from benchmarks.toy_suite import _episode_rows

    torch.set_num_threads(1)
    root = Path(root_text)
    manifest = validate_manifest(root)
    destination = root / "runs" / row["id"]
    destination.mkdir(parents=True)
    result = dict(**row, attempted=[], skipped=[], status="running", shared_gate_eligible=False)
    start = time.perf_counter()
    with (destination / "worker.log").open("x", buffering=1) as logfile, redirect_stdout(logfile), redirect_stderr(logfile):
        try:
            for task in ORDER:
                print(f"START {row['id']} {task}", flush=True)
                tick = time.perf_counter()
                output = destination / task
                observation = episode(root / row["config_file"], task, row["alpha"], output)
                checked = regrade_episode(
                    output, task=task, alpha=row["alpha"],
                    config_sha256=row["config_sha256"],
                    optimizer_source_sha256=manifest["optimizer_source_sha256"],
                    driver_source_sha256=manifest["driver_source_sha256"],
                    regrader_source_sha256=manifest["regrader_source_sha256"],
                    manifest_sha256=_sha(root / "predeclared_manifest.json"),
                    source_commit=manifest["source_commit"],
                )
                frozen = _episode_rows(output, (task,), candidate=True, allow_scratch=True)
                assert frozen["status"] == checked["status"] == observation["status"]
                assert frozen["cases"][task]["noise_applied"] is True
                protocol = json.loads((output / "protocol.json").read_text())
                assert protocol["source_sha256"] == manifest["source_sha256"]
                observation.update(seconds=time.perf_counter() - tick,
                                   final=frozen["cases"][task]["final"],
                                   passing_suffix=frozen["cases"][task]["passing_suffix"])
                result["attempted"].append(observation)
                print(json.dumps(dict(event="DONE", row=row["id"], **observation)), flush=True)
                _write(destination / "row_result.json", result)
                if observation["status"] != "PASS":
                    break
            result["skipped"] = list(ORDER[len(result["attempted"]):])
            result["status"] = "READY_FULL19" if len(result["attempted"]) == len(ORDER) and all(
                item["status"] == "PASS" for item in result["attempted"]) else "STOPPED"
        except Exception as error:
            import traceback
            traceback.print_exc()
            result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        result["wall_seconds"] = time.perf_counter() - start
        _write(destination / "row_result.json", result)
    return result


def run(root: Path) -> dict:
    manifest = validate_manifest(root)
    report = dict(status="running", source_commit=manifest["source_commit"], rows=[])
    _write(root / "screen_result.json", report)
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_candidate, str(root), row): row["id"] for row in manifest["rows"]}
        for future in as_completed(futures):
            result = future.result()
            report["rows"].append(result)
            report["rows"].sort(key=lambda row: row["id"])
            _write(root / "screen_result.json", report)
            print(json.dumps(dict(event="ROW_DONE", id=result["id"], status=result["status"],
                                  attempted=len(result["attempted"]),
                                  final=result["attempted"][-1] if result["attempted"] else result.get("error"))), flush=True)
    report["status"] = "complete"
    _write(root / "screen_result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.root) if args.action == "prepare" else run(args.root)
    print(json.dumps(dict(status=args.action, rows=len(result["rows"]))), flush=True)


if __name__ == "__main__":
    main()
