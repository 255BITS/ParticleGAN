"""Predeclare and screen target-agnostic output/input noise variants.

This bounded experiment keeps the f0595c4 shared optimizer, native model card,
frozen host budgets, targets, seeds, and gates. A row stops on its first failed
host. All attempted episodes are independently regraded from saved evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_suite import _episode_rows
from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite.public_default_verification import load_declaration


BASE_CONFIG = ROOT / "configs/toy100/shared_candidate.json"
SCRIPT = Path(__file__).resolve()
GATE = ROOT / "benchmarks/toy_suite.py"
MODEL = ROOT / "benchmarks/toy100/models.py"
ORDER = (
    "trajectory", "residual_student", "img_stripes2", "mode_hold",
    "img_bars4", "vector_overlap", "img_blobs4", "img_intensity2",
    "vector_unequal_mass", "vector_unequal_width",
)
WIDTHS = (0.005, 0.01, 0.05, 0.1, 0.2)
ENV_PIN = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _rows() -> list[dict]:
    rows = []
    for input_std, end in ((0., 1.), (.25, 1.), (.5, .1), (.5, 1.)):
        rows.append(dict(output_std=0., input_std=input_std, input_end=end,
                         warmup=0., learnable=False, reference=False))
    for learnable in (False, True):
        for width in WIDTHS:
            for input_std in (0., .5):
                for warmup in (0., .2):
                    rows.append(dict(output_std=width, input_std=input_std,
                                     input_end=1. if input_std == 0 else .1,
                                     warmup=warmup, learnable=learnable,
                                     reference=False))
            # Constant input noise probes removal of the original end=.1 timing.
            rows.append(dict(output_std=width, input_std=.5, input_end=1.,
                             warmup=0., learnable=learnable, reference=False))
    for input_std in (0., .5):
        rows.append(dict(output_std=.029, input_std=input_std,
                         input_end=1. if input_std == 0 else .1,
                         warmup=0., learnable=False, reference=True))
    assert len(rows) == 56
    return rows


def _name(row: dict) -> str:
    family = "learn" if row["learnable"] else "fixed"
    if row["reference"]:
        family = "reference"
    if row["output_std"] == 0:
        family = "zero"
    return (f"{family}_o{round(row['output_std'] * 1000):03d}"
            f"_i{round(row['input_std'] * 1000):03d}"
            f"_w{round(row['warmup'] * 1000):03d}"
            f"_e{round(row['input_end'] * 1000):04d}")


def prepare(root: Path) -> dict:
    if root.exists():
        raise FileExistsError("bandwidth evidence root must be new")
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "source").mkdir()
    base = json.loads(BASE_CONFIG.read_text())
    rows = []
    for specification in _rows():
        name = _name(specification)
        config = dict(base)
        config.update(name="bandwidth_" + name,
                      output_noise_std=specification["output_std"],
                      input_noise_std=specification["input_std"],
                      input_noise_anneal_end=specification["input_end"],
                      output_noise_warmup=specification["warmup"])
        if specification["output_std"]:
            config["output_noise_rng"] = "isolated"
        else:
            config.pop("output_noise_rng", None)
        if specification["learnable"]:
            config["output_noise_learnable"] = True
        else:
            config.pop("output_noise_learnable", None)
        path = root / "configs" / f"{name}.json"
        _write(path, config)
        rows.append(dict(id=name, config_file=str(path.relative_to(root)),
                         config_sha256=_sha(path), **specification))
    source = suite.snapshot(root)["source_sha256"]
    source[str(MODEL.relative_to(ROOT))] = _sha(MODEL)
    shutil.copyfile(SCRIPT, root / "source/bandwidth_wave.py")
    shutil.copyfile(GATE, root / "source/toy_suite.py")
    manifest = dict(
        status="predeclared", base_commit="f0595c41807731857d215f0134a80bdd12c7e441",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        base_config_sha256=_sha(BASE_CONFIG), source_sha256=source,
        screen_source_sha256=_sha(SCRIPT), gate_source_sha256=_sha(GATE),
        task_order=list(ORDER), rows=rows,
        fixed_host_seed=0, fixed_native_seed=1234, fixed_budget=True,
        threads_per_worker=1, max_workers=6, environment=ENV_PIN,
        promotion="only 10/10 strict host PASS and nonreference policy advances to fresh19",
        references_nonpromotable=True,
    )
    _write(root / "predeclared_manifest.json", manifest)
    (root / "predeclared_manifest.sha256").write_text(
        _sha(root / "predeclared_manifest.json") + "\n",
    )
    return manifest


def validate(root: Path) -> dict:
    manifest = json.loads((root / "predeclared_manifest.json").read_text())
    if _sha(root / "predeclared_manifest.json") != (
        root / "predeclared_manifest.sha256"
    ).read_text().strip():
        raise ValueError("predeclared manifest changed")
    if (manifest["source_commit"] != subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() or manifest["screen_source_sha256"] != _sha(SCRIPT)
            or manifest["gate_source_sha256"] != _sha(GATE)
            or manifest["base_config_sha256"] != _sha(BASE_CONFIG)):
        raise ValueError("source epoch changed")
    if (manifest["task_order"] != list(ORDER) or len(manifest["rows"]) != 56
            or {row["id"] for row in manifest["rows"]} != {
                _name(row) for row in _rows()
            } or manifest["environment"] != ENV_PIN):
        raise ValueError("bounded screen declaration changed")
    for filename, digest in manifest["source_sha256"].items():
        if _sha(ROOT / filename) != digest:
            raise ValueError(f"training source changed: {filename}")
    for row in manifest["rows"]:
        if _sha(root / row["config_file"]) != row["config_sha256"]:
            raise ValueError(f"candidate config changed: {row['id']}")
    declared_names = {job["spec"]["name"] for job in load_declaration()[0]}
    if not set(ORDER) <= declared_names:
        raise ValueError("screen host is not in canonical plan")
    return manifest


def _run_row(root: Path, row: dict, manifest: dict) -> dict:
    result = dict(id=row["id"], config_sha256=row["config_sha256"],
                  reference=row["reference"], attempted=[], skipped=[], status="running")
    environment = os.environ.copy()
    environment.update(ENV_PIN)
    for task in ORDER:
        validate(root)
        output = root / "runs" / row["id"] / task
        output.parent.mkdir(parents=True, exist_ok=True)
        logfile = root / "logs" / f"{row['id']}__{task}.log"
        command = ["ionice", "-c2", "-n4", sys.executable, "-u", "-m",
                   "benchmarks.transfer_suite.toy100_compatibility",
                   "--config", str(root / row["config_file"]),
                   "--output", str(output), "--tasks", task]
        with logfile.open("x") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            observation = dict(task=task, status="ERROR", exit_code=completed.returncode,
                               log=str(logfile.relative_to(root)))
        else:
            grade = _episode_rows(output, (task,), candidate=True)
            protocol = json.loads((output / "protocol.json").read_text())
            if (protocol["config_sha256"] != row["config_sha256"]
                    or protocol["source_sha256"] != manifest["source_sha256"]):
                raise ValueError(f"archived recipe/source differs for {row['id']}/{task}")
            case = grade["cases"].get(task, {})
            observation = dict(task=task, status=grade["status"],
                               reason=grade.get("reason"), final=case.get("final"),
                               passing_suffix=case.get("passing_suffix"),
                               log=str(logfile.relative_to(root)),
                               episode=str(output.relative_to(root)))
        result["attempted"].append(observation)
        if observation["status"] != "PASS":
            break
    result["skipped"] = list(ORDER[len(result["attempted"]):])
    result["status"] = ("READY_FULL19" if not result["skipped"]
                        and not row["reference"] else "STOPPED")
    return result


def screen(root: Path, workers: int = 6) -> dict:
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError("workers must be between one and six")
    manifest = validate(root)
    (root / "runs").mkdir(exist_ok=False)
    (root / "logs").mkdir(exist_ok=False)
    report = dict(status="running", manifest_sha256=_sha(
        root / "predeclared_manifest.json"), source_commit=manifest["source_commit"],
        rows={})
    _write(root / "screen_result.json", report)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_row, root, row, manifest): row["id"]
                   for row in manifest["rows"]}
        for future in as_completed(futures):
            row_id = futures[future]
            report["rows"][row_id] = future.result()
            _write(root / "screen_result.json", report)
            observations = report["rows"][row_id]["attempted"]
            print(row_id, [(item["task"], item["status"]) for item in observations],
                  "skipped", len(report["rows"][row_id]["skipped"]), flush=True)
    report["status"] = "complete"
    _write(root / "screen_result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "screen"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = prepare(args.root) if args.action == "prepare" else screen(
        args.root, args.workers,
    )
    print(json.dumps(dict(status=result["status"], rows=len(result["rows"]))), flush=True)


if __name__ == "__main__":
    main()
