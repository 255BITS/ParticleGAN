"""Strict scratch screen of two-batch LOO bandwidth on two frozen recipe cores.

Each host receives the same estimator, applied to its first two unlabeled real
discriminator batches from a discarded preflight. The scalar output std can
therefore differ across hosts; these experiments are common-gate ineligible
until a production policy implements and verifies that algorithm directly.
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
from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100.bandwidth_kde_two_batch import capture_two_real_batches


SCRIPT = Path(__file__).resolve()
CAPTURE = ROOT / "reports/toy100/bandwidth_kde_two_batch.py"
ESTIMATOR = ROOT / "reports/toy100/bandwidth_kde_probe.py"
MODEL = ROOT / "benchmarks/toy100/models.py"
BASE_CONFIGS = {
    "incumbent": ROOT / "configs/toy100/shared_candidate.json",
    "simple": Path("/ml2/hypergan/ParticleGAN-toy-constraints-ablation/configs/toy100/constraints_simple_regularization.json"),
}
ORDER = (
    "trajectory", "residual_student", "img_stripes2", "mode_hold",
    "img_bars4", "vector_overlap", "img_blobs4", "img_intensity2",
    "vector_unequal_mass", "vector_unequal_width",
)
ENV_PIN = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(root: Path, receipt_paths: dict[str, Path]) -> dict:
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for folder in ("configs", "base_configs", "source", "calibration"):
        (root / folder).mkdir()
    source = suite.snapshot(root)["source_sha256"]
    source[str(MODEL.relative_to(ROOT))] = _sha(MODEL)
    for path in (SCRIPT, CAPTURE, ESTIMATOR):
        shutil.copyfile(path, root / "source" / path.name)
    receipts = {}
    for core, source_path in receipt_paths.items():
        destination = root / "calibration" / f"{core}.json"
        shutil.copyfile(source_path, destination)
        receipts[core] = json.loads(destination.read_text())
    if [(x["host"], x["first_two_real_sha256"], x["width"])
            for x in receipts["incumbent"]["rows"]] != [
            (x["host"], x["first_two_real_sha256"], x["width"])
            for x in receipts["simple"]["rows"]]:
        raise ValueError("the two core recipes changed the two real calibration batches")
    estimates = {row["host"]: row for row in receipts["incumbent"]["rows"]}
    rows = []
    for core, source_path in BASE_CONFIGS.items():
        base_path = root / "base_configs" / f"{core}.json"
        shutil.copyfile(source_path, base_path)
        base = json.loads(base_path.read_text())
        for task in ORDER:
            value = dict(base)
            value["name"] = f"kde_two_{core}_{task}"
            value["output_noise_std"] = estimates[task]["width"]
            destination = root / "configs" / f"{core}__{task}.json"
            _write(destination, value)
            rows.append(dict(core=core, task=task,
                             config_file=str(destination.relative_to(root)),
                             config_sha256=_sha(destination),
                             width=estimates[task]["width"],
                             first_two_real_sha256=estimates[task]["first_two_real_sha256"]))
    manifest = dict(
        status="predeclared", source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        source_sha256=source, script_sha256=_sha(SCRIPT),
        capture_source_sha256=_sha(CAPTURE), estimator_source_sha256=_sha(ESTIMATOR),
        calibration_receipt_sha256={core: _sha(root / "calibration" / f"{core}.json")
                                    for core in receipts},
        base_config_sha256={core: _sha(root / "base_configs" / f"{core}.json")
                            for core in BASE_CONFIGS},
        task_order=list(ORDER), rows=rows,
        policy="Gaussian LOO KDE on exactly the first two real D batches; preflight discarded",
        common_gate_eligible=False, fixed_host_seed=0, fixed_budget=True,
        max_workers=2, environment=ENV_PIN,
        promotion="only ten strict saved-evidence host PASS may advance to fresh19",
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
    ).strip() or manifest["script_sha256"] != _sha(SCRIPT)
            or manifest["capture_source_sha256"] != _sha(CAPTURE)
            or manifest["estimator_source_sha256"] != _sha(ESTIMATOR)
            or manifest["environment"] != ENV_PIN
            or manifest["task_order"] != list(ORDER)):
        raise ValueError("source epoch changed")
    for filename, digest in manifest["source_sha256"].items():
        if _sha(ROOT / filename) != digest:
            raise ValueError(f"training source changed: {filename}")
    for core in BASE_CONFIGS:
        if (_sha(root / "calibration" / f"{core}.json") !=
                manifest["calibration_receipt_sha256"][core]
                or _sha(root / "base_configs" / f"{core}.json") !=
                manifest["base_config_sha256"][core]):
            raise ValueError(f"calibration or base config changed: {core}")
    for row in manifest["rows"]:
        if _sha(root / row["config_file"]) != row["config_sha256"]:
            raise ValueError(f"candidate config changed: {row['core']}/{row['task']}")
    return manifest


def _screen_core(root: Path, core: str, manifest: dict) -> dict:
    environment = os.environ.copy()
    environment.update(ENV_PIN)
    jobs, profile = load_declaration()
    receipt = json.loads((root / "calibration" / f"{core}.json").read_text())
    estimate = {row["host"]: row for row in receipt["rows"]}
    observations = []
    for task in ORDER:
        validate(root)
        row = next(row for row in manifest["rows"]
                   if row["core"] == core and row["task"] == task)
        config = json.loads((root / row["config_file"]).read_text())
        base, noise, _ = declared_recipe(config)
        job = next(job for job in jobs if job["spec"]["name"] == task)
        spec, card, _ = declared_spec(job, profile, base)
        calibration = capture_two_real_batches(spec, card, base, noise)
        actual_hash = hashlib.sha256(calibration.tobytes()).hexdigest()
        if actual_hash != estimate[task]["first_two_real_sha256"]:
            raise ValueError(f"candidate changed calibration real batches: {core}/{task}")
        output = root / "runs" / core / task
        output.parent.mkdir(parents=True, exist_ok=True)
        logfile = root / "logs" / f"{core}__{task}.log"
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
                raise ValueError(f"archived source/config mismatch: {core}/{task}")
            case = grade["cases"].get(task, {})
            observation = dict(task=task, status=grade["status"],
                               final=case.get("final"),
                               passing_suffix=case.get("passing_suffix"),
                               width=row["width"],
                               episode=str(output.relative_to(root)),
                               log=str(logfile.relative_to(root)))
        observations.append(observation)
        if observation["status"] != "PASS":
            break
    return dict(core=core, attempted=observations,
                skipped=list(ORDER[len(observations):]),
                status="READY_FULL19" if len(observations) == len(ORDER) else "STOPPED")


def screen(root: Path) -> dict:
    manifest = validate(root)
    (root / "runs").mkdir(exist_ok=False)
    (root / "logs").mkdir(exist_ok=False)
    report = dict(status="running", manifest_sha256=_sha(root / "predeclared_manifest.json"),
                  source_commit=manifest["source_commit"], rows={})
    _write(root / "screen_result.json", report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_screen_core, root, core, manifest): core
                   for core in BASE_CONFIGS}
        for future in as_completed(futures):
            core = futures[future]
            report["rows"][core] = future.result()
            _write(root / "screen_result.json", report)
            print(core, [(x["task"], x["status"]) for x in
                         report["rows"][core]["attempted"]], flush=True)
    report["status"] = "complete"
    _write(root / "screen_result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "screen"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--incumbent-receipt", type=Path)
    parser.add_argument("--simple-receipt", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.incumbent_receipt is None or args.simple_receipt is None:
            parser.error("prepare requires both calibration receipts")
        result = prepare(args.root, {"incumbent": args.incumbent_receipt,
                                     "simple": args.simple_receipt})
    else:
        result = screen(args.root)
    print(json.dumps(dict(status=result["status"], rows=len(result["rows"]))), flush=True)


if __name__ == "__main__":
    main()
