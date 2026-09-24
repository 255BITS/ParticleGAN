"""Bounded, production-ineligible GN discriminator screen on three frozen hosts.

The candidate changes only the discriminator's scalar-logit map to paper
eq. 11 and disables the old gradient penalty. All host data, architectures,
noise, optimizer settings, seeds, budgets, and gates stay frozen. Every episode
records the active GN source and call counts; its saved config carries an
unsupported scratch marker so removing a protocol stamp alone cannot make an
ordinary common-gate artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_suite import _episode_rows
from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite import toy100_compatibility as compatibility
from reports.toy100.gradient_normalization_probe import GNReceipt, legacy_gn_patch


SCRIPT = Path(__file__).resolve()
GN_SOURCE = ROOT / "reports/toy100/gradient_normalization_probe.py"
MODEL = ROOT / "benchmarks/toy100/models.py"
BASE_CONFIGS = {
    "incumbent": ROOT / "configs/toy100/shared_candidate.json",
    "simple": Path("/ml2/hypergan/ParticleGAN-toy-constraints-ablation/configs/toy100/constraints_simple_regularization.json"),
}
ORDER = ("trajectory", "residual_student", "mode_hold")
SCRATCH_SELECTOR = "gradient_normalization_eq11_fnone_v1"
ENV_PIN = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2", "ONEDNN_MAX_CPU_ISA": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _archive_source(output: Path) -> dict:
    sources = {str(path.relative_to(ROOT)): path for path in (SCRIPT, GN_SOURCE)}
    hashes = {name: _sha(path) for name, path in sources.items()}
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sources.items():
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
    return hashes


def prepare(root: Path) -> dict:
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for directory in ("configs", "evidence_configs", "base_configs"):
        (root / directory).mkdir()
    source = suite.snapshot(root)["source_sha256"]
    source[str(MODEL.relative_to(ROOT))] = _sha(MODEL)
    rows = []
    for core, path in BASE_CONFIGS.items():
        base_path = root / "base_configs" / f"{core}.json"
        shutil.copyfile(path, base_path)
        base = json.loads(base_path.read_text())
        for task in ORDER:
            config = dict(base)
            config.update(name=f"scratch_gn_{core}_{task}",
                          reg_arm="f_none", reg_coeff=0.0, reg_kappa=0.0)
            training = root / "configs" / f"{core}__{task}.json"
            evidence = root / "evidence_configs" / training.name
            _write(training, config)
            _write(evidence, config | {"scratch_discriminator_policy": SCRATCH_SELECTOR})
            rows.append(dict(core=core, task=task,
                             training_config=str(training.relative_to(root)),
                             training_config_sha256=_sha(training),
                             evidence_config=str(evidence.relative_to(root)),
                             evidence_config_sha256=_sha(evidence)))
    manifest = dict(
        status="predeclared", source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(), source_sha256=source,
        gn_source_sha256=_sha(GN_SOURCE), driver_source_sha256=_sha(SCRIPT),
        base_config_sha256={core: _sha(root / "base_configs" / f"{core}.json")
                            for core in BASE_CONFIGS},
        order=list(ORDER), rows=rows, fixed_host_seed=0, fixed_budgets=True,
        unchanged_noise=True, environment=ENV_PIN, max_workers=2,
        new_formulation="f/(||grad_data f||_2+abs(f)+finfo(dtype).eps)",
        denominator_detached=False,
        old_penalty=dict(reg_arm="f_none", reg_coeff=0.0, reg_kappa=0.0),
        common_gate_eligible=False,
        promotion="only 3/3 strict numerical PASS warrants broader adaptation review",
        batch_coupled_discriminator="unsupported; exact per-sample Jacobian needed",
    )
    _write(root / "predeclared_manifest.json", manifest)
    (root / "predeclared_manifest.sha256").write_text(
        _sha(root / "predeclared_manifest.json") + "\n",
    )
    return manifest


def validate(root: Path) -> dict:
    manifest = json.loads((root / "predeclared_manifest.json").read_text())
    if (_sha(root / "predeclared_manifest.json") !=
            (root / "predeclared_manifest.sha256").read_text().strip()):
        raise ValueError("predeclared manifest changed")
    if (manifest["source_commit"] != subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() or manifest["gn_source_sha256"] != _sha(GN_SOURCE)
            or manifest["driver_source_sha256"] != _sha(SCRIPT)
            or manifest["environment"] != ENV_PIN or manifest["order"] != list(ORDER)):
        raise ValueError("GN source epoch changed")
    for name, digest in manifest["source_sha256"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"training source changed: {name}")
    for core in BASE_CONFIGS:
        if _sha(root / "base_configs" / f"{core}.json") != manifest["base_config_sha256"][core]:
            raise ValueError(f"base recipe changed: {core}")
    for row in manifest["rows"]:
        for key in ("training_config", "evidence_config"):
            if _sha(root / row[key]) != row[key + "_sha256"]:
                raise ValueError(f"predeclared {key} changed: {row['core']}/{row['task']}")
    return manifest


def _row(manifest: dict, core: str, task: str) -> dict:
    return next(row for row in manifest["rows"]
                if row["core"] == core and row["task"] == task)


def _check_gn_receipt(receipt: dict, *, task: str, steps: int) -> None:
    if (receipt.get("calls", 0) < 2 * steps
            or receipt.get("grad_enabled_calls", 0) < 2 * steps
            or receipt.get("no_grad_calls", 0) < 0
            or receipt.get("data_elements", 0) <= 0
            or receipt.get("denominator_detached") is not False
            or receipt.get("formula") !=
            "f/(||grad_data f||_2+abs(f)+finfo(dtype).eps)"
            or receipt.get("routes") != {"wrapped_critic": receipt.get("calls")}
            or receipt.get("data_indices") != {
                "1" if task in ("trajectory", "residual_student") else "0":
                receipt.get("calls")
            }
            or not math.isfinite(receipt.get("denominator_min", math.nan))
            or not math.isfinite(receipt.get("denominator_max", math.nan))
            or receipt["denominator_min"] <= 0
            or receipt["denominator_max"] < receipt["denominator_min"]):
        raise ValueError(f"GN did not normalize every expected data route: {task}")


def run_episode(root: Path, core: str, task: str) -> dict:
    manifest = validate(root)
    if core not in BASE_CONFIGS or task not in ORDER:
        raise ValueError("undeclared GN row")
    row = _row(manifest, core, task)
    output = root / "runs" / core / task
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = GNReceipt()
    with legacy_gn_patch(receipt):
        compatibility.run(root / row["training_config"], output, tasks=(task,))
    numeric = _episode_rows(output, (task,), candidate=True)
    if numeric["status"] not in ("PASS", "FAIL"):
        raise ValueError(f"pre-marker original frozen gate rejected GN episode: {numeric}")
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("saved episode source differs from declaration")
    index = json.loads((output / "index.json").read_text())
    record = json.loads(gzip.decompress((output / index["records"][0]["artifact"]).read_bytes()))
    steps = record["spec"]["steps"]
    receipt_data = receipt.as_dict()
    _check_gn_receipt(receipt_data, task=task, steps=steps)
    receipt_data.update(task=task, steps=steps, core=core,
                        numerical_frozen_gate=numeric["status"])
    _write(output / "scratch_gn_receipt.json", receipt_data)
    archive_hashes = _archive_source(output / "scratch_source.tar.gz")
    if archive_hashes != {
        str(SCRIPT.relative_to(ROOT)): manifest["driver_source_sha256"],
        str(GN_SOURCE.relative_to(ROOT)): manifest["gn_source_sha256"],
    }:
        raise ValueError("archived scratch GN source differs from declaration")
    marker = dict(
        selector=SCRATCH_SELECTOR, common_gate_eligible=False,
        source_commit=manifest["source_commit"], source_sha256=archive_hashes,
        source_archive_file="scratch_source.tar.gz",
        source_archive_sha256=_sha(output / "scratch_source.tar.gz"),
        receipt_file="scratch_gn_receipt.json",
        receipt_sha256=_sha(output / "scratch_gn_receipt.json"),
        original_training_config_sha256=row["training_config_sha256"],
        saved_evidence_config_sha256=row["evidence_config_sha256"],
        task=task, core=core,
    )
    # Retain both exact predeclared config variants. The saved one contains an
    # unsupported selector; deleting only a receipt/protocol flag cannot make
    # this look like a production common-recipe candidate.
    saved_config = output / protocol["config_file"]
    saved_config.write_bytes((root / row["evidence_config"]).read_bytes())
    protocol.update(config_sha256=row["evidence_config_sha256"],
                    scratch_discriminator_policy=marker,
                    shared_gate_eligible=False)
    _write(output / "protocol.json", protocol)
    summary = json.loads((output / "summary.json").read_text())
    summary.update(config_sha256=row["evidence_config_sha256"],
                   scratch_discriminator_policy=marker,
                   shared_gate_eligible=False)
    _write(output / "summary.json", summary)
    index.update(scratch_discriminator_policy=marker, shared_gate_eligible=False)
    _write(output / "index.json", index)
    saved = dict(status=numeric["status"], task=task, core=core,
                 final=numeric["cases"][task]["final"],
                 passing_suffix=numeric["cases"][task]["passing_suffix"],
                 steps=steps, marker=marker)
    _write(output / "scratch_result.json", saved)
    if _episode_rows(output, (task,), candidate=True)["status"] != "INVALID":
        raise ValueError("production combined gate accepted scratch GN episode")
    checked = regrade_episode(root, output, core=core, task=task)
    if checked["status"] != saved["status"]:
        raise ValueError("scratch GN independent regrade differs")
    return checked


def regrade_episode(root: Path, output: Path, *, core: str, task: str) -> dict:
    manifest = json.loads((root / "predeclared_manifest.json").read_text())
    row = _row(manifest, core, task)
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    index = json.loads((output / "index.json").read_text())
    marker = protocol.get("scratch_discriminator_policy")
    if (not isinstance(marker, dict) or marker.get("selector") != SCRATCH_SELECTOR
            or marker != summary.get("scratch_discriminator_policy")
            or marker != index.get("scratch_discriminator_policy")
            or protocol.get("shared_gate_eligible") is not False
            or summary.get("shared_gate_eligible") is not False
            or index.get("shared_gate_eligible") is not False
            or protocol["source_sha256"] != manifest["source_sha256"]
            or marker.get("saved_evidence_config_sha256") != row["evidence_config_sha256"]
            or marker.get("original_training_config_sha256") != row["training_config_sha256"]
            or marker.get("source_commit") != manifest["source_commit"]
            or _sha(output / protocol["config_file"]) != row["evidence_config_sha256"]
            or json.loads((output / protocol["config_file"]).read_text()).get(
                "scratch_discriminator_policy") != SCRATCH_SELECTOR):
        raise ValueError("saved GN episode policy/config/source binding differs")
    receipt_path = output / marker["receipt_file"]
    if _sha(receipt_path) != marker["receipt_sha256"]:
        raise ValueError("GN application receipt is missing or tampered")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("task") != task or receipt.get("core") != core:
        raise ValueError("GN receipt bound to another row")
    _check_gn_receipt(receipt, task=task, steps=receipt["steps"])
    if marker["source_sha256"] != {
        str(SCRIPT.relative_to(ROOT)): manifest["driver_source_sha256"],
        str(GN_SOURCE.relative_to(ROOT)): manifest["gn_source_sha256"],
    } or _sha(output / marker["source_archive_file"]) != marker["source_archive_sha256"]:
        raise ValueError("GN executable source archive differs")
    with tarfile.open(output / marker["source_archive_file"], "r:gz") as archive:
        members = archive.getmembers()
        if {member.name for member in members} != set(marker["source_sha256"]):
            raise ValueError("GN archive member list differs")
        for member in members:
            stream = archive.extractfile(member) if member.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != marker[
                "source_sha256"
            ][member.name]:
                raise ValueError("GN archive source bytes differ")
    saved = json.loads((output / "scratch_result.json").read_text())
    if (saved.get("marker") != marker or saved.get("task") != task
            or saved.get("core") != core or saved.get("steps") != receipt["steps"]):
        raise ValueError("GN scratch result is not bound to its receipt")
    if _episode_rows(output, (task,), candidate=True)["status"] != "INVALID":
        raise ValueError("production combined gate accepted scratch GN evidence")
    # Reconstruct the exact ordinary training declaration in a temporary copy
    # and re-run the original frozen metric gate. The real saved copy remains
    # marked ineligible and source/receipt checks above are required first.
    with tempfile.TemporaryDirectory(prefix="gn-regrade-") as temporary:
        clean = Path(temporary) / "episode"
        shutil.copytree(output, clean)
        (clean / protocol["config_file"]).write_bytes((root / row["training_config"]).read_bytes())
        clean_protocol = dict(protocol)
        clean_protocol.pop("scratch_discriminator_policy")
        clean_protocol.pop("shared_gate_eligible")
        clean_protocol["config_sha256"] = row["training_config_sha256"]
        _write(clean / "protocol.json", clean_protocol)
        clean_summary = dict(summary)
        clean_summary.pop("scratch_discriminator_policy")
        clean_summary.pop("shared_gate_eligible")
        clean_summary["config_sha256"] = row["training_config_sha256"]
        _write(clean / "summary.json", clean_summary)
        clean_index = dict(index)
        clean_index.pop("scratch_discriminator_policy")
        clean_index.pop("shared_gate_eligible")
        _write(clean / "index.json", clean_index)
        grade = _episode_rows(clean, (task,), candidate=True)
    if (grade["status"] not in ("PASS", "FAIL")
            or grade["status"] != saved["status"]
            or grade["cases"][task]["final"] != saved["final"]
            or grade["cases"][task]["passing_suffix"] != saved["passing_suffix"]):
        raise ValueError(f"GN numerical frozen gate differs: {grade}")
    return saved


def _screen_core(root: Path, core: str) -> dict:
    env = os.environ.copy()
    env.update(ENV_PIN)
    attempts = []
    for task in ORDER:
        validate(root)
        output = root / "runs" / core / task
        logfile = root / "logs" / f"{core}__{task}.log"
        command = ["ionice", "-c2", "-n4", sys.executable, "-u", "-m",
                   "reports.toy100.gradient_normalization_screen", "episode",
                   "--root", str(root), "--core", core, "--task", task]
        with logfile.open("x") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            row = dict(task=task, status="ERROR", exit_code=completed.returncode,
                       log=str(logfile.relative_to(root)))
        else:
            checked = regrade_episode(root, output, core=core, task=task)
            row = dict(task=task, status=checked["status"], final=checked["final"],
                       passing_suffix=checked["passing_suffix"],
                       log=str(logfile.relative_to(root)),
                       episode=str(output.relative_to(root)))
        attempts.append(row)
        if row["status"] != "PASS":
            break
    return dict(core=core, attempted=attempts,
                skipped=list(ORDER[len(attempts):]),
                status="READY_BROAD_REVIEW" if len(attempts) == len(ORDER)
                and all(x["status"] == "PASS" for x in attempts) else "STOPPED")


def screen(root: Path) -> dict:
    manifest = validate(root)
    (root / "runs").mkdir(exist_ok=False)
    (root / "logs").mkdir(exist_ok=False)
    report = dict(status="running", manifest_sha256=_sha(root / "predeclared_manifest.json"),
                  source_commit=manifest["source_commit"], rows={})
    _write(root / "screen_result.json", report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_screen_core, root, core): core for core in BASE_CONFIGS}
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
    parser.add_argument("action", choices=("prepare", "episode", "screen", "regrade"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core", choices=tuple(BASE_CONFIGS))
    parser.add_argument("--task", choices=ORDER)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root)
    elif args.action == "screen":
        result = screen(args.root)
    else:
        if args.core is None or args.task is None:
            parser.error("episode/regrade requires --core and --task")
        output = args.root / "runs" / args.core / args.task
        result = (run_episode(args.root, args.core, args.task)
                  if args.action == "episode" else
                  regrade_episode(args.root, output, core=args.core, task=args.task))
    print(json.dumps(dict(status=result["status"], rows=len(result.get("rows", {})))), flush=True)


if __name__ == "__main__":
    main()
