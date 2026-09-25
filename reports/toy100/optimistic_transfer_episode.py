"""Run one frozen transfer host with the scratch Optimistic Adam adapter.

The declared host, data, budgets, noise, controller, losses, and loop order are
left to ``toy100_compatibility.run``. Only Adam's update direction changes.
This wrapper archives the exact optimizer source and every applied group LR,
and marks the resulting episode ineligible for the production shared gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.transfer_suite import toy100_compatibility as compatibility
from benchmarks.transfer_suite.protocol import test_verdict
from reports.toy100.optimistic_adam_scratch import optimistic_adam


SOURCE = ROOT / "reports/toy100/optimistic_adam_scratch.py"
DRIVER = Path(__file__)
REGRADER = ROOT / "reports/toy100/optimistic_regrade.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(config_path: Path, task: str, alpha: float, output: Path,
        *, amsgrad: bool = False) -> dict:
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    source_bytes = SOURCE.read_bytes()
    driver_bytes = DRIVER.read_bytes()
    regrader_bytes = REGRADER.read_bytes()
    manifest_root = output.resolve().parents[2]
    from reports.toy100.optimistic_screen import validate_manifest
    manifest = validate_manifest(manifest_root)
    row = next((item for item in manifest["rows"]
                if item["id"] == output.parent.name), None)
    if (row is None or config_path != (manifest_root / row["config_file"]).resolve()
            or row["config_sha256"] != _sha(config_bytes)
            or row["alpha"] != alpha or row.get("amsgrad", False) != amsgrad
            or task not in row["stage_order"]
            or task != output.name):
        raise RuntimeError("scratch host differs from predeclared row")
    manifest_sha256 = _sha((manifest_root / "predeclared_manifest.json").read_bytes())
    with optimistic_adam(alpha, amsgrad=amsgrad, diagnostics=True) as recorder:
        records = compatibility.run(config_path, output, tasks=(task,))
    if len(records) != 1 or records[0]["name"] != task:
        raise RuntimeError("scratch episode did not return exactly the declared host")
    if config_path.read_bytes() != config_bytes or SOURCE.read_bytes() != source_bytes:
        raise RuntimeError("declared config or Optimistic Adam source changed during episode")
    if DRIVER.read_bytes() != driver_bytes or REGRADER.read_bytes() != regrader_bytes:
        raise RuntimeError("scratch driver or regrader source changed during episode")
    validate_manifest(manifest_root)
    record = records[0]
    artifact = output / record["artifact"]
    saved = json.loads(gzip.decompress(artifact.read_bytes()))
    if saved["result"].get("error"):
        raise RuntimeError("frozen host error: " + saved["result"]["error"])
    receipt = recorder.receipt()
    if receipt["optimizer_count"] < 1 or receipt["optimizer_step_calls"] < 1:
        raise RuntimeError("no Adam optimizer received scratch updates")
    if alpha > 0 and receipt["parameter_updates"] != sum(
        row["optimistic_parameter_update_count"] for row in receipt["optimizers"]
    ):
        raise RuntimeError("an Adam parameter update missed the optimistic correction")
    if any(row["step_calls"] != len(row["group_lrs"])
           or any(count == 0 for count in row["group_parameter_updates"])
           for row in receipt["optimizers"]):
        raise RuntimeError("incomplete optimizer group application receipt")
    receipt.update(
        task=task,
        config_sha256=_sha(config_bytes),
        source_sha256=_sha(source_bytes),
        driver_sha256=_sha(driver_bytes),
        manifest_sha256=manifest_sha256,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        applied_roles=sorted({item["role"] for item in record["applied"]}),
    )
    _json(output / "optimizer_receipt.json", receipt)
    optimizer_hash = _sha((output / "optimizer_receipt.json").read_bytes())
    shutil.copyfile(SOURCE, output / "optimistic_adam_source.py")
    shutil.copyfile(DRIVER, output / "optimistic_transfer_driver.py")
    shutil.copyfile(REGRADER, output / "optimistic_regrade_source.py")

    binding = {
        "shared_gate_eligible": False,
        "scratch_optimizer": "optimistic_adam_algorithm1_damped_v1",
        "alpha": float(alpha),
        "optimizer_source_sha256": _sha(source_bytes),
        "driver_source_sha256": _sha(driver_bytes),
        "regrader_source_sha256": _sha(regrader_bytes),
        "optimizer_receipt_file": "optimizer_receipt.json",
        "optimizer_receipt_sha256": optimizer_hash,
        "config_sha256": _sha(config_bytes),
        "manifest_sha256": manifest_sha256,
    }
    if amsgrad:
        binding["amsgrad"] = True
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["scratch_optimizer_policy"] = binding
    protocol["shared_gate_eligible"] = False
    _json(protocol_path, protocol)

    index_path = output / "index.json"
    index = json.loads(index_path.read_text())
    if len(index["records"]) != 1 or index["records"][0]["name"] != task:
        raise RuntimeError("compatibility index differs from declared host")
    if artifact != output / index["records"][0]["artifact"]:
        raise RuntimeError("compatibility index artifact differs from returned record")
    saved["scratch_optimizer_policy"] = binding
    saved["shared_gate_eligible"] = False
    raw = (json.dumps(saved, sort_keys=True, allow_nan=False) + "\n").encode()
    artifact.write_bytes(gzip.compress(raw, mtime=0))

    index["records"][0].update(
        scratch_optimizer_policy=binding,
        shared_gate_eligible=False,
        uncompressed_sha256=_sha(raw),
    )
    _json(index_path, index)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["scratch_optimizer_policy"] = binding
    summary["shared_gate_eligible"] = False
    summary["cases"][0]["scratch_optimizer_policy"] = binding
    _json(summary_path, summary)
    # Recompute the host verdict using its frozen specification and raw curve;
    # the scratch policy itself does not change any threshold.
    verdict = test_verdict(saved["spec"], saved["result"])
    if verdict != saved["verdict"] or verdict != record["verdict"]:
        raise RuntimeError("frozen host verdict changed during scratch binding")
    result = {"task": task, "alpha": float(alpha), "status": verdict["status"],
              "config_sha256": _sha(config_bytes), "optimizer_receipt_sha256": optimizer_hash,
              "optimizer_step_calls": receipt["optimizer_step_calls"],
              "parameter_updates": receipt["parameter_updates"],
              "shared_gate_eligible": False}
    _json(output / "scratch_result.json", result)
    from reports.toy100.optimistic_regrade import regrade_episode
    checked = regrade_episode(
        output, task=task, alpha=alpha, config_sha256=_sha(config_bytes),
        optimizer_source_sha256=_sha(source_bytes),
        driver_source_sha256=_sha(driver_bytes),
        regrader_source_sha256=_sha(regrader_bytes),
        manifest_sha256=manifest_sha256,
        amsgrad=amsgrad,
        source_commit=receipt["source_commit"],
    )
    if checked["status"] != verdict["status"]:
        raise RuntimeError("scratch regrade changed frozen host verdict")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amsgrad", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.task, args.alpha, args.output, amsgrad=args.amsgrad)), flush=True)


if __name__ == "__main__":
    main()
