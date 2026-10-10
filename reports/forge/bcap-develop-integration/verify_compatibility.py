"""Generate separate final-tree software evidence after the paid runs finish.

The earlier compatibility.json remains immutable. No scientific qualification,
full-budget training, or GPU parity claim is produced by this CPU workflow.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


AE_HOST = "benchmarks/locked_shared/hosts/ae_gan_hold.py"
AE_CARD = "configs/forge/tasks/ae_gan_hold.json"
AE_PATCH = "68127e2b5cde98c72ff50e4c42255535eab83842"
TEST = "tests/test_bcap_integration_compatibility.py"
WRAPPER = ("tests/test_host_integration_bcap.py::"
           "test_declared_ae_consumer_preserves_four_argument_evaluation_wrappers")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root)


def file_identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="Fresh directory outside Git for stdout, JUnit and state dumps")
    parser.add_argument("--receipt", type=Path,
                        help="Defaults to compatibility-post-execution.json beside the original receipt")
    args = parser.parse_args()
    root, artifacts = args.root.resolve(), args.artifacts.resolve()
    original_path = root / "reports/forge/bcap-develop-integration/compatibility.json"
    receipt_path = (args.receipt.resolve() if args.receipt else
                    original_path.with_name("compatibility-post-execution.json"))
    if receipt_path == original_path or receipt_path.exists() or artifacts.exists():
        raise ValueError("Preserve the original receipt and use fresh receipt/artifact destinations")
    if artifacts.is_relative_to(root):
        raise ValueError("Raw compatibility artifacts must stay outside the source checkout")
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    old_files = original["source_files_sha256"]
    if len(old_files) != 52:
        raise ValueError("Expected the preserved 52-file measured source inventory")
    if digest(old_files) != original["tested_source_snapshot_sha256"]:
        raise ValueError("The preserved source inventory does not match its measured identity")
    if sha((root / TEST).read_bytes()) != original["test_source_sha256"]:
        raise ValueError("Preserve the exact 63-case compatibility test source")
    files = {name: sha((root / name).read_bytes()) for name in old_files}
    changed = {name for name in files if files[name] != old_files[name]}
    if changed != {AE_HOST, AE_CARD}:
        raise ValueError(f"Expected solely the AE wrapper and its source binding; found {sorted(changed)}")
    if files[AE_HOST] != sha(git(root, "show", f"{AE_PATCH}:{AE_HOST}")):
        raise ValueError("AE wrapper does not match the declared held patch bytes")
    task = json.loads((root / AE_CARD).read_text())
    if task["evaluation"]["sources"][AE_HOST] != files[AE_HOST]:
        raise ValueError("Final AE task must bind the actual patched evaluator source")
    old_task = json.loads(git(root, "show", f"HEAD:{AE_CARD}"))
    if old_task != task:
        raise ValueError("Commit the final AE source binding before recording its identity")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files, TEST,
                            WRAPPER.split("::")[0]], cwd=root)
    if dirty.returncode:
        raise ValueError("Commit the measured source and test files before verification")
    tested_commit = git(root, "rev-parse", "HEAD").decode().strip()
    artifacts.mkdir(parents=True)
    junit, log = artifacts / "pytest.xml", artifacts / "pytest.log"
    command = [sys.executable, "-m", "pytest", "-q", TEST, WRAPPER,
               f"--basetemp={artifacts / 'pytest'}", f"--junitxml={junit}"]
    environment = dict(os.environ, BCAP_COMPATIBILITY_ROOT=str(root), PYTHONPATH=str(root),
                       CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    print(f"tail -F {log}", flush=True)
    with log.open("w") as stdout:
        result = subprocess.run(command, cwd=root, env=environment, stdout=stdout,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Compatibility verification failed; inspect {log}")
    cases = ET.parse(junit).getroot().findall(".//testcase")
    if len(cases) != 64 or any(case.find(tag) is not None for case in cases
                              for tag in ("failure", "error", "skipped")):
        raise ValueError("Require exactly 63 compatibility passes plus one wrapper pass, with no skips")
    compatibility_cases = [case for case in cases if "test_bcap_integration_compatibility" in
                           case.attrib.get("classname", "")]
    if len(compatibility_cases) != 63:
        raise ValueError("The expected 63 inactive develop comparisons did not all execute")
    if files != {name: sha((root / name).read_bytes()) for name in files}:
        raise ValueError("Measured source changed during verification")
    if tested_commit != git(root, "rev-parse", "HEAD").decode().strip():
        raise ValueError("Commit changed during verification; rerun on a stable tree")
    if original_path.read_bytes() != original_bytes:
        raise ValueError("The preserved original receipt changed")
    # Import the common comparator and content-identity helper from the exact
    # candidate tree. All expected state comes from its separate develop probe.
    sys.path.insert(0, str(root))
    os.environ["BCAP_COMPATIBILITY_ROOT"] = str(root)
    import torch
    from experiments.forge.state import state_digest
    spec = importlib.util.spec_from_file_location("bcap_compatibility_receipt_probe", root / TEST)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    directories = sorted(path for path in (artifacts / "pytest").glob("bcap-compatibility*")
                         if path.is_dir() and not path.is_symlink())
    if len(directories) != 1:
        raise ValueError("Expected one independently executed develop/candidate packet pair")
    directory = directories[0]
    reference = torch.load(directory / "develop.pt", weights_only=False)
    candidate = torch.load(directory / "candidate.pt", weights_only=False)
    state_identities = {}

    def exact(name, old, new):
        probe._assert_equal(old, new, name)
        old_hash, new_hash = state_digest(old), state_digest(new)
        assert old_hash == new_hash
        state_identities[name] = {"reference_sha256": old_hash, "candidate_sha256": new_hash}

    for case in probe.CASES:
        for prior in probe.PRIORS:
            key = f"{case}/{prior}"
            exact(key, reference[key]["final"], candidate[key]["final"])
        exact(f"word/{case}", reference["words"][case]["final"], candidate["words"][case]["final"])
    for case, host in probe.COMPONENT_CASES:
        key = f"{case}/{host}"
        old, new = deepcopy(reference["components"][key]), deepcopy(candidate["components"][key])
        for packet in (old, new):
            evaluation = packet["applied"]["field_ownership"]["task_contract"]["evaluation"]["value"]
            evaluation.pop("sources", None)
            evaluation.pop("evaluator_revision", None)
        exact(f"component/{key}", old, new)
    raw_paths = [log, junit, *(directory / name for name in
                 ("develop.pt", "candidate.pt", "develop.log", "candidate.log"))]
    raw_paths.extend((artifacts / "pytest").rglob("constraint_geometry-scored-outputs.pt"))
    suites = ET.parse(junit).getroot().findall(".//testsuite")
    host_acceptance = {}
    for case, matrix in candidate["host_matrix"].items():
        probe._assert_equal(reference["host_matrix"][case], matrix, case)
        probe._assert_equal(reference["host_matrix"][case], candidate["without_hooks"][case], case)
        host_acceptance[case] = {"ready": sum(not reasons for reasons in matrix.values()),
                                "blocked": sum(bool(reasons) for reasons in matrix.values()),
                                "exact_baseline_blockers_retained": True}
    receipt = {
        "schema_version": 1, "scope": "post_execution_bounded_software_compatibility", "verdict": "PASS",
        "baseline_commit": probe.DEVELOP, "tested_candidate_commit": tested_commit,
        "preserved_measured_receipt": {"path": str(original_path.relative_to(root)),
            "sha256": sha(original_bytes), "git_blob": git(root, "rev-parse", f"HEAD:{original_path.relative_to(root)}").decode().strip(),
            "tested_source_snapshot_sha256": original["tested_source_snapshot_sha256"]},
        "source_files_sha256": files, "tested_source_snapshot_sha256": digest(files),
        "source_delta_since_measured_receipt": {name: {"before_sha256": old_files[name], "after_sha256": files[name]}
                                                for name in sorted(changed)},
        "declared_delta": "AE evaluation/media capture retains original four-argument evaluator calls; task source pin rebound. No trainer, recipe, prior, initialization, budget, cadence, sampling law or numerical gate change.",
        "ae_wrapper_patch": AE_PATCH,
        "test_source_sha256": {name: sha((root / name).read_bytes())
                               for name in (TEST, WRAPPER.split("::")[0])},
        "reproduction_source_sha256": sha(Path(__file__).read_bytes()),
        "tests": {**{key: value for key, value in original["tests"].items()
                      if key not in ("passed", "duration_seconds")},
                  "inactive_compatibility_passed": 63, "four_argument_wrapper_passed": 1,
                  "passed": 64, "failed": 0, "skipped": 0,
                  "duration_seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
                  "additional_wrapper_software_outer_updates": 2},
        "protocol": original["protocol"], "verified_invariants": original["verified_invariants"],
        "state_identities": state_identities,
        "host_acceptance": host_acceptance,
        "runtime": {"device": "cpu", "python": sys.version.split()[0], "torch": torch.__version__,
                    "torch_threads": 1, "deterministic_algorithms": True},
        "limitations": [value for value in original["limitations"]
                        if value != "Final integrated host production tree must rerun these checks before publication"],
        "scientific_training_reruns": 0, "qualification_input": False,
        "artifact_archive": str(artifacts),
        "raw_artifact_identities": {str(path.relative_to(artifacts)): file_identity(path) for path in raw_paths},
        "reproduction": [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                         "--artifacts", "<fresh outside-Git directory>", "--receipt", "<new receipt path>"],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"verdict": "PASS", "passed": 64, "receipt": str(receipt_path),
                      "source_snapshot_sha256": digest(files)}), flush=True)


if __name__ == "__main__":
    main()
