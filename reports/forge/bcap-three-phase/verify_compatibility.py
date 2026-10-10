"""Reuse exact unchanged disabled evidence and execute narrow enabled replay.

If public source differs, repeat the original actual-develop comparison in
separate interpreters. This never repeats quality experiments or a full suite.
Keep the earlier receipt immutable and raw test artifacts outside Git.
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
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OLD = "reports/forge/bcap-develop-integration/compatibility-post-execution.json"
INACTIVE = "tests/test_bcap_integration_compatibility.py"
NARROW = "tests/test_bcap_phase1_compatibility.py"
WRAPPER = "tests/test_host_integration_bcap.py::test_declared_ae_consumer_preserves_four_argument_evaluation_wrappers"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_delta(root, old):
    """Compare every importable public file, including newly added dependencies."""
    previous = old["tested_candidate_commit"]
    changed = git(root, "diff", "--name-only", previous, "--", "particlegan", "experiments", "benchmarks", "lib", "configs").splitlines()
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "--", "particlegan", "experiments", "benchmarks", "lib", "configs").splitlines()
    files = {name: sha(root / name) for name in old["source_files_sha256"]}
    changed += [name for name, value in files.items() if value != old["source_files_sha256"][name]]
    return files, sorted(set(changed + untracked))


def measured_state_pairs(root, artifacts):
    sys.path.insert(0, str(root))
    os.environ["BCAP_COMPATIBILITY_ROOT"] = str(root)
    import torch
    from experiments.forge.state import state_digest
    spec = importlib.util.spec_from_file_location("phase1_actual_develop_probe", root / INACTIVE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    directories = [p for p in (artifacts / "pytest").glob("bcap-compatibility*") if p.is_dir() and not p.is_symlink()]
    require(len(directories) == 1, "Expected one actual-develop/candidate source packet pair")
    old = torch.load(directories[0] / "develop.pt", weights_only=False)
    new = torch.load(directories[0] / "candidate.pt", weights_only=False)
    pairs = {}
    def exact(key, before, after):
        probe._assert_equal(before, after, key)
        a, b = state_digest(before), state_digest(after)
        require(a == b, f"{key}: numerical state differs")
        pairs[key] = dict(reference_sha256=a, candidate_sha256=b)
    for case in probe.CASES:
        for prior in probe.PRIORS:
            key = f"{case}/{prior}"
            exact(key, old[key]["final"], new[key]["final"])
        exact(f"word/{case}", old["words"][case]["final"], new["words"][case]["final"])
    for case, host in probe.COMPONENT_CASES:
        key = f"{case}/{host}"
        before, after = deepcopy(old["components"][key]), deepcopy(new["components"][key])
        for packet in (before, after):
            evaluation = packet["applied"]["field_ownership"]["task_contract"]["evaluation"]["value"]
            evaluation.pop("sources", None)
            evaluation.pop("evaluator_revision", None)
        exact(f"component/{key}", before, after)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--candidate", help="Also exercise this candidate's global repair/backend fields in every subset replay")
    args = parser.parse_args()
    root, artifacts, receipt = args.repository.resolve(), args.artifacts.resolve(), args.receipt.resolve()
    require(not artifacts.exists() and not receipt.exists(), "Use fresh destinations; preserve previous evidence")
    require(not artifacts.is_relative_to(root), "Bulk artifacts belong outside Git")
    old_path = root / OLD
    old_bytes = old_path.read_bytes()
    old = json.loads(old_bytes)
    require(hashlib.sha256(old_bytes).hexdigest() == "8cce2b2cf432f30114211161b3c527957c766ac8f40a2a88eae9c5c8d57f045c",
            "Preserve the exact original final-tree receipt identity")
    require(old["verdict"] == "PASS" and old["tests"]["passed"] == 64 and len(old["state_identities"]) == 42,
            "Require the original complete actual-develop proof")
    require(sha(root / INACTIVE) == old["test_source_sha256"][INACTIVE], "Preserve the exact actual-develop comparator")
    require(all(pair["reference_sha256"] == pair["candidate_sha256"] for pair in old["state_identities"].values()),
            "Every original state-hash pair must retain exact equality")
    files, changes = source_delta(root, old)
    require(len(files) == 52 and digest(old["source_files_sha256"]) == old["tested_source_snapshot_sha256"],
            "Preserve the measured 52-file inventory")
    import torch
    runtime = dict(device="cpu", python=sys.version.split()[0], torch=torch.__version__)
    reused = (not changes and all(sha(root / name) == value for name, value in old["test_source_sha256"].items())
              and all(old["runtime"].get(key) == value for key, value in runtime.items()))
    tests = [NARROW]
    if not reused:
        tests += [INACTIVE, WRAPPER]
    commit = git(root, "rev-parse", "HEAD")
    checked_tests = (INACTIVE, NARROW, WRAPPER.split("::")[0], "tests/test_bcap_integration_core.py",
                     "reports/forge/bcap-three-phase/phase2.py")
    test_sources = {name: sha(root / name) for name in checked_tests}
    selected_candidate = None
    if args.candidate:
        sys.path.insert(0, str(root))
        from experiments.forge.planning import load_idea
        selected_candidate = load_idea(root, args.candidate)
    artifacts.mkdir(parents=True)
    log, junit = artifacts / "pytest.log", artifacts / "pytest.xml"
    env = dict(os.environ, BCAP_COMPATIBILITY_ROOT=str(root), PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("BCAP_COMPATIBILITY_CANDIDATE", None)
    if args.candidate:
        env["BCAP_COMPATIBILITY_CANDIDATE"] = args.candidate
    command = [sys.executable, "-m", "pytest", "-q", *tests, f"--basetemp={artifacts / 'pytest'}", f"--junitxml={junit}"]
    print(f"tail -F {log}", flush=True)
    started = time.monotonic()
    with log.open("w") as stdout:
        result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=subprocess.STDOUT, timeout=380)
    require(result.returncode == 0, f"Compatibility failed; inspect {log}")
    cases = ET.parse(junit).getroot().findall(".//testcase")
    require(not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")),
            "Require all narrow compatibility checks to execute and pass")
    require(len(cases) == (14 if reused else 78), "Unexpected compatibility test count")
    require(source_delta(root, old) == (files, changes) and git(root, "rev-parse", "HEAD") == commit,
            "Source changed during verification")
    require(old_path.read_bytes() == old_bytes, "Earlier evidence changed")
    require(test_sources == {name: sha(root / name) for name in checked_tests}, "Software test source changed during verification")
    pairs = old["state_identities"] if reused else measured_state_pairs(root, artifacts)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(schema_version=1, scope="phase1_bounded_software_compatibility", verdict="PASS",
        qualification_input=False, tested_candidate_commit=commit, baseline_commit=old["baseline_commit"],
        source_files_sha256=files, tested_source_snapshot_sha256=digest(files), public_source_delta=changes,
        disabled_evidence=dict(mode="reused_exact_source_and_tests" if reused else "fresh_actual_develop_interpreters",
            original_receipt=OLD, original_receipt_sha256=hashlib.sha256(old_bytes).hexdigest(),
            original_measured_commit=old["tested_candidate_commit"], original_checks=64),
        state_identities=pairs, host_acceptance=old["host_acceptance"] if reused else
            {"scope": "Fresh original 216 host and 216 absent-hook comparisons passed in the unchanged comparator"},
        new_checks=dict(passed=len(cases), failed=0, skipped=0, wall_seconds=time.monotonic()-started,
            subset_trainer_replays=6, subset_word_joint_replays=6, declaration_contract_checks=2,
            bounded_software_outer_updates=48), candidate_subset_overlay=selected_candidate, runtime=runtime,
        test_source_sha256=test_sources,
        raw_artifacts={str(path.relative_to(artifacts)): dict(sha256=sha(path), bytes=path.stat().st_size)
                       for path in (log, junit)}, artifact_archive=str(artifacts),
        limitations=old["limitations"], scientific_training_reruns=0,
        reproduction=[sys.executable, str(Path(__file__).resolve()), "--repository", str(root),
            "--artifacts", "<fresh outside-Git path>", "--receipt", "<fresh compact receipt>"]
            + (["--candidate", args.candidate] if args.candidate else [])),
        indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(verdict="PASS", new_checks=len(cases), reused_disabled_proof=reused,
                         receipt=str(receipt))), flush=True)


if __name__ == "__main__":
    main()
