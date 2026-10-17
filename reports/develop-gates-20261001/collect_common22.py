"""Collect a compact, independently regraded legacy BCap common-22 receipt.

No training. Raw streams and clouds remain in the existing artifact root.
This record supplies no Atlas, E22 or clean-sampler qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy_suite import regrade
from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
from benchmarks.toy100.gate import evaluate_suite as coverage_suite


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing common-22 artifact root")
    parser.add_argument("--actions-run", type=int)
    parser.add_argument("--training-source-revision", help="commit whose executable bytes match the captured source map")
    args = parser.parse_args()
    output = args.output.resolve()
    aggregate = regrade(output)
    coverage = coverage_suite(output / "toy100", write=False)
    accuracy = accuracy_suite(output / "toy100", write=False)
    natives = []
    for name, row in aggregate["toy100"]["cases"].items():
        directory = output / "toy100" / name
        summary = read(directory / "summary.json")
        provenance = read(directory / "provenance.json")
        samples = sorted((directory / "quality_checks").glob("*.npz"))
        samples += [directory / "final_samples.npz", directory / "holdout_samples.npz"]
        natives.append(dict(task=name, status=row["status"],
            completed_steps=summary["completed_steps"], eval_output_noise=summary["eval_output_noise"],
            terminal_passes=sum(point["passed"] for point in accuracy["problems"][name]["terminal_checks"]),
            terminal_steps=[point["step"] for point in accuracy["problems"][name]["terminal_checks"]],
            holdout=row["holdout"], coverage_stable_checks=coverage["problems"][name]["stable_checks"],
            captured_git_head=provenance["git_sha"],
            source_archive_sha256=provenance["source_archive_sha256"],
            sample_sha256={str(path.relative_to(directory)): sha(path) for path in samples},
            summary_sha256=sha(directory / "summary.json"), config_sha256=sha(directory / "config.json")))
    canonical = []
    for name, row in aggregate["candidate19"]["cases"].items():
        path = Path(row["artifact"])
        canonical.append(dict(task=name, status=row["status"], observations=row["observations"],
            passing_suffix=row["passing_suffix"], noise_applied=row["noise_applied"],
            eval_scope=row["eval_scope"], final=row["final"],
            artifact=str(path.relative_to(output)), artifact_sha256=sha(path)))
    protocol = read(output / "candidate19/protocol.json")
    manifest = read(output / "toy100/run_manifest.json")
    if args.training_source_revision:
        for name, expected in protocol["source_sha256"].items():
            data = subprocess.check_output(["git", "show", args.training_source_revision + ":" + name], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == expected, name
    receipt = dict(schema_version=1, status=aggregate["status"], required=22,
        passes=aggregate["observed_passes"], native=natives, canonical=canonical,
        candidate="legacy BCap constraints_simple_regularization", artifact_root=str(output),
        actions_run=args.actions_run, config_sha256=manifest["config_sha256"],
        sampling_law="original scheduled output-noise serving", sampling_law_identical=aggregate["sampling_law_identical"],
        global_recipe_identical=aggregate["global_recipe_identical"],
        full_public_source_coverage=aggregate["full_public_source_coverage"],
        run_manifest_sha256=sha(output / "toy100/run_manifest.json"),
        transfer_protocol_sha256=sha(output / "candidate19/protocol.json"),
        training_source_sha256=protocol["source_sha256"],
        verified_training_source_revision=args.training_source_revision,
        collector_source_sha256=sha(Path(__file__)),
        grader_source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        grader_source_sha256={name: sha(ROOT / name) for name in (
            "benchmarks/toy_suite.py", "benchmarks/toy100/gate.py", "benchmarks/toy100/accuracy_gate.py")},
        thresholds_changed=False, budgets_changed=False, seeds_changed=False,
        atlas_qualification=False, e22_qualification=False, public_clean_qualification=False)
    path = output / "compact-common22.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=receipt["status"], passes=receipt["passes"], required=22, receipt=str(path))))


if __name__ == "__main__":
    main()
