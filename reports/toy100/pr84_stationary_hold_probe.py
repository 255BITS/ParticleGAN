"""Same-dataset long warm continuation; diagnostic, not cold qualification."""

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from reports.toy100.pr84_smoothed_candidate import METHOD, pr84_smoothed_candidate


def untimed(value):
    if isinstance(value, dict):
        return {key: untimed(item) for key, item in value.items() if "seconds" not in key}
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def validate(output):
    declaration = json.loads((output / "declaration.json").read_text())
    for name, digest in declaration["source"].items():
        if hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"archived source changed: {name}")
        if name != "reports/toy100/pr84_stationary_hold_probe.py":
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"candidate or host source changed: {name}")
    result = json.loads((output / "forks/summary.json").read_text())
    reference = ROOT / "reports/toy100/continuous-evidence/pr84-independent-audit/smooth40-warm/alternating_curvature.json.gz"
    if hashlib.sha256(reference.read_bytes()).hexdigest() != declaration["reference_sha256"]:
        raise RuntimeError("original reference changed")
    expected = json.loads(gzip.decompress(reference.read_bytes()))
    observed = json.loads((output / "forks/original.json").read_text())
    if observed["warm_state_sha256"] != expected["warm_state_sha256"]:
        raise RuntimeError("extended run changed the archived passing state")
    old_diagnostic = {row["step"]: row for row in expected["diagnostic"]}
    new_diagnostic = {row["step"]: row for row in observed["diagnostic"]}
    if any(untimed(new_diagnostic[step]) != untimed(row) for step, row in old_diagnostic.items()):
        raise RuntimeError("extended run changed an archived first-1200 diagnostic")
    # The extracted adapter removes inactive oracle diagnostics from original
    # PR84 records. Compare all fields against its already-audited clean run.
    clean_reference = ROOT / "reports/toy100/continuous-evidence/coverage-projection-round4/warm/forks/original.json.gz"
    clean = json.loads(gzip.decompress(clean_reference.read_bytes()))
    for key in ("warm_state_sha256", "final_state_sha256", "diagnostic"):
        if untimed(clean[key]) != untimed(expected[key]):
            raise RuntimeError(f"clean reference differs from original PR84: {key}")
    old_records = clean["dynamics_receipt"]["records"]
    new_records = observed["dynamics_receipt"]["records"]
    if len(old_records) != 200 or untimed(new_records[:200]) != untimed(old_records):
        raise RuntimeError("extended run changed the first 200 candidate update records")
    result.update(declaration=declaration, original_warm_state_exact=True,
                  original_first1200_diagnostics_exact=True,
                  clean_first200_update_records_exact=True,
                  clean_reference_sha256=hashlib.sha256(clean_reference.read_bytes()).hexdigest(),
                  validation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **result)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    if args.validate_existing:
        validate(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sources = (
        "reports/toy100/pr84_stationary_hold_probe.py",
        "reports/toy100/pr84_smoothed_candidate.py",
        "reports/toy100/alternating_curvature_scratch.py",
        "reports/toy100/extra_adam_scratch.py",
        "benchmarks/toy100/warm_equilibrium_probe.py",
        "benchmarks/toy100/continuous_probe.py",
        "benchmarks/locked_shared/mode_hold.py",
        "configs/toy100/constraints_simple_regularization.json",
    )
    hashes = {}
    for name in sources:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        path = args.output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    reference = ROOT / "reports/toy100/continuous-evidence/pr84-independent-audit/smooth40-warm/alternating_curvature.json.gz"
    declaration = dict(
        method=METHOD, scope="same_dataset_passing_state_continuation_only",
        shared_gate_eligible=False, cold_acquisition_still_failed=True,
        total_steps=2400, scheduled_prefix_steps=1000, noise_horizon=1200,
        variants=["identity", "constant", "original"],
        dense_checks=[1001, 1200], later_check_every=10, seed=0,
        nominal_rates=dict(g=.00425, d=.00425, prior=.0085),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), source=hashes)
    (args.output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    print(json.dumps(dict(event="DECLARED", **declaration)), flush=True)

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == "original"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False,
                       scratch_optimizer_policy=METHOD if recorder.enabled else "control")
        print(json.dumps(dict(event="VARIANT_START", variant=method, completed=completed)), flush=True)
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", variant=method)), flush=True)

    factories = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                 for name in declaration["variants"]}
    config = json.loads((ROOT / sources[-1]).read_text())
    run_warm_variants(config, factories, output_dir=args.output / "forks",
        steps=2400, prefix_context=lambda: pr84_smoothed_candidate(start_step=1000))
    validate(args.output)


if __name__ == "__main__":
    main()
