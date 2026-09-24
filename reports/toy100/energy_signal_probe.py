"""Bounded energy-signal diagnostic and warm-state stability screen."""

import argparse
from contextlib import contextmanager, nullcontext
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.toy100.continuous_probe import DEFAULT_CONFIG, prepared_config, run_probe
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from benchmarks.transfer_suite.toy100_compatibility import run as run_transfer
from reports.toy100.energy_signal_scratch import energy_signal, warm_energy_signal


ROOT = Path(__file__).resolve().parents[2]
FRACTION_STEPS = frozenset((*range(1, 17), 25, 50, 75, 100, 150, 200,
                             400, 600, 800, 1000, 1200))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows):
    if not rows:
        return dict(updates=0)
    improvements = [row["accepted_by_signal"] for row in rows]
    kept = [row["proposal_kept"] for row in rows]
    deltas = [sum(after - before for before, after in zip(row["before"], row["after"]))
              for row in rows]
    return dict(updates=len(rows), improving_both=sum(improvements),
                improving_fraction=sum(improvements) / len(rows),
                proposals_kept=sum(kept),
                scale_counts={str(scale): sum(row["accepted_scale"] == scale
                                               for row in rows)
                              for scale in (0., .125, .25, .5, 1.)},
                mean_energy_delta=sum(deltas) / len(deltas),
                minimum_energy_delta=min(deltas), maximum_energy_delta=max(deltas))


@contextmanager
def _warm_variant(state, prefix, *, observe_only, backtracking=False):
    with constant_rate_context(state, lr=.00425) as rates, warm_energy_signal(
            state, observe_only=observe_only, backtracking=backtracking) as receipt:
        receipt["fixed_rate"] = rates
        yield receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cold_observer", "warm_observer",
                                          "warm_gate", "warm_backtrack",
                                          "cold_trajectory", "cold_trajectory_backtrack",
                                          "cold_gate", "cold_backtrack",
                                          "cold_fraction_diagnostic"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(DEFAULT_CONFIG.read_text())
    declaration = dict(mode=args.mode, config=config, seed=0,
                       shared_gate_eligible=False,
                       fractional_diagnostic_steps=sorted(FRACTION_STEPS)
                       if args.mode == "cold_fraction_diagnostic" else [],
                       source_sha256={name: _hash(ROOT / name) for name in (
                           "reports/toy100/energy_signal_scratch.py",
                           "reports/toy100/energy_signal_probe.py",
                           "benchmarks/toy100/continuous_probe.py",
                           "benchmarks/toy100/warm_equilibrium_probe.py",
                           "benchmarks/toy100/models.py")})
    (args.output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    (args.output / "adapter_at_execution.py").write_bytes(
        (ROOT / "reports/toy100/energy_signal_scratch.py").read_bytes())
    (args.output / "runner_at_execution.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(event="declared", mode=args.mode, output=str(args.output))), flush=True)
    if args.mode in ("cold_trajectory", "cold_trajectory_backtrack"):
        effective = prepared_config(config, "constant")
        config_file = args.output / "effective-config.json"
        config_file.write_text(json.dumps(effective, indent=2) + "\n")
        with energy_signal(backtracking=args.mode == "cold_trajectory_backtrack") as receipt:
            records = run_transfer(config_file, args.output / "suite", tasks=("trajectory",))
        row = records[0]
        summary = dict(scope="cold_trajectory_acquisition_gate",
                       verdict=row["verdict"], live=row.get("live"),
                       noise_applied=row["noise_applied"],
                       proposal_signal=_summary(receipt["rows"]))
        (args.output / "receipt.json.gz").write_bytes(gzip.compress(
            json.dumps(receipt, allow_nan=False).encode(), mtime=0))
    elif args.mode in ("cold_observer", "cold_gate", "cold_backtrack",
                       "cold_fraction_diagnostic"):
        observe_only = args.mode == "cold_observer"
        with energy_signal(observe_only=observe_only,
                           diagnostic_steps=FRACTION_STEPS if args.mode ==
                           "cold_fraction_diagnostic" else frozenset(),
                           backtracking=args.mode == "cold_backtrack") as receipt:
            evidence = run_probe(config, mode="constant", diagnostic_every=50)
        summary = dict(scope="cold_observer_only" if observe_only else
                       "cold_fraction_diagnostic" if args.mode ==
                       "cold_fraction_diagnostic" else "cold_mode_hold_acquisition_gate",
                       final=evidence["final"],
                       stationary=evidence["stationary"],
                       first_200=_summary(receipt["rows"][:200]),
                       last_200=_summary(receipt["rows"][-200:]),
                       all_updates=_summary(receipt["rows"]))
        if args.mode == "cold_fraction_diagnostic":
            selected = [row for row in receipt["rows"] if row["fractional_diagnostic"]]
            summary["fractional_diagnostic"] = dict(
                rejected_checked=len(selected),
                improvement_counts={str(scale): sum(
                    next(point for point in row["fractional_diagnostic"]
                         if point["scale"] == scale)["improves_both"]
                    for row in selected) for scale in (.5, .25, .125)},
                steps=[row["step"] for row in selected])
        (args.output / "cold.json.gz").write_bytes(gzip.compress(json.dumps(
            dict(evidence=evidence, receipt=receipt), allow_nan=False).encode(), mtime=0))
    else:
        observe_only = args.mode == "warm_observer"
        backtracking = args.mode == "warm_backtrack"
        variants = {
            "identity": lambda state, prefix: nullcontext(),
            "ordinary_observer" if observe_only else
            "energy_backtrack" if backtracking else "energy_gate":
                lambda state, prefix: _warm_variant(state, prefix,
                                                    observe_only=observe_only,
                                                    backtracking=backtracking),
        }
        result = run_warm_variants(config, variants, output_dir=args.output / "forks")
        key = "ordinary_observer" if observe_only else \
              "energy_backtrack" if backtracking else "energy_gate"
        branch = json.loads((args.output / "forks" / (key + ".json")).read_text())
        receipt = branch["dynamics_receipt"]
        summary = dict(scope="passing_state_local_stability_only", variant=key,
                       local_stability=branch["local_stability"],
                       final=branch["final"],
                       proposal_signal=_summary(receipt["rows"]),
                       identity_cold_parity=result["identity_cold_parity"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2,
                                                           allow_nan=False) + "\n")
    print(json.dumps(dict(event="result", **summary), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
