"""Same passing-state continuation to update 2400 for the value-transport rule.

Noise horizon stays 1200. The scheduled prefix is not restarted. This is the
delayed-stability filter, not a production gate.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from reports.toy100.value_transport_candidate import METHOD, value_transport_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method in ("original", "value")
        recorder.correction = method == "value"
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
                fired = sum(bool(row.get("fired")) for row in recorder.transport_records)
                accepted = sum(bool(row.get("accepted")) for row in recorder.transport_records)
                receipt.update(recorder.receipt())
                receipt["transport_fired"] = fired
                receipt["transport_accepted"] = accepted
                # Full per-step rows stay in the fork file; the summary stays small.
                receipt.pop("transport_records", None)
        print(json.dumps(dict(event="VARIANT_DONE", variant=method,
                              transport_fired=receipt.get("transport_fired"),
                              transport_accepted=receipt.get("transport_accepted"))), flush=True)

    variants = ("identity", "constant", "original", "value")
    factories = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                 for name in variants}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    print(json.dumps(dict(event="DECLARED", method=METHOD, steps=2400, noise_horizon=1200)), flush=True)
    result = run_warm_variants(
        config, factories, output_dir=args.output / "forks", steps=2400,
        prefix_context=lambda: value_transport_candidate(start_step=1000))
    compact = dict(method=METHOD, torch=__import__("torch").__version__,
                   shared_gate_eligible=False, steps=2400,
                   identity_cold_parity=result["identity_cold_parity"],
                   warm_state_sha256=result["warm_state_sha256"],
                   variants={name: dict(status=row["status"],
                                        local_stability=row["local_stability"],
                                        long_hold=row.get("long_hold"),
                                        final=row["final"])
                             for name, row in result["variants"].items()})
    (args.output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(dict(event="HOLD_DONE", **compact)), flush=True)


if __name__ == "__main__":
    main()
