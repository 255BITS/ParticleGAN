"""Warm continuation of the exit clip against frozen PR84 on one prefix.

Gate order is the dense 200 checks, then every-10 checks through the requested
horizon. Identity is the scheduled host. ``original`` is PR84 with the clip
off. ``exitclip`` is the same stencil with the particle-local clip on.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from reports.toy100.exit_aware_step_clip import METHOD, exit_aware_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = (
        "reports/toy100/exit_aware_hold_probe.py",
        "reports/toy100/exit_aware_step_clip.py",
        "reports/toy100/pr84_smoothed_candidate.py",
        "reports/toy100/alternating_curvature_scratch.py",
        "reports/toy100/extra_adam_scratch.py",
        "benchmarks/toy100/warm_equilibrium_probe.py",
        "benchmarks/toy100/continuous_probe.py",
        "benchmarks/locked_shared/mode_hold.py",
        "configs/toy100/constraints_simple_regularization.json",
    )
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    declaration = dict(
        method=METHOD, steps=args.steps, dense_checks=[1001, 1200],
        later_check_every=10, variants=["identity", "original", "exitclip"],
        margin_proxy="nearest real in the current minibatch",
        support_fence="median NN + 3 * 1.4826 * MAD",
        source=hashes,
    )
    (args.output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    print(json.dumps(dict(event="DECLARED", steps=args.steps, method=METHOD)), flush=True)

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method in ("original", "exitclip")
        recorder.clip_enabled = method == "exitclip"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False,
                       scratch_optimizer_policy=METHOD if recorder.clip_enabled else "pr84")
        print(json.dumps(dict(event="VARIANT_START", variant=method, completed=completed,
                              clip=recorder.clip_enabled)), flush=True)
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", variant=method,
                              clips=getattr(recorder, "clip_count", 0))), flush=True)

    factories = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                 for name in declaration["variants"]}
    config = json.loads((ROOT / sources[-1]).read_text())
    summary = run_warm_variants(
        config, factories, output_dir=args.output / "forks", steps=args.steps,
        prefix_context=lambda: exit_aware_candidate(start_step=1000))
    kept = {
        name: dict(status=row["status"], local=row["local_stability"], hold=row["long_hold"])
        for name, row in summary["variants"].items()
    }
    print(json.dumps(dict(event="DONE", **kept)), flush=True)


if __name__ == "__main__":
    main()
