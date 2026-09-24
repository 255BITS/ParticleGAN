"""Replay frozen PR84 around the first delayed stability failure.

Read-only on the method: same stencil, bounds, and rates as the stationary
hold. Records clean-particle motion for the update that ends at each step in
the two failure windows. No new controller.
"""

from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared.mode_hold import SIGMA, diversity, ring_means
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate

WINDOWS = ((1116, 1140), (1371, 1410), (1521, 1570))
MEANS = ring_means()
RADIUS = 3.0 * SIGMA


def _watch(step):
    return any(lo <= step <= hi for lo, hi in WINDOWS)


def _cloud(generator, prior):
    clean = getattr(generator, "model", generator)
    return clean(prior.z).detach()


def _row(before, proposal, after, record):
    move = (after - before).norm(dim=-1)
    proposed = (proposal - before).norm(dim=-1)
    nearest = torch.cdist(before, MEANS).min(dim=1).values
    after_near = torch.cdist(after, MEANS).min(dim=1).values
    entered = int(((nearest > RADIUS) & (after_near <= RADIUS)).sum())
    left = int(((nearest <= RADIUS) & (after_near > RADIUS)).sum())
    return dict(
        step=1000 + record["outer_step"],
        before=diversity(before, MEANS),
        after=diversity(after, MEANS),
        proposal=diversity(proposal, MEANS),
        applied_rms=float(move.square().mean().sqrt()),
        applied_max=float(move.max()),
        proposal_rms=float(proposed.square().mean().sqrt()),
        proposal_max=float(proposed.max()),
        left_hq=left,
        entered_hq=entered,
        margin_before=float((RADIUS - nearest).min()),
        rho=record["g"]["rho"],
        factor=record["g"]["factor"],
        sharpness=record.get("critic_sharpness"),
        width=record.get("critic_width"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", type=int, default=1570)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    captured = []

    @contextmanager
    def audited(start_step=1000):
        with pr84_smoothed_candidate(start_step=start_step) as (recorder, source):
            delegate = recorder.step
            pending = {}

            def step(optimizer, ordinary_step, closure=None):
                local = recorder._local or {}
                gen, prior = local.get("generator"), local.get("prior")
                phase, enabled = recorder.phase, recorder.enabled and not recorder.passthrough
                opt_d, opt_g = recorder.optimizers or (None, None)
                checkpoint = 1000 + recorder.row.get("outer_step", 0) if enabled else None
                if enabled and _watch(checkpoint) and gen is not None and phase == 0 and optimizer is opt_d:
                    pending["before"] = _cloud(gen, prior)
                if enabled and _watch(checkpoint) and gen is not None and phase == 2 and optimizer is opt_g:
                    pending["proposal"] = _cloud(gen, prior)
                result = delegate(optimizer, ordinary_step, closure)
                if enabled and _watch(checkpoint) and phase == 2 and optimizer is opt_g and "before" in pending:
                    after = _cloud(gen, prior)
                    row = _row(pending.pop("before"), pending.pop("proposal"), after, recorder.row)
                    captured.append(row)
                    print(json.dumps(dict(event="MOTION", step=row["step"], before=row["before"]["hq"],
                                          after=row["after"]["hq"], modes=row["after"]["modes"],
                                          applied_max=row["applied_max"], factor=row["factor"],
                                          left=row["left_hq"])), flush=True)
                return result

            recorder.step = step
            yield recorder, source

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == "original"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False)
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
                (args.output / "motion.json").write_text(json.dumps(captured, indent=2) + "\n")

    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    factories = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                 for name in ("identity", "original")}
    result = run_warm_variants(config, factories, output_dir=args.output / "forks", steps=args.until,
                               prefix_context=lambda: audited())
    (args.output / "summary.json").write_text(json.dumps(dict(
        fork=result["variants"]["original"], n_motion=len(captured)), indent=2) + "\n")
    print(json.dumps(dict(event="DONE", n=len(captured), status=result["variants"]["original"]["status"])), flush=True)


if __name__ == "__main__":
    main()
