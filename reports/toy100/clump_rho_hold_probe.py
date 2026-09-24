"""Fail-fast warm 200 then hold through 2400 for the low-rho clump ball.

Scheduled prefix through update 1000, then constant rates and the exit clip.
The clump HQ ball is on only when generator rho <= 0.25. Stops at the first
failed warm or hold check.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.exit_aware_step_clip import CLUMP_RHO_MAX, exit_aware_candidate


class HoldRegressed(RuntimeError):
    def __init__(self, point):
        super().__init__(f"hold regressed at {point['step']}")
        self.point = point


def main():
    import torch
    from benchmarks.locked_shared.mode_hold import N_MODES, PASS_HQ
    from benchmarks.toy100.continuous_probe import run_probe
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context

    torch.set_num_threads(1)
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    failures = []
    rates = {}

    def passes(point):
        return point["modes"] == N_MODES and point["hq"] >= PASS_HQ

    with exit_aware_candidate(start_step=1000) as (recorder, _):
        held = {}

        def hook(state):
            recorder.clip_enabled = True
            completed, target = state["completed_steps"], state["target_steps"]
            recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
                calls=completed + calls + target - completed - outer,
                moment_updates=target)
            held["ctx"] = constant_rate_context(state)
            rates.update(held["ctx"].__enter__())
            print(json.dumps(dict(event="RATES", **rates, clump_rho_max=CLUMP_RHO_MAX)), flush=True)

        def log(event):
            if event.get("event") != "checkpoint":
                return
            step = int(event["step"])
            if step < 1000:
                return
            row = dict(event="CHECKPOINT", step=step, modes=event["modes"],
                       hq=round(float(event["hq"]), 4), clips=recorder.clip_count)
            print(json.dumps(row), flush=True)
            if step <= 1000 or passes(event):
                return
            failures.append(dict(step=step, modes=event["modes"], hq=float(event["hq"])))
            raise HoldRegressed(event)

        try:
            result = run_probe(
                config, mode="scheduled", steps=2400, diagnostic_every=10,
                dense_after=1000, dense_until=1200,
                checkpoint_hook_step=1000, checkpoint_hook=hook, log=log,
            )
        except HoldRegressed as error:
            point = error.point
            kept = dict(status="STOP", reason="hold regressed",
                        step=point["step"], modes=point["modes"], hq=point["hq"],
                        clips=recorder.clip_count, rates=rates,
                        clump_rho_max=CLUMP_RHO_MAX)
            (output / "stop.json").write_text(json.dumps(kept, indent=2) + "\n")
            print(json.dumps(dict(event="STOP", **kept)), flush=True)
            return
        finally:
            if "ctx" in held:
                held["ctx"].__exit__(None, None, None)

    warm = [p for p in result["diagnostic"] if 1000 < p["step"] <= 1200]
    hold = [p for p in result["diagnostic"] if 1200 < p["step"] <= 2400]
    kept = dict(
        status=result["status"],
        warm_checks=len(warm),
        warm_pass=sum(passes(p) for p in warm),
        warm_min_hq=min(p["hq"] for p in warm),
        hold_checks=len(hold),
        hold_pass=sum(passes(p) for p in hold),
        hold_min_hq=min((p["hq"] for p in hold), default=None),
        hold_min_modes=min((p["modes"] for p in hold), default=None),
        failures=failures,
        clips=recorder.clip_count,
        rates=rates,
        clump_rho_max=CLUMP_RHO_MAX,
    )
    (output / "hold.json").write_text(json.dumps(kept, indent=2) + "\n")
    print(json.dumps(dict(event="HOLD_DONE", **{k: v for k, v in kept.items() if k != "failures"})), flush=True)


if __name__ == "__main__":
    main()
