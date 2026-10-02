"""Cold trajectory, cold ring, then own-state hold for the fence-restore clip.

Constant rates, clip on from update 0. Stops at the first failed host.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.exit_aware_step_clip import exit_aware_candidate


def cold_hosts(output: Path):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="fence_restore_clump_radius", lr_floor=1.0, lr_anneal_start=0.0)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    stages = []
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        with exit_aware_candidate(task=task, start_step=0) as (recorder, _):
            recorder.clip_enabled = True
            result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        verdict = test_verdict(spec, result)
        row = dict(task=task, verdict=verdict, live=result["live"], seconds=result["seconds"],
                   clips=recorder.clip_count)
        stages.append(row)
        print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
        (output / f"{task}.json").write_text(json.dumps(
            dict(verdict=verdict, live=result["live"], dynamics=recorder.receipt()),
            allow_nan=False) + "\n")
        if not verdict["passed"]:
            break
    (output / "cold.json").write_text(json.dumps(dict(stages=stages), indent=2) + "\n")
    return stages


def own_state_hold(output: Path):
    import torch
    from benchmarks.toy100.continuous_probe import run_probe

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    with exit_aware_candidate(start_step=0) as (recorder, _):
        recorder.clip_enabled = True

        def hook(state):
            recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
                calls=calls + state["target_steps"] - outer, moment_updates=state["target_steps"])

        result = run_probe(
            config, mode="constant", steps=2400, diagnostic_every=10,
            checkpoint_hook_step=1, checkpoint_hook=hook,
            log=lambda event: print(json.dumps(event), flush=True) if event.get("step", 0) % 200 == 0 else None,
        )
    kept = dict(status=result["status"], stationary=result["stationary"],
                continued_hold=result["continued_hold"], clips=recorder.clip_count)
    (output / "hold.json").write_text(json.dumps(kept, indent=2) + "\n")
    print(json.dumps(dict(event="HOLD_DONE", **kept)), flush=True)
    return kept


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    stages = cold_hosts(output)
    if len(stages) < 2 or any(not row["verdict"]["passed"] for row in stages):
        print(json.dumps(dict(event="STOP", reason="cold host failed")), flush=True)
        return
    own_state_hold(output)


if __name__ == "__main__":
    main()
