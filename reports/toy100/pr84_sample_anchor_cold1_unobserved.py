"""One native cold update with the sample-anchor factory and no saved-state observer.

The only stop hook fires after the candidate's first complete three-phase
update, before host EMA/checkpoint code. This is a factory integration smoke,
not a completed host episode or shared gate. The PR84 adapter still uses its
own required internal ``locals()`` phase binding.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class StopAfterOneNativeUpdate(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--saved-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    from reports.toy100.pr84_critic_refinement_capture import snapshot, _sha
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.sample_anchor_candidate import sample_anchor_candidate, METHOD

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = json.loads((args.saved_filter / "declaration.json").read_text())
    if declaration["method"] != METHOD:
        raise RuntimeError("source-bound sample-anchor method differs")
    for name, expected in declaration["sources"].items():
        raw = (root / name).read_bytes()
        if sha(raw) != expected:
            raise RuntimeError(f"source changed: {name}")
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    config = json.loads((root / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name=METHOD, lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    spec = next(item["spec"] for item in plan() if item["spec"]["name"] == "mode_hold")
    if spec["steps"] != 1200:
        raise RuntimeError("native mode_hold budget changed")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, states = {}, {}
    factories = (
        ("original", lambda: pr84_smoothed_candidate(task="mode_hold")),
        ("disabled", lambda: sample_anchor_candidate(task="mode_hold", correction=False)),
        ("active", lambda: sample_anchor_candidate(task="mode_hold", correction=True)),
    )
    for name, factory in factories:
        ordinary = torch.optim.Adam.step
        calls = {}

        def audited_step(optimizer, closure=None):
            calls.setdefault(optimizer, []).append(tuple(group["lr"]
                                                         for group in optimizer.param_groups))
            return ordinary(optimizer, closure=closure)

        with patch.object(torch.optim.Adam, "step", audited_step), factory() as (recorder, generated):
            if name == "original":
                (args.output / "generated-mode_hold.py").write_text(generated)
            elif generated != (args.output / "generated-mode_hold.py").read_text():
                raise RuntimeError("factory changed generated native host")

            def stop_after_one(d_calls, outer):
                if outer != 1:
                    raise RuntimeError("unexpected native update count")
                raise StopAfterOneNativeUpdate()

            recorder.accounting = stop_after_one
            try:
                run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
            except StopAfterOneNativeUpdate:
                pass
            else:
                raise RuntimeError("single-update stop callback did not fire")
            if recorder.outer_steps != 1 or len(recorder.records) != 1:
                raise RuntimeError("factory did not complete exactly one native update")
            if recorder._local is None or recorder._local["noise_policy"]._step_calls != 1:
                raise RuntimeError("native one-update noise clock differs")
            applied = {}
            for role, optimizer, rates in zip(("d", "g_prior"), recorder.optimizers,
                                               ((.00425,), (.00425, .0085))):
                observed = calls[optimizer]
                moments = {int(optimizer.state[p]["step"])
                           for group in optimizer.param_groups for p in group["params"]}
                if observed != [rates] or moments != {1}:
                    raise RuntimeError(f"{name}: actual Adam rate/call/moment differs for {role}")
                applied[role] = dict(rate=observed[0], calls=len(observed), moments=1)
            state = snapshot(recorder._local)
            states[name] = state
            value = recorder.receipt()
            corrections = value.get("corrections", [])
            if name == "active":
                if (len(corrections) != 1 or value["correction_owner_checks"] != 1
                        or value["correction_rng_checks"] != 1
                        or value["native_batch_checks"] != 3
                        or len(corrections[0]["centers"]) != 8):
                    raise RuntimeError("active factory failed one-update sampler/owner checks")
            elif corrections:
                raise RuntimeError("disabled/original factory unexpectedly corrected")
            rows[name] = dict(state_sha256=_sha(state), applied=applied,
                              source_sha256=sha(generated.encode()),
                              method=value["method"],
                              phase_replays=value["rng_replay_verified"],
                              selected=corrections[0]["selected"] if corrections else None,
                              group_count=len(corrections[0]["centers"]) if corrections else None,
                              fit_status=corrections[0]["fit"]["status"] if corrections else None,
                              anchor_cost_before=corrections[0]["pre_cost"] if corrections else None,
                              anchor_cost_final=corrections[0]["final_cost"] if corrections else None)
            print(json.dumps(dict(event="COLD_ONE_UPDATE", arm=name,
                                  state_sha256=rows[name]["state_sha256"],
                                  selected=rows[name]["selected"])), flush=True)
    if _sha(states["original"]) != _sha(states["disabled"]):
        raise RuntimeError("disabled factory differs from ordinary PR84 full state")
    for key in ("critic", "optimizer_d", "optimizer_g", "rng", "noise", "ema_g", "ema_z"):
        if _sha(states["original"][key]) != _sha(states["active"][key]):
            raise RuntimeError(f"active correction changed non-owner {key}")
    buffer = io.BytesIO()
    torch.save(states, buffer)
    raw = buffer.getvalue()
    (args.output / "states.pt.gz").write_bytes(gzip.compress(raw, mtime=0))
    result = dict(scope="native cold mode_hold first complete update only; stopped before EMA/checkpoint",
                  no_saved_state_continuation_observer=True,
                  internal_pr84_locals_phase_binding=True,
                  native_recipe_steps=1200, native_noise_horizon=1200,
                  original_disabled_full_state_parity=True,
                  active_nonowner_state_parity=True,
                  states_raw_sha256=sha(raw),
                  source=declaration["sources"],
                  generated_host_sha256=sha((args.output / "generated-mode_hold.py").read_bytes()),
                  arms=rows)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="COLD_FACTORY_SMOKE_DONE", parity=True,
                          active=rows["active"])), flush=True)


if __name__ == "__main__":
    main()
