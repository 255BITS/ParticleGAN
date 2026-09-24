"""One predeclared fresh-data D:G allocation arm on the frozen ring host.

Warm children share the scheduled state at update 1000. A separate cold run
starts the same allocation from initialization at fixed nominal rates. Warm
failure does not imply cold impossibility; its cold run is then explicitly a
new-attractor diagnostic, not promotion evidence.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100 import schedule as schedule_module
from benchmarks.toy100.continuous_probe import _window, prepared_config
from benchmarks.toy100.warm_equilibrium_probe import training_state_sha256
from benchmarks.transfer_suite import compare_defaults
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from reports.toy100.adaptive_d_allocation_scratch import adaptive_d_allocation


class _ParentForked(Exception):
    pass


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               allow_nan=False) + "\n")


def _write_gzip(path: Path, value) -> None:
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True,
                                              allow_nan=False).encode(), mtime=0))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host_state(frame) -> dict:
    if frame is None or frame.f_code.co_name != "train_mode_hold":
        raise RuntimeError("checkpoint did not come from frozen mode-hold")
    local = frame.f_locals
    names = ("generator", "critic", "prior", "opt_g", "opt_d",
             "ema_g", "ema_z", "stream", "noise_policy")
    if any(name not in local for name in names):
        raise RuntimeError("required host state disappeared")
    return {name: local[name] for name in names}


def _moment_steps(state: dict) -> dict:
    answer = {}
    for role, opt in (("g", state["opt_g"]), ("d", state["opt_d"])):
        counts = [int(row["step"].item() if isinstance(row["step"], torch.Tensor)
                      else row["step"])
                  for row in opt.state.values() if "step" in row]
        if not counts or min(counts) != max(counts):
            raise RuntimeError(f"{role} Adam moment counters disagree")
        answer[role] = counts[0]
    return answer


def _rates_after(receipt: dict, first_outer_step: int) -> dict:
    rates = {}
    for row in receipt["rate_records"]:
        if row["outer_step"] <= first_outer_step:
            continue
        rates.setdefault(row["role"], []).append(row["lr"])
    return {role: dict(min=min(values), max=max(values), calls=len(values))
            for role, values in rates.items()}


def _validate_rates(receipt: dict, config: dict, first_outer_step: int) -> dict:
    expected = {"g": config["lr"],
                "d": config["lr"] * config["d_lr_mult"],
                "prior": config["lr"] * config["prior_lr_mult"]}
    observed = _rates_after(receipt, first_outer_step)
    if set(observed) != set(expected):
        raise RuntimeError("a live G, D, or prior rate is missing")
    for role, value in expected.items():
        if observed[role]["min"] != value or observed[role]["max"] != value:
            raise RuntimeError(f"{role} applied Adam rate is not constant")
    return observed


def _run_episode(config: dict, *, variant: str, output: Path,
                 fork_variants: tuple[str, ...] = ()) -> dict | None:
    """Run original mode-hold through run_legacy; only D step hook can add work."""
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "mode_hold")
    recipe, noise, _ = declared_recipe(config)
    original_checkpoint = mode_hold.checkpoint
    dense = []
    hashes = {}
    moments = {}
    is_parent = bool(fork_variants)
    children = {}
    child_name = None
    receipt = None
    with ExitStack() as stack:
        controller = stack.enter_context(adaptive_d_allocation(
            max_d=3 if variant == "adaptive_cold" else 1,
            active_from=0 if variant == "adaptive_cold" else 1000))

        def checkpoint(step, measure):
            nonlocal child_name
            original_checkpoint(step, measure)
            if step >= 1000:
                with torch.random.fork_rng(devices=[]):
                    value = measure()
                dense.append(dict(step=step, modes=value["modes"],
                                  hq=value["hq"], hq_counts=value["hq_counts"]))
            if step not in (1000, 1200):
                return
            state = _host_state(inspect.currentframe().f_back)
            hashes[step] = training_state_sha256(state)
            moments[step] = _moment_steps(state)
            if step == 1000 and is_parent:
                for name in fork_variants:
                    pid = os.fork()
                    if pid == 0:
                        child_name = name
                        if name != "identity":
                            stack.enter_context(patch.object(
                                schedule_module, "policy_multipliers",
                                lambda *args, **kwargs: (1.0, 1.0)))
                            stack.enter_context(patch.object(
                                compare_defaults, "learning_rate_scale",
                                lambda *args, **kwargs: 1.0))
                        if name == "adaptive":
                            controller.max_d = 3
                        return
                    children[name] = pid
                raise _ParentForked()

        stack.enter_context(patch.object(mode_hold, "checkpoint", checkpoint))
        try:
            result, context = run_legacy(spec, recipe, noise,
                                         model_policy=declared_model_policy(config))
        except _ParentForked:
            errors = {}
            for name, pid in children.items():
                _, status = os.waitpid(pid, 0)
                if not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                    errors[name] = status
            if errors:
                raise RuntimeError(f"warm child failure: {errors}")
            return None
        except BaseException:
            if child_name is not None:
                _write_json(output / f"{child_name}.error.json",
                            dict(error=traceback.format_exc()))
                traceback.print_exc()
                os._exit(1)
            raise
        else:
            receipt = controller.receipt()
            if moments[1200] != receipt["adam_moment_updates"]:
                raise RuntimeError("recorded Adam calls differ from actual moments")
            phase = "warm" if child_name else "cold"
            rates = (_validate_rates(receipt, prepared_config(config, "constant"),
                                     1000 if child_name else 0)
                     if (child_name not in (None, "identity") or
                         variant in ("adaptive_cold", "constant_cold")) else None)
            terminal = _window([point for point in dense
                                if point["step"] in (1000, 1050, 1100, 1150, 1200)])
            local = _window([point for point in dense if 1000 < point["step"] <= 1200])
            if len(dense) != 201 or terminal["checks"] != 5 or local["checks"] != 200:
                raise RuntimeError("frozen dense or terminal checkpoint is missing")
            verdict = test_verdict(spec, result)
            row = dict(variant=child_name or variant, phase=phase,
                       shared_gate_eligible=False, result=result,
                       verdict=verdict, terminal=terminal, local_stability=local,
                       checkpoint_hashes=hashes, checkpoint_moments=moments,
                       receipt=receipt, post_prefix_rate_ranges=rates,
                       applied=context["applied"], noise=context["noise_receipt"])
            _write_gzip(output / f"{row['variant']}.json.gz", row)
            print(json.dumps(dict(event="RUN_DONE", variant=row["variant"],
                                  phase=phase, terminal=terminal,
                                  local_passes=local["passing_checks"],
                                  d_calls=receipt["d_gradient_evaluations"],
                                  g_calls=receipt["g_gradient_evaluations"])), flush=True)
            if child_name:
                os._exit(0)
            return row


def _read_gzip(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def _without_timing(value):
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items()
                if key not in ("seconds", "first_pass_seconds",
                               "confirmed_seconds", "stable_from_seconds")}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    constant_config = prepared_config(base, "constant")
    source_names = ["reports/toy100/adaptive_d_allocation_scratch.py",
                    "reports/toy100/adaptive_d_allocation_probe.py",
                    "benchmarks/locked_shared/mode_hold.py",
                    "benchmarks/transfer_suite/legacy_noise_adapters.py",
                    "benchmarks/toy100/continuous_probe.py",
                    "benchmarks/toy100/warm_equilibrium_probe.py"]
    source_hashes = {}
    for name in source_names:
        path = ROOT / name
        target = output / "source_archive" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[name] = _hash(path)
    declaration = dict(seed=0, host="mode_hold", steps=1200, warm_prefix=1000,
                       warm_children=["identity", "constant", "adaptive"],
                       cold_arms=["scheduled_cold", "constant_cold",
                                  "adaptive_cold_only_after_warm_pass"],
                       candidate=dict(max_d=3, min_d=1, eval_batch=16,
                                      a_d=.01, alpha_d=.1, rho_d=.5, eval_seed=20103),
                       unchanged_G_updates_and_noise_clock=True,
                       scratch_extra_D_updates=True,
                       shared_gate_eligible=False,
                       source_sha256=source_hashes)
    _write_json(output / "declaration.json", declaration)
    print(json.dumps(dict(event="PREDECLARED", declaration_sha256=_hash(
        output / "declaration.json"))), flush=True)
    scheduled = _run_episode(base, variant="scheduled_cold", output=output)
    constant = _run_episode(constant_config, variant="constant_cold", output=output)
    _run_episode(base, variant="warm_parent", output=output,
                 fork_variants=("identity", "constant", "adaptive"))
    warm = {name: _read_gzip(output / f"{name}.json.gz")
            for name in ("identity", "constant", "adaptive")}
    if (warm["identity"]["checkpoint_hashes"]["1200"] !=
            scheduled["checkpoint_hashes"][1200]):
        raise RuntimeError("forked identity state differs from separate cold control")
    if (_without_timing(warm["identity"]["result"]) !=
            _without_timing(scheduled["result"])):
        raise RuntimeError("forked identity metrics differ from separate cold control")
    # A warm failure is a conservative stop for this particular arm. A future
    # cold-new-attractor diagnostic requires a separate explicit declaration.
    warm_pass = warm["adaptive"]["local_stability"]["pass_all"]
    cold = (_run_episode(constant_config, variant="adaptive_cold", output=output)
            if warm_pass else None)
    summary = dict(identity_parity=True,
                   controls=dict(scheduled=dict(terminal=scheduled["terminal"],
                                                 local=scheduled["local_stability"]),
                                 constant=dict(terminal=constant["terminal"],
                                               local=constant["local_stability"])),
                   warm={name: dict(terminal=row["terminal"],
                                    local=row["local_stability"],
                                    d_updates=row["receipt"]["d_gradient_evaluations"])
                         for name, row in warm.items()},
                   cold_candidate=(dict(terminal=cold["terminal"],
                                        local=cold["local_stability"],
                                        d_updates=cold["receipt"]["d_gradient_evaluations"],
                                        label="COLD_MODE_HOLD_SCREEN") if cold else
                                   dict(label="SKIPPED_AFTER_WARM_FAIL")),
                   full_gate_winner=False)
    _write_json(output / "summary.json", summary)
    print(json.dumps(dict(event="DONE", **summary)), flush=True)


if __name__ == "__main__":
    main()
