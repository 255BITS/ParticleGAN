"""Fast, production-derived sustained mode-hold probe.

The 1,200-update case calls the frozen transfer host unchanged. Longer runs
reuse that same host training body and optimizer policy, extending only its
budget and observation window. Noise keeps its original 1,200-update horizon;
changing the training horizon must not postpone noise burn-in. A ring shift
mutates the target tensor in place after a recorded update, so model, Adam
moments, EMA, and random streams continue without reconstruction.

Example::

    python -u -m benchmarks.toy100.continuous_probe --mode constant \
        --steps 2400 --shift-step 1200 --output /tmp/constant-shift.json
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline, mode_hold
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.compare_defaults import candidate, optimizer_defaults
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.protocol import required_tasks, test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/toy100/constraints_simple_regularization.json"
FROZEN_STEPS = baseline.BUDGETS["mode_hold"]


def prepared_config(source: dict, mode: str) -> dict:
    """Keep one shared recipe; the constant arm changes only its LR policy."""
    if mode not in ("scheduled", "constant"):
        raise ValueError("mode must be scheduled or constant")
    config = deepcopy(source)
    if mode == "constant":
        config.pop("network_lr_horizon_cap", None)
        config.pop("network_lr_floor", None)
        config.update(lr_anneal_start=0.0, lr_floor=1.0)
    return config


def _noise_policy(noise: dict, horizon: int) -> NoisePolicy:
    return NoisePolicy(
        noise["output_noise_std"], noise["input_noise_std"],
        noise["input_noise_anneal_end"], horizon, seed=0,
        output_noise_warmup=noise.get("output_noise_warmup", 0.0),
        output_noise_learnable=noise.get("output_noise_learnable", False),
        output_noise_rng=noise.get("output_noise_rng"),
    )


def _passes(point: dict) -> bool:
    return point["modes"] == mode_hold.N_MODES and point["hq"] >= mode_hold.PASS_HQ


def _window(points: list[dict], *, minimum: int = 5) -> dict:
    """Show every failure and the length of the passing terminal suffix."""
    passing = [_passes(point) for point in points]
    start = len(points)
    while start and passing[start - 1]:
        start -= 1
    suffix = points[start:]
    return dict(checks=len(points), passing_checks=sum(passing),
                failing_steps=[point["step"] for point, okay in zip(points, passing)
                               if not okay],
                min_modes=min((point["modes"] for point in points), default=None),
                min_hq=min((point["hq"] for point in points), default=None),
                passing_suffix=len(suffix),
                stable_from_step=suffix[0]["step"] if len(suffix) >= minimum else None,
                pass_all=bool(points) and all(passing),
                pass_suffix=len(suffix) >= minimum)


def _run_extended(spec: dict, recipe, noise: dict, config: dict, *,
                  noise_horizon: int, diagnostic_every: int,
                  shift_step: int | None, shift: tuple[float, float],
                  freeze_after_shift: bool, log) -> tuple[dict, dict]:
    """Extend the frozen host with scoped instrumentation only."""
    steps = spec["steps"]
    policy = _noise_policy(noise, noise_horizon)
    original_recipe = mode_hold.ModeHoldRecipe
    original_means = mode_hold.ring_means
    original_checkpoint = mode_hold.checkpoint
    original_adam_step = torch.optim.Adam.step
    tracked: dict[int, dict] = {}
    rate_ranges: dict[str, dict] = {}
    ring: torch.Tensor | None = None
    diagnostic: list[dict] = []
    shift_pair: dict | None = None
    frozen = False

    def extended_recipe(**kwargs):
        return original_recipe(steps=steps, **kwargs)

    def retain_ring(*args, **kwargs):
        nonlocal ring
        ring = original_means(*args, **kwargs)
        return ring

    def step_optimizer(optimizer, *args, **kwargs):
        row = tracked.setdefault(id(optimizer), {"optimizer": optimizer, "calls": 0,
                                                  "updates": 0})
        row["calls"] += 1
        for group in optimizer.param_groups:
            role = ("prior" if group.get("_comparison_prior") else
                    "g" if len(optimizer.param_groups) > 1 else "d")
            rate = float(group["lr"])
            seen = rate_ranges.setdefault(role, {"min": rate, "max": rate,
                                                 "observations": 0})
            seen["min"] = min(seen["min"], rate)
            seen["max"] = max(seen["max"], rate)
            seen["observations"] += 1
        if frozen:
            return None
        result = original_adam_step(optimizer, *args, **kwargs)
        row["updates"] += 1
        return result

    def optimizer_counts() -> list[dict]:
        return sorted((dict(calls=row["calls"], updates=row["updates"],
                            rates=[group["lr"] for group in row["optimizer"].param_groups])
                       for row in tracked.values()), key=lambda row: len(row["rates"]))

    def observe(step: int, measure):
        nonlocal frozen, shift_pair
        # Always preserve the host's own 24-point recorder and its timing.
        original_checkpoint(step, measure)
        if step % diagnostic_every == 0 or step == shift_step:
            with torch.random.fork_rng(devices=[]):
                measured = measure()
                point = {"step": step, **{key: measured[key] for key in
                         ("modes", "hq", "effective_modes", "hq_counts")}}
            diagnostic.append(point)
            if log is not None:
                log({"event": "checkpoint", **point})
        if step == shift_step:
            if ring is None:
                raise RuntimeError("mode-hold target was not captured")
            before = diagnostic[-1]
            with torch.no_grad():
                ring.add_(torch.tensor(shift, dtype=ring.dtype, device=ring.device))
            with torch.random.fork_rng(devices=[]):
                measured = measure()
                after = {"step": step, **{key: measured[key] for key in
                         ("modes", "hq", "effective_modes", "hq_counts")}}
            shift_pair = {"before": before, "after": after,
                          "optimizer_at_shift": optimizer_counts()}
            frozen = freeze_after_shift
            if log is not None:
                log({"event": "shift", **after, "frozen": frozen})

    started = time.perf_counter()
    applied: list[dict] = []
    with ExitStack() as stack:
        stack.enter_context(patch.dict(baseline.BUDGETS, {"mode_hold": steps}))
        stack.enter_context(patch.object(mode_hold, "ModeHoldRecipe", extended_recipe))
        stack.enter_context(patch.object(mode_hold, "ring_means", retain_ring))
        stack.enter_context(patch.object(mode_hold, "checkpoint", observe))
        stack.enter_context(patch.object(torch.optim.Adam, "step", step_optimizer))
        stack.enter_context(optimizer_defaults(
            recipe, applied,
            network_lr_horizon_cap=config.get("network_lr_horizon_cap"),
            network_lr_floor=config.get("network_lr_floor"),
        ))
        control = evaluate.FixedControl(vector_tasks.fixed_policy("cosine"),
                                        noise_horizon)
        stack.enter_context(bridge.control_host_schedules(control))
        result = baseline.run_toy("mode_hold", candidate(recipe), noise_policy=policy)
    result["seconds"] = time.perf_counter() - started
    receipt = policy.receipt()
    if receipt["step_calls"] != steps:
        raise RuntimeError("the host did not complete the requested updates")
    context = dict(
        applied=applied, noise_receipt=receipt,
        diagnostic=diagnostic, shift_pair=shift_pair,
        optimizer_final=optimizer_counts(),
        rate_ranges=rate_ranges,
    )
    return result, context


def run_probe(config: dict, *, mode: str = "constant", steps: int = FROZEN_STEPS,
              noise_horizon: int = FROZEN_STEPS,
              diagnostic_every: int = 50, shift_step: int | None = None,
              shift: tuple[float, float] = (1.0, 0.0),
              freeze_after_shift: bool = False, log=None) -> dict:
    """Return a strict 24-check grade and compact evidence for one run."""
    if type(steps) is not int or steps < FROZEN_STEPS:
        raise ValueError("steps must preserve at least the frozen 1,200 updates")
    if type(noise_horizon) is not int or noise_horizon != FROZEN_STEPS:
        raise ValueError("noise horizon must remain the frozen 1,200 updates")
    if type(diagnostic_every) is not int or diagnostic_every < 1 or 50 % diagnostic_every:
        raise ValueError("diagnostic_every must divide the frozen 50-step cadence")
    if shift_step is not None and (type(shift_step) is not int
                                   or not 0 < shift_step < steps):
        raise ValueError("shift_step must be inside the training episode")
    if freeze_after_shift and shift_step is None:
        raise ValueError("a frozen continuation requires shift_step")
    if (len(shift) != 2 or not all(isinstance(x, (int, float))
                                  and math.isfinite(x) for x in shift)
            or (shift_step is not None and shift == (0, 0))):
        raise ValueError("shift must be two coordinates")
    effective = prepared_config(config, mode)
    recipe, noise, _ = declared_recipe(effective)
    spec = next(row for row in required_tasks() if row["name"] == "mode_hold")
    spec = {**spec, "steps": steps}
    torch.set_num_threads(1)
    result, context = _run_extended(
        spec, recipe, noise, effective, noise_horizon=noise_horizon,
        diagnostic_every=diagnostic_every, shift_step=shift_step,
        shift=shift, freeze_after_shift=freeze_after_shift, log=log,
    )
    diagnostic = context["diagnostic"]
    shift_pair = context["shift_pair"]
    optimizer_final = context["optimizer_final"]
    rate_ranges = context["rate_ranges"]
    grade = test_verdict(spec, result)
    receipt = context["noise_receipt"]
    stationary_steps = range(FROZEN_STEPS - 200, FROZEN_STEPS + 1, 50)
    stationary = _window([point for point in diagnostic
                          if point["step"] in stationary_steps])
    if stationary["checks"] != 5:
        raise RuntimeError("the frozen stationary terminal window is incomplete")
    continued = (_window([point for point in diagnostic
                          if point["step"] > FROZEN_STEPS])
                 if steps > FROZEN_STEPS else None)
    recovery = (_window([point for point in diagnostic
                         if point["step"] > shift_step])
                if shift_step is not None else None)
    if recovery is not None:
        recovery["delay_updates"] = (None if recovery["stable_from_step"] is None else
                                     recovery["stable_from_step"] - shift_step)
    rate_expected = dict(g=recipe.lr,
                         prior=recipe.lr * recipe.prior_lr_mult,
                         d=recipe.lr * recipe.d_lr_mult)
    if set(rate_ranges) != set(rate_expected):
        raise RuntimeError("G, D, or prior optimizer rate was not observed")
    if any(rate_ranges[role]["observations"] != steps for role in rate_expected):
        raise RuntimeError("G, D, or prior optimizer rate trace is incomplete")
    if mode == "constant":
        if not (recipe.lr_anneal_start == 0 and recipe.lr_floor == 1
                and "network_lr_horizon_cap" not in effective
                and "network_lr_floor" not in effective):
            raise RuntimeError("constant arm has a residual LR schedule")
        for role, expected in rate_expected.items():
            measured = rate_ranges[role]
            if not (math.isclose(measured["min"], expected, rel_tol=1e-12)
                    and math.isclose(measured["max"], expected, rel_tol=1e-12)):
                raise RuntimeError(f"{role} actual LR changed during constant run")
    for row in optimizer_final:
        if row["calls"] != steps or row["updates"] != (
                shift_step if freeze_after_shift else steps):
            raise RuntimeError("Adam update count differs from declared continuation")
    if shift_pair is not None and any(row["updates"] != shift_step
                                      for row in shift_pair["optimizer_at_shift"]):
        raise RuntimeError("optimizer state reset at distribution shift")
    window_pass = (stationary["pass_all"] and
                   (recovery["pass_suffix"] if recovery is not None else
                    continued["pass_all"] if continued is not None else True))
    return dict(
        mode=mode, config_sha256=hashlib.sha256(json.dumps(
            effective, sort_keys=True).encode()).hexdigest(),
        source_recipe=recipe.to_dict(), steps=steps, noise_horizon=noise_horizon,
        shift_step=shift_step, shift=list(shift) if shift_step else None,
        freeze_after_shift=freeze_after_shift,
        status="PASS" if window_pass else "FAIL",
        terminal_grade=grade["status"], convergence=grade["convergence"],
        stationary=stationary, continued_hold=continued,
        shift_recovery=recovery,
        observations=result["observations"], diagnostic=diagnostic,
        final=result["live"], ema=result.get("ema"),
        shift_pair=shift_pair, optimizer_final=optimizer_final,
        rate_ranges=rate_ranges,
        seconds=result["seconds"],
        noise=dict(step_calls=receipt["step_calls"],
                   horizon=receipt["total_steps"],
                   input_nonzero_steps=receipt["input_nonzero_steps"],
                   output_nonzero_steps=receipt["output_nonzero_steps"],
                   output_sigma_last=receipt["output_sigma_last"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("scheduled", "constant"), default="constant")
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--diagnostic-every", type=int, default=50)
    parser.add_argument("--shift-step", type=int)
    parser.add_argument("--shift-x", type=float, default=1.0)
    parser.add_argument("--shift-y", type=float, default=0.0)
    parser.add_argument("--freeze-after-shift", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    def log(row):
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    evidence = run_probe(
        config, mode=args.mode, steps=args.steps,
        diagnostic_every=args.diagnostic_every,
        shift_step=args.shift_step, shift=(args.shift_x, args.shift_y),
        freeze_after_shift=args.freeze_after_shift, log=log,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        temporary.replace(args.output)
    log({"event": "result", "status": evidence["status"],
         "final": evidence["final"], "seconds": evidence["seconds"],
         "output": str(args.output) if args.output else None})


if __name__ == "__main__":
    main()
