"""Fast, production-derived sustained mode-hold probe.

Run the 1,200-update stationary control first. A survivor then runs an
uninterrupted 2,400-update hold. For adaptation, run 3,600 updates with the
same target through update 2,400, shift the ring in place, and run a matched
3,600-update control that freezes Adam updates after the shift. Pass requires
the frozen stationary window, every diagnostic hold check, recovery by 400
updates after the shift, every later check, and a failing frozen control.

The host training body, optimizer policy, and sampling are unchanged. Noise
keeps its original 1,200-update horizon; extending training must not postpone
burn-in. The shift mutates the target tensor after a recorded update, so model,
Adam moments, EMA, and random streams continue without reconstruction.

Example::

    python -u -m benchmarks.toy100.continuous_probe --mode constant \
        --steps 2400 --diagnostic-every 10 --output /tmp/hold.json
    python -u -m benchmarks.toy100.continuous_probe --mode constant \
        --steps 3600 --shift-step 2400 --output /tmp/frozen.json \
        --freeze-after-shift
    python -u -m benchmarks.toy100.continuous_probe --mode constant \
        --steps 3600 --shift-step 2400 --output /tmp/shift.json \
        --frozen-control /tmp/frozen.json
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import baseline, mode_hold
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.compare_defaults import candidate, optimizer_defaults
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.protocol import required_tasks, test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/toy100/constraints_simple_regularization.json"
FROZEN_STEPS = baseline.BUDGETS["mode_hold"]
RECOVERY_DEADLINE = 400
_SOURCE_FILES = (
    "benchmarks/toy100/continuous_probe.py",
    "benchmarks/toy100/models.py",
    "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/locked_shared/baseline.py",
    "benchmarks/locked_shared/observation.py",
    "benchmarks/transfer_suite/compare_defaults.py",
    "benchmarks/transfer_suite/legacy_noise_adapters.py",
    "benchmarks/transfer_suite/toy100_compatibility.py",
    "benchmarks/transfer_suite/protocol.py",
    "benchmarks/transfer_suite/suite.py",
    "particlegan/grad_regularizers.py",
    "particlegan/recipes.py",
)
_SUPPLEMENTAL_SOURCE = "benchmarks/toy100/models.py"
_SUPPLEMENTAL_ARCHIVE_NAME = "toy100-models-source.py"


def _provenance() -> dict:
    """Bind the exact executable sources and CPU/PyTorch environment."""
    return dict(
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in _SOURCE_FILES},
        runtime=dict(python=platform.python_version(),
                     torch=str(torch.__version__),
                     torch_git_revision=torch.version.git_version,
                     torch_build=torch.__config__.show(),
                     cpu_capability=torch.backends.cpu.get_cpu_capability(),
                     machine=platform.machine(),
                     processor=platform.processor(),
                     threads=torch.get_num_threads(),
                     device="cpu"),
    )


def archive_executable_sources(output: Path, source_sha256: dict[str, str]) -> dict:
    """Archive the standard suite snapshot plus its omitted noise source."""
    output.mkdir(parents=True, exist_ok=True)
    archive = suite.snapshot(output)
    supplemental = {}
    for name, expected in source_sha256.items():
        observed = archive["source_sha256"].get(name)
        if name == _SUPPLEMENTAL_SOURCE and observed is None:
            source = (ROOT / name).read_bytes()
            path = output / _SUPPLEMENTAL_ARCHIVE_NAME
            path.write_bytes(source)
            observed = hashlib.sha256(source).hexdigest()
            supplemental[name] = dict(path=str(path), sha256=observed)
        if observed != expected:
            raise RuntimeError(f"archived source differs from executed source: {name}")
    source_file = output / "source.tar.gz"
    return dict(path=str(source_file),
                sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                source_sha256=archive["source_sha256"],
                supplemental_sources=supplemental)


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
                  dense_after: int | None, dense_until: int | None,
                  shift_step: int | None, shift: tuple[float, float],
                  freeze_after_shift: bool, log,
                  checkpoint_hook_step: int | None = None,
                  checkpoint_hook=None) -> tuple[dict, dict]:
    """Extend the frozen host with scoped instrumentation only."""
    steps = spec["steps"]
    policy = _noise_policy(noise, noise_horizon)
    original_recipe = mode_hold.ModeHoldRecipe
    original_means = mode_hold.ring_means
    original_checkpoint = mode_hold.checkpoint
    original_adam_step = torch.optim.Adam.step
    step_delegate = {"fn": original_adam_step}
    expected_accounting = {"calls": steps,
                           "moment_updates": shift_step if freeze_after_shift else steps}
    tracked: dict[int, dict] = {}
    rate_ranges: dict[str, dict] = {}
    post_checkpoint_rate_ranges: dict[str, dict] = {}
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
                                                  "delegate_calls": 0})
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
            if (checkpoint_hook_step is not None
                    and row["calls"] > checkpoint_hook_step):
                after = post_checkpoint_rate_ranges.setdefault(
                    role, {"min": rate, "max": rate, "observations": 0})
                after["min"] = min(after["min"], rate)
                after["max"] = max(after["max"], rate)
                after["observations"] += 1
        if frozen:
            return None
        result = step_delegate["fn"](optimizer, *args, **kwargs)
        row["delegate_calls"] += 1
        return result

    def optimizer_counts() -> list[dict]:
        rows = []
        for row in tracked.values():
            moments = [int(state["step"].item() if isinstance(state["step"], torch.Tensor)
                           else state["step"])
                       for state in row["optimizer"].state.values() if "step" in state]
            if not moments:
                raise RuntimeError("Adam moment counters are missing")
            if min(moments) != max(moments):
                raise RuntimeError("Adam moment counters disagree within an optimizer")
            rows.append(dict(calls=row["calls"],
                             delegate_calls=row["delegate_calls"],
                             updates=moments[0],
                             moment_steps_min=min(moments),
                             moment_steps_max=max(moments),
                             rates=[group["lr"] for group in
                                    row["optimizer"].param_groups]))
        return sorted(rows, key=lambda row: len(row["rates"]))

    def observe(step: int, measure):
        nonlocal frozen, shift_pair
        # Always preserve the host's own 24-point recorder and its timing.
        original_checkpoint(step, measure)
        if (step % diagnostic_every == 0 or step == shift_step
                or (dense_after is not None and step > dense_after
                    and (dense_until is None or step <= dense_until))):
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
        if step == checkpoint_hook_step:
            caller = inspect.currentframe().f_back
            if caller is None or caller.f_code.co_name != "train_mode_hold":
                raise RuntimeError("warm hook was not called by the frozen host")
            values = caller.f_locals
            required = ("generator", "critic", "prior", "opt_g", "opt_d",
                        "ema_g", "ema_z", "stream", "means")
            if any(name not in values for name in required):
                raise RuntimeError("frozen host locals changed at warm checkpoint")
            def set_step_delegate(fn):
                if not callable(fn):
                    raise TypeError("optimizer step delegate must be callable")
                step_delegate["fn"] = fn
            def declare_optimizer_accounting(*, calls: int,
                                             moment_updates: int = steps,
                                             d_moment_updates: int | None = None):
                if (type(calls) is not int or calls < steps
                        or type(moment_updates) is not int
                        or not step <= moment_updates <= calls):
                    raise ValueError("invalid optimizer accounting")
                if d_moment_updates is not None and (
                        type(d_moment_updates) is not int
                        or not step <= d_moment_updates <= calls):
                    raise ValueError("invalid discriminator moment accounting")
                expected_accounting.update(calls=calls,
                                           moment_updates=moment_updates,
                                           d_moment_updates=d_moment_updates)
            checkpoint_hook({**{name: values[name] for name in required},
                             "noise_policy": policy,
                             "control": control,
                             "base_adam_step": original_adam_step,
                             "set_step_delegate": set_step_delegate,
                             "declare_optimizer_accounting": declare_optimizer_accounting,
                             "completed_steps": step,
                             "target_steps": steps})

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
        post_checkpoint_rate_ranges=post_checkpoint_rate_ranges,
        expected_optimizer_accounting=expected_accounting,
    )
    return result, context


def run_probe(config: dict, *, mode: str = "constant", steps: int = FROZEN_STEPS,
              noise_horizon: int = FROZEN_STEPS,
              diagnostic_every: int = 50, dense_after: int | None = None,
              dense_until: int | None = None,
              shift_step: int | None = None,
              shift: tuple[float, float] = (1.0, 0.0),
              freeze_after_shift: bool = False, log=None,
              checkpoint_hook_step: int | None = None,
              checkpoint_hook=None) -> dict:
    """Return a strict 24-check grade and compact evidence for one run."""
    if type(steps) is not int or steps < FROZEN_STEPS:
        raise ValueError("steps must preserve at least the frozen 1,200 updates")
    if type(noise_horizon) is not int or noise_horizon != FROZEN_STEPS:
        raise ValueError("noise horizon must remain the frozen 1,200 updates")
    if type(diagnostic_every) is not int or diagnostic_every < 1 or 50 % diagnostic_every:
        raise ValueError("diagnostic_every must divide the frozen 50-step cadence")
    if dense_after is not None and (type(dense_after) is not int
                                    or not 0 <= dense_after < steps):
        raise ValueError("dense_after must be inside the episode")
    if dense_until is not None and (dense_after is None
                                    or type(dense_until) is not int
                                    or not dense_after < dense_until <= steps):
        raise ValueError("dense_until must follow dense_after inside the episode")
    if shift_step is not None and (type(shift_step) is not int
                                   or not FROZEN_STEPS <= shift_step < steps
                                   or shift_step % diagnostic_every):
        raise ValueError("shift_step must be at or after 1,200, before the end, "
                         "and on the diagnostic cadence")
    if freeze_after_shift and shift_step is None:
        raise ValueError("a frozen continuation requires shift_step")
    if (checkpoint_hook is None) != (checkpoint_hook_step is None):
        raise ValueError("checkpoint hook and step must be supplied together")
    if checkpoint_hook_step is not None and (
            type(checkpoint_hook_step) is not int
            or not 0 < checkpoint_hook_step <= steps):
        raise ValueError("checkpoint hook step must be inside the episode")
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
        diagnostic_every=diagnostic_every, dense_after=dense_after,
        dense_until=dense_until,
        shift_step=shift_step,
        shift=shift, freeze_after_shift=freeze_after_shift, log=log,
        checkpoint_hook_step=checkpoint_hook_step,
        checkpoint_hook=checkpoint_hook,
    )
    diagnostic = context["diagnostic"]
    shift_pair = context["shift_pair"]
    optimizer_final = context["optimizer_final"]
    rate_ranges = context["rate_ranges"]
    post_checkpoint_rate_ranges = context["post_checkpoint_rate_ranges"]
    expected_accounting = context["expected_optimizer_accounting"]
    grade = test_verdict(spec, result)
    receipt = context["noise_receipt"]
    stationary_steps = range(FROZEN_STEPS - 200, FROZEN_STEPS + 1, 50)
    stationary = _window([point for point in diagnostic
                          if point["step"] in stationary_steps])
    if stationary["checks"] != 5:
        raise RuntimeError("the frozen stationary terminal window is incomplete")
    continued = (_window([point for point in diagnostic
                          if FROZEN_STEPS < point["step"] <=
                          (shift_step if shift_step is not None else steps)])
                 if steps > FROZEN_STEPS else None)
    recovery = (_window([point for point in diagnostic
                         if point["step"] > shift_step])
                if shift_step is not None else None)
    if recovery is not None:
        deadline_step = shift_step + RECOVERY_DEADLINE
        late = _window([point for point in diagnostic
                        if point["step"] >= deadline_step])
        recovery["deadline_step"] = deadline_step
        recovery["deadline_window"] = late
        recovery["deadline_assessable"] = late["checks"] >= 5
        recovery["deadline_pass"] = bool(late["checks"] >= 5 and late["pass_all"])
        recovery["delay_updates"] = (None if recovery["stable_from_step"] is None else
                                     recovery["stable_from_step"] - shift_step)
    rate_expected = dict(g=recipe.lr,
                         prior=recipe.lr * recipe.prior_lr_mult,
                         d=recipe.lr * recipe.d_lr_mult)
    if set(rate_ranges) != set(rate_expected):
        raise RuntimeError("G, D, or prior optimizer rate was not observed")
    if any(rate_ranges[role]["observations"] != expected_accounting["calls"]
           for role in rate_expected):
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
        if row["calls"] != expected_accounting["calls"]:
            raise RuntimeError("Adam update count differs from declared continuation")
        target = expected_accounting["moment_updates"]
        # D is the single-group Adam. A stall-gated extra critic step advances
        # only that counter; G's moment count stays one per outer step.
        if len(row["rates"]) == 1 and expected_accounting.get("d_moment_updates") is not None:
            target = expected_accounting["d_moment_updates"]
        if row["updates"] != target:
            raise RuntimeError("Adam update count differs from declared continuation")
    if shift_pair is not None and any(row["updates"] != shift_step
                                      for row in shift_pair["optimizer_at_shift"]):
        raise RuntimeError("optimizer state reset at distribution shift")
    hold_pass = bool(continued is None or continued["pass_all"])
    shift_quality = bool(recovery is not None and recovery["deadline_pass"])
    window_pass = (stationary["pass_all"] and hold_pass and
                   (shift_quality if recovery is not None else True))
    if recovery is not None and (not continued["checks"]
                                 or not recovery["deadline_assessable"]):
        status = "INCOMPLETE"
    elif recovery is not None and window_pass:
        # A bare shift result is not evidence of adaptation: compare a
        # separately run frozen control with exactly matching pre-shift state.
        status = "UNCONFIRMED"
    else:
        status = "PASS" if window_pass else "FAIL"
    provenance = _provenance()
    return dict(
        mode=mode, config_sha256=hashlib.sha256(json.dumps(
            effective, sort_keys=True).encode()).hexdigest(),
        effective_config=effective, source_recipe=recipe.to_dict(),
        **provenance, steps=steps, noise_horizon=noise_horizon,
        diagnostic_every=diagnostic_every, dense_after=dense_after,
        dense_until=dense_until,
        shift_step=shift_step, shift=list(shift) if shift_step else None,
        freeze_after_shift=freeze_after_shift,
        status=status,
        terminal_grade=grade["status"], convergence=grade["convergence"],
        stationary=stationary, continued_hold=continued,
        shift_recovery=recovery,
        observations=result["observations"], diagnostic=diagnostic,
        final=result["live"], ema=result.get("ema"),
        shift_pair=shift_pair, optimizer_final=optimizer_final,
        expected_optimizer_accounting=expected_accounting,
        rate_ranges=rate_ranges,
        post_checkpoint_rate_ranges=post_checkpoint_rate_ranges,
        seconds=result["seconds"],
        noise=dict(step_calls=receipt["step_calls"],
                   horizon=receipt["total_steps"],
                   input_nonzero_steps=receipt["input_nonzero_steps"],
                   output_nonzero_steps=receipt["output_nonzero_steps"],
                   output_sigma_last=receipt["output_sigma_last"]),
    )


def match_frozen_control(active: dict, frozen: dict) -> dict:
    """Confirm a shift result using a same-initialization no-update control."""
    if active.get("shift_step") is None or active.get("freeze_after_shift"):
        raise ValueError("active evidence must contain an unfrozen shift")
    if not frozen.get("freeze_after_shift"):
        raise ValueError("control must freeze all Adam updates after the shift")
    matching = ("mode", "config_sha256", "source_sha256", "runtime", "steps",
                "noise_horizon", "diagnostic_every", "dense_after", "dense_until",
                "shift_step", "shift")
    changed = [key for key in matching if active.get(key) != frozen.get(key)]
    if changed:
        raise ValueError(f"shift and frozen control differ in {changed}")
    shift_step = active["shift_step"]
    for key in ("stationary", "continued_hold", "shift_pair"):
        if active.get(key) != frozen.get(key):
            raise ValueError(f"pre-shift control state differs in {key}")
    pre_active = [point for point in active["diagnostic"]
                  if point["step"] <= shift_step]
    pre_frozen = [point for point in frozen["diagnostic"]
                  if point["step"] <= shift_step]
    if pre_active != pre_frozen:
        raise ValueError("pre-shift diagnostic curves differ")
    frozen_late = frozen["shift_recovery"]["deadline_window"]
    sensitivity = (frozen_late["checks"] >= 5
                   and frozen_late["passing_checks"] == 0)
    quality = (active["stationary"]["pass_all"]
               and active["continued_hold"]["pass_all"]
               and active["shift_recovery"]["deadline_pass"])
    confirmed = deepcopy(active)
    confirmed["matched_control"] = dict(
        frozen_config_sha256=frozen["config_sha256"],
        frozen_post_deadline=frozen_late,
        frozen_optimizer_updates=[row["updates"] for row in
                                  frozen["optimizer_final"]],
        sensitivity_pass=sensitivity,
    )
    confirmed["status"] = "PASS" if quality and sensitivity else "FAIL"
    return confirmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("scheduled", "constant"), default="constant")
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--diagnostic-every", type=int, default=50)
    parser.add_argument("--dense-after", type=int,
                        help="also record every update after this step")
    parser.add_argument("--dense-until", type=int,
                        help="stop extra per-update recording after this step")
    parser.add_argument("--shift-step", type=int)
    parser.add_argument("--shift-x", type=float, default=1.0)
    parser.add_argument("--shift-y", type=float, default=0.0)
    parser.add_argument("--freeze-after-shift", action="store_true")
    parser.add_argument("--frozen-control", type=Path,
                        help="matched frozen evidence required to confirm adaptation")
    parser.add_argument("--archive-sources", type=Path,
                        help="directory for the standard transfer-suite source archive")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    def log(row):
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    evidence = run_probe(
        config, mode=args.mode, steps=args.steps,
        diagnostic_every=args.diagnostic_every, dense_after=args.dense_after,
        dense_until=args.dense_until,
        shift_step=args.shift_step, shift=(args.shift_x, args.shift_y),
        freeze_after_shift=args.freeze_after_shift, log=log,
    )
    evidence["input_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    evidence["input_config_path"] = str(args.config)
    if args.frozen_control:
        evidence = match_frozen_control(
            evidence, json.loads(args.frozen_control.read_text()),
        )
        evidence["matched_control_path"] = str(args.frozen_control)
        evidence["matched_control_sha256"] = hashlib.sha256(
            args.frozen_control.read_bytes()).hexdigest()
    if args.archive_sources:
        evidence["source_archive"] = archive_executable_sources(
            args.archive_sources, evidence["source_sha256"])
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
