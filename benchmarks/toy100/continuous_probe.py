"""Sustained mode-hold probe on the shared toy runner.

The ring problem is ``benchmarks.locked_shared.mode_hold.ModeHold``; this
module only grades a run of it. Everything else (recipe-built optimizers and
their LR schedule, loss, penalty, prior, noise, EMA, the extended and shift
protocols) is ``benchmarks.toy_runner``.

Grade: every live check in the stationary window (updates 1,000-1,200) holds
all 8 modes at HQ >= 0.90; an extended run must also hold every later check;
a shifted run must hold every check from ``shift_step + 400`` on.

``--mode constant`` is a recipe field change only (no annealing, floor 1, no
network horizon cap); ``--mode scheduled`` is the shipped recipe as is.
Noise keeps the recipe's 1,200-update horizon when training is extended::

    python -u -m benchmarks.toy100.continuous_probe --mode constant \
        --steps 2400 --log runs/toy-refactor/continuous_probe.log
    python -u -m benchmarks.toy100.continuous_probe --mode constant \
        --steps 3600 --shift-step 2400 --output runs/toy-refactor/shift.json

Not expressible on the runner, so removed: the frozen-after-shift control
(freezing Adam updates from outside the optimizer) and the warm-state
checkpoint hook used by ``warm_equilibrium_probe``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.observation import checkpoint
from benchmarks.toy_runner import run

ROOT = Path(__file__).resolve().parents[2]
FROZEN_STEPS = mode_hold.STEPS
STATIONARY_FROM = FROZEN_STEPS - 200
RECOVERY_DEADLINE = 400
MIN_CHECKS = 5
_SOURCE_FILES = (
    "benchmarks/toy100/continuous_probe.py",
    "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/locked_shared/mlp.py",
    "benchmarks/toy_runner.py",
    "particlegan/recipes.py",
    "particlegan/k3p.py",
    "particlegan/training.py",
)


def probe_recipe(mode: str):
    """The problem's shipped recipe; ``constant`` switches its LR schedule off."""
    recipe = mode_hold.ModeHold().recipe()
    if mode == "scheduled":
        return recipe
    if mode == "constant":
        return recipe.replace(lr_anneal_start=0.0, lr_floor=1.0, network_lr_floor=None,
                              network_lr_horizon_cap=None)
    raise ValueError("mode must be scheduled or constant")


def _passes(point: dict) -> bool:
    return point["modes"] == mode_hold.N_MODES and point["hq"] >= mode_hold.PASS_HQ


def _window(points: list[dict], *, minimum: int = MIN_CHECKS) -> dict:
    """Every failure and the length of the passing terminal suffix."""
    passing = [_passes(point) for point in points]
    start = len(points)
    while start and passing[start - 1]:
        start -= 1
    suffix = points[start:]
    return dict(checks=len(points), passing_checks=sum(passing),
                failing_steps=[p["step"] for p, ok in zip(points, passing) if not ok],
                min_modes=min((p["modes"] for p in points), default=None),
                min_hq=min((p["hq"] for p in points), default=None),
                passing_suffix=len(suffix),
                stable_from_step=suffix[0]["step"] if len(suffix) >= minimum else None,
                pass_all=bool(points) and all(passing),
                pass_suffix=len(suffix) >= minimum)


def run_probe(*, mode: str = "constant", steps: int = FROZEN_STEPS, diagnostic_every: int = 50,
              shift_step: int | None = None, device: str = "cpu", log=None,
              log_path: str | Path | None = None) -> dict:
    """Train the ring under ``mode`` and return the strict window grade."""
    if type(steps) is not int or steps < FROZEN_STEPS:
        raise ValueError("steps must keep at least the 1,200-update budget")
    if type(diagnostic_every) is not int or diagnostic_every < 1 or 50 % diagnostic_every:
        raise ValueError("diagnostic_every must divide the 50-step cadence")
    if shift_step is not None and (type(shift_step) is not int or not FROZEN_STEPS <= shift_step < steps
                                   or shift_step % diagnostic_every):
        raise ValueError("shift_step must be on the cadence, at or after 1,200 and before the end")
    recipe = probe_recipe(mode)
    torch.set_num_threads(1)
    result = run(mode_hold.ModeHold(), recipe=recipe, steps=steps, device=device,
                 observe_every=diagnostic_every, shift_step=shift_step, log=log,
                 log_path=log_path, observer=checkpoint)
    curve = [{k: p[k] for k in ("step", "modes", "hq", "effective_modes", "verdict")}
             for p in result["curve"]]
    stationary = _window([p for p in curve if STATIONARY_FROM <= p["step"] <= FROZEN_STEPS])
    if stationary["checks"] != 200 // diagnostic_every + 1:
        raise RuntimeError("the stationary window is incomplete")
    hold_end = shift_step if shift_step is not None else steps
    continued = (_window([p for p in curve if FROZEN_STEPS < p["step"] <= hold_end])
                 if steps > FROZEN_STEPS else None)
    recovery = None
    if shift_step is not None:
        deadline = shift_step + RECOVERY_DEADLINE
        after = _window([p for p in curve if p["step"] >= deadline])
        recovery = dict(_window([p for p in curve if p["step"] > shift_step]),
                        deadline_step=deadline, deadline_window=after,
                        deadline_pass=after["checks"] >= MIN_CHECKS and after["pass_all"],
                        runner=result["shift"])
    if recovery is not None and (continued is None or not continued["checks"]
                                 or recovery["deadline_window"]["checks"] < MIN_CHECKS):
        status = "INCOMPLETE"
    else:
        ok = (stationary["pass_all"] and (continued is None or continued["pass_all"])
              and (recovery is None or recovery["deadline_pass"]))
        status = "PASS" if ok else "FAIL"
    return dict(
        mode=mode, recipe=result["recipe"], steps=steps, diagnostic_every=diagnostic_every,
        shift_step=shift_step, status=status, stationary=stationary, continued_hold=continued,
        shift_recovery=recovery, hold=result["hold"], diagnostic=curve,
        final=result["live"], ema=result["ema"], seconds=result["seconds"],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in _SOURCE_FILES},
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                     threads=torch.get_num_threads(), device=str(device)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("scheduled", "constant"), default="constant")
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--diagnostic-every", type=int, default=50)
    parser.add_argument("--shift-step", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log", type=Path, help="one JSON line per observation (tail -f)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def log(row):
        print(json.dumps(row, sort_keys=True, allow_nan=False, default=float), flush=True)
    evidence = run_probe(mode=args.mode, steps=args.steps, diagnostic_every=args.diagnostic_every,
                         shift_step=args.shift_step, device=args.device, log=log, log_path=args.log)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False,
                                          default=float) + "\n")
    log({"event": "result", "status": evidence["status"],
         "final": {k: evidence["final"][k] for k in ("modes", "hq")},
         "ema": {k: evidence["ema"][k] for k in ("modes", "hq")},
         "seconds": evidence["seconds"], "output": str(args.output) if args.output else None})


if __name__ == "__main__":
    main()
