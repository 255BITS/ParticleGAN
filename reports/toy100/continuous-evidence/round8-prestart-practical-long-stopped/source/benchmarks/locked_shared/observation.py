"""Optional, RNG-isolated measurements shared by the extracted hosts."""

from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

import torch
from particlegan import learning_rate_scale

_active = ContextVar("behavior_observer", default=None)
OBSERVATIONS = 24
MIN_STABLE_CHECKS = 5


def sustained(curve, requirements, *, expected_steps, minimum=MIN_STABLE_CHECKS):
    """Find a passing suffix, never count a transient pass as convergence.

    Only recorded observations are certified. First/confirmation times include
    setup and measurement overhead, and are not inferred for historical rows.
    """
    def passes(point):
        for key, op, bound in requirements:
            value = point.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return False
            if not (value >= bound if op == ">=" else value <= bound):
                return False
        return True
    if minimum < 2:
        raise ValueError("minimum must be at least two observations")
    if any(b["step"] <= a["step"] for a, b in zip(curve, curve[1:])):
        raise ValueError("observations must have unique increasing steps")
    passing = [passes(p) for p in curve]
    complete = [p["step"] for p in curve] == sorted(expected_steps)
    first = next((p for p, ok in zip(curve, passing) if ok), None)
    start = len(curve)
    while start and passing[start - 1]:
        start -= 1
    suffix = curve[start:]
    stable = complete and len(suffix) >= minimum
    return {"complete": complete, "observations": len(curve), "passing_observations": sum(passing),
            "minimum_stable_checks": minimum, "passing_suffix": len(suffix),
            "first_pass_step": first["step"] if first else None,
            "stable_from_step": suffix[0]["step"] if stable else None,
            "stable_from_seconds": suffix[0].get("seconds") if stable else None,
            "confirmed_step": suffix[minimum - 1]["step"] if stable else None,
            "confirmed_seconds": suffix[minimum - 1].get("seconds") if stable else None}


class Recorder:
    def __init__(self, total_steps, *, schedule="host", start=0.6, floor=0.05):
        self.steps = {math.ceil(i * total_steps / OBSERVATIONS) for i in range(1, OBSERVATIONS + 1)}
        self.total_steps = total_steps
        self.schedule = schedule
        self.start = start
        self.floor = floor
        self.base_rates = {}
        self.started = time.monotonic()
        self.curve = []

    def record(self, step, measure):
        if step not in self.steps:
            return
        # Some host evaluators consume the global RNG. Preserve its exact state.
        with torch.random.fork_rng(devices=[]):
            values = measure()
        self.curve.append({**values, "step": step, "seconds": time.monotonic() - self.started})


@contextmanager
def recording(total_steps, **options):
    recorder = Recorder(total_steps, **options)
    token = _active.set(recorder)
    try:
        yield recorder
    finally:
        _active.reset(token)


def checkpoint(step, measure):
    recorder = _active.get()
    if recorder is not None:
        recorder.record(step, measure)


def schedule_optimizer(optimizer, completed_updates):
    """Replace host schedules using initial group LRs; never compound decay."""
    recorder = _active.get()
    if recorder is None or recorder.schedule == "host":
        return
    if optimizer not in recorder.base_rates:
        recorder.base_rates[optimizer] = [group["lr"] for group in optimizer.param_groups]
    scale = learning_rate_scale(completed_updates, recorder.total_steps, recorder.start, recorder.floor)
    for group, initial in zip(optimizer.param_groups, recorder.base_rates[optimizer]):
        group["lr"] = initial * scale
