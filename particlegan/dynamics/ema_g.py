"""Exponential moving average of the generator and the particles.

Yazici et al. 2019, "The Unusual Effectiveness of Averaging in GAN Training":
at a fixed learning rate the iterates orbit a solution, and the average of the
generator sits on that solution. This keeps that average at decay 0.999 and
scores it beside the live weights. The live Adam updates are unchanged.

``K3P_DYNAMICS=ema_g`` is that shadow. The probe's own gate stays on the live
weights. A sidecar records both, and the screen scores the average separately.
An EMA-only pass is labeled EMA-scored.

Declared alternative, written before any run of it: ``K3P_DYNAMICS=ema_g_fake``.
The same average is also the fake the critic trains on. Reason: if the critic
only ever sees the orbiting iterate, it keeps scoring the moving point and
never has to defend the center the average occupies. The generator step stays
on the live weights. Decay stays 0.999. This is not the already-failed
critic-fake average at decay 0.995. No coefficient is introduced and none is swept.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import sys
from pathlib import Path

DECAY = 0.999
NAMES = ("ema_g", "ema_g_fake")

_points: list[dict] = []
_seen: set[int] = set()
_installed = False


def name() -> str | None:
    value = os.environ.get("K3P_DYNAMICS")
    return value if value in NAMES else None


def active() -> bool:
    return name() is not None


def critic_fakes() -> bool:
    """True only for the predeclared alternative: D's fakes come from the average."""
    return name() == "ema_g_fake"


def decay_or(default: float) -> float:
    """0.999 while this mechanism is on; the caller's decay otherwise."""
    return DECAY if active() else default


def reset() -> None:
    """Drop recorded gate points. Tests use this; training does not."""
    _points.clear()
    _seen.clear()


def points() -> list[dict]:
    return _points


def install() -> None:
    """Log the setting and write the live/EMA curve next to ``--output`` at exit."""
    global _installed
    if _installed:
        return
    _installed = True
    which = name()
    if which is None:
        return
    atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": which,
        "decay": DECAY,
        "setting": ("shadow average of G and particles, gates also scored on the average"
                    if which == "ema_g" else
                    "shadow average of G and particles; D's fakes are that average"),
    }), flush=True)


def score(step: int, generator, prior, ema_params, ema_z, measure):
    """Return the live measurement. Also record the averaged model when active.

    Flag off calls ``measure`` once and does not touch the average. Flag on
    measures live, swaps in the average, measures again, and restores the live
    weights before returning the live row. The existing gate therefore stays
    on the live model; the sidecar holds both.
    """
    if not active():
        return measure()
    live = measure()
    if step not in _seen:
        saved_g, saved_z = _capture(generator, prior)
        _swap(generator, prior, ema_params, ema_z)
        try:
            averaged = measure()
        finally:
            _swap(generator, prior, saved_g, [saved_z])
        _seen.add(step)
        row = {
            "step": step,
            "live_modes": int(live["modes"]),
            "live_hq": float(live["hq"]),
            "live_pass": _passes(live),
            "ema_modes": int(averaged["modes"]),
            "ema_hq": float(averaged["hq"]),
            "ema_pass": _passes(averaged),
        }
        _points.append(row)
        if step <= 10 or step % 100 == 0:
            print(json.dumps({"event": "ema_g", **row}), flush=True)
    return live


def averaged_fake(generator, prior, ema_params, ema_z, batch: int, stream):
    """One detached fake from the averaged generator and averaged particles.

    The training stream advances once, the same as ``prior.sample`` then
    ``generator``. Live weights are restored before return.
    """
    saved_g, saved_z = _capture(generator, prior)
    _swap(generator, prior, ema_params, ema_z)
    try:
        latent, _ = prior.sample(batch, generator=stream)
        return generator(latent).detach()
    finally:
        _swap(generator, prior, saved_g, [saved_z])


def _capture(generator, prior):
    return [p.detach().clone() for p in generator.parameters()], prior.z.detach().clone()


def _swap(generator, prior, params, ema_z) -> None:
    import torch
    z = ema_z[0] if isinstance(ema_z, list) else ema_z
    with torch.no_grad():
        for parameter, value in zip(generator.parameters(), params):
            parameter.copy_(value)
        prior.z.copy_(z)


def _passes(row) -> bool:
    return row["modes"] == 8 and row["hq"] >= 0.90


def _kind_rows(points, kind: str, *, after: int | None = None, every: int | None = None):
    rows = []
    seen = set()
    for point in points:
        step = point["step"]
        if step in seen:
            continue
        if after is not None and step <= after:
            continue
        if every is not None and step % every:
            continue
        seen.add(step)
        rows.append({
            "step": step,
            "modes": point[f"{kind}_modes"],
            "hq": point[f"{kind}_hq"],
            "pass": point[f"{kind}_pass"],
        })
    rows.sort(key=lambda row: row["step"])
    return rows


def ring_summary(points, kind: str, *, steps: int = 1200) -> dict:
    """24-check ring on one column. Pass matches the mode_hold sustained rule."""
    expected = sorted({math.ceil(i * steps / 24) for i in range(1, 25)})
    by_step = {row["step"]: row for row in _kind_rows(points, kind)}
    if any(step not in by_step for step in expected):
        return {"status": "MISSING", "pass": False, "modes": None, "hq": None, "passing_suffix": None}
    curve = [by_step[step] for step in expected]
    passing = [row["pass"] for row in curve]
    start = len(curve)
    while start and passing[start - 1]:
        start -= 1
    suffix = len(curve) - start
    final = curve[-1]
    passed = suffix >= 5 and bool(final["pass"])
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "modes": final["modes"],
        "hq": final["hq"],
        "passing_suffix": suffix,
    }


def hold_summary(points, kind: str) -> dict:
    """First 200 consecutive passes after step 1200, then 1200 more. Live rule."""
    rows = _kind_rows(points, kind, after=1200)
    gate = _Convergence()
    if not rows:
        return {**gate.as_dict(), "pass": False}
    expect = 1201
    for row in rows:
        if row["step"] != expect:
            return {"status": "GAP", "pass": False, "hold_checks": gate.hold_checks,
                    "gap_at": row["step"], "expected": expect}
        if gate.done:
            break
        gate.observe(row)
        expect += 1
    out = gate.as_dict()
    out["pass"] = out["status"] == "PASS" and out["hold_checks"] == 1200
    return out


def stay_summary(points, kind: str, *, start: int = 1200, end: int = 2400, every: int = 10) -> dict:
    """Stay window: every ``every`` steps in (start, end], 120 checks at the defaults."""
    rows = _kind_rows(points, kind, after=start, every=every)
    rows = [row for row in rows if row["step"] <= end]
    checks = len(rows)
    passing = sum(row["pass"] for row in rows)
    passed = checks == 120 and passing == 120 and all(row["pass"] for row in rows)
    return {"stay": f"{passing}/{checks}", "pass": passed, "checks": checks, "passing_checks": passing}


class _Convergence:
    """Same rules as the probe's ConvergenceGate, without importing the probe."""

    def __init__(self):
        self.last_step = 1200
        self.settling_checks = self.settling_failures = self.streak = 0
        self.hold_checks = 0
        self.converged_step = None
        self.status = "SETTLING"

    @property
    def done(self) -> bool:
        return self.status in ("PASS", "NOT_CONVERGED", "POST_CONVERGENCE_FAIL")

    def observe(self, point) -> None:
        self.last_step = point["step"]
        passing = point["modes"] == 8 and 0.90 <= point["hq"] <= 1.0
        if self.converged_step is None:
            self.settling_checks += 1
            self.settling_failures += int(not passing)
            self.streak = self.streak + 1 if passing else 0
            if self.streak == 200:
                self.converged_step = point["step"]
                self.status = "HOLDING"
            elif self.settling_checks == 4800:
                self.status = "NOT_CONVERGED"
        else:
            self.hold_checks += 1
            if not passing:
                self.status = "POST_CONVERGENCE_FAIL"
            elif self.hold_checks == 1200:
                self.status = "PASS"

    def as_dict(self) -> dict:
        return {"status": self.status, "hold_checks": self.hold_checks,
                "converged_step": self.converged_step, "settling_checks": self.settling_checks}


def _output_dir():
    if "--output" not in sys.argv:
        return None
    return Path(sys.argv[sys.argv.index("--output") + 1])


def _emit() -> None:
    payload = {"mechanism": name(), "decay": DECAY, "critic_fakes": critic_fakes(),
               "points": len(_points)}
    print(json.dumps({"event": "dynamics_receipt", **payload}), flush=True)
    dest = _output_dir()
    if dest is None:
        return
    dest.mkdir(parents=True, exist_ok=True)
    body = {"mechanism": name(), "decay": DECAY, "critic_fakes": critic_fakes(), "points": _points}
    (dest / "ema_g_scores.json").write_text(json.dumps(body) + "\n")
