"""lrfree custom22: VERBATIM function copies of the frozen verdict (PR #155 checkout).

``score_metrics`` from benchmarks/locked_shared/baseline.py; ``requirements`` and ``test_verdict`` from
benchmarks/transfer_suite/protocol.py. The function bodies below are byte-identical to the originals
(``SOURCES`` holds the sha256 of each function's source text; harness/tests checks them against the
checkout). Only the module-level glue differs: ``sustained`` is the verbatim observation.py copy and
``baseline.score_metrics`` resolves to the copy here.
"""
import math
from types import SimpleNamespace

from benchmarks.locked_shared.observation import sustained

SOURCES = {
 "benchmarks/locked_shared/baseline.py::score_metrics": "c587421a74171ae411997a93cc6725d6fdf115a94f105f6b273baa46e75fbdfc",
 "benchmarks/transfer_suite/protocol.py::requirements": "e723d2398f2d12f078065b569b4e57cf5aed577bba55723f0685ee2c5a2d7268",
 "benchmarks/transfer_suite/protocol.py::test_verdict": "320ac2f0d7889226b45f745317545b74b0d07dbc76e5517172800a7123557e68"
}


def score_metrics(values, requirements):
    cells = []
    for name, op, threshold in requirements:
        value = values.get(name)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        ok = numeric and (value >= threshold if op == ">=" else value <= threshold)
        cells.append({"metric": name, "value": value if numeric else None, "op": op, "threshold": threshold,
                      "status": "PASS" if ok else "FAIL" if numeric else "MISSING",
                      "margin": (value - threshold if op == ">=" else threshold - value) if numeric else None})
    return cells


def requirements(spec):
    """Image hosts additionally declare measurement settings in their native schema."""
    thresholds = spec["thresholds"]
    if isinstance(thresholds, dict):
        return [["modes", ">=", thresholds["modes"]], ["hq", ">=", thresholds["hq_min"]]]
    return thresholds


def test_verdict(spec, result):
    """Recompute sustained success from complete live curves, never trust a stamp."""
    if result is None:
        return dict(status="MISSING", attempted=False, passed=False, confirmation_fraction=2., shortfall=2.)
    if result.get("error"):
        return dict(status="ERROR", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    observations = result.get("observations", result.get("curve", []))
    steps = sorted({math.ceil(i * spec["steps"] / 24) for i in range(1, 25)})
    try:
        convergence = sustained(observations, requirements(spec), expected_steps=steps)
    except (KeyError, TypeError, ValueError):
        return dict(status="INVALID", attempted=True, passed=False, confirmation_fraction=2., shortfall=2.)
    cells = baseline.score_metrics(result.get("live", {}), requirements(spec))
    passed = (convergence["confirmed_step"] is not None
              and all(c["status"] == "PASS" for c in cells))
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    deficits = [2. if c["margin"] is None else min(2., max(0., -c["margin"]) / (abs(c["threshold"]) or 1.))
                for c in cells]
    return dict(status=status, attempted=True, passed=passed, metrics=cells,
                convergence=convergence,
                shortfall=sum(deficits) / len(deficits) if convergence["complete"] else 2.,
                confirmation_fraction=convergence["confirmed_step"] / spec["steps"] if passed else 2.)


baseline = SimpleNamespace(score_metrics=score_metrics)
