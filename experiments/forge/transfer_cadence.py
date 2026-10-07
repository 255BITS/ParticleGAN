"""Explicit task cadence with the unchanged transfer bounds and suffix gate."""
import math

from benchmarks.locked_shared.observation import sustained
from benchmarks.transfer_suite.protocol import test_verdict as legacy_verdict


def test_verdict(spec, result):
    count = spec["observations"]
    if type(count) is not int or not 5 <= count <= spec["steps"]:
        raise ValueError("declared observations must be between five and the update budget")
    # Reuse the exact live metric grading and deficit normalization. The new
    # evaluator identity changes only the expected measurement checkpoints.
    grade = legacy_verdict(spec, result)
    if result is None or result.get("error"):
        return grade
    steps = sorted({math.ceil(i * spec["steps"] / count) for i in range(1, count + 1)})
    convergence = sustained(result.get("observations", result.get("curve", [])),
                            spec["thresholds"], expected_steps=steps)
    cells = grade["metrics"]
    passed = convergence["confirmed_step"] is not None and all(cell["status"] == "PASS" for cell in cells)
    deficits = [2. if cell["margin"] is None else min(2., max(0., -cell["margin"]) / (abs(cell["threshold"]) or 1.))
                for cell in cells]
    grade.update(status="PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE",
                 passed=passed, convergence=convergence,
                 shortfall=sum(deficits) / len(deficits) if convergence["complete"] else 2.,
                 confirmation_fraction=convergence["confirmed_step"] / spec["steps"] if passed else 2.)
    return grade
