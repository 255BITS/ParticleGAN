"""Convergence must precede a disjoint hold; early dips cannot poison it."""
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('convergence_gate', Path(__file__).resolve().parents[1] /
    'reports/toy100/h_stability/convergence_gate.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ConvergenceGate = module.ConvergenceGate


def feed(gate, passing):
    return gate.observe(dict(step=gate.last_step+1, modes=8, hq=.95 if passing else .5))


def test_early_dips_are_allowed_and_confirmation_is_not_part_of_hold():
    gate = ConvergenceGate(confirmation=3, settling_budget=8, hold_budget=2)
    for value in (False, True, False, True, True, True):
        assert not feed(gate, value)
    assert gate.converged_step == 1206 and gate.hold_checks == 0
    assert not feed(gate, True)
    assert feed(gate, True)
    assert gate.status == 'PASS' and gate.settling_failures == 2


def test_first_confirmed_window_cannot_be_replaced_after_hold_failure():
    gate = ConvergenceGate(confirmation=2, settling_budget=5, hold_budget=2)
    for value in (True, True, False):
        feed(gate, value)
    assert gate.status == 'POST_CONVERGENCE_FAIL'
    assert gate.first_hold_failure == 1203
    with pytest.raises(RuntimeError, match='cannot restart'):
        feed(gate, True)


def test_unconverged_budget_is_not_a_stability_failure_or_pass():
    gate = ConvergenceGate(confirmation=3, settling_budget=5, hold_budget=2)
    for value in (True, True, False, True, True):
        feed(gate, value)
    assert gate.status == 'NOT_CONVERGED' and gate.hold_checks == 0


def test_confirmation_at_budget_boundary_still_gets_full_hold():
    gate = ConvergenceGate(confirmation=2, settling_budget=3, hold_budget=2)
    for value in (False, True, True, True, True):
        feed(gate, value)
    assert gate.status == 'PASS' and gate.hold_checks == 2


def test_missing_dense_observation_cannot_count_toward_convergence():
    gate = ConvergenceGate()
    with pytest.raises(ValueError, match='dense'):
        gate.observe(dict(step=1250, modes=8, hq=1.))


def test_good_quality_without_all_modes_does_not_converge():
    gate = ConvergenceGate(confirmation=1, settling_budget=1, hold_budget=1)
    gate.observe(dict(step=1201, modes=7, hq=1.))
    assert gate.status == 'NOT_CONVERGED'
