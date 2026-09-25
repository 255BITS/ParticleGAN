"""Observation-only convergence qualification, separate from frozen toy scores."""
from dataclasses import dataclass


@dataclass
class ConvergenceGate:
    confirmation: int = 200
    settling_budget: int = 4800
    hold_budget: int = 1200
    start_step: int = 1200

    def __post_init__(self):
        if not 0 < self.confirmation <= self.settling_budget or self.hold_budget <= 0:
            raise ValueError('invalid convergence or hold budget')
        self.last_step = self.start_step
        self.settling_checks = self.settling_failures = self.streak = 0
        self.longest_settling_streak = self.hold_checks = 0
        self.converged_step = self.first_hold_failure = None
        self.status = 'SETTLING'

    @property
    def done(self):
        return self.status in ('PASS', 'NOT_CONVERGED', 'POST_CONVERGENCE_FAIL')

    def observe(self, point):
        if self.done:
            raise RuntimeError('completed gate cannot restart or select a later window')
        if point['step'] != self.last_step + 1:
            raise ValueError('convergence requires consecutive dense observations')
        self.last_step = point['step']
        passing = point['modes'] == 8 and .90 <= point['hq'] <= 1.
        if self.converged_step is None:
            self.settling_checks += 1
            self.settling_failures += int(not passing)
            self.streak = self.streak + 1 if passing else 0
            self.longest_settling_streak = max(self.longest_settling_streak, self.streak)
            if self.streak == self.confirmation:
                self.converged_step = point['step']
                self.status = 'HOLDING'
            elif self.settling_checks == self.settling_budget:
                self.status = 'NOT_CONVERGED'
        else:
            self.hold_checks += 1
            if not passing:
                self.first_hold_failure = point['step']
                self.status = 'POST_CONVERGENCE_FAIL'
            elif self.hold_checks == self.hold_budget:
                self.status = 'PASS'
        return self.done

    def declaration(self):
        return dict(name='first_convergence_then_hold_v1', confirmation_checks=self.confirmation,
                    settling_budget=self.settling_budget, hold_budget=self.hold_budget,
                    start_step=self.start_step, modes=8, min_hq=.90,
                    convergence='first consecutive qualifying window; operational quality criterion',
                    hold='the next updates, disjoint from confirmation; no restart after failure',
                    training_feedback=False, frozen_toy_verdicts_unchanged=True)

    def summary(self):
        return dict(**self.declaration(), status=self.status, converged_step=self.converged_step,
                    settling_checks=self.settling_checks, settling_failures=self.settling_failures,
                    longest_settling_streak=self.longest_settling_streak,
                    hold_checks=self.hold_checks, first_hold_failure=self.first_hold_failure,
                    hold_budget_complete=self.hold_checks == self.hold_budget)
