"""Bounded critic-fit policy that rejects a later nonfinite inner trial.

Finite-path L-BFGS arithmetic and budgets are unchanged. A later evaluated
trial with nonfinite loss/gradient ends this single attempt and restores its
lowest-loss finite evaluated point. An initially nonfinite field remains an
error. Failed trials count toward the same hard80 closure budget; there is no
retry, reset, additional attempt or quality-based selection. The original
fitting helper and all previously frozen adapters remain unchanged on disk.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100 import pr84_critic_refinement_cold as cold


METHOD = 'pr84_critic_refinement_with_later_nonfinite_trial_rejection'


class InitialPointNonfinite(FloatingPointError):
    """The first objective/gradient evaluation is invalid; no fallback exists."""
    def __init__(self, audit):
        super().__init__('initial critic-fit loss or gradient is nonfinite')
        self.audit = audit


class _RejectedNonfiniteTrial(Exception):
    pass


def _json_scalar(value):
    number = float(value.detach())
    return number if math.isfinite(number) else None


def finite_trial_fit(critic, bank, gan, regularizer, step, metric):
    """One original40/80 fit; stop on a later invalid numerical trial."""
    optimizer = torch.optim.LBFGS(critic.parameters(), lr=1., max_iter=fit.MAX_ITER,
        max_eval=fit.MAX_CLOSURES, tolerance_grad=1e-7, tolerance_change=1e-12,
        history_size=10, line_search_fn='strong_wolfe')
    records, invalid_trials = [], []
    best, best_loss, attempted = None, float('inf'), 0

    def closure():
        nonlocal best, best_loss, attempted
        if attempted >= fit.MAX_CLOSURES:
            raise fit.ClosureBudget()
        attempted += 1
        optimizer.zero_grad()
        loss, logistic, penalty = fit.d_loss(critic, bank, gan, regularizer, step)
        loss.backward()
        gradient = [parameter.grad.detach().clone() for parameter in critic.parameters()]
        value = float(loss.detach())
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradient):
            audit = dict(closure=attempted, total_loss=_json_scalar(loss),
                         logistic_loss=_json_scalar(logistic), penalty=_json_scalar(penalty),
                         loss_finite=bool(torch.isfinite(loss)),
                         nonfinite_gradient_elements=sum(int((~torch.isfinite(g)).sum()) for g in gradient),
                         finite_parameter_tensors=sum(int(torch.isfinite(p).all()) for p in critic.parameters()),
                         parameter_tensors=len(list(critic.parameters())),
                         policy='end this attempt; restore best finite point; no retry')
            invalid_trials.append(audit)
            if best is None:
                raise InitialPointNonfinite(audit)
            raise _RejectedNonfiniteTrial()
        records.append(dict(closure=attempted, total_loss=value, logistic_loss=float(logistic.detach()),
                            penalty=float(penalty.detach()), gradient=fit.norm_receipt(gradient, metric)))
        if value < best_loss:
            best_loss = value
            best = deepcopy(critic.state_dict())
        return loss

    exhausted = rejected = False
    started = time.perf_counter()
    try:
        optimizer.step(closure)
    except fit.ClosureBudget:
        exhausted = True
    except _RejectedNonfiniteTrial:
        rejected = True
    # InitialPointNonfinite and unrelated exceptions deliberately propagate.
    if best is None:
        raise RuntimeError('local critic fit evaluated no finite point')
    critic.load_state_dict(best)
    if rejected:
        # A rejected trial can leave NaNs in leaf .grad even after restoring
        # finite weights. They are not a usable field at Dhat. Clear only this
        # invalid path; the host's later D block recomputes gradients normally.
        # This has no optimizer/moment update and adds no field query.
        for parameter in critic.parameters():
            parameter.grad = None
    iterations = int(next(iter(optimizer.state.values())).get('n_iter', 0))
    return dict(closure_calls=attempted, iterations=iterations,
                closure_budget_exhausted=exhausted,
                selection='lowest total fixed-bank loss among evaluated points; no quality selection',
                seconds=time.perf_counter()-started, records=records,
                finite_closure_calls=len(records), nonfinite_closure_calls=len(invalid_trials),
                nonfinite_trial_rejected=rejected, invalid_trials=invalid_trials,
                invalid_trial_gradients_cleared=rejected,
                extra_attempts=0, failed_trials_count_against_budget=True)


class FiniteCriticRefinementRecorder(cold.ColdCriticRefinementRecorder):
    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
            inner_solver_policy='stop later nonfinite trial and restore best finite evaluated training-loss point',
            initial_nonfinite_policy='ERROR; no best finite point exists',
            later_nonfinite_rejections=sum(bool(row['nonfinite_trial_rejected'])
                                           for row in self.refinement_records),
            finite_fit_gradient_evaluations=sum(row['finite_closure_calls'] for row in self.refinement_records),
            nonfinite_fit_gradient_evaluations=sum(row['nonfinite_closure_calls'] for row in self.refinement_records),
            original_fitting_source_sha256=hashlib.sha256(Path(fit.__file__).read_bytes()).hexdigest(),
            cold_adapter_source_sha256=hashlib.sha256(Path(cold.__file__).read_bytes()).hexdigest(),
            fit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            solver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return value


@contextmanager
def pr84_critic_refinement_finite(*, task='mode_hold', start_step=0, refinement=True):
    with patch.object(fit, 'relax', finite_trial_fit), \
         patch.object(cold, 'ColdCriticRefinementRecorder', FiniteCriticRefinementRecorder):
        with cold.pr84_critic_refinement_cold(task=task, start_step=start_step,
                                             refinement=refinement) as value:
            yield value
