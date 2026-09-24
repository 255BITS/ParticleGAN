"""Alternating PR84 with one fixed-step critic prediction for the G field.

D materializes its original bounded Adam step D*. Only G evaluates the
temporary opponent D_probe = D* + (D* - D0); D's queries and checkpoints see
D*. G/prior still have one Adam moment update and the existing own-curvature
bound. Prediction-active G base/proposal queries use one frozen stencil width.
This changes game dynamics, not the loss, target, nominal rates or training
clock. It is a scratch candidate, with no convergence or gate claim.
"""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import math
from unittest.mock import patch

import torch

from reports.toy100 import pr84_smoothed_candidate as frozen


METHOD = "pr84_alternating_g_opponent_prediction"


def predicted_opponent(base, materialized):
    """Exactly one extrapolated opponent step, without changing either input."""
    if len(base) != len(materialized):
        raise ValueError("opponent parameter collections differ")
    result = []
    for before, after in zip(base, materialized):
        if before.shape != after.shape or before.dtype != after.dtype or before.device != after.device:
            raise ValueError("opponent parameter metadata differs")
        predicted = after + (after - before)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("nonfinite extrapolated opponent")
        result.append(predicted)
    return result


class OpponentPredictionRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, prediction=True):
        super().__init__(start_step=start_step)
        self.prediction = bool(prediction)
        self._probe_active = False
        self._materialized_d = None
        self._probe_d = None
        self._probe_stencil = None
        self.prediction_queries = 0
        self.prediction_restores = 0
        self.prediction_records = []
        self._current_opponent_g = None

    @torch.no_grad()
    def restore_materialized_critic(self):
        if self._probe_active:
            for parameter, value in zip(self._params(self.optimizers[0]), self._materialized_d):
                parameter.copy_(value)
            self._probe_active = False
            self.prediction_restores += 1

    @torch.no_grad()
    def _arm_smoothed_critic(self):
        if not self.prediction or self.passthrough or self.phase not in (1, 2):
            return super()._arm_smoothed_critic()
        if self._probe_active:
            raise RuntimeError("D field query began while an opponent probe was installed")
        if self.phase == 1:
            self._materialized_d = [p.detach().clone() for p in self._params(self.optimizers[0])]
            if any(not torch.equal(p, saved) for p, saved in zip(self._materialized_d, self.d_star)):
                raise RuntimeError("G prediction did not start from the accepted D point")
            self._probe_d = predicted_opponent(self.d0, self._materialized_d)
        elif self._probe_d is None or self._probe_stencil is None:
            raise RuntimeError("proposal G query lacks its fixed predicted opponent")
        if any(not torch.equal(p, saved) for p, saved in
               zip(self._params(self.optimizers[0]), self._materialized_d)):
            raise RuntimeError("materialized D changed between G field evaluations")
        for parameter, value in zip(self._params(self.optimizers[0]), self._probe_d):
            parameter.copy_(value)
        self._probe_active = True
        self.prediction_queries += 1
        if self.phase == 1:
            super()._arm_smoothed_critic()
            self._probe_stencil = (self._smooth_on, self._smooth_width)
        else:
            # Same predicted D and same detached smoothing width for the G
            # own-field secant. Disabled prediction preserves original PR84.
            self._smooth_on, self._smooth_width = self._probe_stencil

    def phases(self, step, opt_d, opt_g, local):
        active = self.enabled and step >= self.start_step and self.prediction
        if self._probe_active:
            raise RuntimeError("previous update left an opponent probe installed")
        self._probe_d = self._probe_stencil = self._current_opponent_g = None
        try:
            for phase in super().phases(step, opt_d, opt_g, local):
                yield phase
                if self._probe_active:
                    raise RuntimeError("G optimizer callback did not restore materialized D")
        finally:
            self.restore_materialized_critic()
        if active:
            displacement = sum(float((new - old).double().square().sum())
                               for old, new in zip(self.d0, self._materialized_d)) ** .5
            row = dict(outer_step=self.outer_steps, critic_step_norm=displacement,
                       predicted_critic_extra_displacement_norm=displacement,
                       frozen_g_stencil_width=self._probe_stencil[1],
                       g_prediction_queries=2, additional_field_queries=0)
            # Phase zero's G gradient is discarded by the original adapter.
            # It sees ordinary D1; when the D bound is inactive, this is the
            # exact current-opponent comparator for the prediction G gradient.
            if self._current_opponent_g is not None:
                old_norm = sum(float(g.double().square().sum()) for g in self._current_opponent_g)
                new_norm = sum(float(g.double().square().sum()) for g in self.gg0)
                change = sum(float((a - b).double().square().sum())
                             for a, b in zip(self.gg0, self._current_opponent_g))
                dot = sum(float((a.double() * b.double()).sum())
                          for a, b in zip(self.gg0, self._current_opponent_g))
                row.update(current_opponent_gradient_scope="phase0 at ordinary D1; equal to D* when D bound inactive",
                           g_gradient_relative_change=math.sqrt(change / old_norm) if old_norm else None,
                           g_gradient_cosine=dot / math.sqrt(old_norm * new_norm) if old_norm and new_norm else None)
            self.prediction_records.append(row)
            self.records[-1]["opponent_prediction"] = row

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        active_g = (not self.passthrough and self.prediction and self.optimizers is not None
                    and optimizer is self.optimizers[1])
        if active_g and self.phase == 0:
            self._current_opponent_g = [p.grad.detach().clone() for p in self._params(optimizer)]
        try:
            return super().step(optimizer, ordinary_step, closure)
        finally:
            if active_g:
                self.restore_materialized_critic()

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD, scratch_optimizer_policy=METHOD,
                      shared_gate_eligible=False, prediction=bool(self.prediction),
                      opponent_rule="D_probe=D_star+(D_star-D0); final D=D_star",
                      prediction_multiplier=2.,
                      active_stencil_policy="freeze phase1 width through phase2",
                      disabled_stencil_policy="unchanged original PR84 recomputation",
                      prediction_queries=self.prediction_queries,
                      prediction_restores=self.prediction_restores,
                      additional_field_queries=0,
                      prediction_records=self.prediction_records,
                      adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def pr84_opponent_prediction(*, task="mode_hold", start_step=0, prediction=True):
    def factory(*, start_step=0):
        return OpponentPredictionRecorder(start_step=start_step, prediction=prediction)
    with patch.object(frozen, "SmoothedBothBoundRecorder", factory):
        with frozen.pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
            try:
                yield recorder, source
            finally:
                recorder.restore_materialized_critic()
