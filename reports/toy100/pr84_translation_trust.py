"""Stall reach, plus a trust bound on the common-mode translation of G's step.

The curvature-bounded G step is left intact, then split in output space:

- common mode: the mean displacement of the clean particle cloud;
- residual: every particle's motion relative to that mean.

Only the common mode is shrunk. It is removed by an output-bias correction, which
translates every particle by the same vector and does not change relative motion.
The stall-reach stencil, both curvature bounds, and the losses are unchanged.

The shrink comes from D's value along a pure translation of the pre-step cloud,
not from the length of the step. With ``v(a)`` the mean critic score of the cloud
shifted by ``a * t / ||t||``:

    s = (v(+L) - v(-L)) / (2 L)
    q = (v(+L) + v(-L) - 2 v(0)) / L^2

``s <= 0`` means D does not endorse the translation, and it is rejected.
``q < 0`` means D's score is concave along it, and the step is cut to the
quadratic peak ``-s/q``. Otherwise D is still rising through the whole probed
step and the translation is kept. A magnitude clip is not used: the same length
is kept or rejected according to this probe.
"""

from contextlib import contextmanager
import json
import math

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder


METHOD = "pr84_stall_reach_common_mode_trust"
_MIN_TRANSLATION = 1e-8


def translation_trust_factor(v0, v_plus, v_minus, length):
    """Fraction of a pure translation of ``length`` that D's probe endorses."""
    if not length > _MIN_TRANSLATION:
        return 1.
    slope = (v_plus - v_minus) / (2. * length)
    curvature = (v_plus + v_minus - 2. * v0) / (length * length)
    if not math.isfinite(slope) or not math.isfinite(curvature) or slope <= 0.:
        return 0.
    if curvature >= 0.:
        return 1.
    return min(1., max(0., (-slope / curvature) / length))


def _critic_module(critic):
    module = critic
    while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
        module = module.model
    return module if isinstance(module, SimpleMLPDiscriminator) else None


def _output_bias(generator):
    module = getattr(generator, "model", generator)
    last = None
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            last = layer
    if last is None or last.bias is None or int(last.out_features) != 2:
        return None
    return last.bias


class TranslationTrustRecorder(ReachRecorder):
    """Stall reach with the common-mode trust applied after G's curvature bound."""

    ramp = "stall"
    reach = REACH

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self._trans_seen = 0
        self._trans_shrunk = 0
        self._trans_rejected = 0

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        result = super().step(optimizer, ordinary_step, closure)
        if (not self.passthrough and self.phase == 2 and self.optimizers is not None
                and optimizer is self.optimizers[1] and "g" in self.row):
            self._bound_common_mode()
        return result

    def _mean_score(self, critic, points):
        saved = self._smooth_on
        self._smooth_on = False
        try:
            return float(critic(points).mean())
        finally:
            self._smooth_on = saved

    @torch.no_grad()
    def _bound_common_mode(self):
        local = self._local or {}
        generator, prior, critic = (local.get("generator"), local.get("prior"),
                                    local.get("critic"))
        if generator is None or prior is None or not hasattr(prior, "z") or critic is None:
            return
        bias = _output_bias(generator)
        critic = _critic_module(critic)
        if bias is None or critic is None:
            return
        clean = getattr(generator, "model", generator)
        opt_g = self.optimizers[1]
        proposed = [p.detach().clone() for p in self._params(opt_g)]
        y1 = clean(prior.z).detach()
        for param, saved in zip(self._params(opt_g), self.g_base):
            param.copy_(saved)
        y0 = clean(prior.z).detach()
        for param, saved in zip(self._params(opt_g), proposed):
            param.copy_(saved)
        if y0.ndim != 2 or y0.shape[-1] != 2 or y0.shape != y1.shape:
            return
        delta = y1 - y0
        translation = delta.mean(0)
        length = float(translation.norm())
        rms = float(delta.square().sum(1).mean().sqrt())
        if length <= _MIN_TRANSLATION:
            factor, slope, curvature = 1., 0., 0.
        else:
            v0 = self._mean_score(critic, y0)
            v_plus = self._mean_score(critic, y0 + translation)
            v_minus = self._mean_score(critic, y0 - translation)
            factor = translation_trust_factor(v0, v_plus, v_minus, length)
            slope = (v_plus - v_minus) / (2. * length)
            curvature = (v_plus + v_minus - 2. * v0) / (length * length)
            if factor < 1.:
                bias.add_((factor - 1.) * translation.to(dtype=bias.dtype))
        self._trans_seen += 1
        self._trans_shrunk += int(factor < 1.)
        self._trans_rejected += int(factor == 0.)
        self.row["g"].update(translation=length, translation_rms=rms,
                             translation_factor=factor, translation_slope=slope,
                             translation_curvature=curvature)
        step = self.outer_steps + 1
        if step % 50 == 0:
            print(json.dumps(dict(
                event="TRANS", step=step, translation=round(length, 4),
                rms=round(rms, 4), factor=round(factor, 3),
                slope=None if not math.isfinite(slope) else round(slope, 4),
                shrunk=self._trans_shrunk, rejected=self._trans_rejected,
                seen=self._trans_seen)), flush=True)

    def receipt(self):
        value = super().receipt()
        factors = [row["g"]["translation_factor"] for row in self.records
                   if "translation_factor" in row.get("g", {})]
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     translation_trust=True, translation_steps=len(factors),
                     translation_shrunk=sum(factor < 1. for factor in factors),
                     translation_rejected=sum(factor == 0. for factor in factors),
                     translation_factor_mean=(sum(factors) / len(factors) if factors else None))
        return value


@contextmanager
def pr84_translation_trust(*, task="mode_hold", start_step=0):
    original = base.SmoothedBothBoundRecorder
    base.SmoothedBothBoundRecorder = TranslationTrustRecorder
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "TranslationTrustRecorder", "pr84_translation_trust",
           "translation_trust_factor"]
