"""PR84 with a G critic stencil whose reach follows D's slope utilisation.

PR84 sets G's five-point stencil width to ``min(.15, .5 / s)``, where ``s`` is
the critic's RMS input slope on the clean particles. With the host's b_cap
slope limit ``kappa = 1`` the .15 cap binds on every update, so the width never
adapts. Here the width is ``max(.15, (.5 / kappa) * min(s / kappa, kappa / s))``.
When D is well below its slope limit (``s <= .3``: an indistinguishable,
covered cloud) the width is PR84's .15 and the update is PR84's. As D saturates
its Lipschitz budget (still separating, W1-like field), G reads the critic
over up to .5, the distance D needs at full slope to move .5 logit, which is
PR84's own uncapped width at ``s = kappa``. D, both curvature bounds and the
game losses are unchanged. No coverage, assignment, likelihood or clip term.
"""

from contextlib import contextmanager
import math
import os

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from benchmarks.locked_shared.mode_hold import SIGMA, sample_ring
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.alternating_curvature_scratch import _rho


METHOD = "pr84_slope_utilisation_reach"
B_CAP_SLOPE = 1.
REACH = .5


REST_UTILISATION = .3
SATURATED_UTILISATION = .6
STALL_TRUST = .1
STALL_WINDOW = 50

# One recovery width, between stall reach .5 and the killed reach 1.0. Not swept.
RECOVERY_REACH = .75
# Critic-value gap (particles minus a fixed real probe) must fall this far
# below its short EMA. Read off the stall-reach dip trace; not retuned after a kill.
SUPPORT_DROP = .15
SUPPORT_ALPHA = .2
SUPPORT_PROBE_N = 128
SUPPORT_PROBE_SEED = 0


def reach_width(sharpness, reach=REACH, ramp="peak"):
    """``peak``: ``.5·min(u, 1/u)``. ``saturating``: PR84 until u = .3, .5 from u = .6."""
    utilisation = sharpness / B_CAP_SLOPE
    if ramp == "saturating":
        span = (utilisation - REST_UTILISATION) / (SATURATED_UTILISATION - REST_UTILISATION)
        low = base.SMOOTH_WIDTH_CAP
        return low + (reach / B_CAP_SLOPE - low) * min(1., max(0., span))
    utilisation = min(utilisation, 1 / utilisation)
    return max(base.SMOOTH_WIDTH_CAP, reach / B_CAP_SLOPE * utilisation)


def note_support(ema, support, drop=SUPPORT_DROP, alpha=SUPPORT_ALPHA):
    """One short-EMA step. A hard dip is a sharp fall, not G's curvature."""
    if support is None or not math.isfinite(support):
        return False, ema
    if ema is None or not math.isfinite(ema):
        return False, support
    return support <= ema - drop, (1. - alpha) * ema + alpha * support


def recovery_width(sharpness, *, stalled, dip, reach=REACH):
    """Stall reach, or one fixed wider stencil while a hard dip is on."""
    if dip:
        return RECOVERY_REACH
    if stalled:
        return reach / B_CAP_SLOPE
    return reach_width(sharpness, reach)


class ReachRecorder(base.SmoothedBothBoundRecorder):
    reach = REACH
    ramp = "peak"
    game_bound = False
    game_steps = 1

    def _arm_smoothed_critic(self):
        super()._arm_smoothed_critic()
        if self._smooth_on:
            sharpness = self.row["critic_sharpness"]
            dip = self._recovery_dip() if self.ramp == "recovery" else False
            if self.ramp in ("stall", "recovery"):
                self._smooth_width = recovery_width(sharpness, stalled=self._stalled(sharpness), dip=dip,
                                                    reach=self.reach)
            else:
                self._smooth_width = reach_width(sharpness, self.reach, self.ramp)
            self.row["critic_width"] = self._smooth_width
            if self.ramp == "recovery":
                self.row["critic_support"] = getattr(self, "_support_gap", None)
                self.row["support_ema"] = getattr(self, "_support_ema", None)
                self.row["recovery"] = bool(dip)

    def _recovery_dip(self):
        """Hard dip: critic-value support of the clean particles falls sharply.

        Support is the mean critic gap between clean particles and one fixed
        real probe. The short EMA advances once per outer step. G's curvature
        and trust factor are not inputs. A missing probe leaves the dip off.
        """
        if self.phase not in (0, None) and getattr(self, "_dip_token", None) == self.outer_steps:
            return bool(getattr(self, "_recovery", False))
        self._dip_token = self.outer_steps
        support = self._critic_support_gap()
        self._support_gap = support
        self._recovery, self._support_ema = note_support(getattr(self, "_support_ema", None), support)
        return self._recovery

    @torch.no_grad()
    def _critic_support_gap(self):
        local = self._local or {}
        means, critic = local.get("means"), local.get("critic")
        generator, prior = local.get("generator"), local.get("prior")
        if means is None or critic is None or generator is None or prior is None or not hasattr(prior, "z"):
            return None
        module = critic
        while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
            module = module.model
        if not isinstance(module, SimpleMLPDiscriminator):
            return None
        points = getattr(generator, "model", generator)(prior.z).detach()
        if points.ndim != 2 or points.shape[-1] != 2:
            return None
        probe = torch.Generator()
        probe.manual_seed(SUPPORT_PROBE_SEED)
        real = sample_ring(means, SUPPORT_PROBE_N, SIGMA, probe)
        armed = self._smooth_on
        self._smooth_on = False
        try:
            gap = float(module(points).mean() - module(real).mean())
        finally:
            self._smooth_on = armed
        return gap if math.isfinite(gap) else None

    def _stalled(self, sharpness):
        """D near its slope limit while G's own trust region has kept G nearly still."""
        recent = self.records[-STALL_WINDOW:]
        if sharpness / B_CAP_SLOPE < SATURATED_UTILISATION or not recent:
            return False
        return sum(row["g"]["factor"] for row in recent) / len(recent) <= STALL_TRUST

    def phases(self, step, opt_d, opt_g, local):
        if not self.game_bound or not self.enabled or step < self.start_step:
            yield from super().phases(step, opt_d, opt_g, local)
            return
        # BothBoundRecorder.phases with one more replay: D answers G's proposal.
        self._local = local
        if self.optimizers is None:
            self.optimizers = (opt_d, opt_g)
            self.rows = {opt: dict(role=role, calls=0) for role, opt in zip(("d", "g"), self.optimizers)}
        streams = [v for v in local.values() if isinstance(v, torch.Generator)]
        policy = local.get("noise_policy")
        if policy is not None:
            streams.extend(v for name in ("input_stream", "output_stream")
                           if isinstance((v := getattr(policy, name, None)), torch.Generator))
        streams = list({id(s): s for s in streams}.values())
        buffers = [(b, b.detach().clone()) for name in ("generator", "critic", "prior")
                   if isinstance((m := local.get(name)), torch.nn.Module) for b in m.buffers()]
        self.d0 = [p.detach().clone() for p in self._params(opt_d)]
        self.g_base = [p.detach().clone() for p in self._params(opt_g)]
        rng_before = self._rng(streams)
        self.advantage = None
        self.row = dict(outer_step=self.outer_steps + 1)
        rng_after = None
        for phase in range(3 + self.game_steps):
            if phase:
                self._set_rng(streams, rng_before)
                with torch.no_grad():
                    for b, saved in buffers:
                        b.copy_(saved)
            self.phase = phase
            self._smooth_on = False
            yield phase
            state = self._rng(streams)
            if rng_after is None:
                rng_after = state
            elif not all(torch.equal(a, b) for a, b in zip(rng_after, state)):
                raise RuntimeError("replayed block consumed a different RNG pattern")
            else:
                self.rng_replay_verified += 1
        self.row["critic_advantage"] = self.advantage
        self.row["gate_open"] = False
        self.row["rho"], self.row["factor"] = self.row["g"]["rho"], self.row["g"]["factor"]
        self.records.append(self.row)
        self._record_trace(local)
        self.phase = None
        self.outer_steps += 1
        if self.accounting is not None:
            self.accounting(self.rows[opt_d]["calls"], self.outer_steps)

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if (not self.game_bound or self.passthrough or self.phase is None or self.phase < 2
                or (self.phase == 2 and optimizer is self.optimizers[0])):
            return super().step(optimizer, ordinary_step, closure)
        opt_d, opt_g = self.optimizers
        self.rows[optimizer]["calls"] += 1
        grads = [p.grad.detach().clone() for p in self._params(opt_g)] if optimizer is opt_g else None
        if self.phase == 2:
            self._rho_own = _rho(self.g_base, self.g1, self.gg0, grads, self.metric_g)
            return None
        if optimizer is opt_d:
            if self.phase == 3:
                self._d_adam = {p: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                                for p, s in opt_d.state.items()}
            ordinary_step(opt_d)
            self._arm_smoothed_critic()
            return None
        if self.phase < 2 + self.game_steps:
            return None
        rho_game = _rho(self.g_base, self.g1, self.gg0, grads, self.metric_g)
        rho = max(self._rho_own, rho_game)
        factor = min(1., self.curvature_bound / rho) if rho > 0 else 1.
        for p, b, n in zip(self._params(opt_g), self.g_base, self.g1):
            p.copy_(torch.lerp(b, n, factor) if factor < 1 else n)
        for p, v in zip(self._params(opt_d), self.d_star):
            p.copy_(v)
        for p, s in self._d_adam.items():
            opt_d.state[p] = s
        self.row["g"] = dict(rho=rho, factor=factor, rho_own=self._rho_own, rho_game=rho_game)
        return None

    def receipt(self):
        value = super().receipt()
        widths = [row.get("critic_width", 0.) for row in self.records]
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, reach=self.reach, ramp=self.ramp,
                     game_bound=self.game_bound, game_steps=self.game_steps,
                     b_cap_slope=B_CAP_SLOPE,
                     recovery_reach=RECOVERY_REACH, support_drop=SUPPORT_DROP, support_alpha=SUPPORT_ALPHA,
                     recovery_updates=int(sum(bool(row.get("recovery")) for row in self.records)),
                     aten_cpu_capability=os.environ.get("ATEN_CPU_CAPABILITY", ""),
                     cpu_capability=torch.backends.cpu.get_cpu_capability(),
                     widened_updates=int(sum(w > base.SMOOTH_WIDTH_CAP for w in widths)),
                     max_width=max(widths, default=0.))
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1):
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "note_support", "pr84_reach_candidate", "reach_width",
           "recovery_width", "RECOVERY_REACH", "SUPPORT_DROP"]
