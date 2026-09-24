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

An optional post-arm clamp, off unless ``post_arm_g_bound`` is .125, leaves the
pre-arm G own-curvature cap (.25 on stall reach) untouched until the first
logged ring check reports at least 8 modes and HQ >= 0.9. After that arm the
cap used in ``min(1, c / rho)`` is .125 for G only, and it stays there. This is
the trust-region radius, not a scale on the Adam step.
"""

from contextlib import contextmanager
import json
import math

import torch

from benchmarks.locked_shared.observation import set_ring_listener
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.alternating_curvature_scratch import _rho


METHOD = "pr84_slope_utilisation_reach"
B_CAP_SLOPE = 1.
REACH = .5
POST_ARM_G_BOUND = .125
ACQUIRE_BUDGET = 1200


REST_UTILISATION = .3
SATURATED_UTILISATION = .6
STALL_TRUST = .1
STALL_WINDOW = 50


def reach_width(sharpness, reach=REACH, ramp="peak"):
    """``peak``: ``.5·min(u, 1/u)``. ``saturating``: PR84 until u = .3, .5 from u = .6."""
    utilisation = sharpness / B_CAP_SLOPE
    if ramp == "saturating":
        span = (utilisation - REST_UTILISATION) / (SATURATED_UTILISATION - REST_UTILISATION)
        low = base.SMOOTH_WIDTH_CAP
        return low + (reach / B_CAP_SLOPE - low) * min(1., max(0., span))
    utilisation = min(utilisation, 1 / utilisation)
    return max(base.SMOOTH_WIDTH_CAP, reach / B_CAP_SLOPE * utilisation)


class ReachRecorder(base.SmoothedBothBoundRecorder):
    reach = REACH
    ramp = "peak"
    game_bound = False
    game_steps = 1
    post_arm_g_bound = None

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update = None
        self.arm_phase = None
        self.post_arm_fires = 0
        self.post_arm_g_bound = type(self).post_arm_g_bound

    def note_ring(self, step, modes, hq):
        """Arm once on the live ring log: modes ≥ 8 and HQ ≥ 0.9. Not a loss."""
        if self.post_arm_g_bound is None or self.armed:
            return
        try:
            modes, hq, step = int(modes), float(hq), int(step)
        except (TypeError, ValueError):
            return
        if modes < 8 or not math.isfinite(hq) or hq < .9:
            return
        self.armed = True
        self.arm_update = step
        if self.start_step > 0:
            self.arm_phase = "warm"
        elif self.arm_update <= ACQUIRE_BUDGET:
            self.arm_phase = "cold_acquire"
        else:
            self.arm_phase = "stay"

    def _g_bound_apply(self, optimizer):
        """The G own-curvature placement, and the cap before/after the clamp.

        Pre-arm, after equals the #107 cap. After the sticky arm, after is .125.
        D's cap is never this value.
        """
        applying = (self.post_arm_g_bound is not None and not self.passthrough
                    and self.phase == 2 and self.optimizers is not None
                    and optimizer is self.optimizers[1])
        before = float(self.curvature_bound)
        after = float(self.post_arm_g_bound) if applying and self.armed else before
        return before, after, applying

    def _log_g_bound(self, before, after):
        fired = bool(self.armed)
        if fired:
            self.post_arm_fires += 1
        update = int(getattr(self, "_host_step", None) or self.row.get("outer_step", self.outer_steps + 1))
        self.row["armed"] = self.armed
        self.row["g_bound_before"] = before
        self.row["g_bound_after"] = after
        if isinstance(self.row.get("g"), dict):
            self.row["g"]["bound_before"] = before
            self.row["g"]["bound_after"] = after
        print(json.dumps(dict(event="G_BOUND", update=update, armed=self.armed,
                              g_bound_before=before, g_bound_after=after,
                              post_arm_fires=self.post_arm_fires)), flush=True)

    def _arm_smoothed_critic(self):
        super()._arm_smoothed_critic()
        if self._smooth_on:
            sharpness = self.row["critic_sharpness"]
            if self.ramp == "stall":
                self._smooth_width = (self.reach / B_CAP_SLOPE if self._stalled(sharpness)
                                      else reach_width(sharpness, self.reach))
            else:
                self._smooth_width = reach_width(sharpness, self.reach, self.ramp)
            self.row["critic_width"] = self._smooth_width

    def _stalled(self, sharpness):
        """D near its slope limit while G's own trust region has kept G nearly still."""
        recent = self.records[-STALL_WINDOW:]
        if sharpness / B_CAP_SLOPE < SATURATED_UTILISATION or not recent:
            return False
        return sum(row["g"]["factor"] for row in recent) / len(recent) <= STALL_TRUST

    def phases(self, step, opt_d, opt_g, local):
        self._host_step = int(step) + 1
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
        before, after, applying = self._g_bound_apply(optimizer)
        swapped = applying and after != before
        if swapped:
            self.curvature_bound = after
        ok = False
        try:
            result = self._step_body(optimizer, ordinary_step, closure)
            ok = True
            return result
        finally:
            if swapped:
                self.curvature_bound = before
            if ok and applying:
                self._log_g_bound(before, after)

    def _step_body(self, optimizer, ordinary_step, closure=None):
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
                     widened_updates=int(sum(w > base.SMOOTH_WIDTH_CAP for w in widths)),
                     max_width=max(widths, default=0.))
        if self.post_arm_g_bound is not None:
            value.update(post_arm_g_bound=self.post_arm_g_bound,
                         g_bound_pre_arm=float(self.curvature_bound),
                         armed=self.armed, arm_update=self.arm_update, arm_phase=self.arm_phase,
                         post_arm_fires=self.post_arm_fires,
                         cpu=torch.backends.cpu.get_cpu_capability())
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         post_arm_g_bound=None):
    original = base.SmoothedBothBoundRecorder
    if post_arm_g_bound not in (None, POST_ARM_G_BOUND):
        raise ValueError("post-arm G own-curvature bound is fixed at 0.125")

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound
        self.post_arm_g_bound = post_arm_g_bound

    base.SmoothedBothBoundRecorder = type(
        "ReachRecorder", (ReachRecorder,),
        dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
             post_arm_g_bound=post_arm_g_bound, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            recorder = value[0]
            previous = set_ring_listener(recorder.note_ring) if post_arm_g_bound is not None else None
            try:
                yield value
            finally:
                if post_arm_g_bound is not None:
                    set_ring_listener(previous)
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width"]
