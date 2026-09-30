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

An optional post-arm switch, off unless ``post_arm_g_beta1`` is 0, sets Generator
Adam's first-moment coefficient to 0 after one logged ring check reports 8 modes
and HQ >= 0.9. That check is the existing offline mode counter, not a loss. The
flag stays armed. G's existing ``exp_avg`` is zeroed once on the next real G
step so a stale first moment cannot carry; β2, eps, lr and D Adam stay as they
were. Before the flag, G Adam is stall reach.
"""

from contextlib import contextmanager
import json
import math
import os

import torch

from benchmarks.locked_shared.observation import set_ring_listener
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.alternating_curvature_scratch import _rho


METHOD = "pr84_slope_utilisation_reach"
B_CAP_SLOPE = 1.
REACH = .5
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
    post_arm_g_beta1 = None

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update = None
        self.arm_phase = None
        self.exp_avg_zeroed = False
        self.exp_avg_zero_count = 0
        self.momentum_applied = False
        self.momentum_apply_update = None
        self.g_beta1_before = None
        self.post_arm_g_steps = 0
        self.host_step = None

    def note_ring(self, step, modes, hq):
        """Arm once on the live ring log: 8 modes and HQ ≥ 0.9. Not a loss."""
        if self.post_arm_g_beta1 != 0 or self.armed:
            return
        try:
            modes, hq = int(modes), float(hq)
        except (TypeError, ValueError):
            return
        if modes != 8 or not math.isfinite(hq) or hq < .9:
            return
        self.armed = True
        self.arm_update = int(step)
        if self.start_step > 0:
            self.arm_phase = "warm"
        elif self.arm_update <= ACQUIRE_BUDGET:
            self.arm_phase = "cold_acquire"
        else:
            self.arm_phase = "stay"
        self._log(event="ARM", update=self.arm_update, phase=self.arm_phase)

    def _log(self, **row):
        row.setdefault("aten", os.environ.get("ATEN_CPU_CAPABILITY"))
        row.setdefault("cpu", torch.backends.cpu.get_cpu_capability())
        print(json.dumps(row, default=float), flush=True)

    def _g_momentum_kill(self, optimizer):
        return (self.post_arm_g_beta1 == 0 and self.armed and not self.passthrough
                and not self.game_bound and self.phase == 1 and self.optimizers is not None
                and optimizer is self.optimizers[1])

    def _kill_g_momentum(self, optimizer):
        """Set G Adam β1 to 0. Zero exp_avg once, on the first real post-arm G step."""
        if not self.momentum_applied:
            before = [float(group["betas"][0]) for group in optimizer.param_groups]
            self.g_beta1_before = before[0] if before and all(b == before[0] for b in before) else None
            for group in optimizer.param_groups:
                group["betas"] = (0.0, float(group["betas"][1]))
            defaults = optimizer.defaults.get("betas")
            if defaults is not None:
                optimizer.defaults["betas"] = (0.0, float(defaults[1]))
            zeroed = 0
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    moment = None if state is None else state.get("exp_avg")
                    if torch.is_tensor(moment):
                        moment.zero_()
                        zeroed += 1
            self.exp_avg_zeroed = zeroed > 0
            self.exp_avg_zero_count = zeroed
            self.momentum_applied = True
            self.momentum_apply_update = None if self.host_step is None else int(self.host_step) + 1
            self._log(event="G_BETA1_0", arm_update=self.arm_update, phase=self.arm_phase,
                      apply_update=self.momentum_apply_update, exp_avg_zeroed=self.exp_avg_zeroed,
                      exp_avg_zero_count=self.exp_avg_zero_count, g_beta1_before=self.g_beta1_before)
        else:
            for group in optimizer.param_groups:
                if float(group["betas"][0]) != 0.0:
                    group["betas"] = (0.0, float(group["betas"][1]))
            defaults = optimizer.defaults.get("betas")
            if defaults is not None and float(defaults[0]) != 0.0:
                optimizer.defaults["betas"] = (0.0, float(defaults[1]))
        self.post_arm_g_steps += 1

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
        self.host_step = step
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
        if self._g_momentum_kill(optimizer):
            self._kill_g_momentum(optimizer)
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
        if self.post_arm_g_beta1 is not None:
            value.update(mechanism="post_arm_g_adam_beta1_0", post_arm_g_beta1=0,
                         armed=self.armed, arm_update=self.arm_update, arm_phase=self.arm_phase,
                         exp_avg_zeroed=self.exp_avg_zeroed, exp_avg_zero_count=self.exp_avg_zero_count,
                         momentum_applied=self.momentum_applied,
                         momentum_apply_update=self.momentum_apply_update,
                         g_beta1_before=self.g_beta1_before, post_arm_g_steps=self.post_arm_g_steps)
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         post_arm_g_beta1=None):
    original = base.SmoothedBothBoundRecorder
    if post_arm_g_beta1 not in (None, 0):
        raise ValueError("post-arm G Adam beta1 is fixed at 0")

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound
        self.post_arm_g_beta1 = post_arm_g_beta1

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               post_arm_g_beta1=post_arm_g_beta1, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            recorder = value[0]
            previous = set_ring_listener(recorder.note_ring) if post_arm_g_beta1 is not None else None
            try:
                yield value
            finally:
                if post_arm_g_beta1 is not None:
                    set_ring_listener(previous)
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width"]
