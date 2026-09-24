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

Optional G-step rollback, off unless ``trust_rollback`` is set: after a logged
ring check first reports 8 modes and HQ >= 0.9, reject a G Adam step whose
own-curvature trust collapses across that step. D is never restored. Mode
count arms the mechanism and is not the rollback predicate.
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


REST_UTILISATION = .3
SATURATED_UTILISATION = .6
STALL_TRUST = .1
STALL_WINDOW = 50
# Fixed. Pre is the #107 window mean of G's trust factor; post is this step's
# factor from the same own-curvature probe. Not a coefficient sweep.
TRUST_PRE_MIN = .5
TRUST_POST_MAX = .1
TRUST_COLLAPSE_RATIO = .2
ACQUIRE_BUDGET = 1200
DROPOUT_LO = 1720
DROPOUT_HI = 2150


def trust_collapsed(pre, post, *, pre_min=TRUST_PRE_MIN, post_max=TRUST_POST_MAX,
                    ratio=TRUST_COLLAPSE_RATIO):
    """True when G's own-curvature trust fell across this step.

    The predicate reads only the two trust numbers. Mode count is not an input.
    """
    if pre is None or post is None:
        return False
    pre, post = float(pre), float(post)
    if not math.isfinite(pre) or not math.isfinite(post):
        return False
    if pre >= pre_min and post <= post_max:
        return True
    return pre > 0 and post / pre <= ratio


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
    trust_rollback = False

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update = None
        self.arm_phase = None
        self.rollback_steps = []
        self.rollback_checks = 0
        self._g_snap_params = None
        self._g_snap_adam = None
        self._g_pre = None
        self._kept_factor = None
        self._kept_rho = None

    def note_ring(self, step, modes, hq):
        """Arm once on an already-logged ring check. Not a loss and not a predicate."""
        if not self.trust_rollback or self.armed:
            return
        try:
            modes, hq, step = int(modes), float(hq), int(step)
        except (TypeError, ValueError):
            return
        if modes != 8 or not math.isfinite(hq) or hq < .9:
            return
        self.armed = True
        self.arm_update = step
        if self.start_step > 0:
            self.arm_phase = "warm"
        elif step <= ACQUIRE_BUDGET:
            self.arm_phase = "cold_acquire"
        else:
            self.arm_phase = "stay"
        print(json.dumps(dict(event="ROLLBACK_ARM", step=step, phase=self.arm_phase)), flush=True)

    def _window_trust(self):
        recent = [row["g"]["factor"] for row in self.records[-STALL_WINDOW:]
                  if isinstance(row.get("g"), dict) and "factor" in row["g"]]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def _watching_g(self, optimizer):
        return (self.trust_rollback and self.armed and self.enabled and not self.passthrough
                and self.phase is not None and self.optimizers is not None
                and optimizer is self.optimizers[1])

    def _capture_g_rollback(self, optimizer):
        self._g_snap_params = [p.detach().clone() for p in self._params(optimizer)]
        self._g_snap_adam = {p: {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                 for k, v in s.items()}
                             for p, s in optimizer.state.items()}
        self._g_pre = self._window_trust()
        kept = [row["g"] for row in self.records if isinstance(row.get("g"), dict) and "factor" in row["g"]]
        self._kept_factor = kept[-1]["factor"] if kept else None
        self._kept_rho = kept[-1].get("rho") if kept else None

    def _reject_g_on_trust_collapse(self, optimizer):
        post = self.row["g"].get("factor")
        pre = self._g_pre
        self.rollback_checks += 1
        self.row["g"]["trust_pre"] = pre
        self.row["g"]["trust_post"] = post
        if not trust_collapsed(pre, post) or self._g_snap_params is None:
            return
        with torch.no_grad():
            for p, saved in zip(self._params(optimizer), self._g_snap_params):
                p.copy_(saved)
            for p, state in self._g_snap_adam.items():
                optimizer.state[p] = state
        if self._kept_factor is not None:
            self.row["g"]["rejected_factor"] = post
            self.row["g"]["rejected_rho"] = self.row["g"].get("rho")
            self.row["g"]["factor"] = self._kept_factor
            self.row["g"]["rho"] = self._kept_rho
        self.row["g"]["rolled_back"] = True
        step = int(self.row.get("outer_step", self.outer_steps + 1))
        self.rollback_steps.append(step)
        kind = ("dropout" if DROPOUT_LO <= step <= DROPOUT_HI else
                "acquire" if step <= ACQUIRE_BUDGET else "stay")
        print(json.dumps(dict(event="ROLLBACK_FIRE", step=step, kind=kind,
                              pre=pre, post=post)), flush=True)

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
        if self._watching_g(optimizer) and self.phase == 1:
            self._capture_g_rollback(optimizer)
        if (not self.game_bound or self.passthrough or self.phase is None or self.phase < 2
                or (self.phase == 2 and optimizer is self.optimizers[0])):
            result = super().step(optimizer, ordinary_step, closure)
            if (self._watching_g(optimizer) and self.phase == 2
                    and isinstance(self.row.get("g"), dict)):
                self._reject_g_on_trust_collapse(optimizer)
            return result
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
        if self.trust_rollback:
            steps = self.rollback_steps
            value.update(trust_rollback=True, rollback_armed=self.armed,
                         rollback_arm_update=self.arm_update, rollback_arm_phase=self.arm_phase,
                         rollback_checks=self.rollback_checks, rollback_fires=len(steps),
                         rollback_fires_acquire=sum(s <= ACQUIRE_BUDGET for s in steps),
                         rollback_fires_stay=sum(s > ACQUIRE_BUDGET for s in steps),
                         rollback_fires_dropout=sum(DROPOUT_LO <= s <= DROPOUT_HI for s in steps),
                         trust_pre_min=TRUST_PRE_MIN, trust_post_max=TRUST_POST_MAX,
                         trust_collapse_ratio=TRUST_COLLAPSE_RATIO)
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         trust_rollback=False):
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               trust_rollback=bool(trust_rollback), __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            previous = set_ring_listener(value[0].note_ring) if trust_rollback else None
            try:
                yield value
            finally:
                if trust_rollback:
                    set_ring_listener(previous)
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width"]
