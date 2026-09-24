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

Optional post-arm G reject, off unless ``trust_open_reject`` is set. Stall
reach is unchanged until an already-scheduled ring check reports 8 modes and
HQ >= 0.9. That log only arms the mechanism. After the arm, a G Adam step is
rejected when the 50-step mean of G's own-curvature trust factor has opened
by at least 0.10 versus the 50-step mean from 50 updates earlier. D is never
restored. Mode count is not the reject predicate.
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
# Fixed. The 50-step mean of ``factor = min(1, 0.25/ρ)`` must rise by at least
# this much versus the mean from 50 updates earlier. Not a one-step ratio.
TRUST_OPEN_DELTA = .10
ACQUIRE_BUDGET = 1200
DROPOUT_LO = 1720
DROPOUT_HI = 2150


def window_trust_open(factors, post, *, lag=STALL_WINDOW, delta=TRUST_OPEN_DELTA):
    """Whether the rolling trust mean opened by ``delta`` across ``lag`` updates.

    ``factors`` are accepted trust factors before this step, oldest first.
    ``post`` is this step's factor. The two windows are adjacent 50-step means:
    the one ending at this step, and the one ending 50 updates earlier. A short
    history does not open. Mode count is not an input.
    """
    if post is None or len(factors) < 2 * lag - 1:
        return False, None, None
    post = float(post)
    if not math.isfinite(post):
        return False, None, None
    lag_window = [float(v) for v in factors[-(2 * lag - 1):-(lag - 1)]]
    now_window = [float(v) for v in factors[-(lag - 1):]] + [post]
    if (len(lag_window) != lag or len(now_window) != lag
            or any(not math.isfinite(v) for v in lag_window)):
        return False, None, None
    mean_lag = sum(lag_window) / lag
    mean_now = sum(now_window) / lag
    if not math.isfinite(mean_lag) or not math.isfinite(mean_now):
        return False, mean_now, mean_lag
    return mean_now - mean_lag >= delta, mean_now, mean_lag


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
    trust_open_reject = False

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update = None
        self.arm_phase = None
        self.reject_steps = []
        self.reject_checks = 0
        self.open_peak = None
        self.open_peak_dropout = None
        self._g_snap_params = None
        self._g_snap_adam = None
        self._kept_factor = None
        self._kept_rho = None

    def note_ring(self, step, modes, hq):
        """Arm once on an already-logged ring check. Not a loss and not a predicate."""
        if not self.trust_open_reject or self.armed:
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
        print(json.dumps(dict(event="TRUST_OPEN_ARM", step=step, phase=self.arm_phase)), flush=True)

    def _accepted_factors(self):
        return [row["g"]["factor"] for row in self.records
                if isinstance(row.get("g"), dict) and "factor" in row["g"]]

    def _watching_g(self, optimizer):
        return (self.trust_open_reject and self.armed and self.enabled and not self.passthrough
                and self.phase is not None and self.optimizers is not None
                and optimizer is self.optimizers[1])

    def _capture_g_reject(self, optimizer):
        self._g_snap_params = [p.detach().clone() for p in self._params(optimizer)]
        self._g_snap_adam = {p: {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                 for k, v in s.items()}
                             for p, s in optimizer.state.items()}
        kept = [row["g"] for row in self.records
                if isinstance(row.get("g"), dict) and "factor" in row["g"]]
        self._kept_factor = kept[-1]["factor"] if kept else None
        self._kept_rho = kept[-1].get("rho") if kept else None

    def _note_peak(self, step, delta):
        if delta is None or not math.isfinite(delta):
            return
        if self.open_peak is None or delta > self.open_peak[0]:
            self.open_peak = (delta, step)
        if DROPOUT_LO <= step <= DROPOUT_HI and (
                self.open_peak_dropout is None or delta > self.open_peak_dropout[0]):
            self.open_peak_dropout = (delta, step)

    def _consider_trust_open(self, optimizer, watching):
        """Track the rolling mean, and after the arm reject an opening step."""
        post = self.row["g"].get("factor")
        factors = self._accepted_factors()
        opened, mean_now, mean_lag = window_trust_open(factors, post)
        g = self.row["g"]
        g["trust_mean_probe"] = mean_now
        g["trust_mean_lag"] = mean_lag
        g["trust_open"] = None if mean_now is None or mean_lag is None else mean_now - mean_lag
        window = factors + ([float(post)] if post is not None else [])
        window = window[-STALL_WINDOW:]
        g["trust_mean"] = sum(window) / len(window) if window else None
        step = int(self.row.get("outer_step", self.outer_steps + 1))
        if step % 50 == 0:
            print(json.dumps(dict(event="TRUST_MEAN", step=step, armed=bool(self.armed),
                                  mean=g["trust_mean"], open=g["trust_open"],
                                  fires=len(self.reject_steps))), flush=True)
        if not watching:
            return
        self.reject_checks += 1
        self._note_peak(step, g["trust_open"])
        if not opened or self._g_snap_params is None:
            return
        with torch.no_grad():
            for p, saved in zip(self._params(optimizer), self._g_snap_params):
                p.copy_(saved)
            for p, state in self._g_snap_adam.items():
                optimizer.state[p] = state
        if self._kept_factor is not None:
            g["rejected_factor"] = post
            g["rejected_rho"] = g.get("rho")
            g["factor"] = self._kept_factor
            g["rho"] = self._kept_rho
            accepted = factors + [self._kept_factor]
            window = accepted[-STALL_WINDOW:]
            g["trust_mean"] = sum(window) / len(window)
        g["rejected"] = True
        self.reject_steps.append(step)
        if self.start_step > 0:
            kind = "warm"
        elif DROPOUT_LO <= step <= DROPOUT_HI:
            kind = "dropout"
        elif step <= ACQUIRE_BUDGET:
            kind = "acquire"
        else:
            kind = "stay"
        print(json.dumps(dict(event="TRUST_OPEN_REJECT", step=step, kind=kind,
                              mean_now=mean_now, mean_lag=mean_lag,
                              delta=g["trust_open"])), flush=True)

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
        watching = self._watching_g(optimizer)
        if watching and self.phase == 1:
            self._capture_g_reject(optimizer)
        if (not self.game_bound or self.passthrough or self.phase is None or self.phase < 2
                or (self.phase == 2 and optimizer is self.optimizers[0])):
            result = super().step(optimizer, ordinary_step, closure)
            if (self.trust_open_reject and not self.passthrough and self.phase == 2
                    and self.optimizers is not None and optimizer is self.optimizers[1]
                    and isinstance(self.row.get("g"), dict)):
                self._consider_trust_open(optimizer, watching)
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
        if self.trust_open_reject:
            steps = self.reject_steps
            value.update(trust_open_reject=True, trust_open_armed=self.armed,
                         trust_open_arm_update=self.arm_update, trust_open_arm_phase=self.arm_phase,
                         trust_open_checks=self.reject_checks, trust_open_fires=len(steps),
                         trust_open_fires_warm=sum(self.start_step > 0 for _ in steps),
                         trust_open_fires_acquire=sum(self.start_step == 0 and s <= ACQUIRE_BUDGET for s in steps),
                         trust_open_fires_stay=sum(self.start_step == 0 and s > ACQUIRE_BUDGET for s in steps),
                         trust_open_fires_dropout=sum(DROPOUT_LO <= s <= DROPOUT_HI for s in steps),
                         trust_open_delta=TRUST_OPEN_DELTA, trust_open_lag=STALL_WINDOW,
                         trust_open_max=None if self.open_peak is None else self.open_peak[0],
                         trust_open_max_step=None if self.open_peak is None else self.open_peak[1],
                         trust_open_max_dropout=None if self.open_peak_dropout is None else self.open_peak_dropout[0],
                         trust_open_max_dropout_step=(None if self.open_peak_dropout is None
                                                      else self.open_peak_dropout[1]),
                         reject_steps=list(steps))
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         trust_open_reject=False):
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               trust_open_reject=bool(trust_open_reject), __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            previous = set_ring_listener(value[0].note_ring) if trust_open_reject else None
            try:
                yield value
            finally:
                if trust_open_reject:
                    set_ring_listener(previous)
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width", "window_trust_open"]
