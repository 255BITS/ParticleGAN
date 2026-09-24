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

An optional sustained arm, off unless ``sustained_g_scale`` is 0.5, halves each
Generator Adam displacement only after the ring has already been held: 200
consecutive updates with 8 modes and HQ >= 0.9 on the fixed unwrapped draw.
Before that the update is stall reach. After it, the half-step stays on.
"""

from contextlib import contextmanager
import json

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.alternating_curvature_scratch import _rho


METHOD = "pr84_slope_utilisation_reach"
B_CAP_SLOPE = 1.
REACH = .5
SUSTAINED_G_SCALE = .5
HOLD_UPDATES = 200
ARM_MODES = 8
ARM_HQ = .9


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
    sustained_g_scale = None

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self._hold = 0
        self._max_hold = 0
        self.armed = False
        self.arm_update = None
        self.post_arm_g_steps = 0
        self._last_modes = None
        self._last_hq = None

    def _emit(self, **row):
        print(json.dumps(row, default=float), flush=True)

    @torch.no_grad()
    def _ring_coverage(self, local):
        """Fixed 4096-draw through the unwrapped generator. Not a train loss.

        ``None`` when this host has no ring (trajectory): the counter never arms.
        The draw is reseeded, and the global RNG is restored, so training streams
        stay on the host path.
        """
        means = local.get("means") if isinstance(local, dict) else None
        generator = local.get("generator") if isinstance(local, dict) else None
        prior = local.get("prior") if isinstance(local, dict) else None
        if not torch.is_tensor(means) or generator is None or not hasattr(prior, "sample"):
            return None
        from benchmarks.locked_shared.mode_hold import EVAL_N, SIGMA, diversity
        clean = getattr(generator, "model", generator)
        stream = torch.Generator(device="cpu")
        stream.manual_seed(9)
        with torch.random.fork_rng(devices=[]):
            latent, _ = prior.sample(EVAL_N, generator=stream)
            row = diversity(clean(latent), means, SIGMA)
        return int(row["modes"]), float(row["hq"])

    def _note_hold(self, local, step):
        """Count consecutive full-ring updates. Arm once the count hits 200."""
        if self.sustained_g_scale is None:
            return
        measured = self._ring_coverage(local)
        if measured is None:
            return
        modes, hq = measured
        self._last_modes, self._last_hq = modes, hq
        state_step = int(step)
        if not self.armed:
            full = modes == ARM_MODES and hq >= ARM_HQ
            if full:
                self._hold += 1
            else:
                if self._hold >= 20:
                    self._emit(event="HOLD_RESET", state_step=state_step, hold=self._hold,
                               modes=modes, hq=round(hq, 4))
                self._hold = 0
            if self._hold >= HOLD_UPDATES:
                self.armed = True
                self.arm_update = state_step
                self._emit(event="SUSTAINED_ARM", state_step=state_step, hold=self._hold,
                           modes=modes, hq=round(hq, 4), g_scale=self.sustained_g_scale)
        self._max_hold = max(self._max_hold, self._hold)
        if state_step % 100 == 0:
            self._emit(event="HOLD", state_step=state_step, hold=self._hold, armed=self.armed,
                       modes=modes, hq=round(hq, 4), half_steps=self.post_arm_g_steps)

    def _annotate_sustained(self):
        if self.sustained_g_scale is None or not self.records:
            return
        row = self.records[-1]
        if self._last_modes is not None:
            row["ring_modes"] = self._last_modes
            row["ring_hq"] = self._last_hq
        row["hold"] = self._hold
        row["g_half"] = bool(self.armed)

    def _scale_g_adam(self, optimizer):
        return (self.sustained_g_scale is not None and self.armed and not self.passthrough
                and self.phase == 1 and self.optimizers is not None
                and optimizer is self.optimizers[1])

    def _scaled_g_step(self, ordinary_step):
        """Scale the raw G+prior Adam displacement. Moments stay on the full gradient."""
        scale = self.sustained_g_scale

        def run(optimizer, closure=None):
            before = [p.detach().clone() for p in self._params(optimizer)]
            result = ordinary_step(optimizer, closure=closure)
            for p, saved in zip(self._params(optimizer), before):
                p.copy_(saved + scale * (p - saved))
            self.post_arm_g_steps += 1
            return result

        return run

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
        if self.sustained_g_scale is not None and self.enabled and step >= self.start_step:
            self._note_hold(local, step)
        if not self.game_bound or not self.enabled or step < self.start_step:
            yield from super().phases(step, opt_d, opt_g, local)
            self._annotate_sustained()
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
        self._annotate_sustained()

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self._scale_g_adam(optimizer):
            ordinary_step = self._scaled_g_step(ordinary_step)
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
        if self.sustained_g_scale is not None:
            value.update(method=METHOD + "+sustained_g_half",
                         sustained_g_scale=self.sustained_g_scale, hold_updates=HOLD_UPDATES,
                         armed=self.armed, arm_update=self.arm_update,
                         post_arm_g_steps=self.post_arm_g_steps, max_hold=self._max_hold,
                         hold=self._hold)
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         sustained_g_scale=None):
    original = base.SmoothedBothBoundRecorder
    if sustained_g_scale not in (None, SUSTAINED_G_SCALE):
        raise ValueError("sustained G scale is fixed at 0.5")

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound
        self.sustained_g_scale = sustained_g_scale

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               sustained_g_scale=sustained_g_scale, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width"]
