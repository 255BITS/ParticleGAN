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
import json

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.alternating_curvature_scratch import _rho


METHOD = "pr84_slope_utilisation_reach"
B_CAP_SLOPE = 1.
REACH = .5


REST_UTILISATION = .3
SATURATED_UTILISATION = .6
STALL_TRUST = .1
STALL_WINDOW = 50

# Post-acquire cadence only. These are the gate's ring bar, not a loss.
ARM_MODES = 8
ARM_HQ = .9
FREEZE_AT = 6
RESUME_AT = 7
DROPOUT_WINDOW = (1690, 2300)


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
    mode_drop = False

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self._armed = False
        self._armed_at = None
        self._freeze_g = False
        self._skip_open = False
        self._skip_steps = []
        self._last_modes = None
        self._last_hq = None

    def _emit(self, **row):
        print(json.dumps(row, default=float), flush=True)

    @torch.no_grad()
    def _ring_coverage(self, local):
        """Gate ring metric on a fixed 4096-draw, without touching train RNGs.

        Uses the unwrapped generator so output-noise streams and eval receipts
        stay on the host's own measurement path. ``None`` when this host has
        no ring (trajectory): the cadence then never arms.
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

    def _decide_mode_drop(self, local, step):
        """Arm on the first 8-mode HQ≥0.9 state. Later updates with ≤6 modes skip G."""
        measured = self._ring_coverage(local)
        if measured is None:
            return False
        modes, hq = measured
        self._last_modes, self._last_hq = modes, hq
        state_step = int(step)
        if not self._armed:
            if modes == ARM_MODES and hq >= ARM_HQ:
                self._armed = True
                self._armed_at = state_step
                self._emit(event="MODE_DROP_ARMED", state_step=state_step, modes=modes, hq=round(hq, 4))
            return False
        freeze = modes <= FREEZE_AT
        if freeze and not self._skip_open:
            self._skip_open = True
            self._emit(event="G_FREEZE_START", state_step=state_step, modes=modes, hq=round(hq, 4))
        elif not freeze and modes >= RESUME_AT and self._skip_open:
            self._skip_open = False
            self._emit(event="G_FREEZE_END", state_step=state_step, modes=modes, hq=round(hq, 4))
        return freeze

    def _annotate_mode_drop(self):
        if not self.mode_drop or not self.records:
            return
        row = self.records[-1]
        if self._last_modes is not None:
            row["ring_modes"] = self._last_modes
            row["ring_hq"] = self._last_hq
        row["g_skipped"] = bool(self._freeze_g)

    def _revert_frozen_g(self, optimizer):
        """Drop this update's G+prior write. Adam's step counter still advances.

        The host probe requires G and D moment counters to stay locked, so the
        Adam step is taken and the weights are copied back to the pre-update
        snapshot. D's write is left in place. Reach width and the stall
        predicate still see the factor of the discarded step.
        """
        if not self._freeze_g or self.passthrough or self.phase != 2 or self.optimizers is None:
            return
        if optimizer is not self.optimizers[1]:
            return
        with torch.no_grad():
            for param, base in zip(self._params(optimizer), self.g_base):
                param.copy_(base)
        update = int(self.row.get("outer_step", self.outer_steps + 1))
        self._skip_steps.append(update)
        if isinstance(self.row.get("g"), dict):
            self.row["g"]["skipped"] = True

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
        self._freeze_g = False
        if self.mode_drop and self.enabled and step >= self.start_step:
            self._freeze_g = self._decide_mode_drop(local, step)
        if not self.game_bound or not self.enabled or step < self.start_step:
            yield from super().phases(step, opt_d, opt_g, local)
            self._annotate_mode_drop()
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
            result = super().step(optimizer, ordinary_step, closure)
            self._revert_frozen_g(optimizer)
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
        low, high = DROPOUT_WINDOW
        skips = list(self._skip_steps)
        value.update(method=METHOD + ("+mode_drop_g_freeze" if self.mode_drop else ""),
                     scratch_optimizer_policy=METHOD, reach=self.reach, ramp=self.ramp,
                     game_bound=self.game_bound, game_steps=self.game_steps,
                     b_cap_slope=B_CAP_SLOPE,
                     widened_updates=int(sum(w > base.SMOOTH_WIDTH_CAP for w in widths)),
                     max_width=max(widths, default=0.),
                     mode_drop=bool(self.mode_drop),
                     armed_at=self._armed_at,
                     g_skips=len(skips),
                     g_skips_1690_2300=sum(low <= s <= high for s in skips),
                     g_skip_first=(skips[0] if skips else None),
                     g_skip_last=(skips[-1] if skips else None))
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         mode_drop=False):
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound
        self.mode_drop = bool(mode_drop)

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               mode_drop=bool(mode_drop), __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width"]
