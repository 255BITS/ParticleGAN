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

from contextlib import contextmanager, nullcontext
import inspect
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

# Post-arm cadence only. The ring read is a gate, not a coverage loss.
ARM_MODES = 8
ARM_HQ = .9
FIRE_MODES = 6
EXTRA_D_STEPS = 2


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
    # After the first 8-mode HQ≥0.9 ring read, two extra D Adam steps when the
    # current mode count is ≤6. Off by default, so stall reach stays #107.
    mode_ttur = False

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.extra_d_steps = 0
        self._armed = False
        self._armed_at = None
        self._fire_count = 0
        self._fires = []
        self._last_modes = None
        self._last_hq = None
        self._host_step = None
        self._ttur_fire = False
        self._ordinary_step = None
        self._replay_rng = None
        self._replay_streams = None
        self._replay_buffers = None

    def _emit(self, **row):
        print(json.dumps(row, default=float), flush=True)

    @torch.no_grad()
    def _ring_coverage(self, local):
        """Ring metric on a fixed 4096-draw. Does not advance train RNGs.

        ``None`` on hosts without a ring (trajectory): the cadence never arms.
        The unwrapped generator keeps output-noise streams on the host path.
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

    def _decide_mode_ttur(self, local, step):
        """Arm on the first 8-mode HQ≥0.9 read. Later updates with ≤6 modes fire."""
        self._host_step = int(step)
        measured = self._ring_coverage(local)
        if measured is None:
            return False
        modes, hq = measured
        self._last_modes, self._last_hq = modes, hq
        if not self._armed:
            if modes == ARM_MODES and hq >= ARM_HQ:
                self._armed = True
                self._armed_at = int(step)
                self._emit(event="MODE_TTUR_ARMED", step=int(step), modes=modes, hq=round(hq, 4))
            return False
        return modes <= FIRE_MODES

    def _annotate_mode_ttur(self):
        if not self.mode_ttur or not self.records:
            return
        row = self.records[-1]
        if self._last_modes is not None:
            row["ring_modes"] = self._last_modes
            row["ring_hq"] = self._last_hq
        row["armed"] = bool(self._armed)
        row["ttur_fire"] = bool(self._ttur_fire)

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
        # After D* exists, before this phase's G step. Not gated on the stall width.
        if self._ttur_fire and self.phase == 1 and not self.game_bound and not self.passthrough:
            self._extra_discriminator_steps(EXTRA_D_STEPS)

    def _stalled(self, sharpness):
        """D near its slope limit while G's own trust region has kept G nearly still."""
        recent = self.records[-STALL_WINDOW:]
        if sharpness / B_CAP_SLOPE < SATURATED_UTILISATION or not recent:
            return False
        return sum(row["g"]["factor"] for row in recent) / len(recent) <= STALL_TRUST

    @staticmethod
    def _streams(local):
        streams = [v for v in local.values() if isinstance(v, torch.Generator)]
        policy = local.get("noise_policy")
        if policy is not None:
            streams.extend(v for name in ("input_stream", "output_stream")
                           if isinstance((v := getattr(policy, name, None)), torch.Generator))
        return list({id(s): s for s in streams}.values())

    @staticmethod
    def _module_buffers(local):
        return [(b, b.detach().clone()) for name in ("generator", "critic", "prior")
                if isinstance((m := local.get(name)), torch.nn.Module) for b in m.buffers()]

    def _extra_discriminator_steps(self, count):
        """``count`` real D Adam steps on this phase's batch, before G reads D.

        The host has already drawn the batch. Replaying from the phase-start RNG
        redraws it, including input and output noise. G's RNG, buffers and weights
        are put back so its own step still sees the batch stall reach would have used.
        D's learning rate and Adam betas are the host's. G is not frozen or scaled.
        """
        if self.game_bound or self.phase != 1 or self._ordinary_step is None:
            raise RuntimeError("extra D steps are only the phase-1 critic update")
        if self._replay_rng is None:
            raise RuntimeError("extra D steps missing the phase-start replay point")
        frame = self._host_frame()
        streams = self._replay_streams
        post_rng = self._rng(streams)
        post_buffers = self._module_buffers(self._local)
        opt_d, opt_g = self.optimizers
        g_params = [p.detach().clone() for p in self._params(opt_g)]
        saved_smooth, saved_width = self._smooth_on, self._smooth_width
        try:
            self._smooth_on = False
            for _ in range(count):
                self._set_rng(streams, self._replay_rng)
                with torch.no_grad():
                    for buf, saved in self._replay_buffers:
                        buf.copy_(saved)
                with torch.enable_grad():
                    self._rerun_host_d(frame)
                    self._ordinary_step(opt_d)
                self.extra_d_steps += 1
        finally:
            self._set_rng(streams, post_rng)
            with torch.no_grad():
                for buf, saved in post_buffers:
                    buf.copy_(saved)
                for param, saved in zip(self._params(opt_g), g_params):
                    param.copy_(saved)
            self._smooth_on = saved_smooth
            self._smooth_width = saved_width
        self.d_star = [p.detach().clone() for p in self._params(opt_d)]
        self._fire_count += 1
        fire = dict(step=self._host_step, modes=self._last_modes, hq=self._last_hq)
        self._fires.append(fire)
        self.row["ttur_fire"] = True
        self.row["ttur_modes"] = self._last_modes
        self._emit(event="MODE_TTUR_FIRE", step=self._host_step, modes=self._last_modes,
                   hq=None if self._last_hq is None else round(self._last_hq, 4),
                   fires=self._fire_count, extra_d=self.extra_d_steps)

    @staticmethod
    def _host_frame():
        frame = inspect.currentframe()
        while frame is not None:
            local = frame.f_locals
            name = frame.f_code.co_name
            if name == "train_mode_hold" and "gan" in local and "critic" in local:
                return frame
            if name == "train" and "paired" in local and "view" in local and "gan" in local:
                return frame
            frame = frame.f_back
        raise RuntimeError("extra D step could not see the host batch")

    def _rerun_host_d(self, frame):
        """Replay the host's D loss on the restored batch. Does not step."""
        local = frame.f_locals
        gan, opt_d = local["gan"], self.optimizers[0]
        regularizer = local["regularizer"]
        noise = local.get("noise_policy")
        context = noise.discriminator() if noise is not None else nullcontext()
        if frame.f_code.co_name == "train_mode_hold":
            sample_ring = frame.f_globals["sample_ring"]
            sigma = frame.f_globals["SIGMA"]
            real = sample_ring(local["means"], local["batch"], sigma, local["stream"])
            latent, _ = local["prior"].sample(local["batch"], generator=local["stream"])
            with context:
                fake = local["generator"](latent).detach()
            d_loss = gan.d_loss(local["critic"](real), local["critic"](fake))
            d_loss = d_loss + regularizer(local["critic"], real, fake, step=local["step"] + 1)
            opt_d.zero_grad()
        else:
            with context:
                fake = local["generator"](local["slow"], local["prior"].z)
            local["view"].slow = local["slow"].detach()
            d_loss = gan.d_loss(local["critic"](local["slow"], local["paired"]),
                                local["critic"](local["slow"], fake.detach()))
            d_loss = d_loss + regularizer(local["view"], local["paired"], fake.detach(),
                                          step=local["step"])
            opt_d.zero_grad(set_to_none=True)
        d_loss.backward()

    def phases(self, step, opt_d, opt_g, local):
        self._ttur_fire = False
        active = self.mode_ttur and self.enabled and step >= self.start_step and not self.game_bound
        if active:
            self._ttur_fire = self._decide_mode_ttur(local, step)
        if not self.game_bound or not self.enabled or step < self.start_step:
            if not active:
                yield from super().phases(step, opt_d, opt_g, local)
                return
            streams = self._streams(local)
            self._local = local
            for phase in super().phases(step, opt_d, opt_g, local):
                self._replay_streams = streams
                self._replay_rng = self._rng(streams)
                self._replay_buffers = [(b, b.detach().clone()) for b, _saved in self._module_buffers(local)]
                yield phase
            self._annotate_mode_ttur()
            if self.outer_steps % 100 == 0:
                self._emit(event="PROGRESS", outer=self.outer_steps, armed=self._armed,
                           armed_at=self._armed_at, fires=self._fire_count,
                           extra_d=self.extra_d_steps, modes=self._last_modes)
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
        self._ordinary_step = ordinary_step
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
        value.update(method=METHOD + ("+post_arm_mode_ttur" if self.mode_ttur else ""),
                     scratch_optimizer_policy=METHOD, reach=self.reach, ramp=self.ramp,
                     game_bound=self.game_bound, game_steps=self.game_steps,
                     mode_ttur=bool(self.mode_ttur), armed_at=self._armed_at,
                     fires=self._fire_count, extra_d_steps=self.extra_d_steps,
                     b_cap_slope=B_CAP_SLOPE,
                     widened_updates=int(sum(w > base.SMOOTH_WIDTH_CAP for w in widths)),
                     max_width=max(widths, default=0.))
        return value


@contextmanager
def pr84_reach_candidate(*, task="mode_hold", start_step=0, reach=REACH, ramp="peak",
                         g_curvature_bound=base.G_CURVATURE_BOUND, game_bound=False, game_steps=1,
                         mode_ttur=False):
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        ReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = g_curvature_bound
        self.mode_ttur = bool(mode_ttur)

    base.SmoothedBothBoundRecorder = type("ReachRecorder", (ReachRecorder,),
                                          dict(reach=reach, ramp=ramp, game_bound=game_bound, game_steps=game_steps,
                                               mode_ttur=bool(mode_ttur), __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ReachRecorder", "pr84_reach_candidate", "reach_width"]
