"""Stall reach plus one mechanism: delayed-arm extra D TTUR.

Stall reach (width, G own-curvature .25, G Adam lr, D rates, losses) is
unchanged until the arm. The arm sticks on the first diagnostic with
``update_index >= 1200``, ``modes >= 8`` and ``HQ >= 0.9``. An 8/.9 before
1200 does not arm. There is no mode-count latch and no stall predicate.

After the arm, each training turn runs two extra host D Adam steps on this
turn's batch, after the bounded host D* and before G reads D. The steps use
the host D optimizer, the host D loss and penalty, and the host D learning
rate and betas. G's Adam lr and both curvature bounds stay put. G's batch,
RNG and weights are restored so the usual G step still sees the batch stall
reach would have used.

Every apply logs ``armed``, ``update_index``, ``modes``, ``HQ`` and
``extra_d_steps_this_turn``.
"""

from contextlib import contextmanager, nullcontext
import inspect
import json
import math

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import METHOD as REACH_METHOD
from reports.toy100.pr84_reach_candidate import ReachRecorder


METHOD = "stall_reach_delayed_arm_extra_d_ttur"
ARM_UPDATE_INDEX = 1200
ARM_MODES = 8
ARM_HQ = .9
EXTRA_D_STEPS = 2
# Shared continuation dropout cited for this line. Acquire is every fire before it.
DROPOUT_LO = 1690
DROPOUT_HI = 2300


class DelayedArmExtraDRecorder(ReachRecorder):
    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.armed = False
        self.arm_update_index = None
        self.extra_d_steps = 0
        self.fires = 0
        self._fire_updates = []
        self._last_modes = None
        self._last_hq = None
        self._last_diagnostic_index = None
        self._host_step = None
        self._fire_this_turn = False
        self._ordinary_step = None
        self._replay_rng = None
        self._replay_streams = None
        self._replay_buffers = None
        self._local = None

    def consider_diagnostic(self, update_index, modes, hq):
        """Arm once: update_index >= 1200 and modes >= 8 and HQ >= 0.9.

        Earlier full-ring reads are recorded and ignored. Later reads do not
        release the arm and do not decide whether the extra D steps run.
        """
        if not self.enabled or type(update_index) is not int:
            return
        if type(modes) is not int or isinstance(hq, bool) or not isinstance(hq, (int, float)):
            return
        if not math.isfinite(hq):
            return
        self._last_modes = modes
        self._last_hq = float(hq)
        self._last_diagnostic_index = update_index
        if self.armed or update_index < ARM_UPDATE_INDEX or modes < ARM_MODES or hq < ARM_HQ:
            return
        self.armed = True
        self.arm_update_index = update_index
        _log(event="extra_d_arm", armed=True, update_index=update_index, modes=modes,
             HQ=float(hq), extra_d_steps_this_turn=0)

    def phases(self, step, opt_d, opt_g, local):
        self._host_step = step
        self._fire_this_turn = bool(
            self.armed and self.enabled and step >= self.start_step and not self.game_bound)
        if not self.enabled or step < self.start_step or self.game_bound:
            yield from super().phases(step, opt_d, opt_g, local)
            return
        streams = _streams(local)
        self._local = local
        for phase in super().phases(step, opt_d, opt_g, local):
            # Phase start, after the parent's replay restore and before the host block.
            self._replay_streams = streams
            self._replay_rng = self._rng(streams)
            self._replay_buffers = [(buf, buf.detach().clone()) for buf, _saved in _module_buffers(local)]
            yield phase
        self._annotate()
        if self.outer_steps % 100 == 0:
            acquire, dropout, after, _before = self.fire_counts()
            _log(event="PROGRESS", outer=self.outer_steps, armed=self.armed,
                 arm_update_index=self.arm_update_index, fires=self.fires,
                 extra_d_steps=self.extra_d_steps, modes=self._last_modes, HQ=self._last_hq,
                 fires_acquire=acquire, fires_dropout=dropout, fires_after_2300=after)

    def _annotate(self):
        if not self.records:
            return
        row = self.records[-1]
        row["armed"] = bool(self.armed)
        row["extra_d_apply"] = bool(self._fire_this_turn)
        if self._last_modes is not None:
            row["diag_modes"] = self._last_modes
            row["diag_hq"] = self._last_hq

    def _arm_smoothed_critic(self):
        super()._arm_smoothed_critic()
        if self._fire_this_turn and self.phase == 1 and not self.game_bound and not self.passthrough:
            self._extra_discriminator_steps(EXTRA_D_STEPS)

    def step(self, optimizer, ordinary_step, closure=None):
        self._ordinary_step = ordinary_step
        return super().step(optimizer, ordinary_step, closure)

    def _extra_discriminator_steps(self, count):
        """``count`` host D Adam steps on this phase's batch, before G reads D.

        The host has already drawn the batch and placed D*. Replaying from the
        phase-start RNG redraws that batch, including input and output noise.
        G's RNG, buffers and weights are put back afterwards. D's curvature
        bound on the host step is left as stall reach set it; these extra steps
        do not change either curvature bound or any learning rate.
        """
        if self.game_bound or self.phase != 1 or self._ordinary_step is None:
            raise RuntimeError("extra D steps are only the phase-1 critic update")
        if self._replay_rng is None:
            raise RuntimeError("extra D steps missing the phase-start replay point")
        frame = _host_frame()
        streams = self._replay_streams
        post_rng = self._rng(streams)
        post_buffers = _module_buffers(self._local)
        opt_d, opt_g = self.optimizers
        g_params = [p.detach().clone() for p in self._params(opt_g)]
        g_lrs = [float(group["lr"]) for group in opt_g.param_groups]
        d_lrs = [float(group["lr"]) for group in opt_d.param_groups]
        saved_smooth, saved_width = self._smooth_on, self._smooth_width
        try:
            self._smooth_on = False
            for _ in range(count):
                self._set_rng(streams, self._replay_rng)
                with torch.no_grad():
                    for buf, saved in self._replay_buffers:
                        buf.copy_(saved)
                with torch.enable_grad():
                    _rerun_host_d(frame, opt_d)
                    self._ordinary_step(opt_d)
                self.extra_d_steps += 1
        finally:
            self._set_rng(streams, post_rng)
            with torch.no_grad():
                for buf, saved in post_buffers:
                    buf.copy_(saved)
                for param, saved in zip(self._params(opt_g), g_params):
                    param.copy_(saved)
            self._smooth_on, self._smooth_width = saved_smooth, saved_width
        if [float(group["lr"]) for group in opt_g.param_groups] != g_lrs:
            raise RuntimeError("extra D changed a G learning rate")
        if [float(group["lr"]) for group in opt_d.param_groups] != d_lrs:
            raise RuntimeError("extra D changed a D learning rate")
        self.d_star = [p.detach().clone() for p in self._params(opt_d)]
        self.fires += 1
        update_index = None if self._host_step is None else self._host_step + 1
        if update_index is not None:
            self._fire_updates.append(update_index)
        _log(event="extra_d_apply", armed=True, update_index=update_index,
             modes=self._last_modes, HQ=self._last_hq, extra_d_steps_this_turn=count)

    def fire_counts(self):
        updates = self._fire_updates
        acquire = sum(1 for update in updates if update < DROPOUT_LO)
        dropout = sum(1 for update in updates if DROPOUT_LO <= update <= DROPOUT_HI)
        after = sum(1 for update in updates if update > DROPOUT_HI)
        before_arm = sum(1 for update in updates if update < ARM_UPDATE_INDEX)
        return acquire, dropout, after, before_arm

    def receipt(self):
        value = super().receipt()
        acquire, dropout, after, before_arm = self.fire_counts()
        value.update(method=METHOD, parent_method=REACH_METHOD,
                     mechanism="delayed_arm_extra_d_ttur",
                     arm_update_index_min=ARM_UPDATE_INDEX, arm_modes=ARM_MODES, arm_hq=ARM_HQ,
                     extra_d_per_turn=EXTRA_D_STEPS, armed=self.armed,
                     arm_update_index=self.arm_update_index, fires=self.fires,
                     extra_d_steps=self.extra_d_steps, fires_acquire=acquire,
                     fires_dropout=dropout, fires_after_2300=after,
                     fires_before_arm_index=before_arm,
                     last_modes=self._last_modes, last_hq=self._last_hq,
                     cpu_capability=torch.backends.cpu.get_cpu_capability())
        return value


def _streams(local):
    streams = [v for v in local.values() if isinstance(v, torch.Generator)]
    policy = local.get("noise_policy")
    if policy is not None:
        streams.extend(v for name in ("input_stream", "output_stream")
                       if isinstance((v := getattr(policy, name, None)), torch.Generator))
    return list({id(s): s for s in streams}.values())


def _module_buffers(local):
    return [(buf, buf.detach().clone()) for name in ("generator", "critic", "prior")
            if isinstance((module := local.get(name)), torch.nn.Module) for buf in module.buffers()]


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


def _rerun_host_d(frame, opt_d):
    """Replay the host's D loss on the restored batch. Does not step."""
    local = frame.f_locals
    gan = local["gan"]
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


def _log(**row):
    row.setdefault("cpu", torch.backends.cpu.get_cpu_capability())
    print(json.dumps(row, default=float), flush=True)


def diagnostic_log(recorder, user_log=None):
    """Probe checkpoint logger that can arm this recorder and no other state."""
    def log(row):
        if user_log is not None:
            user_log(row)
        if isinstance(row, dict) and row.get("event") == "checkpoint":
            recorder.consider_diagnostic(row.get("step"), row.get("modes"), row.get("hq"))
    return log


@contextmanager
def pr84_delayed_arm_extra_d(*, task="mode_hold", start_step=0):
    """Stall reach, plus sticky +2 host D steps after the delayed full-ring arm."""
    from benchmarks.locked_shared.observation import Recorder
    import benchmarks.toy100.continuous_probe as probe

    holder = {}
    original_record = Recorder.record
    original_run = probe._run_extended

    def record(self, step, measure):
        original_record(self, step, measure)
        target = holder.get("recorder")
        point = self.curve[-1] if self.curve else None
        if target is not None and point is not None and point.get("step") == step:
            target.consider_diagnostic(step, point.get("modes"), point.get("hq"))

    def run_extended(*args, **kwargs):
        target = holder.get("recorder")
        if target is not None:
            kwargs["log"] = diagnostic_log(target, kwargs.get("log"))
        return original_run(*args, **kwargs)

    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        DelayedArmExtraDRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = base.G_CURVATURE_BOUND

    base.SmoothedBothBoundRecorder = type(
        "DelayedArmExtraDRecorder", (DelayedArmExtraDRecorder,),
        dict(ramp="stall", __init__=init))
    Recorder.record = record
    probe._run_extended = run_extended
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            holder["recorder"] = value[0]
            yield value
    finally:
        probe._run_extended = original_run
        Recorder.record = original_record
        base.SmoothedBothBoundRecorder = original


__all__ = ["ARM_HQ", "ARM_MODES", "ARM_UPDATE_INDEX", "DROPOUT_HI", "DROPOUT_LO",
           "EXTRA_D_STEPS", "METHOD", "DelayedArmExtraDRecorder", "diagnostic_log",
           "pr84_delayed_arm_extra_d"]
