"""Stall reach (#107) plus one post-acquire mechanism: EMA discriminator for G.

Until the run first reaches the full 8-mode ring at HQ >= 0.9, every update is
the stall-reach update. After that checkpoint, D still steps on its live
weights. G's adversarial forward uses a slow exponential moving average of
those weights (tau 0.99), through the same stall-reach stencil. No coverage
term, anchor, mode quota, or width retune.
"""

from contextlib import contextmanager, nullcontext
import copy
import json
from unittest.mock import patch

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import ReachRecorder


METHOD = "pr84_stall_reach_post_acquire_ema_d"
TAU = 0.99
ARM_MODES = 8
ARM_HQ = 0.9


def _inner_critic(module):
    while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
        module = module.model
    return module if isinstance(module, SimpleMLPDiscriminator) else None


class PostAcquireEmaRecorder(ReachRecorder):
    """Stall-reach recorder that routes G through an EMA critic after acquire."""

    ramp = "stall"
    tau = TAU

    def __init__(self, *, start_step=0, tau=TAU):
        super().__init__(start_step=start_step)
        if not torch.isfinite(torch.tensor(float(tau))) or not 0. < float(tau) < 1.:
            raise ValueError("invalid EMA tau")
        self.tau = float(tau)
        self._ema_module = None
        self._live_critic = None
        self._g_reads_ema = False
        self._ema_refreshed = False
        self._ema_updates = 0
        self._arm_step = None
        self._arm_hq = None

    def phases(self, step, opt_d, opt_g, local):
        self._ema_refreshed = False
        self._g_reads_ema = False
        self._local = local
        active = bool(self.enabled) and step >= self.start_step and not self.game_bound
        if active and self._ema_module is None:
            self._maybe_arm(step)
        for phase in super().phases(step, opt_d, opt_g, local):
            self._g_reads_ema = False
            yield phase
        if not active or not self.records:
            return
        row = self.records[-1]
        row["ema_on"] = self._ema_module is not None
        if self._arm_step == step:
            row["ema_arm"] = True
        if (step + 1) % 50 == 0:
            factor = (row.get("g") or {}).get("factor")
            print(json.dumps(dict(
                event="STEP", step=step + 1, ema=self._ema_module is not None,
                ema_updates=self._ema_updates, sharp=row.get("critic_sharpness"),
                width=row.get("critic_width"), g_factor=factor,
                adv=row.get("critic_advantage"),
                cpu=torch.backends.cpu.get_cpu_capability())), flush=True)

    def _arm_smoothed_critic(self):
        # Sharpness and the stall width stay on the live critic.
        self._g_reads_ema = False
        super()._arm_smoothed_critic()
        if self._ema_module is None:
            return
        if self.phase == 1:
            self._update_ema()
        self._g_reads_ema = True

    @torch.no_grad()
    def _maybe_arm(self, completed):
        grade = self._grade(completed)
        if grade is None or grade["modes"] != ARM_MODES or grade["hq"] < ARM_HQ:
            return
        module = _inner_critic((self._local or {}).get("critic"))
        if module is None:
            return
        self._live_critic = module
        self._ema_module = copy.deepcopy(module).eval().requires_grad_(False)
        self._arm_step = int(completed)
        self._arm_hq = float(grade["hq"])
        print(json.dumps(dict(
            event="EMA_ARM", acquired_step=int(completed), first_g_step=int(completed) + 1,
            modes=grade["modes"], hq=grade["hq"], tau=self.tau,
            cpu=torch.backends.cpu.get_cpu_capability())), flush=True)

    @torch.no_grad()
    def _grade(self, completed):
        """Same 8-mode HQ read as the mode-hold gate, on a forked RNG."""
        from benchmarks.locked_shared.mode_hold import EVAL_N, diversity

        local = self._local or {}
        means, generator, prior = local.get("means"), local.get("generator"), local.get("prior")
        if means is None or generator is None or prior is None or tuple(means.shape) != (ARM_MODES, 2):
            return None
        policy = local.get("noise_policy")
        context = policy.evaluation(int(completed)) if policy is not None else nullcontext()
        with context, torch.random.fork_rng(devices=[]):
            latent, _ = prior.sample(
                EVAL_N, generator=torch.Generator().manual_seed(int(local.get("seed", 0)) + 9))
            return diversity(generator(latent), means)

    @torch.no_grad()
    def _update_ema(self):
        # Once per outer step, after D* has materialized (phase 1) and before G steps.
        if (self.phase != 1 or self._ema_module is None or self._live_critic is None
                or self._ema_refreshed):
            return
        for ema_p, live_p in zip(self._ema_module.parameters(), self._live_critic.parameters()):
            ema_p.mul_(self.tau).add_(live_p, alpha=1. - self.tau)
        self._ema_refreshed = True
        self._ema_updates += 1

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, ema_tau=self.tau,
                     ema_arm_modes=ARM_MODES, ema_arm_hq=ARM_HQ,
                     ema_armed=self._ema_module is not None, ema_acquired_step=self._arm_step,
                     ema_first_generator_step=None if self._arm_step is None else self._arm_step + 1,
                     ema_acquired_hq=self._arm_hq, ema_updates=self._ema_updates,
                     shared_gate_eligible=False)
        return value


@contextmanager
def pr84_post_acquire_ema_candidate(*, task="mode_hold", start_step=0, tau=TAU):
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        PostAcquireEmaRecorder.__init__(self, start_step=start_step, tau=tau)

    base.SmoothedBothBoundRecorder = type(
        "PostAcquireEmaRecorder", (PostAcquireEmaRecorder,),
        dict(ramp="stall", tau=tau, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
            smoothed = SimpleMLPDiscriminator.forward

            def forward(self, x):
                if (recorder._g_reads_ema and recorder.enabled and not recorder.passthrough
                        and recorder._ema_module is not None):
                    return smoothed(recorder._ema_module, x)
                return smoothed(self, x)

            with patch.object(SimpleMLPDiscriminator, "forward", forward):
                yield recorder, source
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "TAU", "ARM_HQ", "ARM_MODES", "PostAcquireEmaRecorder",
           "pr84_post_acquire_ema_candidate"]
