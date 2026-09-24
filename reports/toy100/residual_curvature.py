"""G steps on the PR #84 stencil. The curvature bound sees only the residual.

v17 measured G's own-curvature on the plain critic and shrank acquisition.
Here the step and its moments still come from the five-point field. The bound
uses the gradient the stencil removes, ``plain - smoothed``, at the same two
parameter states. A flat smoothed field at rest leaves that residual, so the
.25 cap can still bite. Curvature that exists in both fields does not.
"""

from contextlib import contextmanager

import torch

from reports.toy100.alternating_curvature_scratch import _rho
from reports.toy100.pr84_smoothed_candidate import SmoothedBothBoundRecorder, pr84_smoothed_candidate


METHOD = "pr84_stencil_step_residual_curvature_bound"
PHASES = 5


def residual_grads(plain, smooth):
    return [a - b for a, b in zip(plain, smooth)]


class ResidualCurvatureRecorder(SmoothedBothBoundRecorder):
    """Five passes: the original three, then plain grads at G1 and at G0."""

    def phases(self, step, opt_d, opt_g, local):
        self._local = local
        if not self.enabled or step < self.start_step:
            self.passthrough = True
            try:
                yield 0
            finally:
                self.passthrough = False
            return
        if self.optimizers is None:
            self.optimizers = (opt_d, opt_g)
            self.rows = {opt: dict(role=role, calls=0) for role, opt in zip(("d", "g"), self.optimizers)}
        if self.optimizers != (opt_d, opt_g):
            raise RuntimeError("game optimizers changed")
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
        for phase in range(PHASES):
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
        if self.passthrough or self.phase is None or self.phase < 2:
            return super().step(optimizer, ordinary_step, closure)
        if optimizer not in self.rows or closure is not None:
            raise RuntimeError("optimizer call outside the declared game update")
        self.rows[optimizer]["calls"] += 1
        opt_d, opt_g = self.optimizers
        grads = lambda opt: [p.grad.detach().clone() for p in self._params(opt)]
        if optimizer is opt_d:
            for p, v in zip(self._params(opt_d), self.d_star):
                p.copy_(v)
            if self.phase == 2:
                self._arm_smoothed_critic()
            else:
                self._smooth_on = False
                self._smooth_width = 0.
            return None
        if self.phase == 2:
            self.smooth_g1 = grads(opt_g)
            return None
        if self.phase == 3:
            self.plain_g1 = grads(opt_g)
            for p, v in zip(self._params(opt_g), self.g_base):
                p.copy_(v)
            return None
        rho = _rho(self.g_base, self.g1, residual_grads(grads(opt_g), self.gg0),
                   residual_grads(self.plain_g1, self.smooth_g1), self.metric_g)
        factor = min(1., self.curvature_bound / rho) if rho > 0 else 1.
        for p, b, n in zip(self._params(opt_g), self.g_base, self.g1):
            p.copy_(torch.lerp(b, n, factor) if factor < 1 else n)
        self.row["g"] = dict(rho=rho, factor=factor, curvature_critic="residual")
        print(f"residual step={self.row.get('outer_step')} rho={rho:.6f} factor={factor:.6f}", flush=True)
        return None

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
                     curvature_critic="residual", g_step_field="five_point_stencil")
        return value


@contextmanager
def residual_curvature(*, task="mode_hold", start_step=0):
    """Same host patch as the PR #84 candidate, with the residual recorder."""
    with pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
        residual = ResidualCurvatureRecorder(start_step=start_step)
        residual.host_source = recorder.host_source
        # The context already bound `recorder`. Swap the object the patches close over
        # by copying identity-sensitive hooks onto the yielded instance.
        yield _swap(recorder, residual), source


def _swap(bound, residual):
    """Point the already-patched callbacks at ``residual``.

    The PR #84 context closes over its recorder. Mutating that instance's class
    keeps ``_smooth_on`` and ``step`` on the object the patches already hold.
    """
    bound.__class__ = ResidualCurvatureRecorder
    return bound
