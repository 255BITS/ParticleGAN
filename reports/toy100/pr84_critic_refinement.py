"""Bounded empirical D refinement before the alternating PR84 G update.

The original D Adam proposal and own-curvature bound run once. At its accepted
point D*, one fixed-bank L-BFGS attempt minimizes the SAME sharp penalized D
loss, with G/prior frozen. The lowest finite training-bank loss evaluated by
that attempt selects Dhat. G's two own-curvature evaluations then both see
Dhat and the same phase-one stencil width; Dhat is the materialized next D.

The bank contains eight 128-pair draws from clones of the pre-D data/noise
streams; its first pair batch exactly reproduces the host's D gradient. No
training stream, noise clock, Adam rate or Adam moment count is advanced by
the fit. This is extra D optimization, not a certified best response. D's
retained Adam moments describe its original proposal, before refinement.

First scope: CPU mode_hold, zero discriminator input noise, fixed output
scale. Cold input noise and the conditional trajectory host are unsupported
while refinement is active. Disabled refinement preserves original PR84.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100 import pr84_critic_relaxation as fit


METHOD = "pr84_alternating_bounded_empirical_critic_refinement"
BANK_BATCHES = 8
BANK_BATCH_SIZE = 128


def _tensor_sha(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _streams(local):
    values = [v for v in local.values() if isinstance(v, torch.Generator)]
    policy = local.get("noise_policy")
    if policy is not None:
        values.extend(v for name in ("input_stream", "output_stream")
                      if isinstance((v := getattr(policy, name, None)), torch.Generator))
    return list({id(stream): stream for stream in values}.values())


@torch.no_grad()
def fixed_bank(local):
    """Cache detached training-law data from cloned pre-D streams only."""
    policy = local.get("noise_policy")
    if policy is None or policy.input_sigma != 0 or policy.output_scale is not None:
        raise ValueError("critic refinement requires input_sigma=0 and a fixed output scale")
    generator = getattr(local["generator"], "model", local["generator"])
    critic = getattr(local["critic"], "model", local["critic"])
    if (not isinstance(generator, SimpleMLPGenerator)
            or not isinstance(critic, SimpleMLPDiscriminator)
            or local.get("slow") is not None or local["batch"] != BANK_BATCH_SIZE
            or next(generator.parameters()).device.type != "cpu"):
        raise ValueError("critic refinement is scoped to the frozen CPU mode_hold host")
    stream = torch.Generator()
    stream.set_state(local["stream"].get_state())
    output = None
    if policy.output_stream is not None:
        output = torch.Generator()
        output.set_state(policy.output_stream.get_state())
    rows = []
    # fork_rng restores the global stream, including on a failed bank build.
    with torch.random.fork_rng(devices=[]):
        for _ in range(BANK_BATCHES):
            real = mode_hold.sample_ring(local["means"], BANK_BATCH_SIZE, mode_hold.SIGMA, stream)
            latent, _ = local["prior"].sample(BANK_BATCH_SIZE, generator=stream)
            clean = generator(latent)
            if policy.output_sigma:
                noise = (torch.randn_like(clean) if output is None else torch.randn(
                    clean.shape, generator=output, device=clean.device, dtype=clean.dtype))
                fake = clean + policy.output_sigma * noise
            else:
                fake = clean
            rows.append(dict(real=real.detach(), fake=fake.detach()))
    bank = {key: torch.cat([row[key] for row in rows]) for key in ("real", "fake")}
    return rows[0], bank


class CriticRefinementRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, refinement=True):
        super().__init__(start_step=start_step)
        self.refinement = bool(refinement)
        self.refinement_records = []
        self.fit_gradient_evaluations = 0
        self.parity_gradient_evaluations = 0
        self.bank_rng_verified = 0
        self.fit_rng_verified = 0
        self._active = False
        self._bank = self._first_batch = self._fit_row = self._fixed_stencil = None

    def phases(self, step, opt_d, opt_g, local):
        self._active = self.enabled and step >= self.start_step and self.refinement
        self._step = step
        self._fit_row = self._fixed_stencil = None
        if self._active:
            streams = _streams(local)
            before = self._rng(streams)
            self._first_batch, self._bank = fixed_bank(local)
            if not all(torch.equal(a, b) for a, b in zip(before, self._rng(streams))):
                raise RuntimeError("refinement bank advanced a training random stream")
            self.bank_rng_verified += 1
        for phase in super().phases(step, opt_d, opt_g, local):
            yield phase
        if self._active:
            if self._fit_row is None or self._fixed_stencil is None:
                raise RuntimeError("refinement did not finish before the accepted G update")
            self._fit_row.update(outer_step=self.outer_steps, host_update=step + 1,
                                 frozen_g_stencil_width=self._fixed_stencil[1])
            self.refinement_records.append(self._fit_row)
            self.records[-1]["critic_refinement"] = self._fit_row

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if (self._active and not self.passthrough and self.phase == 0
                and optimizer is self.optimizers[0]):
            # The first cached batch must be exactly the actual D batch, not a
            # stale G-phase batch or a new sample. autograd.grad leaves the
            # host .grad values and Adam moments intact.
            critic = getattr(self._local["critic"], "model", self._local["critic"])
            before = [p.grad.detach().clone() for p in critic.parameters()]
            with torch.enable_grad():
                loss = fit.d_loss(critic, self._first_batch, self._local["gan"],
                                  self._local["regularizer"], self._step + 1)[0]
                checked = fit.gradients(loss, critic)
            self.parity_gradient_evaluations += 1
            if not all(torch.equal(a, b) for a, b in zip(before, checked)):
                raise RuntimeError("cloned first bank batch differs from the original D gradient")
        return super().step(optimizer, ordinary_step, closure)

    @torch.no_grad()
    def _arm_smoothed_critic(self):
        if not self._active or self.passthrough or self.phase not in (1, 2):
            return super()._arm_smoothed_critic()
        if self.phase == 2:
            if self._fixed_stencil is None:
                raise RuntimeError("G proposal query lacks its fixed refined opponent")
            if any(not torch.equal(p, saved) for p, saved in
                   zip(self._params(self.optimizers[0]), self.d_star)):
                raise RuntimeError("refined critic changed between G evaluations")
            self._smooth_on, self._smooth_width = self._fixed_stencil
            return
        local = self._local
        critic = getattr(local["critic"], "model", local["critic"])
        self._smooth_on = False  # Fit the original sharp D objective.
        streams = _streams(local)
        rng = self._rng(streams)
        before = [p.detach().clone() for p in critic.parameters()]
        # Current frozen hosts have only immutable Fourier buffers. Refuse a
        # mutable-buffer extension rather than silently changing fit semantics.
        buffers = [(b, b.detach().clone()) for b in critic.buffers()]
        try:
            with torch.enable_grad():
                row = fit.relax(critic, self._bank, local["gan"], local["regularizer"],
                                self._step + 1, self.metric_d)
            if not all(torch.equal(a, b) for a, b in zip(rng, self._rng(streams))):
                raise RuntimeError("fixed-bank critic fit consumed a random stream")
            if any(not torch.equal(b, saved) for b, saved in buffers):
                raise RuntimeError("mutable critic buffers are unsupported by this fit")
        except BaseException:
            for parameter, saved in zip(critic.parameters(), before):
                parameter.copy_(saved)
            for buffer, saved in buffers:
                buffer.copy_(saved)
            self._set_rng(streams, rng)
            raise
        self.fit_rng_verified += 1
        self.fit_gradient_evaluations += row["closure_calls"]
        self.d_star = [p.detach().clone() for p in self._params(self.optimizers[0])]
        row.update(first_bank_gradient_bitwise_equal=True,
                   bank_sha256=_tensor_sha(self._bank["real"], self._bank["fake"]),
                   first_bank_sha256=_tensor_sha(self._first_batch["real"], self._first_batch["fake"]),
                   refinement_parameter_norm=sum(float((p - old).double().square().sum())
                       for p, old in zip(critic.parameters(), before)) ** .5,
                   best_training_loss=min(point["total_loss"] for point in row["records"]),
                   initial_training_loss=row["records"][0]["total_loss"],
                   adam_moments="retained from original D Adam proposal; no refinement moment update",
                   convergence_claim=False)
        self._fit_row = row
        super()._arm_smoothed_critic()
        self._fixed_stencil = self._smooth_on, self._smooth_width

    def receipt(self):
        result = super().receipt()
        base = 3 * self.outer_steps
        total_d = base + self.fit_gradient_evaluations + self.parity_gradient_evaluations
        result.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
            refinement=bool(self.refinement), refinement_records=self.refinement_records,
            refinement_scope="CPU mode_hold, input_sigma=0, fixed output scale only",
            bank_batches=BANK_BATCHES, bank_batch_size=BANK_BATCH_SIZE,
            fit_max_iterations=fit.MAX_ITER, fit_hard_max_closures=fit.MAX_CLOSURES,
            fit_selection="lowest finite SAME penalized D training-bank loss; no heldout/quality selection",
            active_stencil_policy="freeze phase1 width through phase2 at materialized Dhat",
            d_bound_scope="original D Adam proposal only; excludes the explicit refinement",
            adam_moment_scope="one D and one G+prior update; D moments precede refinement",
            bank_rng_verified=self.bank_rng_verified, fit_rng_verified=self.fit_rng_verified,
            additional_d_gradient_evaluations=self.fit_gradient_evaluations + self.parity_gradient_evaluations,
            fit_gradient_evaluations=self.fit_gradient_evaluations,
            parity_gradient_evaluations=self.parity_gradient_evaluations,
            per_role_gradient_evaluations=dict(d=total_d, g=base),
            gradient_evaluations_per_outer_step=(dict(d=total_d / self.outer_steps, g=3.)
                                                 if self.outer_steps else None),
            fit_sample_pairs_evaluated=self.fit_gradient_evaluations * BANK_BATCHES * BANK_BATCH_SIZE,
            gradient_query_size_note="three host batches/player plus one D parity batch and 1024-pair fit closures",
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fit_source_sha256=hashlib.sha256(Path(fit.__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def pr84_critic_refinement(*, task="mode_hold", start_step=0, refinement=True):
    if task != "mode_hold" and refinement:
        raise ValueError("active critic refinement has only been declared for mode_hold")
    def factory(*, start_step=0):
        return CriticRefinementRecorder(start_step=start_step, refinement=refinement)
    with patch.object(frozen, "SmoothedBothBoundRecorder", factory):
        with frozen.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value
