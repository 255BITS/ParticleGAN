"""PR84 alternating game plus one sampled-real bidirectional Chamfer correction.

This is an explicit real-to-support plus support-to-real objective addition. The original
sharp-D/smoothed-G Adam update, same-batch curvature replays, noise clocks, and
EMA remain in the frozen PR84 adapter. Only the mode-hold host's unconditional
real batch is available to this research adapter; conditional trajectory keeps
the original PR84 update, since marginal matching would break identity.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.chamfer_pullback import chamfer_pullback
from reports.toy100.coverage_pullback import prior_adam_metric


METHOD = "pr84_smoothed_g_with_bidirectional_chamfer_prior_pullback"


class ChamferSmoothedRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, task="mode_hold", correction=True):
        super().__init__(start_step=start_step)
        self.task = task
        self.correction = bool(correction)
        self.real_samples = []
        self.phase_zero_samples = None
        self.chamfer_records = []
        self.batch_replays_verified = 0

    def capture_sample(self, value):
        if (self.task == "mode_hold" and self.enabled and not self.passthrough
                and self.phase is not None):
            self.real_samples.append(value.detach().clone())

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, local):
            self.real_samples = []
            yield phase
            if self.task != "mode_hold" or not self.enabled or self.passthrough:
                continue
            expected_batch = local.get("batch", mode_hold.BATCH)
            if len(self.real_samples) != 2 or self.real_samples[0].shape != (expected_batch, 2):
                raise RuntimeError("current-phase D/G real minibatch capture changed")
            if phase == 0:
                self.phase_zero_samples = self.real_samples
            elif not all(torch.equal(a, b) for a, b in zip(self.phase_zero_samples,
                                                             self.real_samples)):
                raise RuntimeError("current real minibatches did not replay exactly")
            else:
                self.batch_replays_verified += 1
            if phase == 2 and self.correction:
                self._correct(self.real_samples[0], opt_g)

    def _correct(self, real, opt_g):
        local = self._local or {}
        if local.get("slow") is not None:
            raise RuntimeError("conditional host cannot use marginal Chamfer")
        generator, prior = local.get("generator"), local.get("prior")
        if generator is None or prior is None or not hasattr(prior, "z"):
            raise RuntimeError("unconditional particle generator is required")
        clean = getattr(generator, "model", generator)
        metric = prior_adam_metric(opt_g, prior.z)
        rng_before = torch.get_rng_state().clone()
        parameters_before = [p.detach().clone() for p in generator.parameters()]
        state_before = {name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for name, value in opt_g.state[prior.z].items()}
        row = chamfer_pullback(clean, prior.z, real, metric)
        if not torch.equal(torch.get_rng_state(), rng_before):
            raise RuntimeError("Chamfer correction consumed global RNG")
        if any(not torch.equal(p, before) for p, before in zip(generator.parameters(),
                                                                parameters_before)):
            raise RuntimeError("Chamfer correction modified generator network")
        for name, before in state_before.items():
            after = opt_g.state[prior.z][name]
            if isinstance(before, torch.Tensor) and not torch.equal(before, after):
                raise RuntimeError("Chamfer correction modified Adam state")
        # Full clean support arrays are retained only for one-state diagnostics.
        row.pop("output_before")
        row.pop("output_after")
        row["outer_step"] = self.outer_steps + 1
        self.chamfer_records.append(row)
        self.row["chamfer"] = {name: row[name] for name in
                              ("accepted", "alpha", "coverage_before", "coverage_after",
                               "backward_before", "backward_after", "objective_before",
                               "objective_after", "empty_cells",
                               "actual_latent_displacement_norm")}

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD, scratch_optimizer_policy=METHOD,
                      helper_sha256=hashlib.sha256(
                          Path(__file__).with_name("chamfer_pullback.py").read_bytes()).hexdigest(),
                      real_to_support_objective="C=mean_real min_particle squared output distance",
                      support_to_real_objective="Q=mean_particle min_real squared output distance",
                      total_objective="S=C+Q, both coefficients exactly one",
                      chamfer_scope="unconditional current real batch, clean G, prior-only correction",
                      conditional_policy="leave original PR84 update unchanged",
                      current_batch_replays_verified=self.batch_replays_verified,
                      correction_enabled=self.correction, chamfer_records=self.chamfer_records,
                      chamfer_accepted=sum(row["accepted"] for row in self.chamfer_records))
        return result


@contextmanager
def chamfer_smoothed_candidate(*, task="mode_hold", start_step=0, correction=True):
    def factory(*, start_step=0):
        return ChamferSmoothedRecorder(start_step=start_step, task=task,
                                      correction=correction)
    with patch.object(frozen, "SmoothedBothBoundRecorder", factory):
        with frozen.pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
            if task != "mode_hold":
                yield recorder, source
            else:
                original = mode_hold.sample_ring
                def observed(means, n, sigma, generator):
                    value = original(means, n, sigma, generator)
                    recorder.capture_sample(value)
                    return value
                with patch.object(mode_hold, "sample_ring", observed):
                    yield recorder, source
