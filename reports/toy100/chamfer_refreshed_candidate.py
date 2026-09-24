"""PR84 alternating update plus a refreshed-target nonlinear prior landing.

Target centers are not inputs. The target is four nearest-neighbor refreshes
on the current real minibatch and clean support. Only prior rows move, and
only when the actual unit-mean Chamfer objective decreases.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.chamfer_refreshed_pullback import nonlinear_refreshed_pullback
from reports.toy100.chamfer_smoothed_candidate import ChamferSmoothedRecorder


METHOD = "pr84_smoothed_g_with_refreshed_chamfer_nonlinear_prior"


class RefreshedChamferRecorder(ChamferSmoothedRecorder):
    def _correct(self, real, opt_g):
        local = self._local or {}
        if local.get("slow") is not None:
            raise RuntimeError("conditional host cannot use marginal Chamfer")
        generator, prior = local.get("generator"), local.get("prior")
        if generator is None or prior is None or not hasattr(prior, "z"):
            raise RuntimeError("unconditional particle generator is required")
        clean = getattr(generator, "model", generator)
        rng_before = torch.get_rng_state().clone()
        parameters_before = [p.detach().clone() for p in generator.parameters()]
        state_before = {name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for name, value in opt_g.state[prior.z].items()}
        row = nonlinear_refreshed_pullback(clean, prior.z, real)
        if not torch.equal(torch.get_rng_state(), rng_before):
            raise RuntimeError("refreshed Chamfer correction consumed global RNG")
        if any(not torch.equal(p, before) for p, before in zip(generator.parameters(),
                                                                parameters_before)):
            raise RuntimeError("refreshed Chamfer correction modified generator network")
        for name, before in state_before.items():
            after = opt_g.state[prior.z][name]
            if isinstance(before, torch.Tensor) and not torch.equal(before, after):
                raise RuntimeError("refreshed Chamfer correction modified Adam state")
        row.pop("output_before")
        row.pop("output_after")
        row.pop("target_points")
        row["outer_step"] = self.outer_steps + 1
        self.chamfer_records.append(row)
        self.row["chamfer"] = {name: row[name] for name in (
            "accepted", "coverage_before", "coverage_after", "backward_before",
            "backward_after", "objective_before", "objective_after", "empty_cells",
            "actual_latent_displacement_norm", "final_target_error", "rounds")}

    def receipt(self):
        result = super().receipt()
        result.update(
            method=METHOD, scratch_optimizer_policy=METHOD,
            helper_sha256=hashlib.sha256(
                Path(__file__).with_name("chamfer_refreshed_pullback.py").read_bytes()).hexdigest(),
            target_rule="four output-space refreshes of the unit-mean Chamfer assignment target",
            landing="prior-only Gauss-Newton onto that frozen target",
            total_objective="accept only if actual S=C+Q decreases; coefficients remain one",
            chamfer_scope="unconditional current real batch, clean G, prior-only correction",
            conditional_policy="leave original PR84 update unchanged",
            current_batch_replays_verified=self.batch_replays_verified,
            correction_enabled=self.correction, chamfer_records=self.chamfer_records,
            chamfer_accepted=sum(row["accepted"] for row in self.chamfer_records))
        return result


@contextmanager
def refreshed_chamfer_candidate(*, task="mode_hold", start_step=0, correction=True):
    def factory(*, start_step=0):
        return RefreshedChamferRecorder(start_step=start_step, task=task, correction=correction)
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
