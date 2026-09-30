"""PR84 alternating update plus one critic-value spare-particle pull.

The relativistic generator step is unchanged. After it, on the unconditional
ring only, one prior particle may take a trust-region step toward real samples
the current critic values above the generated cloud. Conditional trajectory
keeps the original PR84 update.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.coverage_pullback import prior_adam_metric
from reports.toy100.value_transport import spare_particle_pull


METHOD = "pr84_smoothed_g_with_critic_value_spare_particle"


class ValueTransportRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, task="mode_hold", correction=True):
        super().__init__(start_step=start_step)
        self.task = task
        self.correction = bool(correction)
        self.real_samples = []
        self.phase_zero_samples = None
        self.batch_replays_verified = 0
        self.transport_records = []

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
            expected = local.get("batch", mode_hold.BATCH)
            if len(self.real_samples) != 2 or self.real_samples[0].shape != (expected, 2):
                raise RuntimeError("current-phase real minibatch capture changed")
            if phase == 0:
                self.phase_zero_samples = self.real_samples
            elif not all(torch.equal(a, b) for a, b in zip(self.phase_zero_samples, self.real_samples)):
                raise RuntimeError("current real minibatches did not replay exactly")
            else:
                self.batch_replays_verified += 1
            if phase == 2 and self.correction:
                self._correct(self.real_samples[0], opt_g, local)

    def _correct(self, real, opt_g, local):
        if local.get("slow") is not None:
            raise RuntimeError("conditional host cannot use marginal value transport")
        generator, prior, critic = local.get("generator"), local.get("prior"), local.get("critic")
        if generator is None or prior is None or critic is None or not hasattr(prior, "z"):
            raise RuntimeError("unconditional particle generator is required")
        clean = getattr(generator, "model", generator)
        metric = prior_adam_metric(opt_g, prior.z)
        streams = [value for value in local.values() if isinstance(value, torch.Generator)]
        policy = local.get("noise_policy")
        if policy is not None:
            streams.extend(value for name in ("input_stream", "output_stream")
                           if isinstance((value := getattr(policy, name, None)), torch.Generator))
        streams = list({id(stream): stream for stream in streams}.values())
        saved_streams = [stream.get_state() for stream in streams]
        saved_global = torch.get_rng_state()
        parameters = [p.detach().clone() for p in generator.parameters()]
        moments = {name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                   for name, value in opt_g.state[prior.z].items()}
        module = critic
        while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
            module = module.model
        if not isinstance(module, SimpleMLPDiscriminator):
            raise RuntimeError("ring critic must be the unconditional MLP")
        with torch.no_grad():
            fake = clean(prior.z).detach()
            # Score the inner MLP. The noise wrapper is outside this module, and
            # the pull must not draw from the host streams.
            fake_scores = module(fake).detach().reshape(-1)
            real_scores = module(real).detach().reshape(-1)
        row = spare_particle_pull(clean, prior.z, real, fake_scores, real_scores, metric)
        for stream, state in zip(streams, saved_streams):
            stream.set_state(state)
        torch.set_rng_state(saved_global)
        if any(not torch.equal(p, before) for p, before in zip(generator.parameters(), parameters)):
            raise RuntimeError("value transport modified the generator network")
        for name, before in moments.items():
            after = opt_g.state[prior.z][name]
            if isinstance(before, torch.Tensor) and not torch.equal(before, after):
                raise RuntimeError("value transport modified Adam state")
        row["outer_step"] = self.outer_steps + 1
        self.transport_records.append(row)
        self.row["value_transport"] = {
            key: row.get(key) for key in
            ("fired", "accepted", "alpha", "gap", "spread", "nearest", "twin", "ahead",
             "latent_norm", "output_error_before", "output_error_after")}

    def receipt(self):
        result = super().receipt()
        result.update(
            method=METHOD, scratch_optimizer_policy=METHOD,
            helper_sha256=hashlib.sha256(Path(__file__).with_name("value_transport.py").read_bytes()).hexdigest(),
            value_transport_scope="unconditional current real batch, clean G, one prior particle",
            conditional_policy="leave original PR84 update unchanged",
            current_batch_replays_verified=self.batch_replays_verified,
            correction_enabled=self.correction,
            transport_accepted=sum(bool(row["accepted"]) for row in self.transport_records),
            transport_fired=sum(bool(row["fired"]) for row in self.transport_records),
            transport_records=self.transport_records)
        return result


@contextmanager
def value_transport_candidate(*, task="mode_hold", start_step=0, correction=True):
    def factory(*, start_step=0):
        return ValueTransportRecorder(start_step=start_step, task=task, correction=correction)
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
