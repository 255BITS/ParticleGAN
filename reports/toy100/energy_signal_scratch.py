"""Experimental data-space check of each ordinary G/prior Adam proposal.

The discriminator updates normally. At the generator step, use the already
sampled real batch and learned-prior indices to compare the clean generated
cloud before and after the ordinary joint G/prior proposal. For conditional
hosts, the comparison uses observed (condition, output) pairs. A proposal is
retained only if both disjoint half-batches reduce the empirical energy
distance to real data. This is a signal diagnostic and a scratch update rule,
not a convergence theorem or a frozen-gate-eligible production policy.
"""

from contextlib import contextmanager
import inspect
import math
from unittest.mock import patch

import torch


def energy_score(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Return energy distance without the unchanged real-real term."""
    if real.ndim != 2 or fake.ndim != 2 or real.shape != fake.shape or len(real) < 2:
        raise ValueError("paired real/fake batches must have matching 2D shapes")
    return float(2 * torch.cdist(real, fake).mean() - torch.cdist(fake, fake).mean())


def _host_locals() -> dict:
    frame = inspect.currentframe()
    try:
        hosts = {"benchmarks.locked_shared.mode_hold": "train_mode_hold",
                 "benchmarks.locked_shared.trajectory": "train"}
        while frame is not None and hosts.get(frame.f_globals.get("__name__")) != frame.f_code.co_name:
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("energy signal requires a frozen supported host")
        required = ("generator", "prior", "opt_g", "opt_d")
        if any(name not in frame.f_locals for name in required):
            raise RuntimeError("frozen host frame changed")
        return {**{name: frame.f_locals[name] for name in
                   (*required, "latent", "real_g", "_", "slow", "paired")
                   if name in frame.f_locals},
                "host_kind": frame.f_globals["__name__"]}
    finally:
        del frame


class EnergySignalRecorder:
    def __init__(self, *, observe_only: bool = False,
                 diagnostic_steps: frozenset[int] = frozenset(),
                 backtracking: bool = False):
        if observe_only and backtracking:
            raise ValueError("observer and backtracking are mutually exclusive")
        self.observe_only = observe_only
        self.diagnostic_steps = diagnostic_steps
        self.backtracking = backtracking
        self.receipt = dict(policy="paired_split_energy_proposal_v1",
                            shared_gate_eligible=False,
                            observe_only=observe_only, backtracking=backtracking,
                            rows=[], diagnostic_scales=[.5, .25, .125]
                            if diagnostic_steps or backtracking else [])

    def step(self, optimizer, ordinary_step, closure=None):
        if closure is not None:
            raise RuntimeError("energy signal does not support Adam closures")
        host = _host_locals()
        if optimizer is host["opt_d"]:
            return ordinary_step(optimizer)
        if optimizer is not host["opt_g"]:
            raise RuntimeError("unknown optimizer in mode_hold host")
        generator, prior = host["generator"], host["prior"]
        clean = getattr(generator, "model", generator)
        if host["host_kind"] == "benchmarks.locked_shared.mode_hold":
            if any(name not in host for name in ("latent", "real_g", "_")):
                raise RuntimeError("mode-hold generator sampling variables are missing")
            indices = host["_"]
            real = host["real_g"].detach()
            if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
                    or len(indices) != len(real)):
                raise RuntimeError("mode-hold prior indices are missing")
            def generated():
                return clean(prior.z[indices])
        elif host["host_kind"] == "benchmarks.locked_shared.trajectory":
            if any(name not in host for name in ("slow", "paired")):
                raise RuntimeError("trajectory conditional samples are missing")
            slow = host["slow"].detach()
            real = torch.cat((slow, host["paired"].detach()), dim=-1)
            def generated():
                return torch.cat((slow, clean(slow, prior.z)), dim=-1)
        else:
            raise RuntimeError("unsupported host")
        params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if len(params) != len(set(params)):
            raise RuntimeError("joint G/prior optimizer has duplicate parameters")
        if {id(p) for p in clean.parameters()} != {id(p) for p in optimizer.param_groups[0]["params"]}:
            raise RuntimeError("clean generator differs from network optimizer group")
        half = len(real) // 2
        if half < 2 or half * 2 != len(real):
            raise RuntimeError("energy check requires an even batch of at least four")
        with torch.no_grad():
            before_params = [p.detach().clone() for p in params]
            before_fake = generated().detach()
            before = [energy_score(real[s], before_fake[s]) for s in
                      (slice(0, half), slice(half, None))]
        result = ordinary_step(optimizer)
        with torch.no_grad():
            after_fake = generated().detach()
            after = [energy_score(real[s], after_fake[s]) for s in
                     (slice(0, half), slice(half, None))]
            if not all(math.isfinite(value) for value in before + after):
                raise FloatingPointError("nonfinite energy proposal score")
            accepted = all(new < old for old, new in zip(before, after))
            accepted_scale = 1. if accepted else 0.
            fractional = []
            if not accepted and (self.backtracking or
                                 len(self.receipt["rows"]) + 1 in self.diagnostic_steps):
                full_params = [p.detach().clone() for p in params]
                for scale in self.receipt["diagnostic_scales"]:
                    for parameter, original, full in zip(params, before_params, full_params):
                        parameter.copy_(torch.lerp(original, full, scale))
                    fake = generated().detach()
                    values = [energy_score(real[s], fake[s]) for s in
                              (slice(0, half), slice(half, None))]
                    if not all(math.isfinite(value) for value in values):
                        raise FloatingPointError("nonfinite fractional energy score")
                    improves = all(new < old for old, new in zip(before, values))
                    fractional.append(dict(scale=scale, after=values,
                                           improves_both=improves))
                    if improves and self.backtracking:
                        accepted_scale = scale
                        break
                if not self.backtracking:
                    for parameter, full in zip(params, full_params):
                        parameter.copy_(full)
            if accepted_scale == 0 and not self.observe_only:
                for parameter, original in zip(params, before_params):
                    parameter.copy_(original)
        self.receipt["rows"].append(dict(step=len(self.receipt["rows"]) + 1,
                                         before=before, after=after,
                                         accepted_by_signal=accepted,
                                         accepted_scale=accepted_scale,
                                         proposal_kept=accepted_scale > 0 or self.observe_only,
                                         fractional_diagnostic=fractional,
                                         rates=[float(group["lr"]) for group in optimizer.param_groups]))
        return result


@contextmanager
def energy_signal(*, observe_only: bool = False,
                  diagnostic_steps: frozenset[int] = frozenset(),
                  backtracking: bool = False):
    recorder = EnergySignalRecorder(observe_only=observe_only,
                                    diagnostic_steps=diagnostic_steps,
                                    backtracking=backtracking)
    ordinary_step = torch.optim.Adam.step
    def delegated(optimizer, closure=None):
        return recorder.step(optimizer, ordinary_step, closure)
    with patch.object(torch.optim.Adam, "step", delegated):
        yield recorder.receipt


@contextmanager
def warm_energy_signal(state, *, observe_only: bool = False,
                       backtracking: bool = False):
    recorder = EnergySignalRecorder(observe_only=observe_only,
                                    backtracking=backtracking)
    ordinary_step = state["base_adam_step"]
    def delegated(optimizer, closure=None):
        return recorder.step(optimizer, ordinary_step, closure)
    state["set_step_delegate"](delegated)
    try:
        yield recorder.receipt
    finally:
        state["set_step_delegate"](ordinary_step)
