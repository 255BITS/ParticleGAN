"""Scratch D-phase allocation with fixed Adam rates and fresh-data e-values.

One original G/prior update remains at every host outer step. After its ordinary
D update, the current D can receive at most two more full host D updates. The
decision uses Kim et al.'s difference-score e-process on a separate RNG stream.
This is a D-side test of adaptive D:G allocation, not a production adapter or
a claim of e-process validity for finite pseudorandom runs.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import math
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold


class AdaptiveDAllocation:
    def __init__(self, *, max_d: int = 3, eval_batch: int = 16,
                 a_d: float = .01, alpha_d: float = .1, rho_d: float = .5,
                 eval_seed: int = 20103, active_from: int = 0):
        if type(max_d) is not int or not 1 <= max_d <= 3:
            raise ValueError("max_d must be an integer in 1..3")
        if type(eval_batch) is not int or eval_batch < 1:
            raise ValueError("eval_batch must be positive")
        if not (0 <= a_d < 1 and 0 < alpha_d < 1 and 0 < rho_d <= 1):
            raise ValueError("invalid e-process settings")
        self.max_d, self.eval_batch = max_d, eval_batch
        self.a_d, self.alpha_d, self.rho_d = a_d, alpha_d, rho_d
        self.active_from = active_from
        self.eval_stream = torch.Generator(device="cpu").manual_seed(eval_seed)
        self.eval_seed = eval_seed
        self.original_step = None
        self.roles = {"d": [], "g": [], "prior": []}
        self.rate_records = []
        self.moment_calls = {"d": 0, "g": 0}
        self.extra_d_calls = 0
        self.eval_queries = 0
        self.phase_rows = []
        self._last_host_step = -1

    @staticmethod
    def _host_frame():
        frame = inspect.currentframe().f_back
        while frame is not None and frame.f_code.co_name != "train_mode_hold":
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("adaptive D policy was called outside mode-hold")
        local = frame.f_locals
        required = ("opt_d", "opt_g", "step", "generator", "critic", "prior",
                    "means", "batch", "stream", "gan", "regularizer", "noise_policy")
        if any(key not in local for key in required):
            raise RuntimeError("mode-hold training locals changed")
        return local

    def _observe_rates(self, role: str, optimizer):
        for group in optimizer.param_groups:
            name = ("d" if role == "d" else
                    "prior" if group.get("_comparison_prior") else "g")
            lr = float(group["lr"])
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("nonpositive/nonfinite applied Adam rate")
            self.roles[name].append(lr)
            self.rate_records.append(dict(outer_step=self._last_host_step + 1,
                                          role=name, lr=lr))

    def _log_e_increment(self, gap: torch.Tensor) -> float:
        e_pair = 2 * torch.sigmoid(gap) / (1 + 2 * self.a_d - self.a_d**2)
        log_batch = torch.log1p(e_pair).sum().item() - len(gap) * math.log(2)
        e_batch = math.exp(log_batch)
        return math.log1p(self.rho_d * (e_batch - 1))

    @torch.no_grad()
    def _fresh_log_evalue(self, local) -> float:
        """One fresh mini-batch e-value for the current fixed D and G state."""
        policy = local["noise_policy"]
        if policy is None or policy.output_scale is not None:
            raise ValueError("scratch e-process requires fixed output-noise scale")
        real = mode_hold.sample_ring(local["means"], self.eval_batch,
                                     mode_hold.SIGMA, self.eval_stream)
        latent, _ = local["prior"].sample(self.eval_batch,
                                           generator=self.eval_stream)
        generated = local["generator"].model(latent)
        if policy.output_sigma:
            generated = generated + policy.output_sigma * torch.randn(
                generated.shape, generator=self.eval_stream,
                dtype=generated.dtype, device=generated.device)
        if policy.input_sigma:
            real = real + policy.input_sigma * torch.randn(
                real.shape, generator=self.eval_stream,
                dtype=real.dtype, device=real.device)
            generated = generated + policy.input_sigma * torch.randn(
                generated.shape, generator=self.eval_stream,
                dtype=generated.dtype, device=generated.device)
        critic = local["critic"].model
        gap = critic(real).reshape(-1) - critic(generated).reshape(-1)
        self.eval_queries += 1
        return self._log_e_increment(gap)

    def _extra_d_update(self, local):
        policy = local["noise_policy"]
        real = mode_hold.sample_ring(local["means"], local["batch"],
                                     mode_hold.SIGMA, local["stream"])
        latent, _ = local["prior"].sample(local["batch"],
                                           generator=local["stream"])
        with policy.discriminator():
            fake = local["generator"](latent).detach()
        critic = local["critic"]
        loss = local["gan"].d_loss(critic(real), critic(fake))
        loss = loss + local["regularizer"](critic, real, fake,
                                           step=local["step"] + 1)
        optimizer = local["opt_d"]
        optimizer.zero_grad()
        loss.backward()
        # The ordinary host D update has already set this outer step's rate.
        self._observe_rates("d", optimizer)
        self.original_step(optimizer)
        self.moment_calls["d"] += 1
        self.extra_d_calls += 1

    def step(self, optimizer, *, closure=None):
        if closure is not None or self.original_step is None:
            raise RuntimeError("scratch allocation supports ordinary Adam only")
        local = self._host_frame()
        role = ("d" if optimizer is local["opt_d"] else
                "g" if optimizer is local["opt_g"] else None)
        if role is None:
            raise RuntimeError("unknown optimizer in mode-hold")
        outer_step = int(local["step"])
        self._last_host_step = outer_step
        self._observe_rates(role, optimizer)
        result = self.original_step(optimizer)
        self.moment_calls[role] += 1
        if role == "d" and outer_step >= self.active_from and self.max_d > 1:
            log_e = 0.0
            calls = 1
            evidence = []
            while True:
                log_e += self._fresh_log_evalue(local)
                evidence.append(log_e)
                if log_e >= -math.log(self.alpha_d) or calls >= self.max_d:
                    break
                self._extra_d_update(local)
                calls += 1
            self.phase_rows.append(dict(outer_step=outer_step + 1,
                                        d_updates=calls, log_e=evidence))
        return result

    def receipt(self) -> dict:
        rates = {name: dict(min=min(values), max=max(values),
                            observations=len(values))
                 for name, values in self.roles.items() if values}
        return dict(policy="fresh_difference_eprocess_d_allocation_v1",
                    shared_gate_eligible=False, paper="https://arxiv.org/html/2608.10096",
                    max_d=self.max_d, eval_batch=self.eval_batch,
                    a_d=self.a_d, alpha_d=self.alpha_d, rho_d=self.rho_d,
                    eval_seed=self.eval_seed, active_from=self.active_from,
                    eval_queries=self.eval_queries,
                    extra_d_gradient_evaluations=self.extra_d_calls,
                    d_gradient_evaluations=self.moment_calls["d"],
                    g_gradient_evaluations=self.moment_calls["g"],
                    adam_moment_updates=self.moment_calls,
                    applied_rate_ranges=rates, rate_records=self.rate_records,
                    phases=self.phase_rows)


@contextmanager
def adaptive_d_allocation(**options):
    controller = AdaptiveDAllocation(**options)
    controller.original_step = torch.optim.Adam.step
    with patch.object(torch.optim.Adam, "step", controller.step):
        yield controller
