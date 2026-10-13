"""CUDA zero-smoothing compatibility against the pre-smoothing DualNorm rule.

The reference below retains the DualNorm/row-only branches from develop
83b099d1e4330dda953d5fce4f68ce00f75fa9a6, independent of the new polar helper.
This tiny API fixture checks software parity, not scientific acquisition.
"""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from experiments.forge.state import state_digest
from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
import particlegan.optim.dualnorm as dualnorm


class LegacyDualNorm(dualnorm.NormalizedOptimizer):
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None and self.family == "dualnorm" and self.momentum == 0
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                assert not gradient.is_sparse and not gradient.is_complex()
                if group["algorithm"] == "rownorm" and group["sampled_rows_required"]:
                    assert parameter in self._sampled_rows
        for group in self.param_groups:
            algorithm, rate, eps = group["algorithm"], group["lr"], group["eps"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                if algorithm == "rownorm":
                    rows = (self._sampled_rows[parameter] if group["sampled_rows_required"]
                            else torch.arange(len(parameter), device=parameter.device))
                    selected = gradient[rows]
                    update = selected / (selected.norm(dim=1, keepdim=True) + eps)
                    parameter.index_add_(0, rows, update, alpha=-rate)
                else:
                    assert algorithm == "dualnorm"
                    if parameter.ndim == 2:
                        if bool(gradient.norm() < eps):
                            continue
                        value = gradient.float() if gradient.dtype in (torch.float16, torch.bfloat16) else gradient
                        left, singular, right = torch.linalg.svd(value, full_matrices=False)
                        threshold = max(value.shape) * torch.finfo(value.dtype).eps * singular[0]
                        update = ((left * (singular > threshold)) @ right).to(dtype=gradient.dtype)
                        factor = math.sqrt(max(1., parameter.shape[0] / parameter.shape[1]))
                        parameter.add_(update, alpha=-rate * factor)
                    else:
                        parameter.add_(gradient / (gradient.norm() + eps), alpha=-rate)
        self.clear_sampled_rows()
        if hasattr(self, "record"):
            self.record.record_step(self)


def test_zero_smoothing_matches_legacy_public_updates_and_checkpoint_packets(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for default compatibility")

    def build():
        recipe = get_recipe("bcap", num_particles=8, z_dim=2, batch_size=4, total_steps=8,
                            constraint_geometry_mode="none",
                            loss="relativistic", optimizer_smoothing=0., optimizer_convolution="none",
                            prior_kind="mog", sigma_rel=.1, standardize=False)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        critic = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for index, module in enumerate((generator, critic, prior)):
            deterministic_orthogonal_(module, seed=index)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device="cuda:0").manual_seed(0))

    def digest(trainer):
        state = deepcopy(trainer.state_dict())
        # Independent construction consumes ambient RNGs; learned and named
        # stream state, Recipe and complete optimizer packets must match.
        state.pop("cpu_rng")
        state.pop("cuda_rng")
        return state_digest(state)

    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        with monkeypatch.context() as patch:
            patch.setattr(dualnorm, "NormalizedOptimizer", LegacyDualNorm)
            reference = build()
        current = build()
        assert "optimizer_smoothing" not in current.recipe.to_dict()
        assert all("smoothing" not in optimizer.state_dict()["dualnorm"]
                   for optimizer in (current.opt_g, current.opt_d))
        assert digest(reference) == digest(current)
        for index in range(3):
            batch = torch.tensor([[-1.], [-.2], [.4], [1.]]) + index * .1
            reference.step(batch)
            current.step(batch)
            assert digest(reference) == digest(current)
