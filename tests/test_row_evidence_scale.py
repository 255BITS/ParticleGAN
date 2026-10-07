"""Exact parity with the original scale search, including checkpoint continuation."""
from copy import deepcopy
import math

import pytest
import torch

from particlegan.row_evidence import RowEvidence


class OriginalScale(RowEvidence):
    def _scale(self, t2, n_eff, ok):
        lo = torch.zeros((), device=t2.device, dtype=t2.dtype)
        hi = torch.full_like(lo, math.log(1e6))
        nan = torch.full_like(t2, float("nan"))
        for _ in range(24):
            mid = (lo + hi) / 2
            med = torch.where(ok, self._pvalue(t2 / mid.exp(), n_eff), nan).nanmedian()
            below = med < 0.5
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        c = ((lo + hi) / 2).exp()
        self.scale_c = float(c)
        self.counters["scale_sum"] = self.counters.get("scale_sum", 0.0) + self.scale_c
        return c


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            same_state(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 2, 3, 8])
@pytest.mark.parametrize("case", ["invalid", "lower", "upper", "mixed", "ties", "clamp"])
def test_scale_search_exact(dtype, width, case, device="cpu"):
    table = torch.zeros(25, width, dtype=dtype, device=device)
    optimized, original = RowEvidence(table, null="scaled"), OriginalScale(table, null="scaled")
    n_eff = torch.linspace(6, 99, 25, dtype=dtype, device=device)
    t2 = torch.logspace(-6, 10, 25, dtype=dtype, device=device)
    ok = torch.ones(25, dtype=torch.bool, device=device)
    if case == "invalid":
        ok.zero_()
    elif case == "lower":
        t2.zero_()
    elif case == "upper":
        t2.fill_(1e20)
    elif case == "mixed":
        ok[::3] = False
        t2[::3] = float("nan")
    elif case == "ties":
        n_eff.fill_(20)
        t2.fill_(7)
    elif case == "clamp":
        n_eff[:3] = torch.tensor([0., 1., 2.], dtype=dtype, device=device)
        ok[:3] = False
    same_state(optimized._scale(t2, n_eff, ok), original._scale(t2, n_eff, ok))
    same_state(optimized.state_dict(), original.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [2, 3])
def test_flags_resets_and_checkpoint_continuation_exact(dtype, width):
    table = torch.zeros(25, width, dtype=dtype)
    optimized, original = RowEvidence(table, null="scaled"), OriginalScale(table, null="scaled")
    base = torch.arange(table.numel(), dtype=dtype).reshape_as(table) * .1 + 1
    training_rng = torch.get_rng_state().clone()
    flagged = 0
    for step in range(160):
        gradient = torch.sin(base + step * .7)
        gradient[0] = base[0] * 100 + torch.sin(torch.tensor(step * .7, dtype=dtype)) * .01
        gradient[step % 25] = 0
        for evidence in (optimized, original):
            evidence.update(gradient)
            if step == 47:
                evidence.reset(torch.tensor([2, 9]))
        same_state(optimized.state_dict(), original.state_dict())
        flagged += int(optimized.flag.sum())
        if step == 79:
            state = deepcopy(optimized.state_dict())
            optimized = RowEvidence(table, null="scaled")
            optimized.load_state_dict(state)
    assert flagged > 0
    assert torch.equal(training_rng, torch.get_rng_state())


def test_public_trainer_exact_across_settle_moves_and_anchored_release(monkeypatch):
    from particlegan import GANTrainer
    from test_e22_policy import components, native_trajectory, recipe

    optimized_scale = RowEvidence._scale
    results = []
    for implementation in (OriginalScale._scale, optimized_scale):
        monkeypatch.setattr(RowEvidence, "_scale", implementation)
        options = recipe()
        generator, critic, prior = components(options)
        trainer = GANTrainer(options, generator, critic, prior=prior, seed=101, serial_backward=True)
        observed = native_trajectory(trainer)
        results.append((observed, deepcopy(trainer.state_dict()),
                        {role: {name: None if p.grad is None else p.grad.clone()
                                for name, p in module.named_parameters()}
                         for role, module in (("G", generator), ("D", critic), ("prior", prior))}))
        assert trainer.row_evidence.counters["resets"] > 0
        assert trainer.lr_settle.testers[0][1].counts["stationary"] > 0
        assert trainer.lr_settle.testers[0][1].counts["drift"] > 0
    same_state(*results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_scale_search_exact():
    for dtype in (torch.float32, torch.float64):
        for width in (1, 2, 3, 8):
            for case in ("invalid", "lower", "upper", "mixed", "ties", "clamp"):
                test_scale_search_exact(dtype, width, case, device="cuda")
