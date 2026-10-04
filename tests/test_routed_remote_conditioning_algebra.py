"""One pure tensor contract: no fixture/model/native update or noise draw."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "remote_conditioning_algebra_contract", EXAMPLES / "routed_remote_conditioning.py"
)
toy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = toy
SPEC.loader.exec_module(toy)


def test_pair_midpoint_contrast_identity_and_codeblind_reference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "pure algebra contract cannot construct fixture/model/loop"
        )

    monkeypatch.setattr(toy, "fixture", forbidden)
    monkeypatch.setattr(toy, "make_loop", forbidden)
    monkeypatch.setattr(toy, "teacher_policy", forbidden)
    midpoint = torch.tensor([[[[2.0, -1.0]]]], dtype=torch.float64)
    contrast = torch.tensor([[[[0.1, -0.2]]]], dtype=torch.float64)
    target = torch.stack((midpoint + contrast, midpoint - contrast), 1)
    variance = float(contrast.square().mean())
    output = toy.paired_decomposition(target + 3, target, variance)
    assert output["pair_MSE"] == pytest.approx(9)
    assert output["midpoint_loss"] == pytest.approx(9)
    assert output["conditional_error"] == pytest.approx(0, abs=1e-30)
    assert output["alpha"] == pytest.approx(1)
    assert output["predicted_contrast_power_over_V"] == pytest.approx(1)
    assert output["decomposition_identity_error"] < 1e-12
    blind = midpoint.unsqueeze(1).expand_as(target) + 3
    output = toy.paired_decomposition(blind, target, variance)
    assert output["conditional_error_over_V"] == pytest.approx(1)
    assert output["alpha"] == output["predicted_contrast_power_over_V"] == 0
    for bad in (0, float("nan")):
        with pytest.raises(ValueError):
            toy.paired_decomposition(target + 3, target, bad)
    with pytest.raises(ValueError):
        toy.paired_decomposition(target.flatten(), target, variance)
    with pytest.raises(ValueError):
        toy.paired_decomposition(target * float("nan"), target, variance)
    assert not torch.cuda.is_initialized()
