"""Analytic geometry, ownership and checkpoint tests; no training evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from particlegan import get_recipe
from particlegan.optim.constraint_geometry import (
    ConstraintGeometryOptimizer, constraint_geometry_backward, project_nonascent,
)


def test_intersecting_and_opposed_halfspaces():
    d = torch.tensor([2., 1.], dtype=torch.float64)
    a = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(project_nonascent(d, a), d.new_tensor([0., 1.]))
    a = d.new_tensor([[1., 0.], [1., 1.]])
    out = project_nonascent(d, a)
    torch.testing.assert_close(out, d.new_tensor([0., 0.]))
    assert bool((a @ out <= 1e-10).all())
    assert torch.equal(project_nonascent(-d, a), -d)


def test_post_normalization_protection_preserves_unsampled_rows_and_pending_state():
    table = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    vector = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = ConstraintGeometryOptimizer([dict(params=[table], role="prior"),
                                       dict(params=[vector], role="generator")], lr=.1, smoothing=.001)
    opt.set_sampled_rows(table, torch.tensor([0, 2]))
    protected = table.sum() + vector.sum()
    constraint_geometry_backward(-protected, opt, (protected,))
    pending = deepcopy(opt.state_dict())
    restored = ConstraintGeometryOptimizer([dict(params=[table], role="prior"),
                                            dict(params=[vector], role="generator")], lr=.1, smoothing=.001)
    restored.load_state_dict(pending)
    restored.step()
    assert torch.equal(table[1], torch.zeros(2, dtype=torch.float64))
    assert float((table.sum() + vector.sum()).detach()) <= 1e-12
    assert restored.constraint_geometry_stats["projected_steps"] == 1
    assert restored.constraint_geometry_stats["max_derivative_before"] > .1
    assert restored.constraint_geometry_stats["max_derivative_after"] <= 1e-12
    with pytest.raises(ValueError, match="explicit protected"):
        restored.step()
    bad = deepcopy(restored.state_dict())
    bad["constraint_geometry"]["mode"] = "other"
    with pytest.raises(ValueError, match="checkpoint"):
        restored.load_state_dict(bad)


def test_disabled_backward_and_exact_winner_resolution():
    p = torch.nn.Parameter(torch.tensor([.3, .4], dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=.1)
    loss = p.square().sum()
    constraint_geometry_backward(loss, opt, (loss,))
    torch.testing.assert_close(p.grad, p.detach() * 2)
    root = Path(__file__).resolve().parents[1]
    original = json.loads(next((root / "configs/forge/configurations").glob("bcap-dualnorm--5b1ef*.json")).read_text())
    control = json.loads((root / "configs/forge/ideas/constraint_geometry-control-v1.json").read_text())
    recipe = get_recipe(control["recipe_preset"], **control["recipe_overrides"])
    from dataclasses import asdict
    resolved = json.loads(json.dumps(asdict(recipe)))
    resolved.pop("constraint_geometry_mode")
    expected = dict(original["resolved_configuration_recipe"])
    # Public task binding supplies prior implementation/resources in both arms.
    for key in ("prior_kind", "sigma_rel", "standardize", "total_steps", "batch_size", "num_particles", "z_dim"):
        resolved.pop(key, None)
        expected.pop(key, None)
    assert resolved == expected


def test_recipe_rejects_incompatible_update():
    with pytest.raises(ValueError, match="zero-momentum"):
        get_recipe("bcap", optimizer_momentum=.5, constraint_geometry_mode="nonascent")


def test_inactive_projection_preserves_already_applied_step_bitwise():
    from particlegan.optim.dualnorm import NormalizedOptimizer
    old = torch.tensor([.1234567], dtype=torch.float32)
    plain = torch.nn.Parameter(old.clone())
    guarded = torch.nn.Parameter(old.clone())
    options = dict(lr=.12345676, smoothing=.001)
    baseline = NormalizedOptimizer([plain], **options)
    candidate = ConstraintGeometryOptimizer([guarded], **options)
    plain.sum().backward()
    baseline.step()
    loss = guarded.sum()
    constraint_geometry_backward(loss, candidate, (loss,))
    candidate.step()
    # The fixture deliberately crosses zero. Recomposition destroys a real
    # low-order result, so ordinary allclose would miss this regression.
    assert not torch.equal(old + (plain.detach() - old), plain.detach())
    assert torch.equal(guarded.detach(), plain.detach())
    assert candidate.constraint_geometry_stats['projected_steps'] == 0
    legacy = deepcopy(candidate.state_dict())
    legacy['constraint_geometry']['schema'] = 1
    with pytest.raises(ValueError, match='checkpoint'):
        candidate.load_state_dict(legacy)
