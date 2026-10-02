"""The new acquisition gate must discriminate target law from cheap impostors."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit import api_contract
from benchmarks.toy_audit.api_ring16 import CASE_ID, list_cases
from benchmarks.toy_audit.ring16_controls import run_controls
from experiments.forge.adapters import adapter_preflight
from experiments.forge.views import load_tasks, load_view
from particlegan import GANTrainer, MoGParticlePrior

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_target_and_destructive_controls_validate_acquisition_bounds():
    report = run_controls()
    assert report["training_updates"] == 0
    assert report["passed"], report["controls"]
    rows = {row["control"]: row for row in report["controls"]}
    assert rows["independent_target"]["passed"]
    assert "modes >= 16" in rows["missing_cluster"]["failed_bounds"]
    assert "mass_tv <= 0.15" in rows["biased_mass"]["failed_bounds"]
    for name in ("centers_only", "collapsed_width"):
        assert "component_min_eigen_ratio >= 0.15" in rows[name]["failed_bounds"]
    for name in ("continuous_circle", "wrong_location", "inflated_width"):
        assert "hq >= 0.85" in rows[name]["failed_bounds"]
    assert not rows["nonfinite"]["passed"]


def test_declaration_is_shared_and_new_view_preserves_original_mode_hold():
    tasks = load_tasks(ROOT)
    task = tasks["ring16_acquisition"]
    case = list_cases()[0]
    assert case["thresholds"] == task["evaluation"]["thresholds"]
    assert case["default_steps"] == task["execution"]["steps"] == 400
    means = torch.tensor(case["law"]["means"])
    assert means.shape == (16, 2)
    assert torch.allclose(means.norm(dim=1), torch.full((16,), 3.))
    assert case["law"]["masses"] == [1/16]*16
    view = load_view(ROOT, "ring16_acquisition")
    assert view["calibration"]["status"] == "provisional"
    assert view["assignments"] == [dict(task="ring16_acquisition", qualification_tier=1,
                                        importance="required", order=0)]
    original = load_view(ROOT, "discriminator_stability")
    assert original["revision"] == 2
    assert len(original["assignments"]) == 24
    assert next(a for a in original["assignments"] if a["task"] == "mode_hold")["qualification_tier"] == 2
    candidate = json.loads((ROOT / "configs/forge/ideas/k3p.json").read_text())
    assert adapter_preflight(task, candidate) == []


def test_shared_public_caller_has_exact_mog_and_clean_live_sampling():
    case = api_contract.discover()[CASE_ID]
    fixture = api_contract.build(case, seed=0, max_steps=1)
    assert isinstance(fixture.trainer, GANTrainer)
    assert isinstance(fixture.trainer.prior, MoGParticlePrior)
    assert float(fixture.trainer.prior.sigma) == pytest.approx(.025)
    assert fixture.recipe.standardize is False
    assert fixture.recipe.total_steps == 400 and fixture.recipe.num_particles == 256
    assert fixture.completed_steps == 0
    result = fixture.observe(n=4096, seed=10000)
    assert not result["passed"]
    assert len(result["views"]) == 3  # Whole target, masses, fixed local width.
    assert result["views"][0]["target"].shape == (4096, 2)


def rendering_failure_receipt():
    from benchmarks.toy_audit.api_vectors import _bounds
    from benchmarks.toy_audit.ring16_publish import MEDIA_ERROR
    case = api_contract.discover()[CASE_ID]
    metrics = dict(sample_count=4096, modes=0, mass_tv=1., hq=0.,
                   component_covariance_error=1., component_min_eigen_ratio=0.)
    bounds = _bounds(metrics, case["thresholds"])
    media = api_contract.evaluation_steps(400, 9)
    scoring = api_contract.evaluation_steps(400, 25)
    schedule = sorted(set(media) | set(scoring))
    return dict(case=case, status="ERROR", verdict="FAIL", passed=False, source_unchanged=True,
                completed_updates=400, metric_passed=False, sustained_metric_passed=False,
                default_protocol_complete=True, gif_frames=0, artifacts={},
                failed_bounds=bounds + ["last 5 post-update metric observations do not all pass", MEDIA_ERROR],
                protocol=dict(updates=400, default_updates=400, evaluation_samples=4096,
                              default_evaluation_samples=4096, metric_observations=24,
                              terminal_observations=5, media_frames=9, media_steps=media,
                              metric_evaluation_steps=scoring, evaluation_steps=schedule),
                observations=[dict(step=step, metrics=metrics, passed=False, failed_bounds=bounds,
                                   views=[dict(kind="scatter", title="Actual output")]) for step in schedule])


def test_renderer_recovery_preserves_original_error_and_numeric_fail():
    from benchmarks.toy_audit.ring16_publish import verify_rendering_failure
    receipt = rendering_failure_receipt()
    original = deepcopy(receipt)
    assert verify_rendering_failure(receipt) == receipt["protocol"]["media_steps"]
    assert receipt == original


@pytest.mark.parametrize("mutation", ["partial", "other_error", "missing_check", "forged_pass"])
def test_renderer_recovery_rejects_incomplete_or_changed_evidence(mutation):
    from benchmarks.toy_audit.ring16_publish import verify_rendering_failure
    receipt = rendering_failure_receipt()
    if mutation == "partial":
        receipt["default_protocol_complete"] = False
    elif mutation == "other_error":
        receipt["failed_bounds"].append("unrelated optimizer exception")
    elif mutation == "missing_check":
        receipt["observations"].pop(1)
    else:
        receipt["metric_passed"] = True
    with pytest.raises(ValueError):
        verify_rendering_failure(receipt)
