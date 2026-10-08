"""Convolution adaptation has one active delta and cannot regrade old gates."""
from dataclasses import asdict
from pathlib import Path

import pytest

from particlegan import get_recipe
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.configuration_search import recipe_identity_fields
from experiments.forge.planning import resolve_idea
from experiments.forge.techniques import validate_same_technique
from experiments.forge.views import qualify

ROOT = Path(__file__).resolve().parents[1]


def test_convolution_is_structural_and_default_recipe_identity_is_compatible():
    base = get_recipe("bcap", optimizer_smoothing=1e-5)
    enabled = base.replace(optimizer_convolution="per_offset")
    assert recipe_field_owner("optimizer_convolution") == "technique"
    old = asdict(base)
    old.pop("optimizer_convolution")
    assert recipe_identity_fields(old) == recipe_identity_fields(asdict(base))
    with pytest.raises(ValueError, match="structural idea"):
        validate_same_technique(base, enabled)


def test_four_image_diagnostic_retains_task_gates_budget_and_exact_mechanism_delta():
    request = resolve_idea(ROOT, "bcap-dualnorm-convolution-v1", study="bcap-convolution-images-v1")
    assert not request["preflight_blockers"]
    assert request["protocol"]["seed"] == 0
    assert request["view"]["evidence_scope"] == "research_diagnostic"
    assert len(request["jobs"]) == 4
    assert sum(job["budget_seconds"] for job in request["jobs"]) == 7200
    assert all(job["science"]["evidence_use"] == "research_diagnostic" for job in request["jobs"])
    deltas = request["study_review"]["expected"]["substantive_delta"]
    assert len(deltas) == 4
    assert all(delta["path"][0] == "recipe" and delta["path"][-1] == "optimizer_convolution"
               and delta["before"] == "none" and delta["after"] == "per_offset" for delta in deltas)
    verdict = qualify(request["view"], request["tasks"], [], candidate=request["candidate"])
    assert verdict["status"] == "DIAGNOSTIC"
    assert verdict["qualified_tier"] == 0 and not verdict["qualification_reuse"]
    for task in request["tasks"].values():
        assert task["execution"]["steps"] == 600
        assert task["evaluation"]["minimum_stable_checks"] == 5
        assert task["evaluation"]["observations"] == 24
