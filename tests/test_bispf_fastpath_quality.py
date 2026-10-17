"""Freeze the wide-host question and validate its actual numerical scorer."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.transfer_suite.image_tasks import image_metrics, templates
from benchmarks.transfer_suite.protocol import test_verdict as grade
from experiments.forge.imageprofiles import resolve_image_spec
from experiments.forge.views import load_tasks, load_view


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("original", ["img_stripes2", "img_bars4"])
def test_wide_host_preserves_parent_question_and_scoring(original):
    tasks = load_tasks(ROOT)
    parent = deepcopy(tasks[original])
    wide = tasks[original + "_w64_optimizer_diagnostic"]
    parent["id"] = wide["id"]
    parent["execution"]["host_definition"]["width"] = 64
    parent["resources"]["timeout_seconds"] = 300
    assert wide == parent
    assert resolve_image_spec(wide)["width"] == 64
    assert wide["execution"]["steps"] == 600
    view = load_view(ROOT, "bispf_fastpath_quality")
    assert view["evidence_scope"] == "research_diagnostic"
    assert {row["importance"] for row in view["assignments"]} == {"diagnostic"}


@pytest.mark.parametrize("original", ["img_stripes2", "img_bars4"])
def test_actual_wide_image_scorer_distinguishes_oracle_collapse_and_blur(original):
    task = json.loads((ROOT / "configs/forge/tasks" / (original + "_w64_optimizer_diagnostic.json")).read_text())
    host = task["execution"]["host_definition"]
    thresholds = task["evaluation"]["measurement"]
    centers = templates(host)
    samples = centers.repeat(host["particles"] // len(centers), 1, 1, 1)
    def verdict(values, *, transient=False):
        metrics = image_metrics(values, centers, thresholds)
        curve = [{"step": 25 * index, **metrics} for index in range(1, 25)]
        if transient:
            curve[-1].update(hq=0., modes=0)
        return metrics, grade(dict(steps=600, thresholds=task["evaluation"]["thresholds"]),
                              dict(observations=curve, live=curve[-1]))
    oracle, oracle_grade = verdict(samples)
    collapsed, collapse_grade = verdict(centers[:1].expand_as(samples))
    blur, blur_grade = verdict(torch.full_like(samples, .5))
    assert oracle["hq"] == 1. and oracle["modes"] == host["modes"]
    assert oracle_grade["status"] == "PASS"
    assert collapsed["hq"] == 1. and collapsed["modes"] == 1
    assert collapse_grade["status"] == "FAIL"
    assert blur["hq"] == 0. and blur_grade["status"] == "FAIL"
    assert verdict(samples, transient=True)[1]["status"] == "FAIL"
    with pytest.raises(ValueError, match="finite"):
        image_metrics(torch.full_like(samples, float("nan")), centers, thresholds)
