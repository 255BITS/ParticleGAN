from copy import deepcopy
import math

import pytest

from benchmarks.transfer_suite import image_tasks, vector_tasks
from benchmarks.transfer_suite.formulations import architecture_cell, axes


def result(spec, passed):
    metrics = dict(modes=spec["modes"] if passed else 0, hq=1. if passed else 0.)
    return dict(live=metrics, observations=[dict(step=math.ceil(i * spec["steps"] / 24), **metrics)
                                         for i in range(1, 25)])


def test_discriminator_changes_are_architecture_only():
    spec = deepcopy(vector_tasks.TASKS[0])
    changed = spec | dict(d_hidden=128, d_layers=3, fourier=4,
                          research_discriminator=dict(activation="softplus", beta=5., projection="axis"))
    before, after = axes(spec, "vector"), axes(changed, "vector")
    assert before["formulation"] == after["formulation"]
    assert before["training"] == after["training"]
    assert before["architecture"]["generator"] == after["architecture"]["generator"]
    assert before["architecture"]["discriminator"] != after["architecture"]["discriminator"]
    assert after["architecture"]["discriminator"]["research"] == changed["research_discriminator"]


def test_architecture_support_is_one_pass_with_failures_visible():
    spec = deepcopy(image_tasks.TASKS[0])
    residual = spec | dict(architecture="residual_upsample", width=16)
    cell = architecture_cell([dict(label="transpose", spec=spec, result=result(spec, False)),
                              dict(label="residual", spec=residual, result=result(residual, True))], "image")
    assert cell["supported"] and cell["status"] == "PASS"
    assert not cell["all_tested_architectures_pass"]
    assert [t["verdict"]["status"] for t in cell["trials"]] == ["FAIL", "PASS"]
    assert cell["trials"][0]["architecture"]["generator"] != cell["trials"][1]["architecture"]["generator"]


@pytest.mark.parametrize("change,axis", [
    (dict(penalty_coeff=10.), "formulation"),
    (dict(gradient_penalty="a_r1r2"), "formulation"),
    (dict(prior_weight=1.), "formulation"),
    (dict(lr_g=.0005), "training"),
    (dict(g_every=2), "training"),
    (dict(steps=1200), "resources"),
    (dict(noise_std=0.), "target"),
])
def test_settings_or_target_changes_cannot_rescue_architecture_cell(change, axis):
    spec = deepcopy(image_tasks.TASKS[0])
    with pytest.raises(ValueError, match=axis):
        architecture_cell([dict(label="a", spec=spec, result=result(spec, False)),
                           dict(label="b", spec=spec | change, result=result(spec | change, True))], "image")
