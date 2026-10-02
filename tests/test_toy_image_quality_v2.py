"""Finite-template gates must reject biased mass without changing old gates."""
from copy import deepcopy
import json

import numpy as np
import pytest

from benchmarks.toy_audit import image_quality_v2 as image
from benchmarks.transfer_suite import image_tasks


def fixture():
    templates = np.stack([np.zeros((1, 8, 8)), np.ones((1, 8, 8))])
    spec = dict(particles=32, modes=2, steps=24, thresholds=dict(
        quality_rmse=0.05, hq_min=0.9, modes=2, min_mode_fraction=0.25,
        observations=24, minimum_stable_checks=5))
    return templates, spec


def test_exact_oracles_and_known_accepted_mass_error():
    templates, spec = fixture()
    controls = image.controls(templates, spec)
    assert controls["exact_uniform"]["v2_accepts"]
    assert controls["uniform_permuted"] == controls["exact_uniform"]
    unequal = controls["mass_imbalance_tv_025"]
    assert unequal["hq"] == 1 and unequal["modes"] == 2
    assert unequal["distribution_tv"] == unequal["finite_template_tv"] == 0.25
    assert unequal["historical_accepts"] and not unequal["v2_accepts"]
    assert not controls["one_mode_collapse"]["v2_accepts"]
    assert not controls["global_template_mean"]["v2_accepts"]
    assert controls["within_tolerance_mutation"]["v2_accepts"]


def test_reject_bin_does_not_hide_bad_images_in_nearest_assignment():
    templates, spec = fixture()
    # Nearest assignments have 13/19 mass and just one bad atom. Each old gate
    # and the nearest TV gate passes, but the valid-bin deficit is four atoms.
    cloud = np.concatenate([np.repeat(templates[:1], 13, axis=0),
                            np.repeat(templates[1:], 19, axis=0)])
    cloud[0] = 0.1
    measured = image.score(cloud, templates, spec["thresholds"])
    assert measured["mode_counts"] == [13, 19]
    assert measured["quality_mode_counts"] == [12, 19]
    assert measured["hq"] == 31 / 32
    assert measured["distribution_tv"] == 3 / 32
    assert measured["finite_template_tv"] == 4 / 32
    assert image.accepted(measured, spec, strengthened=False)
    assert not image.accepted(measured, spec)


def test_full_cadence_and_five_terminal_passes_are_required():
    templates, spec = fixture()
    good = image.score(np.repeat(templates, 16, axis=0), templates, spec["thresholds"])
    bad = image.score(np.concatenate([np.repeat(templates[:1], 8, axis=0),
                                     np.repeat(templates[1:], 24, axis=0)]),
                      templates, spec["thresholds"])
    curve = [dict(step=i, **(bad if i <= 19 else good)) for i in range(1, 25)]
    assert image.verdict(curve, spec)["passed"]
    curve[19] = dict(step=20, **bad)
    assert image.verdict(curve, spec)["final_pass"]
    assert not image.verdict(curve, spec)["passed"]
    assert image.verdict(curve[:-1], spec)["status"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="increasing"):
        image.verdict([curve[0], curve[0]], spec)


def test_overlapping_templates_cannot_supply_an_unambiguous_oracle():
    templates, spec = fixture()
    templates[1] = 0.09
    with pytest.raises(ValueError, match="overlap"):
        image.score(np.repeat(templates, 16, axis=0), templates, spec["thresholds"])
    templates[1] = 0
    with pytest.raises(ValueError, match="overlap"):
        image.score(np.repeat(templates, 16, axis=0), templates, spec["thresholds"])


def test_expected_negative_hosts_have_independent_analytic_limits():
    mean_spec = next(s for s in image_tasks.TASKS if s["name"] == "img_mean_discriminator")
    mean_limit = image.host_limits(image_tasks.templates(mean_spec), mean_spec)
    assert mean_limit["all_projected_templates_equal"]
    assert mean_limit["template_means"] == [4 / 64] * 4
    uniform_spec = next(s for s in image_tasks.TASKS if s["name"] == "img_uniform_generator")
    uniform_limit = image.host_limits(image_tasks.templates(uniform_spec), uniform_spec)
    # Each stripe has 16 bright and 48 dark pixels. Its optimal constant has
    # variance (16/64)*(48/64), independent of training or scorer output.
    assert uniform_limit["best_constant_rmse_per_template"] == [np.sqrt(3 / 16)] * 2
    assert uniform_limit["all_modes_unreachable_at_quality_bound"]


@pytest.mark.parametrize("corruption", ["empty", "nonfinite", "bad_shape"])
def test_invalid_atoms_cannot_shrink_the_denominator(corruption):
    templates, spec = fixture()
    cloud = np.repeat(templates, 16, axis=0)
    if corruption == "empty":
        cloud = cloud[:0]
    elif corruption == "nonfinite":
        cloud[0, 0, 0, 0] = np.nan
    else:
        cloud = cloud[:, 0]
    with pytest.raises(ValueError, match="finite nonempty"):
        image.score(cloud, templates, spec["thresholds"])


def test_reanalysis_checks_retained_hashes_and_preserves_historical_verdict(tmp_path):
    templates, spec = fixture()
    cloud = np.concatenate([np.repeat(templates[:1], 8, axis=0),
                            np.repeat(templates[1:], 24, axis=0)])
    measured = image.score(cloud, templates, spec["thresholds"])
    curve = [dict(step=i, **measured) for i in range(1, 25)]
    source = tmp_path / "source"
    source.mkdir()
    np.savez(source / "observations.npz", live=np.repeat(cloud[None], 24, axis=0),
             templates=templates, steps=np.arange(1, 25))
    (source / "result.json").write_text(json.dumps(dict(observations=curve)))
    record = dict(spec=spec, artifact=str(source),
                  capture_sha256=image.sha256(source / "observations.npz"),
                  result_sha256=image.sha256(source / "result.json"),
                  verdict=image.verdict(curve, spec, strengthened=False))
    compact, _ = image.reanalyze(record, output=tmp_path / "raw", label="retained")
    assert compact["historical"]["passed"] and not compact["revised"]["passed"]
    assert compact["historical_metric_max_difference"] == 0
    tampered = deepcopy(record)
    tampered["capture_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        image.reanalyze(tampered, output=tmp_path / "raw", label="tampered")
