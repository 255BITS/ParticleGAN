"""Oracle noise floors are reporting only; evaluation-draw contracts are declared."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks.transfer_suite import image_tasks, metric_uncertainty as mu, suite, vector_tasks as vectors

ROOT = Path(__file__).resolve().parents[1]
FLOOR_KEYS = {"noise_floor", "sw1_projections", "sw1_finite_n_bias"}


def _strip(metrics):
    return {k: v for k, v in metrics.items() if k not in FLOOR_KEYS}


def _shifted(spec, n=4096, seed=993, shift=0.):
    return vectors.sample_target(spec, n, torch.Generator().manual_seed(seed), spec["steps"]) * 1.03 + shift


def test_frozen_evaluator_sources_are_byte_identical_to_forge_pins():
    # Floors must not revise the evaluator bytes that Forge task cards pin.
    card = json.loads((ROOT / "configs/forge/tasks/vector_two_broad.json").read_text())
    for path, digest in card["evaluation"]["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("spec", vectors.TASKS, ids=lambda x: x["name"])
@pytest.mark.parametrize("n", [4096, 256])
def test_reference_transcription_matches_frozen_scorer(spec, n):
    spec = vectors.resolve(spec)
    fake = _shifted(spec, n, seed=5, shift=.03)
    for step in (1, spec["steps"] // 3, spec["steps"]):
        real = vectors.sample_target(spec, n, torch.Generator().manual_seed(991), step)
        assert mu.score_against_reference(fake, real, spec, step, 992) == vectors.score_samples(fake, spec, step)


@pytest.mark.parametrize("spec", vectors.TASKS, ids=lambda x: x["name"])
def test_floor_is_reporting_only_and_isolated(spec):
    torch.set_num_threads(1)
    spec = vectors.resolve(spec)
    for shift in (0., .4):
        fake = _shifted(spec, shift=shift)
        frozen = vectors.score_samples(fake, spec, spec["steps"])
        state = torch.get_rng_state().clone()
        scored = mu.score_vector(fake, spec, spec["steps"])
        assert torch.equal(state, torch.get_rng_state())
        assert _strip(scored) == frozen and set(scored) - set(frozen) == FLOOR_KEYS
        assert vectors.passes(scored, spec["thresholds"]) == vectors.passes(frozen, spec["thresholds"])
        json.dumps(scored, allow_nan=False)
        floor = scored["noise_floor"]
        assert floor["draws"] == mu.ORACLE_DRAWS and floor["unscored_gates"] == []
        assert scored["sw1_projections"] == 32
        assert scored["sw1_finite_n_bias"] == floor["metrics"]["sw1_normalized"]["oracle_mean"] > 0
        for key, op, bound in spec["thresholds"]:
            entry = floor["metrics"][key]
            assert (entry["op"], entry["threshold"]) == (op, bound)
            assert entry["oracle_std"] > 0 and entry["oracle_pass_fraction"] == 1.
            assert (entry["margin_oracle_std"] >= 0) == vectors.passes(frozen, [[key, op, bound]])
        assert floor["oracle_all_gates_pass_fraction"] == 1.
    shifted = mu.score_vector(_shifted(spec, shift=1.), spec, spec["steps"])["noise_floor"]["metrics"]
    assert shifted["sw1_normalized"]["margin_oracle_std"] < 0  # a visibly wrong draw sits far below the floor


def test_summarize_signs_unscored_gates_and_degenerate_spread():
    rows = [dict(a=1., b=5), dict(a=3., b=5)]
    floor = mu.summarize(dict(a=0., b=5, c=1.), rows, [["a", "<=", 2.], ["b", ">=", 5], ["c", "<=", 1.]],
                         report=("a",))
    a = floor["metrics"]["a"]
    assert (a["oracle_mean"], a["oracle_pass_fraction"]) == (2., .5)
    assert a["margin_oracle_std"] == pytest.approx(2 / 2 ** .5) and a["oracle_margin_oracle_std"] == 0.
    assert floor["metrics"]["b"]["oracle_std"] == 0. and floor["metrics"]["b"]["margin_oracle_std"] is None
    assert floor["unscored_gates"] == ["c"] and floor["oracle_all_gates_pass_fraction"] == .5
    json.dumps(floor, allow_nan=False)


def test_wrapper_gates_without_an_oracle_value_are_declared_unscored():
    spec = vectors.resolve(vectors.TASKS[0]) | dict(thresholds=vectors.TASKS[0]["thresholds"] + [["modes", ">=", 2]])
    floor = mu.score_vector(_shifted(spec), spec, spec["steps"])["noise_floor"]
    assert floor["unscored_gates"] == ["modes"] and "modes" not in floor["metrics"]


def test_nonfinite_draw_keeps_frozen_none_metrics_and_reports_the_floor():
    spec = vectors.resolve(vectors.TASKS[0])
    fake = _shifted(spec, 128)
    fake[0, 0] = float("nan")
    scored = mu.score_vector(fake, spec, spec["steps"])
    assert _strip(scored) == vectors.score_samples(fake, spec, spec["steps"])
    assert all(e["margin_oracle_std"] is None for k, e in scored["noise_floor"]["metrics"].items() if "op" in e)


def test_eval_draw_contracts_are_declared_versioned_and_fail_closed():
    assert mu.evaluation_seeds("fixed_v1", 7) == dict(latent=990, reference=991, projection=992)
    first, later = mu.evaluation_seeds("per_observation_v2", 50), mu.evaluation_seeds("per_observation_v2", 100)
    assert first != later and set(first.values()).isdisjoint({990, 991, 992})
    assert mu.evaluation_seeds("per_observation_v2", 50) == first
    with pytest.raises(ValueError, match="eval_draw_contract"):
        mu.evaluation_seeds("silent_v3", 1)
    assert mu.evaluation_contract()["comparable_with_archived_fixed_v1"] is True
    assert mu.evaluation_contract("per_observation_v2")["comparable_with_archived_fixed_v1"] is False


def test_per_observation_contract_varies_reference_draw_only_when_opted_in():
    spec = vectors.resolve(vectors.TASKS[0])
    fake = _shifted(spec, 1024)
    fixed = [mu.score_vector(fake, spec, step, noise_floor=False)["sw1_normalized"] for step in (600, 1200)]
    assert fixed[0] == fixed[1] == vectors.score_samples(fake, spec, 1200)["sw1_normalized"]
    varied = [mu.score_vector(fake, spec, step, contract="per_observation_v2") for step in (600, 1200)]
    assert varied[0]["sw1_normalized"] != varied[1]["sw1_normalized"]
    assert varied[0]["noise_floor"]["reference_seed"] != varied[1]["noise_floor"]["reference_seed"]
    assert varied[0]["noise_floor"]["eval_draw_contract"] == "per_observation_v2"


def test_suite_episode_annotation_keeps_frozen_values_curve_and_verdict(monkeypatch):
    monkeypatch.setattr(vectors, "EVAL_SAMPLES", 128)
    spec = dict(vectors.TASKS[0], steps=24, hidden=8, layers=1, particles=12, batch=16, runner="vector",
                split="development")
    frozen = vectors.run_episode(deepcopy(spec), vectors.fixed_policy(), fixed=True)
    annotated = suite.run_episode(deepcopy(spec), vectors.fixed_policy(), fixed=True)
    assert "error" not in annotated and "noise_floor_error" not in annotated
    assert annotated["convergence"] == frozen["convergence"] and annotated["status"] == frozen["status"]
    assert annotated["actions"] == frozen["actions"]
    for name in ("live", "ema"):
        assert _strip(annotated[name]) == frozen[name]
        assert annotated[name]["noise_floor"]["sample_count"] == 128
    for got, want in zip(annotated["observations"], frozen["observations"]):
        assert {k: v for k, v in _strip(got).items() if k not in ("seconds", "ema")} == \
            {k: v for k, v in want.items() if k not in ("seconds", "ema")}
        assert _strip(got["ema"]) == want["ema"] and got["noise_floor"]["draws"] == mu.ORACLE_DRAWS
    assert annotated["evaluation_contract"]["eval_draw_contract"] == "fixed_v1"
    json.dumps(annotated, allow_nan=False)


def test_image_floor_reports_modes_and_hq_without_touching_rng():
    spec = image_tasks.TASKS[0]
    centers = image_tasks.templates(spec)
    state = torch.get_rng_state().clone()
    exact = centers.repeat_interleave(spec["particles"] // len(centers) + 1, 0)[:spec["particles"]]
    metrics = image_tasks.image_metrics(exact, centers, spec["thresholds"])
    floor = mu.image_noise_floor(metrics, spec, centers, spec["particles"])
    assert torch.equal(state, torch.get_rng_state())
    assert set(floor["metrics"]) == {"modes", "hq", "mean_rmse", "distribution_tv"}
    assert floor["metrics"]["hq"]["threshold"] == spec["thresholds"]["hq_min"]
    assert floor["unscored_gates"] == [] and floor["sample_count"] == spec["particles"]
    json.dumps(floor, allow_nan=False)


def test_forge_transfer_grade_passes_floor_fields_through():
    from experiments.forge.views import grade_result
    from tests.test_forge_views import receipt, simple_task
    task = simple_task("a")
    raw = receipt(task)
    plain = grade_result(task, deepcopy(raw))
    floor = mu.summarize({"error": .5}, [{"error": .4}, {"error": .6}], [["error", "<=", 1.]])
    for row in raw["evidence"]["observations"]:
        row["noise_floor"] = floor
    raw["evidence"]["live"].update(noise_floor=floor, sw1_projections=32, sw1_finite_n_bias=.01)
    graded = grade_result(task, raw)
    assert graded["status"] == plain["status"] == "PASS"
    assert graded["metrics"]["noise_floor"] == floor and graded["metrics"]["sw1_projections"] == 32


def test_forge_vector_observations_carry_floors_over_frozen_metrics(tmp_path):
    from experiments.forge import adapters
    from tests.test_forge_adapters import request, task
    torch.set_num_threads(1)
    value = task("vector_two_broad")
    global_rng = torch.get_rng_state().clone()
    raw = adapters._vector(request(value), value, tmp_path, "cpu")
    assert torch.equal(global_rng, torch.get_rng_state())
    evidence = raw["evidence"]
    spec = dict(value["execution"]["host_definition"], thresholds=value["evaluation"]["thresholds"])
    archived = torch.load(tmp_path / "observed-samples.pt", weights_only=False)
    assert len(archived) == len(evidence["observations"]) == 2
    for row, record in zip(evidence["observations"], archived):
        frozen = vectors.score_samples(record["samples"], spec, record["step"])
        assert {k: v for k, v in _strip(row).items() if k != "step"} == frozen
        assert row["noise_floor"]["eval_draw_contract"] == "fixed_v1"
        assert set(row["noise_floor"]["metrics"]) >= {key for key, _, _ in spec["thresholds"]}
    assert evidence["evaluation_contract"] == mu.evaluation_contract("fixed_v1")


def test_forge_vector_rejects_undeclared_draw_contracts(tmp_path):
    from experiments.forge import adapters
    from tests.test_forge_adapters import request, task
    value = task("vector_two_broad")
    value["evaluation"]["eval_draw_contract"] = "silent_v3"
    with pytest.raises(ValueError, match="eval_draw_contract"):
        adapters._vector(request(value), value, tmp_path, "cpu")
    value["evaluation"]["eval_draw_contract"] = "per_observation_v2"
    raw = adapters._vector(request(value), value, tmp_path, "cpu")
    rows = raw["evidence"]["observations"]
    assert raw["evidence"]["evaluation_contract"]["comparable_with_archived_fixed_v1"] is False
    assert rows[0]["noise_floor"]["reference_seed"] != rows[1]["noise_floor"]["reference_seed"]
