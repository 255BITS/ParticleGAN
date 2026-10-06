"""Offline cadence and gate guards; these tests launch no training."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from experiments.forge.adapters import _checkpoints
from experiments.forge.budget_diagnostics import EVALUATOR, KIND, checkpoint_steps, validate_declaration
from experiments.forge.views import _validate_task, grade_result


def task(base=1000):
    return {"schema_version": 1, "id": "offline-budget-diagnostic", "adapter": "transfer_vector",
            "execution": {"steps": base * 10, "original_schedule_horizon": base,
                          "initializer": "deterministic_orthogonal", "produces_state": True,
                          "prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}},
            "evaluation": {"kind": KIND, "evaluator": EVALUATOR, "base_steps": base, "factor": 10,
                           "observations": 240, "minimum_stable_checks": 5, "scoring_weights": "live",
                           "thresholds": [["quality", ">=", .9], ["modes", "==", 16]],
                           "guards": {"finite_state": True, "exact_optimizer_updates": True,
                                      "optimizer_roles": ["generator", "discriminator", "prior"],
                                      "rng_isolation": True}},
            "resources": {}, "requires_capabilities": [], "dependencies": []}


def receipt(value, bad_checks=()):
    points = [{"step": step, "quality": .5 if index in bad_checks else .95, "modes": 16}
              for index, step in enumerate(checkpoint_steps(value), 1)]
    return {"status": "PASS", "evidence": {"observations": points, "live": dict(points[-1]),
            "scoring_weights": "live", "guards": {"all_finite": True,
                "optimizer_updates": {role: value["execution"]["steps"]
                                      for role in ("generator", "discriminator", "prior")},
                "unintended_rng_deviations": 0}}}


@pytest.mark.parametrize("base", [400, 1000])
def test_cadence_keeps_original_spacing_and_all_prefix_endpoints(base):
    value = task(base)
    _validate_task(value)
    steps = _checkpoints(value)
    original = {"execution": {"steps": base}, "evaluation": {"kind": "transfer_sustained"}}
    assert len(steps) == len(set(steps)) == 240
    assert steps[:24] == _checkpoints(original)
    assert steps == [math.ceil(index * base / 24) for index in range(1, 241)]
    assert [steps[index * 24 - 1] for index in (1, 2, 4, 10)] == [base * index for index in (1, 2, 4, 10)]
    grade = grade_result(value, receipt(value))
    assert grade["status"] == "PASS"
    snapshots = grade["evaluator_result"]["prefix_snapshots"]
    assert list(snapshots) == ["1x", "2x", "4x", "10x"]
    for factor in (1, 2, 4, 10):
        assert snapshots[f"{factor}x"]["status"] == "PASS"
        assert snapshots[f"{factor}x"]["convergence"]["observations"] == 24 * factor
        assert snapshots[f"{factor}x"]["step_budget"] == base * factor


@pytest.mark.parametrize("section,key,bad", [
    ("evaluation", "base_steps", True), ("evaluation", "base_steps", 23),
    ("evaluation", "base_steps", 1000.), ("evaluation", "factor", 2),
    ("evaluation", "factor", 10.), ("evaluation", "observations", 24),
    ("evaluation", "observations", 240.), ("evaluation", "minimum_stable_checks", 6),
    ("evaluation", "scoring_weights", "ema"), ("evaluation", "evaluator", "ordinary:grade"),
    ("evaluation", "thresholds", []), ("evaluation", "thresholds", [["quality", ">=", float("nan")]]),
    ("evaluation", "thresholds", [["quality", ">=", True]]),
    ("evaluation", "prefixes", [1, 3, 4, 10]),
    ("execution", "steps", 1000), ("execution", "steps", 10000.),
    ("execution", "original_schedule_horizon", 10000),
    ("execution", "original_schedule_horizon", True),
])
def test_unsupported_cadence_is_rejected_before_execution(section, key, bad):
    value = task(); value[section][key] = bad
    with pytest.raises(ValueError):
        validate_declaration(value)
    with pytest.raises(ValueError):
        _checkpoints(value)
    assert grade_result(value, {"status": "PASS"})["status"] == "INVALID"


@pytest.mark.parametrize("key", ["base_steps", "factor", "observations", "minimum_stable_checks",
                                 "scoring_weights", "evaluator"])
def test_cadence_fields_must_be_explicit(key):
    value = task(); del value["evaluation"][key]
    with pytest.raises(ValueError):
        _validate_task(value)


def test_each_prefix_uses_its_own_terminal_five_joint_checks():
    value = task()
    grade = grade_result(value, receipt(value, bad_checks=(24, 48)))
    assert grade["status"] == "PASS"
    snapshots = grade["evaluator_result"]["prefix_snapshots"]
    assert {name: snapshot["status"] for name, snapshot in snapshots.items()} == {
        "1x": "FAIL", "2x": "FAIL", "4x": "PASS", "10x": "PASS"}
    assert snapshots["1x"]["convergence"]["first_pass_step"] == 42
    assert snapshots["1x"]["convergence"]["passing_suffix"] == 0


@pytest.mark.parametrize("last_bad,status,suffix", [(235, "PASS", 5), (236, "FAIL", 4), (240, "FAIL", 0)])
def test_five_terminal_checks_cannot_be_replaced_by_a_good_endpoint(last_bad, status, suffix):
    value = task(); grade = grade_result(value, receipt(value, (last_bad,)))
    assert grade["status"] == status
    assert grade["evaluator_result"]["convergence"]["passing_suffix"] == suffix


def test_a_metric_passes_only_when_all_thresholds_pass_together():
    value = task(); raw = receipt(value)
    for index, point in enumerate(raw["evidence"]["observations"][-5:]):
        point["quality"] = .95 if index % 2 else .5
        point["modes"] = 15 if index % 2 else 16
    raw["evidence"]["live"] = dict(raw["evidence"]["observations"][-1])
    assert grade_result(value, raw)["status"] == "FAIL"


@pytest.mark.parametrize("mutation,status", [
    (lambda e: e["observations"].pop(10), "INCOMPLETE"),
    (lambda e: e["observations"].pop(), "INVALID"),
    (lambda e: e["observations"].append(dict(e["observations"][-1])), "INVALID"),
    (lambda e: e["observations"][5].update(step=1), "INVALID"),
    (lambda e: e["observations"][5].update(step=250.5), "INVALID"),
    (lambda e: e["observations"][5].update(quality=float("inf")), "INVALID"),
    (lambda e: e["observations"][5].update(extra=float("nan")), "INVALID"),
    (lambda e: e["observations"][5].update(quality=True), "INVALID"),
    (lambda e: e["observations"][5].pop("modes"), "INVALID"),
    (lambda e: e["live"].clear(), "INCOMPLETE"),
    (lambda e: e.update(observations=[]), "INCOMPLETE"),
    (lambda e: e.update(scoring_weights="ema"), "INVALID"),
])
def test_incomplete_nonfinite_or_wrong_cadence_evidence_cannot_pass(mutation, status):
    value = task(); raw = receipt(value); mutation(raw["evidence"])
    assert grade_result(value, raw)["status"] == status


def test_24_checks_spread_across_long_budget_cannot_replace_240_checks():
    value = task(); raw = receipt(value)
    raw["evidence"]["observations"] = raw["evidence"]["observations"][9::10]
    assert grade_result(value, raw)["status"] == "INCOMPLETE"


def test_independent_live_override_cannot_change_the_final_observation():
    value = task(); raw = receipt(value, (240,))
    raw["evidence"]["live"]["quality"] = .95
    assert grade_result(value, raw)["status"] == "INVALID"
    raw = receipt(value); raw["evidence"]["live"]["quality"] = .96
    assert grade_result(value, raw)["status"] == "INVALID"


@pytest.mark.parametrize("mutation,status", [
    (lambda e: e.pop("guards"), "INCOMPLETE"),
    (lambda e: e["guards"].update(all_finite=False), "FAIL"),
    (lambda e: e["guards"]["optimizer_updates"].update(prior=9999), "INCOMPLETE"),
    (lambda e: e["guards"].update(unintended_rng_deviations=1), "INVALID"),
])
def test_shared_finite_update_and_rng_guards_still_apply(mutation, status):
    value = task(); raw = receipt(value); mutation(raw["evidence"])
    assert grade_result(value, raw)["status"] == status


def test_grading_does_not_mutate_task_or_recorded_evidence():
    value = task(); raw = receipt(value, (48,)); before = deepcopy((value, raw))
    grade_result(value, raw)
    assert (value, raw) == before


def test_existing_transfer_sustained_retains_24_checks_and_terminal_rule():
    value = task()
    value["execution"]["steps"] = 1000
    value["evaluation"] = {"kind": "transfer_sustained", "thresholds": [["quality", ">=", .9]],
                           "observations": 24, "minimum_stable_checks": 5, "scoring_weights": "live"}
    points = [{"step": step, "quality": .95} for step in _checkpoints(value)]
    raw = {"evidence": {"observations": points, "live": dict(points[-1])}}
    assert len(points) == 24 and grade_result(value, raw)["status"] == "PASS"
    points[-5]["quality"] = .5
    assert grade_result(value, raw)["status"] == "FAIL"
    assert "prefix_snapshots" not in grade_result(value, raw)["evaluator_result"]
    assert _checkpoints({"execution": {"steps": 2}}) == [1, 2]


@pytest.mark.parametrize("kind", [KIND, "transfer_sustained"])
def test_vector_adapter_keeps_schedule_horizon_but_executes_declared_budget(monkeypatch, tmp_path, kind):
    """Mock only the update engine: consume the adapter loop without training."""
    import torch
    from benchmarks.transfer_suite import vector_tasks
    from experiments.forge import adapters, vectorprofiles

    value = task(400)
    if kind == "transfer_sustained":
        value["evaluation"]["kind"] = kind
        value["execution"]["steps"] = 400
    build_options, observations, consumed = [], [], []
    context = SimpleNamespace(recipe=SimpleNamespace(total_steps=400, batch_size=4),
                              streams=SimpleNamespace(generator=lambda *args, **kwargs: torch.Generator()))

    def build_trainer(generator, discriminator, **options):
        build_options.append(options)
        return SimpleNamespace(max_steps=options.get("max_steps", context.recipe.total_steps))

    context.build_trainer = build_trainer

    class Run:
        def __init__(self, context, trainer, output, task):
            self.trainer, self.completed_steps = trainer, 0

        def step(self, batch):
            if self.completed_steps >= self.trainer.max_steps:
                raise RuntimeError("execution budget exhausted")
            self.completed_steps += 1
            consumed.append(self.completed_steps)

        def sample(self, count):
            return torch.zeros(count, 1)

        def evaluate(self, evaluator):
            observations.append(self.completed_steps)
            return evaluator()

        def receipt(self, evidence, **options):
            return evidence

    monkeypatch.setattr(adapters, "_context", lambda *args: context)
    monkeypatch.setattr(adapters, "_Run", Run)
    monkeypatch.setattr(adapters, "_host_receipt", lambda *args: {})
    monkeypatch.setattr(adapters, "_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(vectorprofiles, "resolve_vector_spec", lambda task: {"particles": 12, "z_dim": 2, "batch": 4})
    monkeypatch.setattr(vectorprofiles, "build_vector_models", lambda *args: (None, None))
    monkeypatch.setattr(vector_tasks, "sample_target", lambda spec, count, data, step: torch.zeros(count, 1))
    monkeypatch.setattr(vector_tasks, "score_samples", lambda *args: {"quality": .95, "modes": 16})
    raw = adapters._vector({}, value, tmp_path, "cpu", retain_scored_outputs=False)
    assert context.recipe.total_steps == 400
    assert build_options == ([{"max_steps": 4000}] if kind == KIND else [{}])
    assert consumed[-1] == value["execution"]["steps"]
    assert observations == _checkpoints(value)
    assert len(raw["observations"]) == (240 if kind == KIND else 24)
