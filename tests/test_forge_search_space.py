"""Lazy finite draws, categorical admission, shared budgets and immutable manifests."""
from copy import deepcopy
import random

import pytest

from test_forge_configuration_search import checkout
from experiments.forge import configuration_search as search
from experiments.forge import search_space as spaces
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue


@pytest.fixture
def definition(checkout):
    base = read_json(checkout / "configs/forge/ideas/base.json")
    second = {**base, "id": "second", "recipe_preset": "bcap", "recipe_overrides": {"loss": "least_squares"}}
    atomic_json(checkout / "configs/forge/ideas/second.json", second)
    return {"schema": spaces.SPACE_SCHEMA, "id": "finite-space", "hypothesis": "Compare declared complete trainers",
            "protocol": "screening", "view": "stability", "execution_backend": "cpu", "tuning_through_tier": 1,
            "campaign": {"id": "finite-space", "budget_seconds": 40, "candidate_budget_seconds": 10},
            "samples": 4, "parameters": {"lr": {"kind": "choice", "values": [.003, .005]}},
            "candidates": [{"base_candidate": "base", "trainer_family": "first"},
                           {"base_candidate": "second", "trainer_family": "second"}]}


def test_compilation_is_readonly_reproducible_and_does_not_consume_global_rng(checkout, definition):
    before = {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    rng = random.getstate()
    manifest = spaces.compile_space(checkout, checkout / "runs", definition)
    assert random.getstate() == rng
    assert stable_hash(manifest) == stable_hash(spaces.compile_space(checkout, checkout / "runs", definition))
    assert {d["index"] for d in manifest["draws"]} == set(range(4))
    assert manifest["declared_worst_case_seconds"] == 40
    assert len(manifest["searches"]) == 2
    assert manifest["rng"]["initial_state"] != manifest["rng"]["final_state"]
    assert before == {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}


def test_sampler_handles_huge_populations_without_expansion():
    indices = spaces._sample_indices(random.Random(0), 10**40, 256)
    assert len(set(indices)) == 256 and all(0 <= index < 10**40 for index in indices)
    assert spaces._at([[{"lr": 1}, {"lr": 2}], [{"eps": 3}, {"eps": 4}]], 3) == {"lr": 2, "eps": 4}


def test_prose_and_budget_changes_do_not_change_the_search_stream(checkout, definition):
    initial = spaces.compile_space(checkout, checkout / "runs", definition)
    definition["hypothesis"] = "Reworded hypothesis"
    definition["campaign"]["budget_seconds"] = 100
    changed = spaces.compile_space(checkout, checkout / "runs", definition)
    assert initial["draws"] == changed["draws"]
    assert initial["rng"] == changed["rng"]
    assert initial["manifest_hash"] != changed["manifest_hash"]


def test_logspace_can_span_extreme_finite_values():
    values = spaces._values({"kind": "logspace", "low": 1e-300, "high": 1e300, "count": 3})
    assert values == pytest.approx([1e-300, 1., 1e300])


def test_literal_pairs_logspace_and_conditional_effective_knobs(checkout, definition):
    definition["parameters"] = {"lr": {"kind": "logspace", "low": .001, "high": .01, "count": 3},
                                "betas": {"kind": "literal", "value": [0, .9]}}
    definition["candidates"][1]["parameters"] = {"lr": {"kind": "literal", "value": .002}}
    manifest = spaces.compile_space(checkout, checkout / "runs", definition)
    assert manifest["population"] == 4
    rates = sorted(d["settings"]["lr"] for d in manifest["draws"] if d["base_candidate"] == "base")
    assert rates == pytest.approx([.001, .001 * 10**.5, .01])
    assert all("betas" not in d["settings"] for d in manifest["draws"] if d["base_candidate"] == "second")


def test_aggregate_budget_checks_all_categories_before_writing(checkout, definition):
    definition["campaign"]["budget_seconds"] = 30
    with pytest.raises(ValueError, match="shared campaign"):
        spaces.write_compilation(checkout, checkout / "runs", definition, checkout / "manifest.json")
    assert not (checkout / "manifest.json").exists()


def test_manifest_tampering_and_source_drift_block_admission(checkout, definition):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    manifest = spaces.compile_space(checkout, queue.root, definition)
    corrupt = deepcopy(manifest)
    corrupt["draws"][0]["settings"]["lr"] = .99
    with pytest.raises(ValueError, match="changed"):
        spaces.enqueue_compilation(checkout, queue.root, corrupt, queue=queue)
    base = read_json(checkout / "configs/forge/ideas/base.json")
    base["recipe_overrides"]["lr"] = .004
    atomic_json(checkout / "configs/forge/ideas/base.json", base)
    with pytest.raises(ValueError, match="changed"):
        spaces.enqueue_compilation(checkout, queue.root, manifest, queue=queue)
    assert not queue.inspect()["submissions"]


def test_all_categories_freeze_before_submission_and_recovery_is_idempotent(checkout, definition, monkeypatch):
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    manifest = spaces.compile_space(checkout, queue.root, definition)
    original_resolve, original_submit = search.resolve_idea, queue.submit
    frozen = []
    def resolve(*args, **kwargs):
        result = original_resolve(*args, **kwargs)
        if kwargs.get("freeze_source"):
            frozen.append(result)
        return result
    def submit(*args, **kwargs):
        assert len(frozen) == 4
        return original_submit(*args, **kwargs)
    monkeypatch.setattr(search, "resolve_idea", resolve)
    monkeypatch.setattr(queue, "submit", submit)
    summary = spaces.enqueue_compilation(checkout, queue.root, manifest, queue=queue)
    assert summary["submitted_count"] == 4
    first = set(queue.inspect()["submissions"])
    frozen.clear()
    spaces.enqueue_compilation(checkout, queue.root, manifest, queue=queue)
    assert set(queue.inspect()["submissions"]) == first
    assert summary["selection"]["selection_kind"] == "pending"


@pytest.mark.parametrize("parameters", [
    {"lr": [.1]}, {"lr": {"kind": "choice", "values": [.1, .1]}},
    {"loss": {"kind": "choice", "values": ["hinge"]}},
    {"seed": {"kind": "literal", "value": 0}},
    {"lr": {"kind": "logspace", "low": 0, "high": .1, "count": 2}},
])
def test_invalid_space_values_are_rejected(checkout, definition, parameters):
    definition["parameters"] = parameters
    with pytest.raises(ValueError):
        spaces.compile_space(checkout, checkout / "runs", definition)
