"""Reporting guards use saved compact evidence, never new model execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.model_toy_publication import PUBLICATION, PROTOCOL, load_publication

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def publication(tmp_path):
    document = read_json(ROOT / PUBLICATION)
    for path in (PROTOCOL, PUBLICATION):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    for row in document["readouts"]:
        if row.get("gif"):
            path = PUBLICATION.parent / row["gif"]
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, tmp_path / path)
    return tmp_path, document


def test_inactive_recipe_fields_rehydrate_without_training(publication):
    root, document = publication
    assert any("adam_variant" not in row["recipe"] for row in document["runs"])
    assert load_publication(root) == document
    assert document["qualification_input"] is False
    assert len(document["readouts"]) == 8


def test_changed_global_recipe_cannot_hide_behind_a_recomputed_hash(publication):
    root, document = publication
    row = document["runs"][0]
    row["recipe"]["lr"] *= .5
    row["recipe_sha256"] = stable_hash(row["recipe"])
    atomic_json(root / PUBLICATION, document)
    with pytest.raises(ValueError, match="global baseline"):
        load_publication(root)


def test_blocked_prefix_cannot_be_published_as_full_quality(publication):
    root, document = publication
    row = next(row for row in document["readouts"] if not row["full_protocol_complete"])
    row["verdict"] = "PASS"
    atomic_json(root / PUBLICATION, document)
    with pytest.raises(ValueError, match="partial model research"):
        load_publication(root)


@pytest.mark.parametrize("changed_core", [False, True])
def test_reporting_adapter_accepts_only_the_frozen_science(tmp_path, monkeypatch, changed_core):
    from experiments.forge import sources

    path = ROOT / "reports/forge/api-model-toys/run.py"
    spec = importlib.util.spec_from_file_location("model_mode_research_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Later develop merges cannot qualify changed package code using this run.
    source = read_json(ROOT / "reports/forge/api-model-toys/execution-source.json")
    assert source["digest"] == stable_hash(source["files"])
    repair = read_json(runner.REPORTING_REPAIR)
    for name, patch in repair["reporting_only_changes"].items():
        assert source["files"][name] == patch["frozen_sha256"]
        assert file_hash(ROOT / name) == patch["repaired_sha256"]
        source["files"][name] = patch["repaired_sha256"]
    if changed_core:
        source["files"]["particlegan/recipes.py"] = "changed-core-code"
    monkeypatch.setattr(sources, "inspect_source", lambda *a, **k: source)
    monkeypatch.setattr(sys, "argv", [str(path), "--output", str(tmp_path)])
    def forbidden(*args, **kwargs):
        raise AssertionError("this metadata proof must not launch a worker")
    monkeypatch.setattr(runner, "execute_stage", forbidden)
    # Source verification passes, then the existing output directory refuses a
    # duplicate execution before any process, optimizer, scorer or sampler runs.
    error = ValueError if changed_core else FileExistsError
    with pytest.raises(error, match="scientific source" if changed_core else None):
        runner.main()


def test_recall_keeps_blocked_prefixes_separate_from_full_results(publication, monkeypatch):
    from collections import Counter
    from experiments.forge import knowledge, publication_memory

    root, document = publication
    monkeypatch.setattr(knowledge, "_attempts", lambda *a: pytest.fail("recall must not hydrate execution evidence"))
    records = publication_memory.normalize(root)
    assert len(records) == 1
    record = records[0]
    assert record["qualification_input"] is record["qualification_reuse"] is False
    assert record["source"]["sha256"] == file_hash(root / PUBLICATION)
    assert record["record_type"] == "published_study"
    assert Counter(row["gate_status"] for row in record["task_results"]) == {"PASS": 3, "FAIL": 3, "BLOCKED": 2}
    for row in record["task_results"]:
        if row["gate_status"] == "BLOCKED":
            assert row["metrics"] == {}
            assert row["software_prefix"]["status"] == "INCOMPLETE"
            assert row["completed_updates"] == 3 < row["required_updates"]
    for query in ("model_vae_hard", "worst_transition_cdf", "api-model-toys-v1"):
        assert knowledge.recall(root, query, "discriminator_stability")
    assert knowledge._records(root) == ([], [])


def test_memory_tracks_protocol_publication_and_actual_media(publication):
    from experiments.forge.publication_memory import input_paths

    root, document = publication
    paths = set(input_paths(root))
    assert {root / PROTOCOL, root / PUBLICATION} <= paths
    assert {root / PUBLICATION.parent / row["gif"] for row in document["readouts"]} <= paths
