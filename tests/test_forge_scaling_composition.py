"""Incremental performance updates preserve complete, source-bound matrices."""
from copy import deepcopy

import pytest

from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge import scaling_publication as publication
from test_forge_scaling_publication import receipt, write_receipt


def archive(root, name, choices, *, sizes=(32, 64)):
    entries = []
    for family, candidate in choices:
        for size in sizes:
            entries.append(write_receipt(root, receipt(size=size, candidate=candidate),
                                         name=f"{name}-{family}-{size}.json", family=family,
                                         task=f"r{size}"))
    document = publication.build_manifest(root, entries)
    document["export"] = {"registration": name, "family_selections": [
        {"trainer_family": family, "candidate_id": candidate} for family, candidate in choices]}
    document.pop("input_digest")
    document["input_digest"] = stable_hash(document)
    path = f"reports/forge/scaling-history/{name}.json"
    atomic_json(root / path, document)
    return path, document


def test_overlay_preserves_other_families_and_replaces_whole_recipe(tmp_path):
    old, original = archive(tmp_path, "original", [("bispf", "bispf-v2"), ("atlas", "atlas")])
    new, optimized = archive(tmp_path, "optimized", [("bispf", "bispf-optimized-v2")])
    before = file_hash(tmp_path / old)
    result = publication.compose_manifests(tmp_path, [old, new])
    assert file_hash(tmp_path / old) == before
    assert len(result["entries"]) == 4
    assert [row for row in result["entries"] if row["trainer_family"] == "atlas"] == [
        row for row in original["entries"] if row["trainer_family"] == "atlas"]
    assert [row for row in result["entries"] if row["trainer_family"] == "bispf"] == optimized["entries"]
    assert result["export"]["sources"][0]["sha256"] == before
    assert result["qualification_input"] is False
    assert not (tmp_path / publication.MANIFEST).exists()


def test_overlay_rejects_different_tasks_and_incomplete_roster(tmp_path):
    old, _ = archive(tmp_path, "original", [("bispf", "bispf-v2")])
    smaller, _ = archive(tmp_path, "smaller", [("bispf", "bispf-optimized-v2")], sizes=(32,))
    with pytest.raises(ValueError, match="different performance task"):
        publication.compose_manifests(tmp_path, [old, smaller])
    path, document = archive(tmp_path, "incomplete", [("bispf", "bispf-optimized-v2")])
    document["entries"].pop()
    document.pop("input_digest")
    document["input_digest"] = stable_hash(document)
    atomic_json(tmp_path / path, document)
    with pytest.raises(ValueError, match="every registered recipe/task cell"):
        publication.compose_manifests(tmp_path, [old, path])


def test_overlay_rejects_changed_original_receipt(tmp_path):
    path, document = archive(tmp_path, "original", [("bispf", "bispf-v2")])
    row = document["entries"][0]
    changed = deepcopy(receipt())
    changed["metrics"]["update_ms_median"] = 1.
    atomic_json(tmp_path / row["receipt"], changed)
    with pytest.raises(ValueError, match="receipt bytes changed"):
        publication.compose_manifests(tmp_path, [path])


def test_loading_composite_rechecks_exact_archived_manifest_bytes(tmp_path):
    path, document = archive(tmp_path, "original", [("bispf", "bispf-v2")])
    result = publication.compose_manifests(tmp_path, [path])
    atomic_json(tmp_path / publication.MANIFEST, result)
    assert publication.load_performance(tmp_path)["records"]
    document["export"]["registration"] = "changed"
    atomic_json(tmp_path / path, document)
    with pytest.raises(ValueError, match="archived composite source bytes changed"):
        publication.load_performance(tmp_path)
