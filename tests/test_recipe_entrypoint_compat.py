"""Source-refusal expectations confer no compatibility or qualification credit."""
from hashlib import sha256

import pytest

from recipe_entrypoint_compat import frozen_restoration_source, model_source_blockers


def source(tmp_path, relative, data):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def task(relative, expected):
    return {"evaluation": {"sources": {relative: sha256(expected).hexdigest()}}}


def test_matching_source_needs_no_restoration_or_refusal(tmp_path):
    relative = "benchmarks/toy_audit/api_contract.py"
    original = b"exact declared source\n"
    source(tmp_path, relative, original)
    assert model_source_blockers(tmp_path, task(relative, original)) == []
    assert frozen_restoration_source(tmp_path, relative, sha256(original).hexdigest()) == original


def test_unrelated_source_change_is_refused_without_becoming_a_restoration(tmp_path):
    relative = "benchmarks/toy_audit/api_contract.py"
    original, changed = b"original contract\n", b"new contract and capabilities\n"
    path = source(tmp_path, relative, changed)
    assert model_source_blockers(tmp_path, task(relative, original)) == [
        f"model source binding differs: {relative}"]
    with pytest.raises(AssertionError, match=relative):
        frozen_restoration_source(tmp_path, relative, sha256(original).hexdigest())
    assert path.read_bytes() == changed


def test_exact_rename_proof_still_returns_original_bytes_and_refuses_current(tmp_path):
    relative = "benchmarks/toy_audit/api_model_ae.py"
    original, changed = b"recipe = Recipe(**saved)\n", b"recipe = Recipe.from_dict(saved)\n"
    path = source(tmp_path, relative, changed)
    assert frozen_restoration_source(tmp_path, relative, sha256(original).hexdigest()) == original
    assert model_source_blockers(tmp_path, task(relative, original)) == [
        f"model source binding differs: {relative}"]
    assert path.read_bytes() == changed


def test_extra_change_cannot_hide_in_a_declared_restoration_module(tmp_path):
    relative = "benchmarks/toy_audit/api_model_ae.py"
    original = b"recipe = Recipe(**saved)\n"
    changed = b"recipe = Recipe.from_dict(saved)\n# another scientific change\n"
    source(tmp_path, relative, changed)
    assert model_source_blockers(tmp_path, task(relative, original)) == [
        f"model source binding differs: {relative}"]
    with pytest.raises(AssertionError, match=relative):
        frozen_restoration_source(tmp_path, relative, sha256(original).hexdigest())
