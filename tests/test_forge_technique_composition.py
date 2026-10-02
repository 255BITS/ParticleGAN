"""Composed displays preserve independent publication rows and cohort identities."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash

SCRIPT = Path(__file__).resolve().parents[1] / "reports/forge/regenerate_technique_inventory.py"
spec = importlib.util.spec_from_file_location("forge_technique_composition", SCRIPT)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)
NEW = "r3gan-stacked-training-toy-v1"


def _seal(report):
    report = deepcopy(report)
    report.setdefault("provenance", {}).pop("input_digest", None)
    report["provenance"]["input_digest"] = stable_hash(report)
    return report


def _row(candidate, source, *, passed=0, attempt=None, hardware="gpu-old"):
    return {"candidate_id": candidate, "candidate_revision": "revision-" + source,
            "cohort": "cohort-" + source + "-" + candidate, "technique": candidate,
            "bindings": {"source_digest": source, "prior": {"kind": "mog", "sigma": .025},
                         "claim_contract": {"sampling_law": "clean"}},
            "runtime_cohort": {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": hardware}}},
            "tiers": {"1": {"passed": passed, "total": 3}, "2": {"passed": 0, "total": 19},
                      "3": {"passed": 0, "total": 2}}, "qualified_tier": 1 if passed == 3 else 0,
            "tasks": [{"task_id": "two_pole", "status": "PASS" if passed else "FAIL"}],
            "attempt_ids": [attempt] if attempt else [], "cost": {"wall_seconds": 5.25}}


def _report(rows, *, frozen=False):
    return {"schema_version": 1, "view": "discriminator_stability", "view_revision": 2,
            "policy_fingerprint": "same-policy", "execution_backend": "cuda",
            "tier_requirements": {"1": ["two_pole", "unused_token_hold", "ae_gan_hold"],
                                  "2": [f"quality-{i}" for i in range(19)], "3": ["hold", "extension"]},
            "publication_scope": "frozen_source" if frozen else "live_current",
            "frozen_source": {"commit": "original-commit"} if frozen else None,
            "rows": rows, "provenance": {"qualified_receipts": {}},
            "recipe_contracts": {}, "protocol_contracts": {}, "task_contracts": {}, "status_reasons": {}}


def _summary(root, row, attempt):
    proof = {"canonical_result_hash": "result-" + attempt, "source_digest": row["bindings"]["source_digest"],
             "original_file_sha256": {name: name + "-hash" for name in ("request", "evidence", "result")}}
    compact = {"candidate_id": row["candidate_id"], "candidate_revision": row["candidate_revision"],
               "certificate_validated": True, "qualification_input": False, "qualification_reuse": False,
               "provenance": {"canonical_result_hash": proof["canonical_result_hash"],
                              "source_digest": proof["source_digest"],
                              "original_files": {name: {"sha256": value} for name, value in proof["original_file_sha256"].items()}}}
    atomic_json(root / f"reports/forge/technique-receipts/{attempt}.json", compact)
    return proof


@pytest.fixture
def inputs(tmp_path):
    old = _row("bcap", "a" * 64, passed=3, attempt="old-attempt")
    new = _row(NEW, "b" * 64, attempt="new-attempt", hardware="gpu-new")
    new["bindings"]["prior"] = {"kind": "particle_cloud", "sigma": 0.}
    new["bindings"]["claim_contract"] = {"sampling_law": "noisy"}
    original = _report([old], frozen=True)
    current = _report([_row("bcap", "b" * 64), new])
    original["provenance"]["qualified_receipts"]["old-attempt"] = _summary(tmp_path, old, "old-attempt")
    current["provenance"]["qualified_receipts"]["new-attempt"] = _summary(tmp_path, new, "new-attempt")
    for name, report in (("original", original), ("current", current)):
        atomic_json(tmp_path / f"reports/forge/{name}.json", _seal(report))
        atomic_text(tmp_path / f"reports/forge/{name}.md", f"# {name} publication\n")
    return tmp_path


def _compose(root, **kwargs):
    return publication.compose(root, original_report="reports/forge/original.json",
                               current_report="reports/forge/current.json", output_prefix="reports/forge/expanded", **kwargs)


def test_composition_preserves_frozen_rows_and_separates_new_source_runtime_and_sampling(inputs):
    root = inputs
    originals = {path: path.read_bytes() for path in (root / "reports/forge").glob("*.*")}
    first = _compose(root)
    result = read_json(first["json"])
    assert len(result["rows"]) == 2
    frozen_row = read_json(root / "reports/forge/original.json")["rows"][0]
    assert all(result["rows"][0][key] == value for key, value in frozen_row.items())
    assert result["rows"][0]["tiers"]["1"] == {"passed": 3, "total": 3}
    assert result["rows"][1]["tiers"]["1"] == {"passed": 0, "total": 3}
    assert result["rows"][0]["bindings"]["claim_contract"]["sampling_law"] == "clean"
    assert result["rows"][1]["bindings"]["claim_contract"]["sampling_law"] == "noisy"
    assert result["rows"][0]["runtime_cohort"] != result["rows"][1]["runtime_cohort"]
    assert result["qualification_reuse"] is False and result["qualification_input"] is False
    assert all(row["qualification_reuse"] is False for row in result["rows"])
    assert result["source_publications"]["original"]["json_sha256"] == file_hash(root / "reports/forge/original.json")
    assert originals == {path: path.read_bytes() for path in originals}
    assert not (root / "reports/forge/attempts").exists()
    markdown = Path(first["report"]).read_text()
    assert "R3GAN Stacked-MNIST recipe (toy-host adaptation)" in markdown
    assert "a" * 12 in markdown and "b" * 12 in markdown
    assert "gpu-old" in markdown and "gpu-new" in markdown
    assert "(original.md)" in markdown and "(current.md)" in markdown
    assert "does not pool passes" in markdown
    assert "--compose-original reports/forge/original.json" in markdown
    before = {Path(first[key]): Path(first[key]).read_bytes() for key in ("report", "json")}
    times = {path: path.stat().st_mtime_ns for path in before}
    assert first == _compose(root)
    assert before == {path: path.read_bytes() for path in before}
    assert times == {path: path.stat().st_mtime_ns for path in before}


def test_tampered_input_cannot_overwrite_existing_composition(inputs):
    root = inputs
    first = _compose(root)
    before = Path(first["json"]).read_bytes()
    path = root / "reports/forge/original.json"
    report = read_json(path)
    report["rows"][0]["tiers"]["1"]["passed"] = 0
    atomic_json(path, report)
    with pytest.raises(ValueError, match="input digest mismatch"):
        _compose(root)
    assert Path(first["json"]).read_bytes() == before


@pytest.mark.parametrize("field", ["view", "policy_fingerprint", "tier_requirements", "execution_backend"])
def test_incompatible_declared_protocols_keep_separate_reports(inputs, field):
    path = inputs / "reports/forge/current.json"
    report = read_json(path)
    report[field] = "different"
    atomic_json(path, _seal(report))
    with pytest.raises(ValueError, match="incompatible " + field):
        _compose(inputs)
    assert not (inputs / "reports/forge/expanded.json").exists()


def test_missing_current_denominator_rows_are_rejected(inputs):
    path = inputs / "reports/forge/current.json"
    report = read_json(path)
    report["rows"] = report["rows"][1:]
    atomic_json(path, _seal(report))
    with pytest.raises(ValueError, match="every original technique denominator row"):
        _compose(inputs)


def test_multiple_runtime_rows_cannot_be_pooled(inputs):
    path = inputs / "reports/forge/current.json"
    report = read_json(path)
    extra = deepcopy(report["rows"][1])
    extra["runtime_cohort"]["execution_backend"] = "cpu"
    report["rows"].append(extra)
    atomic_json(path, _seal(report))
    with pytest.raises(ValueError, match="exactly one requested technique/runtime cohort"):
        _compose(inputs)


def test_compact_receipt_proof_must_match_selected_source_and_revision(inputs):
    path = inputs / "reports/forge/technique-receipts/new-attempt.json"
    summary = read_json(path)
    summary["candidate_revision"] = "unrelated"
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="candidate/source cohort differs"):
        _compose(inputs)
    assert not (inputs / "reports/forge/expanded.json").exists()


def test_composition_cannot_overwrite_either_source_publication(inputs):
    before = (inputs / "reports/forge/original.json").read_bytes()
    with pytest.raises(ValueError, match="cannot overwrite"):
        publication.compose(inputs, original_report="reports/forge/original.json", current_report="reports/forge/current.json",
                            output_prefix="reports/forge/original")
    assert (inputs / "reports/forge/original.json").read_bytes() == before


def test_forged_contract_catalog_hash_is_rejected(inputs):
    path = inputs / "reports/forge/current.json"
    report = read_json(path)
    report["recipe_contracts"] = {"invalid-hash": {"reg_arm": "other"}}
    atomic_json(path, _seal(report))
    with pytest.raises(ValueError, match="invalid recipe_contracts identity"):
        _compose(inputs)
