"""Canonical selection must reduce duplicate listing without deleting controls."""
from copy import deepcopy
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.toy_audit import canonical_selection as selection


@pytest.fixture
def inventory():
    return selection.load()


def find(catalog, name):
    return next(case for case in catalog["cases"] if case["id"] == name)


def test_default_view_retains_every_named_case_and_original_evidence(inventory):
    manifest, catalog = inventory
    before = deepcopy((manifest, catalog))
    problems = selection.list_problems(manifest, catalog)
    cases = selection.list_cases(manifest, catalog)
    assert len(problems) == 97 and len(cases) == 109
    assert len({case for problem in problems for case in problem["retained_cases"]}) == 109
    assert sum(case["retention"] == "control" for case in cases) == 12
    assert all(case["reason"].strip() for case in cases)
    for case in cases:
        original = find(catalog, case["catalog_id"])
        assert (case["original_name"], case["original_rating"], case["original_status"], case["original_media"]) == (
            original["name"], original["rating"], original["status"], original.get("media"))
    assert (manifest, catalog) == before
    assert manifest["counts"]["deleted_merged_source_files"] == 0
    assert manifest["counts"]["removed_evidence_cases"] == 0


def test_all_named_architecture_and_ring_controls_remain_accessible(inventory):
    manifest, catalog = inventory
    image = selection.list_problems(manifest, catalog, "develop-img_bars4")[0]
    assert set(image["retained_cases"]) == {"develop-img_bars4", "develop-img_tiny_generator", "develop-img_residual_bars4"}
    ring = selection.list_problems(manifest, catalog, "develop-stress_fast_critic")[0]
    assert len(ring["retained_cases"]) == 8
    for name in ("develop-img_uniform_generator", "develop-img_mean_discriminator", "pr58",
                 "develop-stress_long_horizon", "develop-reserved_alternating_critic_updates"):
        row = selection.list_cases(manifest, catalog, name)[0]
        assert row["retention"] == "control"
        assert row["original_status"] == find(catalog, name)["status"]


def test_active_names_are_bounded_and_original_negative_grades_are_archival(inventory):
    manifest, catalog = inventory
    rows = {row["catalog_id"]: row for row in selection.list_cases(manifest, catalog)}
    for name in ("pr61", "pr63", "pr65"):
        assert rows[name]["display_name"].startswith("Unconditional")
        assert rows[name]["original_name"] == find(catalog, name)["name"]
    assessment = json.loads((selection.ROOT / manifest["current_definition_assessments"]["path"]).read_text())
    latest = {row["id"]: row for row in assessment["rows"]}
    for name in ("develop-img_mean_discriminator", "develop-img_uniform_generator"):
        assert rows[name]["display_name"].endswith("unit")
        assert rows[name]["original_rating"] == 2 and latest[name]["followup_rating"] == 4
        assert rows[name]["original_status"] == "FAIL" and rows[name]["retention"] == "control"


@pytest.mark.parametrize("edit", ("missing", "unknown", "duplicate"))
def test_unknown_missing_and_duplicate_retained_ids_are_rejected(inventory, edit):
    manifest, catalog = deepcopy(inventory)
    if edit == "missing":
        manifest["records"].pop()
    elif edit == "unknown":
        manifest["records"][0]["catalog_id"] = "invented-problem"
    else:
        manifest["records"].append(deepcopy(manifest["records"][0]))
    with pytest.raises(ValueError, match="IDs"):
        selection.validate(manifest, catalog)


def test_conflicting_group_membership_is_rejected(inventory):
    manifest, catalog = deepcopy(inventory)
    duplicate = deepcopy(manifest["groups"][0])
    duplicate["canonical_id"] = "develop-img_uniform_generator"
    manifest["groups"].append(duplicate)
    with pytest.raises(ValueError, match="conflicting aliases"):
        selection.validate(manifest, catalog)


@pytest.mark.parametrize("case_id,key,value", (
    ("develop-img_uniform_generator", "pattern", "intensity2"),
    ("develop-img_uniform_generator", "noise_std", .02),
    ("develop-img_uniform_generator", "modes", 3),
    ("develop-stress_slow_critic", "covariances", [[[.36, 0.], [0., .36]]] * 8),
    ("develop-stress_slow_critic", "masses", [.2] + [.8 / 7] * 7),
    ("develop-stress_slow_critic", "scale_end", 2.),
))
def test_changed_target_law_cannot_hide_under_an_existing_alias(inventory, case_id, key, value):
    manifest, catalog = deepcopy(inventory)
    find(catalog, case_id)["spec"][key] = value
    with pytest.raises(ValueError, match="different.*law|different image mode"):
        selection.validate(manifest, catalog)


def test_same_looking_overlap_target_cannot_join_the_static_ring(inventory):
    manifest, catalog = deepcopy(inventory)
    manifest["groups"][-1]["members"].append("develop-stress_overlapping_data")
    with pytest.raises(ValueError, match="different or unproved law"):
        selection.validate(manifest, catalog)


def test_an_unreviewed_bank_proof_cannot_be_injected(inventory):
    manifest, catalog = deepcopy(inventory)
    original = manifest["template_banks"]["stripes2"]
    manifest["template_banks"]["unreviewed_bank"] = deepcopy(original)
    with pytest.raises(ValueError, match="registered law proofs changed"):
        selection.validate(manifest, catalog)


def test_shared_kernels_and_unit_changes_remain_separate(inventory):
    manifest, catalog = inventory
    rows = {row["catalog_id"]: row for row in selection.list_cases(manifest, catalog)}
    for names in (("source-family-04", "source-family-05"), ("source-family-06", "source-family-07"),
                  ("develop-trajectory", "develop-residual_student"),
                  ("develop-mode_hold", "source-family-10", "develop-stress_fast_critic"),
                  ("pr45-adapted", "pr47-adapted", "develop-vector_two_broad"),
                  ("pr64", "develop-img_stripes2")):
        assert len({rows[name]["canonical_id"] for name in names}) == len(names)
    unknown = next(p for p in selection.list_problems(manifest, catalog) if p["canonical_id"] == "source-family-14")
    assert unknown["equivalence"] == "separately retained; global equivalence unproved"


def test_dynamic_target_budget_is_a_law_field_but_static_horizon_is_not(inventory):
    manifest, catalog = inventory
    proof = dict(kind="vector_sampler")
    static = deepcopy(find(catalog, "develop-stress_fast_critic"))
    baseline = selection.law_fingerprint(static, proof, manifest["template_banks"])
    static["spec"]["steps"] *= 2
    assert selection.law_fingerprint(static, proof, manifest["template_banks"]) == baseline
    dynamic = deepcopy(find(catalog, "develop-vector_scale_drift"))
    baseline = selection.law_fingerprint(dynamic, proof, manifest["template_banks"])
    dynamic["spec"]["steps"] *= 2
    assert selection.law_fingerprint(dynamic, proof, manifest["template_banks"]) != baseline


def test_image_proofs_match_actual_source_templates_and_ignore_only_mode_order(inventory):
    from benchmarks.transfer_suite import image_tasks
    manifest, catalog = inventory
    for pattern, proof in manifest["template_banks"].items():
        spec = next(c["spec"] for c in catalog["cases"] if c.get("spec", {}).get("pattern") == pattern)
        bank = image_tasks.templates(spec).numpy()
        def fingerprint(images):
            return hashlib.sha256(str(images.shape).encode() + images.dtype.str.encode()
                                  + b"".join(sorted(np.ascontiguousarray(image).tobytes() for image in images))).hexdigest()
        assert fingerprint(bank) == proof["sha256"]
        assert fingerprint(bank[::-1]) == proof["sha256"]
        damaged = bank.copy()
        damaged[0, 0, 0, 0] += .01
        assert fingerprint(damaged) != proof["sha256"]


def test_cli_defaults_to_canonical_list_and_exposes_retained_named_case(inventory, capsys):
    assert selection.main(["--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 97
    assert selection.main(["cases", "--id", "pr58", "--json"]) == 0
    case = json.loads(capsys.readouterr().out)[0]
    assert case["canonical_id"] == "develop-img_intensity2" and case["original_status"] == "FAIL / PASS"
    with pytest.raises(SystemExit) as exit_info:
        selection.main(["cases", "--id", "unknown"])
    assert exit_info.value.code == 2


def test_direct_script_metadata_cli_listing_does_not_import_training_stack():
    # Benchmark package initialization imports Torch for serial scheduling.
    # Execute the metadata script directly to isolate its actual CLI contract.
    script = ("import runpy, sys; "
              "s = runpy.run_path('benchmarks/toy_audit/canonical_selection.py'); "
              "s['main'](['cases','--id','pr58','--json']); assert 'torch' not in sys.modules")
    completed = subprocess.run([sys.executable, "-c", script], cwd=selection.ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    case, = json.loads(completed.stdout)
    assert case['catalog_id'] == 'pr58'
    assert case['canonical_id'] == 'develop-img_intensity2'
    assert case['original_status'] == 'FAIL / PASS'


def test_source_drift_blocks_selection_even_with_unchanged_catalog(inventory, tmp_path):
    manifest, _ = inventory
    names = [manifest["catalog"]["path"], *manifest["law_source_sha256"]]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selection.ROOT / name, path)
    path = tmp_path / "benchmarks/transfer_suite/image_tasks.py"
    path.write_bytes(path.read_bytes() + b"\n# changed source\n")
    with pytest.raises(ValueError, match="law source changed"):
        selection.load(tmp_path)
