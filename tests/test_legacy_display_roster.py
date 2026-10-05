"""Exact prior declaration display survives refresh without granting evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from test_forge_current_technique_inventory import evidence, publication, _outputs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("legacy_roster", ROOT / "reports/forge/legacy_display_roster.py")
roster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roster)


@pytest.fixture
def legacy(evidence):
    root, _ = evidence
    current = read_json(publication.publish_current(root)["json"])
    row = deepcopy(current["configuration_rows"][0])
    row.update(candidate_id="legacy-proposed", candidate_revision="legacy-revision", cohort="legacy-cohort",
               attempt_ids=[], status="INCOMPLETE", qualified_tier=0, selected_configuration=False,
               alternative_scope="archived_alternative", qualification_input=False, qualification_reuse=False,
               cost={"wall_seconds": None, "measured_tasks": 0})
    for tier in row["tiers"].values():
        tier["passed"] = 0
    for task in row["tasks"]:
        task["status"] = "UNKNOWN"
    current["configuration_rows"].append(row)
    current["provenance"].pop("input_digest")
    current["provenance"]["input_digest"] = stable_hash(current)
    atomic_json(root / roster.PUBLICATION, current)
    shutil.copyfile(ROOT / "reports/forge/regenerate_technique_inventory.py", root / "reports/forge/regenerate_technique_inventory.py")
    for arguments in (("init", "--quiet"), ("config", "user.name", "Publication Fixture"),
                      ("config", "user.email", "fixture@example.invalid"), ("add", "."),
                      ("commit", "--quiet", "-m", "Exact prior unmeasured declaration display")):
        subprocess.run(["git", *arguments], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    registered = roster.register(root, commit)
    assert len(registered["scientific_row_sha256"]) == 1
    return root, current, row


@pytest.mark.parametrize("objects_available", [True, False], ids=["git-history", "shallow-no-legacy-object"])
def test_exact_legacy_display_refresh_preserves_every_row_without_measurement(legacy, objects_available):
    root, before, _ = legacy
    if not objects_available:
        shutil.rmtree(root / ".git")
    protected = {path: path.read_bytes() for path in (root / publication.EVIDENCE_MANIFEST.parent).glob("*.json")}
    result = read_json(publication.refresh_publication(root)["json"])
    for key in ("rows", "configuration_rows", "evidence_rows", *publication.POLICY_FIELDS):
        assert result.get(key) == before.get(key)
    assert all(path.read_bytes() == contents for path, contents in protected.items())
    assert not (root / "reports/forge/attempts").exists()


@pytest.mark.parametrize("tamper", ["recipe", "pass", "attempt", "paid", "selected", "qualification", "selected-row", "evidence-row"])
def test_roster_cannot_admit_changed_science_or_promote_a_declaration(legacy, tamper):
    root, current, _ = legacy
    row = current["configuration_rows"][-1]
    if tamper == "recipe":
        row["bindings"]["recipe_sha256"] = "forged"
    elif tamper == "pass":
        row["tasks"][0]["status"] = "PASS"
    elif tamper == "attempt":
        row["attempt_ids"] = ["new-attempt"]
    elif tamper == "paid":
        row["cost"]["wall_seconds"] = 1
    elif tamper == "selected":
        row["selected_configuration"] = True
    elif tamper == "qualification":
        row["qualification_input"] = True
    elif tamper == "selected-row":
        current["rows"].append(deepcopy(row))
    else:
        current["evidence_rows"].append(deepcopy(row))
    current["provenance"].pop("input_digest")
    current["provenance"]["input_digest"] = stable_hash(current)
    atomic_json(root / roster.PUBLICATION, current)
    before = _outputs(root)
    with pytest.raises(ValueError, match="unregistered scientific row"):
        publication.refresh_publication(root)
    assert _outputs(root) == before


@pytest.mark.parametrize("tamper", ["git_blob", "row_hash", "policy"])
def test_roster_provenance_and_allowed_row_identity_are_exact(legacy, tamper):
    root, _, _ = legacy
    registry = read_json(root / roster.REGISTRY)
    if tamper == "git_blob":
        registry["source_publication"]["git_blob"] = "f" * 40
    elif tamper == "row_hash":
        registry["scientific_row_sha256"] = ["f" * 64]
    else:
        registry["policy"]["view_revision"] += 1
    atomic_json(root / roster.REGISTRY, registry)
    before = _outputs(root)
    with pytest.raises(ValueError, match="identity mismatch|prior scientific rows|cross qualification"):
        publication.refresh_publication(root)
    assert _outputs(root) == before


def test_exact_unselected_snapshot_declaration_needs_no_legacy_receipt(legacy):
    root, current, row = legacy
    manifest = read_json(root / publication.EVIDENCE_MANIFEST)
    entry = manifest["cohorts"][0]
    path = root / entry["snapshot"]
    snapshot = read_json(path)
    snapshot["rows"].append(deepcopy(row))
    snapshot["provenance"].pop("input_digest")
    snapshot["provenance"]["input_digest"] = stable_hash(snapshot)
    atomic_json(path, snapshot)
    entry["json_sha256"] = file_hash(path)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    current["provenance"]["evidence_manifest_sha256"] = stable_hash(manifest)
    current["provenance"].pop("input_digest")
    current["provenance"]["input_digest"] = stable_hash(current)
    atomic_json(root / roster.PUBLICATION, current)
    (root / roster.REGISTRY).unlink()
    refreshed = read_json(publication.refresh_publication(root)["json"])
    assert refreshed["rows"] == current["rows"]
    assert refreshed["evidence_rows"] == current["evidence_rows"]
    assert refreshed["configuration_rows"] == current["configuration_rows"]
