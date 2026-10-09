"""Destructive controls for the explicit archival software fixture boundary."""
import json

import pytest

from tests import archived_forge_contracts as archived


@pytest.mark.parametrize("scope", sorted(archived.COMMITS))
def test_all_archived_contract_bytes_retain_their_sha_and_git_blob(scope):
    packet = json.loads(archived.FIXTURE.read_text())
    for relative in packet[scope]["files"]:
        assert archived.archived_contract_bytes(scope, relative)


@pytest.mark.parametrize("change", ["bytes", "git_blob", "source_commit"])
def test_archived_fixture_mutation_is_rejected_before_restore(tmp_path, monkeypatch, change):
    packet = json.loads(archived.FIXTURE.read_text())
    cohort = packet["published_develop"]
    relative = "configs/forge/tasks/two_pole.json"
    if change == "bytes":
        cohort["files"][relative]["text"] += " "
    elif change == "git_blob":
        cohort["files"][relative]["git_blob"] = "0" * 40
    else:
        cohort["source_commit"] = "0" * 40
    fixture = tmp_path / "changed-fixture.json"
    fixture.write_text(json.dumps(packet))
    monkeypatch.setattr(archived, "FIXTURE", fixture)
    with pytest.raises(AssertionError):
        archived.archived_contract_bytes("published_develop", relative)


def test_archive_restore_refuses_live_repository():
    with pytest.raises(AssertionError):
        archived.restore_archived_contracts(archived.ROOT, "published_develop")


def test_unchanged_archive_input_drift_cannot_rebind_its_byte_pins(tmp_path, monkeypatch):
    packet = json.loads(archived.FIXTURE.read_text())
    relative = next(path for path, record in packet["tier1_completion"]["files"].items()
                    if record.get("repository_bytes") is True)
    data = archived.archived_contract_bytes("tier1_completion", relative)
    changed = tmp_path / relative
    changed.parent.mkdir(parents=True)
    changed.write_bytes(data + b" ")
    monkeypatch.setattr(archived, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        archived.archived_contract_bytes("tier1_completion", relative)
