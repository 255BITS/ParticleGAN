"""Original v1 case cards for archived coordinator software tests only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from benchmarks.toy_audit import api_contract, api_run

CURRENT_DISCOVERY = api_contract.discover
DISCOVERY_FIXTURES = Path(__file__).resolve().parent / "fixtures/archived-discovery"


def archived_discovery_bytes(relative, expected_sha):
    """Byte-exact historical data; never replace an archived runner's pins."""
    receipt = json.loads((DISCOVERY_FIXTURES / "provenance.json").read_text())
    provenance = receipt["files"][relative]
    data = (DISCOVERY_FIXTURES / provenance["fixture"]).read_bytes()
    assert provenance["sha256"] == expected_sha == hashlib.sha256(data).hexdigest()
    # A Git blob identity includes the original byte count in its header.
    header = b"blob " + str(len(data)).encode() + b"\0"
    assert hashlib.sha1(header + data).hexdigest() == provenance["git_blob"]
    return data


def archived_runner_source(runner, tmp_path, monkeypatch):
    """Expose original discovery data in a scoped software source fixture.

    Runner files remain linked to their current, strictly hashed bytes. Only
    the pinned discovery inputs use their original archived data. The live
    checkout and all production hash checks remain unchanged.
    """
    original = runner.ROOT
    copied = tmp_path / "archived-runner-inputs"
    for relative in {runner.RELATIVE, runner.CAPACITY_RELATIVE,
                     getattr(runner, "DELEGATED_RELATIVE", runner.RELATIVE)}:
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(original / relative)
    for relative, expected_sha in runner.DISCOVERY_INPUTS.items():
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archived_discovery_bytes(relative, expected_sha))
    monkeypatch.setattr(runner, "ROOT", copied)
    return copied


def archived_cases(names, monkeypatch):
    found = CURRENT_DISCOVERY()
    cards = {name: found[name] for name in names}
    for case in cards.values():
        case.pop("protocol_seed", None)
        case.pop("comparison_version", None)
        if case["provider"] == "api_vectors":
            case.pop("initialization", None)
    # Archived coordinators verify their original hard-coded case hashes.
    # Their parser tests retain the old seed without modifying live defaults.
    monkeypatch.setattr(api_contract, "discover", lambda: deepcopy(cards))
    monkeypatch.setattr(api_run, "DEFAULT_SEED", 24002)
    return cards
