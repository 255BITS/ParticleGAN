"""Original v1 case cards for archived coordinator software tests only."""
from copy import deepcopy

from benchmarks.toy_audit import api_contract, api_run

CURRENT_DISCOVERY = api_contract.discover


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
