"""Current questions retain initial conditions and gates after ownership moves."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "03466efa4de8271b9a2c964406dc6f4f0f260792"


def test_current_prior_questions_change_only_declared_ownership_and_source_bindings():
    migrated = []
    for path in sorted((ROOT / "configs/forge/tasks").glob("*.json")):
        current = json.loads(path.read_text())
        if current["execution"].get("prior_contract") != "recipe_owned_v1":
            continue
        relative = path.relative_to(ROOT).as_posix()
        original = json.loads(subprocess.check_output(
            ["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT))
        expected = deepcopy(original)
        assert expected["execution"]["prior"].pop("learnable") is True
        expected["execution"]["prior_contract"] = "recipe_owned_v1"
        expected["requires_capabilities"] = [name for name in expected["requires_capabilities"]
                                              if name != "learned_locations"]
        observed = deepcopy(current)
        # Source rebinding authorizes current software only. Every architecture,
        # target, initial prior, sampling law, budget, cadence and bound stays.
        for card in (expected, observed):
            card["evaluation"].pop("sources", None)
            card["evaluation"].pop("evaluator_revision", None)
        assert observed == expected, current["id"]
        migrated.append(current["id"])
    assert len(migrated) == 33

