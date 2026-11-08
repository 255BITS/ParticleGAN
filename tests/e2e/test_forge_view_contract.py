"""Forge N-tier view policy, exercised through the production validator.

Each case starts from the live ``discriminator_stability`` view and the real
task catalog, applies one defect, and requires
``experiments.forge.views.validate_view`` to refuse it with the specific
reason. The unmodified view and an explicit Tier 5 extension are the positive
controls. These replace the earlier tier 1-4 e2e files, which validated a
test-local mirror of the contract and so could not fail.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from experiments.forge.tier_report import tier_name
from experiments.forge.views import load_tasks, load_view, validate_view

VIEW = "discriminator_stability"
SMOKE, RAIL = "zero_tanh_smoke", "zero_tanh_rail_escape"
TIER4_HELD = ("grid100_mlp16", "grid100_res16_tanh", "misgan_block_recovery",
              "misgan_block_neural_imputer", "ring16_bigan_continuous")
HOLD = {
    "schema": "forge_hold_v1",
    "previous_importance": "required",
    "reason": "unsolved",
    "blocked_by": [],
    "summary": "Held while no recorded recipe passes this declaration.",
    "explanation": "reports/forge/problems.md#held",
    "reentry": "Re-enter when a global recipe passes the full declaration.",
}


@pytest.fixture(scope="module")
def catalog(repo_root):
    return load_tasks(repo_root), load_view(repo_root, VIEW)


@pytest.fixture
def live(catalog):
    tasks, view = catalog
    return deepcopy(tasks), deepcopy(view)


def _assignment(view, name):
    return next(a for a in view["assignments"] if a["task"] == name)


def test_live_view_validates_and_declares_the_ntier_target_tasks(live):
    tasks, view = live
    validate_view(view, tasks)
    smoke, rail = _assignment(view, SMOKE), _assignment(view, RAIL)
    assert (smoke["qualification_tier"], smoke["importance"]) == (1, "required")
    assert (rail["qualification_tier"], rail["importance"], rail["hold"]["reason"]) == (3, "held", "unsolved")
    assert tasks[RAIL]["dependencies"] == [{"task": SMOKE, "kind": "gate"}]
    for name in TIER4_HELD:
        assignment = _assignment(view, name)
        assert (assignment["qualification_tier"], assignment["importance"]) == (4, "held"), name
    for name in ("grid100_mlp16", "grid100_res16_tanh"):
        assert tasks[name]["dependencies"] == [{"task": "grid100", "kind": "gate"}]
    highest = max(a["qualification_tier"] for a in view["assignments"])
    assert (highest, tier_name(highest)) == (4, "stress_and_formulations")


def test_tier_five_extension_validates_and_uses_the_dynamic_tier_name(live):
    tasks, view = live
    extra = deepcopy(tasks[SMOKE])
    extra["id"] = "zero_tanh_smoke_tier5_probe"
    tasks[extra["id"]] = extra
    view["assignments"].append({"task": extra["id"], "qualification_tier": 5, "importance": "required", "order": 0})
    validate_view(view, tasks)
    assert [tier_name(n) for n in (1, 2, 3, 4, 5, 12)] == [
        "smoke", "quality", "endurance", "stress_and_formulations", "tier-5", "tier-12"]


@pytest.mark.parametrize("tier", [0, -1, 1.5, "1", None, True])
def test_rejects_invalid_qualification_tier(live, tier):
    tasks, view = live
    _assignment(view, SMOKE)["qualification_tier"] = tier
    with pytest.raises(ValueError, match="qualification_tier must be an integer >= 1"):
        validate_view(view, tasks)


def _optional(view):
    _assignment(view, SMOKE)["importance"] = "optional"


def _held_without_metadata(view):
    del _assignment(view, RAIL)["hold"]


def _unknown_hold_reason(view):
    _assignment(view, RAIL)["hold"]["reason"] = "invalid_random_reason"


def _hold_on_required(view):
    _assignment(view, SMOKE)["hold"] = deepcopy(HOLD)


def _duplicate(view):
    view["assignments"].append(deepcopy(_assignment(view, SMOKE)))


def _unknown_task(view):
    view["assignments"].append({"task": "no_such_task", "qualification_tier": 1, "importance": "required", "order": 0})


def _bad_revision(view):
    view["revision"] = 0


def _string_revision(view):
    view["revision"] = "14"


def _empty(view):
    view["assignments"] = []


def _tier_gap(view):
    # Every Tier 3 assignment moves to Tier 4: counted tiers become {1, 2, 4}.
    for assignment in view["assignments"]:
        if assignment["qualification_tier"] == 3:
            assignment["qualification_tier"] = 4


def _dependency_in_later_tier(view):
    _assignment(view, SMOKE)["qualification_tier"] = 4


def _missing_dependency(view):
    view["assignments"].remove(_assignment(view, SMOKE))


def _held_prerequisite_not_listed(view):
    smoke = _assignment(view, SMOKE)
    smoke.update(importance="held", hold=deepcopy(HOLD))


@pytest.mark.parametrize("mutate,match", [
    (_optional, "unknown task importance"),
    (_held_without_metadata, "held assignments need hold metadata"),
    (_unknown_hold_reason, "hold reason must be one of"),
    (_hold_on_required, "hold metadata requires importance held"),
    (_duplicate, f"duplicate assignment {SMOKE}"),
    (_unknown_task, "missing task reference 'no_such_task'"),
    (_bad_revision, "view revision must be an integer >= 1"),
    (_string_revision, "view revision must be an integer >= 1"),
    (_empty, "view needs nonempty assignments"),
    (_tier_gap, "qualification tiers must be contiguous from Tier 1"),
    (_dependency_in_later_tier, f"{RAIL}: dependency {SMOKE} is in a later tier"),
    (_missing_dependency, f"dependency {SMOKE} missing from view"),
    (_held_prerequisite_not_listed, f"{RAIL}: hold blocked_by must list exactly its held prerequisites"),
], ids=lambda value: getattr(value, "__name__", "")[1:] or None)
def test_rejects_view_defect(live, mutate, match):
    tasks, view = live
    mutate(view)
    with pytest.raises(ValueError, match=match):
        validate_view(view, tasks)


def test_rejects_gate_dependency_cycle(live):
    tasks, view = live
    rail = _assignment(view, RAIL)
    del rail["hold"]
    rail.update(importance="required", qualification_tier=1)
    tasks[SMOKE]["dependencies"] = [{"task": RAIL, "kind": "gate"}]
    with pytest.raises(ValueError, match="dependency cycle involving"):
        validate_view(view, tasks)


def _tier_at_task_root(task):
    task["qualification_tier"] = 1


def _soft_dependency(task):
    task["dependencies"] = [{"task": SMOKE, "kind": "soft"}]


def _repeated_dependency(task):
    task["dependencies"] = [{"task": SMOKE, "kind": "gate"}] * 2


def _missing_capabilities(task):
    del task["requires_capabilities"]


@pytest.mark.parametrize("mutate,match", [
    (_tier_at_task_root, "qualification_tier belongs only in view assignments"),
    (_soft_dependency, "dependency needs task and gate/checkpoint kind"),
    (_repeated_dependency, "duplicate dependency"),
    (_missing_capabilities, r"task lacks \['requires_capabilities'\]"),
], ids=lambda value: getattr(value, "__name__", "")[1:] or None)
def test_rejects_task_declaration_defect(live, mutate, match):
    tasks, view = live
    mutate(tasks[RAIL])
    with pytest.raises(ValueError, match=match):
        validate_view(view, tasks)


def test_every_previous_revision_names_an_existing_matching_archive(repo_root):
    checked = 0
    for path in sorted((repo_root / "configs/forge/views").glob("*.json")):
        previous = json.loads(path.read_text(encoding="utf-8")).get("previous_revision")
        if previous is None:
            continue
        archive = repo_root / previous["archive"]
        assert archive.is_file(), f"{path.name}: missing {previous['archive']}"
        archived = json.loads(archive.read_text(encoding="utf-8"))
        assert (archived["id"], archived["revision"]) == (path.stem, previous["revision"]), path.name
        checked += 1
    assert checked >= 1
