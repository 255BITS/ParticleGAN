"""Named reforms cannot shrink a family's required scientific denominator."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.forge import contracts, named_policy_planning as planning, views
from experiments.forge import conditional_policy_contracts as conditional
from experiments.forge import routed_policy_contracts as routed
from experiments.forge import multibank_policy_contracts as multibank
from experiments.forge import ae_routed_policy_contracts as ae
from experiments.forge import word_joint_policy_contracts as word
from experiments.forge.policy_contracts import PARENT_TASK_IDS


ROOT = Path(__file__).resolve().parents[1]


def declared(cohort):
    if cohort == conditional.COHORT:
        return [conditional.make_conditional_variant(ROOT, host) for host in conditional.HOSTS]
    return [{routed.COHORT: routed.make_unused_variant,
             multibank.COHORT: multibank.make_variant,
             ae.COHORT: ae.make_ae_variant,
             word.COHORT: word.make_variant}[cohort](ROOT)]


def inputs(cohort):
    parents = {name: contracts.read_json(ROOT / "configs/forge/tasks" / f"{name}.json")
               for name in PARENT_TASK_IDS}
    tasks = {**parents, **{task["id"]: task for task in declared(cohort)}}
    return contracts.read_json(ROOT / "configs/forge/views/discriminator_stability.json"), tasks


@pytest.mark.parametrize("cohort", tuple(planning.NAMED_PARENTS))
def test_real_variants_keep_every_original_required_slot_and_do_not_change_inputs(cohort):
    view, tasks = inputs(cohort)
    before = deepcopy((view, tasks))
    resolved, selected = planning.resolve_task_view(view, tasks, {"task_cohort": cohort})
    assert (view, tasks) == before
    assert len(selected) == 26
    assert [sum(a["qualification_tier"] == tier for a in resolved["assignments"])
            for tier in (1, 2, 3)] == [5, 19, 2]
    assert resolved["parent_view_fingerprint"] == views.view_fingerprint(view)
    for original, actual in zip(view["assignments"], resolved["assignments"]):
        parent = original["task"]
        expected = parent + "_" + cohort if parent in planning.NAMED_PARENTS[cohort] else parent
        assert actual == {**original, "task": expected}
        if parent not in planning.NAMED_PARENTS[cohort]:
            assert selected[parent] == tasks[parent]
    # A measured adapted subset does not qualify an incomplete whole family.
    quality = views.qualify(resolved, selected, [])
    assert quality["qualified_tier"] == 0
    assert len(quality["tasks"]) == 26


@pytest.mark.parametrize("field,value", [
    ("importance", "optional"), ("qualification_tier", 2), ("order", 4),
])
def test_named_resolution_rejects_denominator_tier_or_order_changes(field, value):
    view, tasks = inputs(routed.COHORT)
    view["assignments"][0][field] = value
    with pytest.raises(ValueError, match="unchanged main"):
        planning.resolve_task_view(view, tasks, {"task_cohort": routed.COHORT})


def test_named_resolution_rejects_missing_unadapted_questions():
    view, tasks = inputs(routed.COHORT)
    tasks.pop("ring_extension")
    with pytest.raises(ValueError, match="full parent/variant"):
        planning.resolve_task_view(view, tasks, {"task_cohort": routed.COHORT})


def test_unknown_policy_cohort_does_not_load_files(tmp_path):
    with pytest.raises(ValueError, match="unknown explicit"):
        planning.load_task_variants(tmp_path, {}, "pretend_atlas")


def test_explicit_named_cohort_is_an_identity_not_a_family_label():
    idea = contracts.read_json(ROOT / "configs/forge/ideas/atlas-c6-observed-policy-current-v1.json")
    for cohort in planning.NAMED_PARENTS:
        contracts.validate_idea({**idea, "task_cohort": cohort})
    with pytest.raises(ValueError, match="task_cohort"):
        contracts.validate_idea({**idea, "task_cohort": "atlas_conditional"})
