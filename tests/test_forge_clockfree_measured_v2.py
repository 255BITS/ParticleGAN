"""Proposed measured-parity clock audit (v2): measurement decides, declarations pre-screen.

The v1 task and grader stay unchanged; v2 applies only to tasks declaring
evaluation.audit_rule == "measured_parity_v2".
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.clockfree import (PROGRESS_COUNTER_ENTRY, power_witness, run_clockfree,
                                         v2_declarations)
from experiments.forge.views import grade_result
from particlegan import Recipe


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs/forge/tasks"
GUARD0 = {"d_guard_min_steps": 0}


@pytest.fixture
def tasks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    v1 = json.loads((TASKS / "clockfree_audit_measurement_v1.json").read_text())
    v2 = json.loads((TASKS / "clockfree_audit_measured_v2-clockdig.json").read_text())
    yield v1, v2
    torch.set_num_threads(previous)


def declared(task):
    strict = deepcopy(task)
    strict["evaluation"]["internal_counter_policy"] = "declared"
    return strict


def probe(task, tmp_path, overrides, formulation="k3p"):
    request = {"candidate": {"recipe_overrides": {"critic_formulation": formulation, **overrides}},
               "protocol": {"seed": 0}}
    return run_clockfree(request, task, tmp_path, "cpu")


def test_v2_task_is_a_separate_revision_with_the_same_probe(tasks):
    v1, v2 = tasks
    assert v2["id"] != v1["id"] and v2["execution"] == v1["execution"]
    assert {k: v for k, v in v2["evaluation"].items()
            if k not in {"audit_rule", "internal_counter_policy", "rule_status", "previous_task"}} == v1["evaluation"]
    assert v2["evaluation"]["rule_status"] == "proposed_requires_sign_off"


def test_internal_counter_anneal_fails_v1_by_declaration_and_passes_v2_by_measurement(tasks, tmp_path):
    v1, v2 = tasks
    raw = probe(v1, tmp_path / "a", {"progress_schedule": "updates", "progress_rate": 0.01, **GUARD0})
    assert all(row["reference_sha256"] == row["changed_sha256"] for row in raw["evidence"]["comparisons"])
    assert grade_result(v1, raw)["status"] == "FAIL"
    grade = grade_result(v2, raw)
    assert grade["status"] == "PASS", grade
    assert grade["metrics"]["overridden_entries"] == 4 and grade["metrics"]["standing_entries"] == 0
    strict = grade_result(declared(v2), raw)
    assert strict["status"] == "FAIL" and PROGRESS_COUNTER_ENTRY in strict["reasons"][0]


def test_state_driven_progress_passes_both_counter_policies(tasks, tmp_path):
    _, v2 = tasks
    raw = probe(v2, tmp_path, {"progress_schedule": "critic_order", **GUARD0})
    assert grade_result(v2, raw)["status"] == "PASS"
    assert grade_result(declared(v2), raw)["status"] == "PASS"


def test_step_scheduled_recipe_still_fails_by_measurement(tasks, tmp_path):
    _, v2 = tasks
    grade = grade_result(v2, probe(v2, tmp_path, GUARD0))
    assert grade["status"] == "FAIL" and "step_label changed" in grade["reasons"][0]
    assert grade["metrics"]["overridden_entries"] == 0


def test_fixed_count_guard_is_a_counter_policy_choice(tasks, tmp_path):
    _, v2 = tasks
    raw = probe(v2, tmp_path, {"lr_floor": 1, "network_lr_floor": 1, "input_noise_std": 0,
                               "output_noise_warmup": 0, "d_guard_min_steps": 200})
    assert grade_result(v2, raw)["status"] == "PASS"
    assert grade_result(declared(v2), raw)["status"] == "FAIL"


def test_ka2_call_counter_is_witnessed_from_saved_state(tasks, tmp_path):
    _, v2 = tasks
    raw = probe(v2, tmp_path, {"progress_schedule": "updates", **GUARD0}, formulation="ka2")
    grade = grade_result(v2, raw)
    assert grade["status"] == "PASS", grade
    assert grade["metrics"]["declared_entries"] == 5


def test_unpowered_probe_keeps_the_step_schedule_declaration():
    recipe = Recipe(critic_formulation="k3p", total_steps=128).to_dict()
    weak = {"warmup_steps": 4, "probe_steps": 4, "step_label_offset": 1, "perturbed_horizon": 129}
    strong = {"warmup_steps": 4, "probe_steps": 4, "step_label_offset": 8192, "perturbed_horizon": 4096}
    lr_only = Recipe(critic_formulation="k3p", total_steps=128, input_noise_std=0,
                     output_noise_warmup=0).to_dict()
    assert power_witness(lr_only, weak)["learning_rate"] is False
    assert power_witness(recipe, strong) == {"learning_rate": True, "input_noise": True, "output_noise": True}


def test_extensions_and_lazy_penalty_never_override():
    recipe = Recipe(critic_formulation="k3p", lr_floor=1, network_lr_floor=1, input_noise_std=0,
                    output_noise_warmup=0, d_guard_min_steps=0, reg_every=2).to_dict()
    entries = v2_declarations(recipe, {"unreviewed": True})
    assert len(entries) == 2
    from experiments.forge.clockfree import COUNTER_ENTRIES, STEP_SCHEDULE_ENTRIES
    assert not any(entry in COUNTER_ENTRIES or entry in STEP_SCHEDULE_ENTRIES for entry in entries)


def test_unknown_rule_is_invalid(tasks, tmp_path):
    _, v2 = tasks
    raw = probe(v2, tmp_path, {"progress_schedule": "critic_order", **GUARD0})
    bad = deepcopy(v2)
    bad["evaluation"]["audit_rule"] = "anything_goes"
    assert grade_result(bad, raw)["status"] == "INVALID"
