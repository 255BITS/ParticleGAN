"""Explicit tied-starter selection preserves the scientific search objective."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bcap_optimizer_selection", Path(__file__).resolve().parents[1]
    / "reports/forge/dualnorm-tier1/select_measurements.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def trial(config, passes=3):
    return {
        "candidate_id": "bcap-dualnorm--" + config, "configuration_id": config,
        "trainer_family": "bcap-dualnorm", "submission_status": "completed",
        "resolved_recipe": {"optimizer_family": "dualnorm", "optimizer_momentum": 0.,
                            "lr": .01, "d_lr_mult": 1.5, "prior_lr_mult": 3.},
        "tasks": [{"task": str(index), "qualification_tier": 1,
                   "importance": "required" if index < 6 else "diagnostic",
                   "gate_status": "PASS" if index < passes or index == 6 else "FAIL",
                   "compatibility_key": str(index) + config}
                  for index in range(7)],
    }


def test_explicit_tied_starter_does_not_mutate_or_rerank_original_search():
    trials = [trial("b"), trial("a")]
    saved = deepcopy(trials)
    assert selection.select_configuration(trials, 1)["selected_configuration_id"] == "a"
    selected = selection.explicit_starter_trial(trials, "bcap-dualnorm--b")
    assert selected["configuration_id"] == "b"
    assert selected["tasks"] == saved[0]["tasks"]
    assert trials == saved
    assert selection.select_configuration(trials, 1)["selected_configuration_id"] == "a"
    assert not selection.select_configuration(trials, 1)["qualified"]


def test_starter_cannot_trade_a_lower_whole_recipe_score_for_one_task_metric():
    trials = [trial("a", 3), trial("b", 2)]
    with pytest.raises(ValueError, match="tying the best required PASS count"):
        selection.explicit_starter_trial(trials, "bcap-dualnorm--b")


@pytest.mark.parametrize("status", ["UNKNOWN", "INCOMPLETE", "INVALID"])
def test_starter_requires_every_current_tier_cell_to_be_measured(status):
    trials = [trial("a"), trial("b")]
    trials[1]["tasks"][-1]["gate_status"] = status
    with pytest.raises(ValueError, match="fully measured whole recipe"):
        selection.explicit_starter_trial(trials, "bcap-dualnorm--b")


@pytest.mark.parametrize("field,value", [("optimizer_momentum", .5), ("lr", .03),
                                         ("d_lr_mult", 2.), ("prior_lr_mult", 2.)])
def test_starter_binds_the_requested_recipe_instead_of_reusing_another_rate(field, value):
    trials = [trial("a"), trial("b")]
    trials[1]["resolved_recipe"][field] = value
    with pytest.raises(ValueError, match="requested mu=0"):
        selection.explicit_starter_trial(trials, "bcap-dualnorm--b")


def test_starter_requires_an_actual_recorded_configuration():
    with pytest.raises(ValueError, match="one recorded dualnorm configuration"):
        selection.explicit_starter_trial([trial("a")], "missing")
