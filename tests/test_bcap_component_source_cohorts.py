"""Exact two-update component regression from a1e8073a to the current source.

The original 5737ade4 comparator and historical incompatibilities stay separate.
This runnable software cohort uses its implementation-neutral probe, original
task/data/prior/initializer contracts and protocol seed 0, with an explicit
two-update budget. Every model, optimizer and consumed named stream remains
bitwise checked. These controls make no trained quality or qualification claim.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from conftest import _archived_inactive_comparison_scope
from tests.archived_forge_contracts import bcap_pinned_software_checkout

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "a1e8073ae8352d3aff25f04bb51446454e873d90"
PROBE = ROOT / "tests/test_bcap_integration_compatibility.py"
spec = importlib.util.spec_from_file_location("bcap_original_component_probe", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(scope="module")
def source_cohort(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bcap-component-source-cohort")
    with _archived_inactive_comparison_scope(probe, directory):
        reference = bcap_pinned_software_checkout(directory / "reference", REFERENCE)
        outputs = []
        env = {**os.environ, "PYTHONPATH": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for label, source in (("a1e8073a", reference), ("current", probe.ROOT)):
            output = directory / f"{label}.pt"
            with (directory / f"{label}.log").open("w") as stdout:
                result = subprocess.run([sys.executable, probe.__file__, str(source), str(output)],
                    cwd=source, env=env, stdout=stdout, stderr=subprocess.STDOUT, timeout=180)
            assert result.returncode == 0, (directory / f"{label}.log").read_text()
            outputs.append(torch.load(output, weights_only=False))
        (directory / "comparison-scope.json").write_text(json.dumps({
            "reference_source_commit": REFERENCE, "question_source_commit": probe.DEVELOP,
            "protocol_seed": 0, "updates": 2, "qualification_input": False,
            "comparison": "complete component model/optimizer/named-stream state; exact equality",
        }, indent=2) + "\n")
    return tuple(outputs)


def _component(packet, key):
    state = deepcopy(packet["components"][key])
    # Source-binding locations differ by the declared code cohort. Numerical
    # task conditions and every tensor/state entry remain in the comparison.
    evaluation = state["applied"]["field_ownership"]["task_contract"]["evaluation"]["value"]
    evaluation.pop("sources", None)
    evaluation.pop("evaluator_revision", None)
    return state


@pytest.mark.parametrize("case,host", probe.COMPONENT_CASES)
def test_complete_component_state_is_bitwise_a1e8073a(source_cohort, case, host):
    old, new = source_cohort
    key = f"{case}/{host}"
    probe._assert_equal(_component(old, key), _component(new, key), key)


def test_direct_response_counter_matches_its_completed_update_budget(source_cohort):
    for packet in source_cohort:
        state = packet["components"]["ka2/two_pole"]
        budget = state["applied"]["field_ownership"]["task_contract"]["budget"]["value"]["steps"]
        optimizer = state["optimizers"]["generator"][0]
        counter = optimizer["regularizer"]["direct"]["state"]["step_count"]
        assert budget == counter == 2
        assert all(int(moment["step"]) == budget for moment in optimizer["state"].values())


def test_component_comparator_still_rejects_changed_model_tensors(source_cohort):
    state = _component(source_cohort[1], "ka2/ae_gan_hold")
    changed = deepcopy(state)
    tensor = next(value for model in changed["models"].values()
                  for value in model.values() if value.is_floating_point())
    tensor.add_(1)
    with pytest.raises(AssertionError, match="models"):
        probe._assert_equal(state, changed)
