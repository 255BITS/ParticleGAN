"""Every field except two independently counted eval counters stays exact."""

from copy import deepcopy
import gzip
from io import BytesIO
from pathlib import Path

import pytest
import torch

from reports.toy100.allocation_snapshot_reconciliation import expected_difference, reconcile


def states():
    hold = dict(generator={"weight": torch.tensor([1., 2.])}, optimizer_g={"step": 1200},
                rng={"global": torch.tensor([1, 2, 3])}, noise={"step_calls": 1200},
                snapshot_scope={"host_loop_step": 1199},
                noise_policy=dict(total_steps=1200, _step_calls=1200, history=[.01, .029],
                    _counts=dict(output_eval_calls=480, output_eval_elements=1971840, input_train_calls=7200)))
    warm = deepcopy(hold)
    warm["noise_policy"]["_counts"].update(output_eval_calls=510, output_eval_elements=2095080)
    return warm, hold


def test_original_observation_schedule_plus_native_tail_predict_exact_difference():
    row = expected_difference()
    assert len(row["extra_native_observation_steps"]) == 12
    assert row["extra_native_live_curve_steps"] == [1050, 1100, 1150]
    assert row["counter_deltas"] == dict(output_eval_calls=30, output_eval_elements=123240)
    warm, hold = states()
    before = deepcopy(warm)
    receipt = reconcile(warm, hold)
    assert receipt["training_state_parity"] is True
    assert receipt["warm_raw_sha256"] != receipt["hold_raw_sha256"]
    assert receipt["adjusted_warm_sha256"] == receipt["hold_raw_sha256"]
    assert warm["noise_policy"] == before["noise_policy"]


@pytest.mark.parametrize("field", ["model", "moment", "rng", "clock", "history", "input_count", "output_count", "extra_field"])
def test_no_other_state_difference_or_counter_delta_is_accepted(field):
    warm, hold = states()
    if field == "model": warm["generator"]["weight"][0] += 1
    elif field == "moment": warm["optimizer_g"]["step"] += 1
    elif field == "rng": warm["rng"]["global"][0] += 1
    elif field == "clock": warm["noise_policy"]["total_steps"] = 2400
    elif field == "history": warm["noise_policy"]["history"][0] += .01
    elif field == "input_count": warm["noise_policy"]["_counts"]["input_train_calls"] += 1
    elif field == "output_count": warm["noise_policy"]["_counts"]["output_eval_calls"] += 1
    else: warm["unreviewed_metadata"] = True
    with pytest.raises(RuntimeError):
        reconcile(warm, hold)


def test_exact_captured_experiment_diff_is_only_the_independently_enumerated_counters():
    directory = Path(__file__).resolve().parents[1] / "reports/toy100/continuous-evidence/anchor-prefix-hash-diagnosis"
    load = lambda horizon: torch.load(BytesIO(gzip.decompress(
        (directory / f"snapshot-horizon{horizon}.pt.gz").read_bytes())), weights_only=True)
    receipt = reconcile(load(1200), load(2400))
    assert receipt["warm_raw_sha256"] == "99a21069ff1b9869e186b003f35430c26943247cad0dca981178105a7276c546"
    assert receipt["hold_raw_sha256"] == "d9c0c76f144373d6cd728e3c4bbb00a42902c72782eb89b952bcb9fe6015328b"
    assert receipt["adjusted_warm_sha256"] == receipt["hold_raw_sha256"]


def test_v2_cold_gate_requires_named_training_parity_and_explicit_reconciliation():
    from reports.toy100.allocation_continuous_probe_v2 import require_previous
    row = dict(method="m", phase="hold", factory="f", source={"s": "h"}, saved_state_filter_sha256="filter",
        identity_cold_parity=True, original_control_exact_parity=True, status="PASS",
        variants=dict(candidate=dict(status="PASS", local_stability=dict(pass_all=True, checks=200),
                                     long_hold=dict(pass_all=True, checks=1200))),
        first200_training_parity=True,
        first200_snapshot_reconciliation=dict(status="EXACT_TRAINING_STATE_WITH_ENUMERATED_EVALUATION_COUNTERS"))
    kwargs = dict(phase="cold", method="m", factory="f", sources={"s": "h"}, filter_sha="filter")
    require_previous(row, **kwargs)
    row["first200_snapshot_reconciliation"]["status"] = "UNKNOWN_DIFFERENCE"
    with pytest.raises(RuntimeError):
        require_previous(row, **kwargs)
