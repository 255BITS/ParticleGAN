"""Exact reconciliation of two independently measured evaluation counters.

The original host records 24 observations across its declared horizon and
extra native live-tail diagnostics. These alter evaluation counters, while
NoisePolicy.evaluation restores training RNG. No training-state field is
ignored: subtract only the independently enumerated warm-minus-hold counters,
then require the complete snapshot hash to match.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from reports.toy100.pr84_critic_refinement_capture import _sha


def expected_difference():
    from benchmarks.locked_shared.observation import Recorder
    observation = {
        horizon: {step for step in Recorder(horizon).steps if step < 1200}
        for horizon in (1200, 2400)
    }
    tail = {
        horizon: {step for step in range(1, 1200)
                  if step % 200 == 0 or (step >= horizon-200 and step % 50 == 0)}
        for horizon in (1200, 2400)
    }
    if not observation[2400] <= observation[1200] or not tail[2400] <= tail[1200]:
        raise RuntimeError("native observation schedule no longer has the reviewed containment")
    extra_observations = sorted(observation[1200]-observation[2400])
    extra_tail = sorted(tail[1200]-tail[2400])
    count = len(extra_observations)+len(extra_tail)
    expected = dict(output_eval_calls=2*count, output_eval_elements=2*(4096+12)*count)
    if (extra_observations != list(range(50, 1200, 100))
            or extra_tail != [1050, 1100, 1150]
            or expected != dict(output_eval_calls=30, output_eval_elements=123240)):
        raise RuntimeError("independently enumerated evaluation difference changed")
    return dict(counter_deltas=expected, extra_native_observation_steps=extra_observations,
                extra_native_live_curve_steps=extra_tail,
                per_measure_output_calls=2, per_measure_output_elements=2*(4096+12),
                boundary="after active update1200, before its EMA/checkpoint")


def reconcile(warm, hold):
    declared = expected_difference()
    for state in (warm, hold):
        if (state["noise"]["step_calls"] != 1200 or state["noise_policy"]["_step_calls"] != 1200
                or state["noise_policy"]["total_steps"] != 1200
                or state["snapshot_scope"]["host_loop_step"] != 1199):
            raise RuntimeError("snapshot is not the declared pre-EMA update1200 boundary")
    adjusted = deepcopy(warm)
    observed = {}
    for key, expected in declared["counter_deltas"].items():
        a, b = warm["noise_policy"]["_counts"][key], hold["noise_policy"]["_counts"][key]
        if type(a) is not int or type(b) is not int or b < 0 or a-b != expected:
            raise RuntimeError(f"unexpected evaluation counter difference: {key}")
        adjusted["noise_policy"]["_counts"][key] -= expected
        observed[key] = dict(warm=a, hold=b, warm_minus_hold=a-b)
    adjusted_sha, hold_sha = _sha(adjusted), _sha(hold)
    if adjusted_sha != hold_sha:
        raise RuntimeError("snapshot differs beyond the two enumerated evaluation counters")
    return dict(status="EXACT_TRAINING_STATE_WITH_ENUMERATED_EVALUATION_COUNTERS",
        training_state_parity=True, raw_snapshots_identical=False,
        warm_raw_sha256=_sha(warm), hold_raw_sha256=hold_sha,
        adjusted_warm_sha256=adjusted_sha, ignored_fields=[], adjusted_fields=observed,
        independent_count=declared,
        scope="all model/Adam/EMA/RNG/noise clocks/histories and other metadata exact; only two eval counters reconciled")


def load_and_verify(directory, receipt):
    import torch
    path = directory / receipt["first200_snapshot_file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["first200_snapshot_file_sha256"]:
        raise RuntimeError("first200 snapshot file changed")
    value = torch.load(path, weights_only=True, map_location="cpu")
    if _sha(value) != receipt["first200_raw_snapshot_sha256"]:
        raise RuntimeError("first200 raw snapshot hash changed")
    return value
