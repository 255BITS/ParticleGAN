"""The replay diagnostic must not perturb the cross-only trajectory."""

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.cross_competitive_scratch import cross_competitive
from reports.toy100.cross_drift_replay import replay_cross


def _run(context):
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    with context as (recorder, _):
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy,
                                           diagnostics=True)
    return result, policy.receipt(), torch.get_rng_state().clone(), \
        policy.input_stream.get_state().clone(), recorder


def test_replay_matches_plain_cross_only_and_decomposes_field():
    plain = _run(cross_competitive(start_step=1, krylov_dim=2))
    replay = _run(replay_cross(start_step=1, krylov_dim=2))
    assert plain[0] == replay[0]
    counters = ("output_train_calls", "input_train_calls", "output_train_elements", "input_train_elements")
    assert ({k: v for k, v in plain[1].items() if k not in counters}
            == {k: v for k, v in replay[1].items() if k not in counters})
    # Three extra same-sample evaluations per recorded update are counted.
    assert replay[1]["output_train_calls"] > plain[1]["output_train_calls"]
    assert all(torch.equal(a, b) for a, b in zip(plain[2:4], replay[2:4]))
    recorder = replay[4]
    assert recorder.replay_restores_verified == len(recorder.replays) == 2
    for row in recorder.replays:
        for role in ("d", "g"):
            player = row["players"][role]
            assert player["step"] > 0 and player["explicit"] > 0
            assert player["own_curvature"] >= 0 and player["cross_response"] >= 0
        assert len(row["functional"]["occupancy_before"]) == 8
    kinds = [query["kind"] for query in recorder.queries]
    assert kinds.count("replay_joint") == 2
    assert recorder.rng_replay_verified == len(recorder.queries)
