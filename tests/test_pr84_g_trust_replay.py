from reports.toy100.pr84_g_trust_replay import approaches, verdict


def test_approach_requires_closure_and_positive_projection():
    assert approaches([2.0, 2.1, 2.2], [1.9, 2.0, 2.1], [0.1, 0.1, 0.1])
    assert not approaches([2.0, 2.1, 2.2], [2.0, 2.1, 2.2], [0.1, 0.0, 0.0])
    assert not approaches([2.0, 2.1, 2.2], [1.5, 1.6, 1.7], [-0.2, -0.1, -0.1])


def test_verdict_follows_the_predeclared_rule():
    assert verdict([False], [True], 1) == "build"
    assert verdict([False], [False], 1) == "kill_shared_network"
    assert verdict([True], [True], 1) == "kill_premise"
    assert verdict([True], [False], 1) == "kill_premise"
    assert verdict([], [], 0) == "not_stuck"
    assert verdict([False, False], [True, False], 2) == "kill_shared_network"
