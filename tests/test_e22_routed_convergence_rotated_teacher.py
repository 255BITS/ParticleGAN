"""Software and receipt guards for one distinct reachable teacher-span law.

Only brief optimizer updates verify real native replay and live particle paths.
These tests do not train or qualify the fixed 6400-update quality experiment.
"""
from copy import deepcopy
import io
import json

import pytest
import torch

from examples import e22_routed_convergence as base
from examples import e22_routed_convergence_neutral as neutral
from examples import e22_routed_convergence_rotated_teacher as law
from examples import run_e22_routed_convergence_rotated_teacher as runner


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


@pytest.fixture(scope="module")
def data():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original = base.make_data()
        return original, law.make_rotated_data(original)
    finally:
        torch.set_num_threads(threads)


def roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("corruption", (None, "archive", "source"))
def test_source_archive_preserves_bytes_and_rejects_drift(tmp_path, monkeypatch, corruption):
    root = tmp_path / "input"
    native_path = root / "particlegan" / "api.py"
    native_path.parent.mkdir(parents=True)
    native_path.write_text("API_VERSION = 1\n")
    card_path = root / "card.json"
    card_path.write_text('{"task": "software-only"}\n')
    hashes = {"card.json": runner.common.sha(card_path)}
    native = {"particlegan/api.py": runner.common.sha(native_path)}
    bindings = {"source_hashes": hashes, "native_source_hash": base.digest(native)}
    monkeypatch.setattr(runner, "ROOT", root)
    copyfile = runner.shutil.copyfile
    if corruption:
        def corrupt_copy(source, destination):
            copyfile(source, destination)
            (destination if corruption == "archive" else source).write_text("altered bytes\n")
        monkeypatch.setattr(runner.shutil, "copyfile", corrupt_copy)
        with pytest.raises(RuntimeError, match="snapshot differs"):
            runner.archive_sources(tmp_path / "run", bindings)
    else:
        archive = runner.archive_sources(tmp_path / "run", bindings)
        assert archive == {**hashes, "native/particlegan/api.py": native["particlegan/api.py"]}
        assert (tmp_path / "run/source/card.json").read_bytes() == card_path.read_bytes()
        assert (tmp_path / "run/source/native/particlegan/api.py").read_bytes() == native_path.read_bytes()


def test_rotation_preserves_teacher_gram_up_inputs_and_exact_reachability(data):
    original, rotated = data
    before, rng = base.digest(original), torch.get_rng_state().clone()
    witness = law.feasibility_witness(original)
    assert witness["optimizer_updates"] == 0
    assert witness["rho"] == law.RHO
    assert all(value for pools in witness["reachability"].values() for value in pools.values())
    assert witness["data_digest"] == rotated["digest"]
    assert all(item["within_rank_gram_max_absolute_error"] <= 1e-7 for item in witness["geometry"].values())
    assert base.digest(original) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert rotated["digest"] != original["digest"]
    for site in law.SITES:
        assert not torch.equal(rotated["teacher"][site + ".down.weight"], original["teacher"][site + ".down.weight"])
        assert torch.equal(rotated["teacher"][site + ".up.weight"], original["teacher"][site + ".up.weight"])


def test_rotation_rejects_missing_complement_or_rank_deficiency():
    with pytest.raises(ValueError, match="width"):
        law.rotated_down(torch.eye(2))
    with pytest.raises(ValueError, match="row rank"):
        law.rotated_down(torch.zeros(2, 16))


@pytest.mark.parametrize("arm", law.ARMS)
def test_fresh_arms_preserve_common_base_and_native_particle_guard(arm, data):
    _, rotated = data
    loop = law.make_rotated_loop(arm, rotated, bindings={"source": "software-only"})
    assert loop.law["task"] == law.TASK and loop.law["arm"] == arm
    assert loop.law["common_task_data_digest"] == rotated["digest"]
    with torch.no_grad():
        for pool in base.SPLITS:
            context = rotated[pool]["context"][:base.BATCH_SIZE]
            assert torch.equal(base.forward(loop, context), rotated[pool]["base"][:base.BATCH_SIZE])
    if arm in law.ARMS[1:]:
        assert loop.arm == "particle_native_game"
        assert loop.policy.routed_control.spec.max_context_harm == 0
        assert not loop.policy.routed_control.spec.output_error_guard
        assert loop.policy.table.requires_grad
        for site in law.SITES:
            bridge = getattr(loop.G, site).bridge
            assert bridge.weight.requires_grad and bridge.bias.requires_grad
            assert bridge.weight[:, base.RANK:].count_nonzero() > 0


def test_neutral_changes_only_h_b_in_fast_ema_and_keeps_all_other_native_state(data):
    _, rotated = data
    reference = law.make_rotated_loop(law.ARMS[1], rotated)
    candidate = law.make_rotated_loop(law.ARMS[2], rotated)
    expected, actual = deepcopy(reference.policy.state_dict()), candidate.policy.state_dict()
    for family in ("models", "averages"):
        for key, value in expected[family]["generator"].items():
            if key.endswith("bridge.weight"):
                value[:, :base.RANK].zero_()
            elif key.endswith("bridge.bias"):
                value.zero_()
        assert base.digest(expected[family]["generator"]) == base.digest(actual[family]["generator"])
    assert base.digest(expected) == base.digest(actual)
    assert candidate.law["data_digest"] != reference.law["data_digest"]
    assert base.digest(candidate.data["teacher"]) == base.digest(reference.data["teacher"])
    assert torch.equal(candidate.data["scale"], reference.data["scale"])


@pytest.mark.parametrize("arm", law.ARMS)
def test_real_native_short_recovery_is_exact_and_teacher_laws_cannot_mix(arm, data):
    original, rotated = data
    loop = law.make_rotated_loop(arm, rotated, bindings={"source": "software-only"})
    for _ in range(3):
        base.update(loop)
    saved = roundtrip(base.checkpoint(loop))
    expected_rows = [base.update(loop), base.update(loop)]
    expected = base.checkpoint(loop)
    restored = law.make_rotated_loop(arm, rotated, bindings={"source": "software-only"})
    base.restore(restored, saved)
    assert base.digest([base.update(restored), base.update(restored)]) == base.digest(expected_rows)
    assert base.digest(base.checkpoint(restored)) == base.digest(expected)
    parent_arm = law.ARMS[1] if arm == law.ARMS[2] else arm
    parent = base.checkpoint(base.make_loop(parent_arm, original))
    before = base.digest(base.checkpoint(restored))
    with pytest.raises(ValueError, match="match"):
        base.restore(restored, parent)
    assert base.digest(base.checkpoint(restored)) == before
    with pytest.raises(ValueError, match="law"):
        law.make_rotated_loop(arm, law.make_rotated_data(original, rho=1.))


def test_actual_learned_heads_score_without_consuming_state_and_reject_mislabeled_checkpoints(data):
    _, rotated = data
    judges = {}
    for arm in law.ARMS[:2]:
        loop = law.make_rotated_loop(arm, rotated)
        for step in range(1, 5):
            base.update(loop)
            if step in (1, 4):
                state = base.checkpoint(loop)
                judges[f"{arm}@{step}"] = runner.load_judge(state, rotated, expected_step=step)
        with pytest.raises(ValueError, match="mislabeled"):
            runner.load_judge(base.checkpoint(loop), rotated, expected_step=800)
    loop = law.make_rotated_loop(law.ARMS[2], rotated)
    initial = base.checkpoint(loop)
    for _ in range(4):
        row = base.update(loop)
    assert row["bank_gradient_rows"] == 128 and row["query_gradient_norm"] > 0
    for site in law.SITES:
        bridge = getattr(loop.G, site).bridge
        assert bridge.weight.grad[:, base.RANK:].norm() > 0
        assert bridge.weight.grad[:, :base.RANK].norm() > 0 and bridge.bias.grad.norm() > 0
    resolved = base.checkpoint(loop)
    panels = base.evaluation_panels(rotated)
    scores = runner.immutable_scores(loop, judges, panels)
    ablated = runner.immutable_scores(loop, judges, panels, code_ablation=True)
    assert set(scores) == set(judges) == set(ablated)
    assert all(set(pools) == set(base.SPLITS) for pools in scores.values())
    assert base.digest(base.checkpoint(loop)) == base.digest(resolved)
    fresh = law.make_rotated_loop(law.ARMS[2], rotated)
    base.restore(fresh, initial)
    runner.immutable_scores(fresh, judges, panels)
    assert base.digest(base.checkpoint(fresh)) == base.digest(initial)
    base.restore(fresh, resolved)
    assert base.digest(base.checkpoint(fresh)) == base.digest(resolved)


def gate_fixture(particle_games, neutral_game=1.1):
    values = {}
    for arm, games in zip(law.ARMS, ([1.] * 4, particle_games, [neutral_game] * 4)):
        values[f"{arm}@6400"] = {name: {"test": {"paired_game": value}}
                                 for name, value in zip(runner.JUDGES, games)}
    witnesses = {arm: {"bridge_still_trainable": True, "bank_still_trainable": True,
                      "router_still_trainable": True, "C_norms": dict.fromkeys(law.SITES, .1),
                      "live_bank_updates": 4, "live_query_updates": 4,
                      "zero_code_minus_live_test_game": dict.fromkeys(runner.JUDGES, .1)}
                 for arm in law.ARMS[1:]}
    return values, witnesses


@pytest.mark.parametrize("particle", ([1.] * 4, [.9] * 4, [1.2, 1.2, 1.2, .9]))
def test_nonpositive_or_mixed_baseline_gap_has_no_reduction_or_remaining_gap_claim(particle):
    values, witnesses = gate_fixture(particle)
    result = runner.final_gates(values, witnesses)
    assert not result["original_particle_gap_reproduced"]
    assert not result["support_gate_applicable"]
    assert result["endpoint_gap_reduction"] is None
    assert not result["H_b_support_gate"]
    assert not result["remaining_neutral_gap_witness"]
    assert set(result["paired_game_H_b_improvement"]) == set(runner.JUDGES)


def test_all_four_heads_required_for_particle_retention_and_wins_are_separate():
    values, witnesses = gate_fixture([1.5] * 4, neutral_game=.9)
    result = runner.final_gates(values, witnesses)
    assert result["H_b_support_gate"] and result["neutral_beats_ordinary_all_four"]
    assert not result["remaining_neutral_gap_witness"]
    witnesses[law.ARMS[2]]["zero_code_minus_live_test_game"].pop(runner.JUDGES[0])
    result = runner.final_gates(values, witnesses)
    assert not result["retained_particle_gate"][law.ARMS[2]]
    assert not result["H_b_support_gate"] and not result["neutral_beats_ordinary_all_four"]
    witnesses[law.ARMS[2]]["zero_code_minus_live_test_game"][runner.JUDGES[0]] = .1
    witnesses[law.ARMS[2]]["bank_still_trainable"] = False
    assert not runner.final_gates(values, witnesses)["retained_particle_gate"][law.ARMS[2]]


def test_frozen_contract_includes_actual_extra_endpoint_and_source_hold():
    card = json.loads(runner.CARD.read_text())
    # A signed evaluation repair cannot relabel the archived unsigned training
    # sources. Fresh execution under that frozen card must remain blocked.
    with pytest.raises(ValueError, match="declared source changed"):
        runner.validate_contract(card)
    assert isinstance(card["execution"]["execution_authorized"], bool)
    assert len(runner.checkpoint_steps()) == 35
    assert len(runner.curve_steps()) == 34
    assert {0, 800, 802, 5120, 6400} <= set(runner.checkpoint_steps())
    assert 802 not in runner.curve_steps()
    bad = deepcopy(card)
    bad["evaluation"]["mandatory_common_judges"] = list(runner.JUDGES[:-1])
    with pytest.raises(ValueError, match="frozen"):
        runner.validate_contract(bad)
