"""Software gates for the fixed learned-game toy, without quality selection."""
from copy import deepcopy
import hashlib
import io
import math

import pytest
import torch

from examples import e22_routed_convergence as toy


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    # This is a locally constructed native checkpoint, including native types.
    return torch.load(buffer, map_location="cpu", weights_only=False)


@pytest.fixture(autouse=True)
def single_threaded():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


@pytest.fixture(scope="module")
def data():
    return toy.make_data()


def test_public_named_initialization_and_full_pool_teacher_reachability(data, monkeypatch):
    calls, initialize = [], toy.init.initialize_

    def observed(module, **kwargs):
        generators = kwargs["parameter_generators"]
        expected = {name for name, p in module.named_parameters() if p.requires_grad and p.numel()}
        assert kwargs["method"] == "sample_distributions_v1"
        assert set(generators) == expected
        assert len({id(g) for g in generators.values()}) == len(generators)
        assert all(g.device.type == "cpu" and g is not torch.default_generator for g in generators.values())
        calls.append(module)
        return initialize(module, **kwargs)

    monkeypatch.setattr(toy.init, "initialize_", observed)
    rng = torch.get_rng_state().clone()
    ordinary = toy.make_loop("ordinary_native_game", data)
    particle = toy.make_loop("particle_native_game", data)
    assert_tree_equal(torch.get_rng_state(), rng)
    assert any(module is ordinary.G for module in calls) and any(module is particle.G for module in calls)
    assert any(module is ordinary.policy.D for module in calls)
    assert any(module is particle.policy.D for module in calls)
    assert any(module is particle.policy.router for module in calls)
    assert_tree_equal(ordinary.policy.D.state_dict(), particle.policy.D.state_dict())
    for site in ("first", "second"):
        plain, routed = getattr(ordinary.G, site), getattr(particle.G, site)
        for field in ("base", "down", "up"):
            assert_tree_equal(getattr(plain, field).state_dict(), getattr(routed, field).state_dict())
        assert routed.bridge.weight.count_nonzero() > 0
        assert plain.up.weight.count_nonzero() == routed.up.weight.count_nonzero() == 0
        name = site + ".down.weight"
        expected_seed = int.from_bytes(hashlib.sha256(("convergence-v1:generator:" + name).encode()).digest()[:8],
                                       "little") % (2**63 - 1)
        assert data["initialization"]["generator"][name] == expected_seed
        assert data["initialization"]["particle_generator"][name] == expected_seed
    for pool in toy.SPLITS:
        context = data[pool]["context"]
        initial_plain = torch.cat([toy.forward(ordinary, context[i:i + toy.BATCH_SIZE])
                                   for i in range(0, len(context), toy.BATCH_SIZE)])
        initial_routed = torch.cat([toy.forward(particle, context[i:i + toy.BATCH_SIZE])
                                    for i in range(0, len(context), toy.BATCH_SIZE)])
        assert_tree_equal(initial_plain, data[pool]["base"])
        assert_tree_equal(initial_routed, initial_plain)
    # Install the teacher independently rather than trusting a Boolean receipt.
    with torch.no_grad():
        ordinary.G.load_state_dict(data["teacher"])
        for site in ("first", "second"):
            branch = getattr(particle.G, site)
            branch.down.weight.copy_(data["teacher"][site + ".down.weight"])
            branch.up.weight.copy_(data["teacher"][site + ".up.weight"])
            branch.bridge.weight.zero_()
            branch.bridge.bias.zero_()
        for pool in toy.SPLITS:
            context = data[pool]["context"]
            for start in range(0, len(context), toy.BATCH_SIZE):
                expected = data[pool]["targets"][start:start + toy.BATCH_SIZE]
                assert_tree_equal(toy.forward(ordinary, context[start:start + toy.BATCH_SIZE]), expected)
                assert_tree_equal(toy.forward(particle, context[start:start + toy.BATCH_SIZE]), expected)


def test_native_particles_remain_live_after_neutral_start_with_frozen_owners(data, monkeypatch):
    loop = toy.make_loop("particle_native_game", data)
    p, table = loop.policy, loop.policy.table.detach().clone()
    penalty_type, penalty_units = type(p.penalty.regularizer), []
    native_penalty = penalty_type._k3p_penalty

    def observed_penalty(self, critic, real, fake, *args, **kwargs):
        penalty_units.append((real.shape, fake.shape, real[0].numel()))
        return native_penalty(self, critic, real, fake, *args, **kwargs)

    monkeypatch.setattr(penalty_type, "_k3p_penalty", observed_penalty)
    frozen = {role: {name: tensor.detach().clone() for name, tensor in module.state_dict().items()
                     if name.endswith("base.weight") or role == "encoder" or name == "scale"}
              for role, module in toy.modules(loop).items()}
    rows = [toy.update(loop) for _ in range(4)]
    assert rows[0]["bank_gradient_norm"] == 0  # Zero up factors give the neutral fresh model.
    assert any(row["bank_gradient_norm"] > 0 and row["bank_gradient_rows"] == toy.PARTICLES for row in rows[1:])
    assert not torch.equal(table, p.table)
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in p.router.parameters())
    assert p.routed_control.spec.sites == ("first", "second")
    assert p.routed_control.spec.max_context_harm == 0 and not p.routed_control.spec.output_error_guard
    assert p.recipe.row_evidence_gate and p.recipe.particle_birth_death
    assert p.recipe.birth_death_backend == "auto" and p.recipe.reopen_guard == "settled"
    assert p._feature_selection.state["actual_backend"] == "routed"
    assert p._feature_selection.state["selection_reason"] == "routed_rows_owns_controls"
    assert p.opt_d.record.calls == 4
    assert penalty_units == [(torch.Size([toy.BATCH_SIZE * toy.TOKENS, toy.WIDTH]),
                              torch.Size([toy.BATCH_SIZE * toy.TOKENS, toy.WIDTH]), toy.WIDTH)] * 4
    assert p.log_output_sigma.grad is not None and torch.isfinite(p.log_output_sigma.grad)
    assert p.D.scale.shape == (toy.WIDTH,)
    for role, saved in frozen.items():
        actual = toy.modules(loop)[role].state_dict()
        for name, expected in saved.items():
            assert_tree_equal(actual[name], expected)
    for module in (p.G, p.ema_G):
        for site in ("first", "second"):
            base = getattr(module, site).base.weight
            assert base.dtype == torch.bfloat16 and not base.requires_grad


@pytest.mark.parametrize("arm", toy.ARMS)
def test_private_common_judge_evaluation_preserves_owned_state_and_game_anchor(data, arm):
    loop = toy.make_loop(arm, data)
    for _ in range(3):
        toy.update(loop)
    judge_loop = toy.make_loop("ordinary_native_game", data)
    toy.update(judge_loop)
    judge = deepcopy(judge_loop.policy.D).eval().requires_grad_(False)
    panels = toy.evaluation_panels(data)
    before, judge_before, rng = toy.checkpoint(loop), deepcopy(judge.state_dict()), torch.get_rng_state().clone()
    first = toy.evaluate(loop, judge, "test", panels)
    assert_tree_equal(toy.checkpoint(loop), before)
    assert_tree_equal(judge.state_dict(), judge_before)
    assert_tree_equal(torch.get_rng_state(), rng)
    assert_tree_equal(toy.evaluate(loop, judge, "test", panels), first)
    zero = torch.zeros_like(data["test"]["targets"])
    anchor = toy.score_residual(judge, data["test"]["context"], zero, panels["test"])
    assert anchor["paired_game"] == pytest.approx(math.log(2), abs=1e-7)
    assert anchor["clean_zero_noise_game"] == pytest.approx(math.log(2), abs=1e-7)
    assert anchor["feature_proxy"] == 0
    # Reconstruct the shared paired game with the public native loss.
    context, targets = data["test"]["context"], data["test"]["targets"]
    prediction = torch.cat([toy.forward(loop, context[i:i + toy.BATCH_SIZE])
                            for i in range(0, len(context), toy.BATCH_SIZE)])
    residual, condition = (prediction - targets) / data["scale"], context[:, 0, toy.WIDTH:]
    loss = toy.get_recipe("e22").make_loss()
    expected = torch.stack([loss.g_loss(judge(base + residual, condition), judge(base, condition))
                            for base in panels["test"]]).mean()
    assert first["paired_game"] == pytest.approx(float(expected.detach()), abs=1e-7)
    if arm == "particle_native_game":
        toy.evaluate(loop, judge, "test", panels, code_ablation=True)
        assert_tree_equal(toy.checkpoint(loop), before)


@pytest.mark.parametrize("arm", toy.ARMS)
def test_checkpoint_roundtrip_replays_two_actual_updates_and_rejects_law_mismatch(data, arm):
    loop = toy.make_loop(arm, data, bindings={"protocol": "fixed-test-card", "source": "fixed-test-source"})
    for _ in range(3):
        toy.update(loop)
    saved = cpu_roundtrip(toy.checkpoint(loop))
    expected_rows = [toy.update(loop) for _ in range(2)]
    expected = toy.checkpoint(loop)
    resumed = toy.make_loop(arm, data, bindings={"protocol": "fixed-test-card", "source": "fixed-test-source"})
    toy.restore(resumed, saved)
    assert_tree_equal(toy.checkpoint(resumed), saved)
    assert_tree_equal([toy.update(resumed) for _ in range(2)], expected_rows)
    assert_tree_equal(toy.checkpoint(resumed), expected)
    for field, value in (("task", "unknown-task-law"), ("data_digest", "changed-data"),
                         ("bindings", {"protocol": "different-card", "source": "fixed-test-source"})):
        invalid = deepcopy(saved)
        invalid["law"][field] = value
        before = toy.checkpoint(resumed)
        with pytest.raises(ValueError, match="match"):
            toy.restore(resumed, invalid)
        assert_tree_equal(toy.checkpoint(resumed), before)


def test_all_arms_use_declared_generator7_fit_stream_and_native_paired_draws(data):
    loops = [toy.make_loop(arm, data) for arm in toy.ARMS]
    stream, paired = torch.Generator().manual_seed(7), torch.Generator().manual_seed(43)
    for _ in range(3):
        indices = torch.randint(len(data["fit"]["context"]), (toy.BATCH_SIZE,), generator=stream)
        expected_base = tuple(torch.randn((toy.BATCH_SIZE, toy.TOKENS, toy.WIDTH), generator=paired) for _ in range(2))
        rows = [toy.update(loop) for loop in loops]
        for row in rows:
            assert row["batch_indices"] == indices.tolist()
        for row in rows[:2]:
            assert row["paired_base_digest"] == toy.digest(expected_base)
        assert_tree_equal(loops[0].paired_rng.get_state(), loops[1].paired_rng.get_state())
    assert set(data["fit"]["latent_ids"].tolist()).isdisjoint(data["guard"]["latent_ids"].tolist())
    assert set(data["fit"]["latent_ids"].tolist()).isdisjoint(data["test"]["latent_ids"].tolist())
    assert set(data["fit"]["times"].tolist()).isdisjoint(data["test"]["times"].tolist())


def test_actual_data_drift_and_malformed_recovery_are_rejected_before_mutation(data):
    changed = deepcopy(data)
    changed["fit"]["targets"][0, 0, 0] += .25
    with pytest.raises(ValueError, match="digest"):
        toy.make_loop("particle_native_game", changed)
    loop = toy.make_loop("particle_native_game", data)
    toy.update(loop)
    saved = toy.checkpoint(loop)
    toy.update(loop)
    for mutation in (lambda state: state.update(unknown_schema_field=True),
                     lambda state: state.update(step=state["step"] + 1),
                     lambda state: state["modes"].pop(next(iter(state["modes"]))),
                     lambda state: state.update(data_rng=torch.zeros(1, dtype=torch.uint8))):
        invalid = deepcopy(saved)
        mutation(invalid)
        before = toy.checkpoint(loop)
        with pytest.raises((ValueError, RuntimeError)):
            toy.restore(loop, invalid)
        assert_tree_equal(toy.checkpoint(loop), before)
