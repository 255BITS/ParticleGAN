import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lib.lunar_training import (load_fast_policy, load_world_model,
                                train_fast_policy, train_world_model, FastPolicy,
                                DynamicsModel, engine_power)


def _transitions(count=96, seed=14):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(count, 8)).astype(np.float32)
    actions = np.tanh(states[:, :2] * .6).astype(np.float32)
    next_states = states.copy()
    next_states[:, :2] += .1 * actions
    next_states[:, 2:6] += .01
    return dict(states=states, actions=actions, next_states=next_states)


def test_world_checkpoint_learns_action_conditioned_transition(tmp_path):
    torch.set_num_threads(1)
    records = _transitions()
    validation = _transitions(32, 35)
    path = tmp_path / "world.pt"
    metrics = train_world_model(records, path, validation_records=validation,
                                steps=80, batch_size=32, width=32)
    assert metrics["validation"]["next_state_mse"] < .002
    world = load_world_model(path)
    state = torch.from_numpy(validation["states"][:1])
    action = torch.from_numpy(validation["actions"][:1]).requires_grad_()
    world(state, action)[:, 0].backward()
    assert action.grad.abs().sum() > 0


def test_policy_warm_start_and_adversarial_updates(tmp_path):
    torch.set_num_threads(1)
    records = _transitions()
    validation = _transitions(32, 35)
    world_path = tmp_path / "world.pt"
    train_world_model(records, world_path, steps=40, batch_size=32, width=32)
    slow_path = tmp_path / "slow.pt"
    fast_path = tmp_path / "fast.pt"
    first = train_fast_policy(records, world_path, slow_path, validation_records=validation,
                              warmup_steps=25, steps=8, batch_size=32, width=32)
    second = train_fast_policy(records, world_path, fast_path, validation_records=validation,
                               initial_policy=slow_path, warmup_steps=0, steps=4,
                               batch_size=32, width=32)
    assert first["warmup_updates"] == 25 and first["rpgan_updates"] == 8
    assert second["warmup_updates"] == 0 and second["rpgan_updates"] == 4
    assert first["validation"]["action_mse"] < .3
    action = load_fast_policy(fast_path).act(validation["states"][0])
    assert action.shape == (2,) and np.all(np.isfinite(action)) and np.max(np.abs(action)) <= 1
    slow = load_fast_policy(slow_path).act(validation["states"][0])
    assert not np.allclose(action, slow)


def test_main_engine_deadband_preserves_training_gradient():
    policy = FastPolicy(np.zeros(8), np.ones(8), width=8)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.net[-1].bias[0] = .05
    output = policy(torch.zeros(1, 8))
    assert output[0, 0].item() == 0.
    output[0, 0].backward()
    assert policy.net[-1].bias.grad[0].item() > 0


def test_world_engine_features_preserve_ignition_and_side_dead_zone():
    commands = torch.tensor([[-.3, -.8], [0., -.5], [.001, .5], [1., .8]], requires_grad=True)
    expected = torch.tensor([[-.3, -.8], [0., 0.], [.5005, 0.], [1., .8]])
    torch.testing.assert_close(engine_power(commands), expected)
    engine_power(commands).sum().backward()
    torch.testing.assert_close(commands.grad[:, 0], torch.tensor([1., 1., .5, .5]))


def test_world_checkpoint_preserves_original_or_explicit_action_features(tmp_path):
    model = DynamicsModel(np.zeros(8), np.ones(8), np.zeros(8), np.ones(8), width=8)
    path = tmp_path / "world.pt"
    saved = {"format": "lunar_world_v1", "state_dict": model.state_dict(), "width": 8}
    torch.save(saved, path)
    assert load_world_model(path).action_features == "raw"
    saved.update(format="lunar_world_v2", action_features="engine_power", weight_kind="live")
    torch.save(saved, path)
    assert load_world_model(path).action_features == "engine_power"
    del saved["action_features"]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="action features"):
        load_world_model(path)


def test_checkpoint_declares_action_semantics(tmp_path):
    policy = FastPolicy(np.zeros(8), np.ones(8), width=8)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.net[-1].bias[0] = .05
    path = tmp_path / "policy.pt"
    torch.save({"format": "lunar_policy_rpgan_v1", "state_dict": policy.state_dict(),
                "width": 8}, path)
    assert load_fast_policy(path).act(np.zeros(8))[0] > 0.
    torch.save({"format": "lunar_policy_rpgan_v2", "state_dict": policy.state_dict(),
                "width": 8}, path)
    with pytest.raises(ValueError, match="deadband"):
        load_fast_policy(path)


def test_checkpoints_save_exact_live_parameters_and_reject_ema(tmp_path, monkeypatch):
    import lib.lunar_training as training
    torch.set_num_threads(1)
    data = _transitions()
    world_path, policy_path = tmp_path / "world.pt", tmp_path / "policy.pt"
    train_world_model(data, world_path, steps=4, width=8, batch_size=16)
    instances = []
    original = training.FastPolicy

    def capture_policy(*args, **kwargs):
        policy = original(*args, **kwargs)
        instances.append(policy)
        return policy

    monkeypatch.setattr(training, "FastPolicy", capture_policy)
    metrics = train_fast_policy(data, world_path, policy_path, warmup_steps=4, steps=4,
                                width=8, batch_size=16)
    saved = torch.load(policy_path, weights_only=True)
    assert saved["weight_kind"] == "live"
    assert metrics["weight_kind"] == "live" and metrics["recipe"]["ema_decay"] == 0.
    for name, value in instances[0].state_dict().items():
        torch.testing.assert_close(saved["state_dict"][name], value, rtol=0, atol=0)
    del saved["weight_kind"]
    torch.save(saved, policy_path)
    with pytest.raises(ValueError, match="live weights"):
        load_fast_policy(policy_path)
    saved["weight_kind"] = "ema"
    torch.save(saved, policy_path)
    with pytest.raises(ValueError, match="live weights"):
        load_fast_policy(policy_path)
    world = torch.load(world_path, weights_only=True)
    assert world["weight_kind"] == "live"
    del world["weight_kind"]
    torch.save(world, world_path)
    with pytest.raises(ValueError, match="live weights"):
        load_world_model(world_path)
    world["weight_kind"] = "ema"
    torch.save(world, world_path)
    with pytest.raises(ValueError, match="live weights"):
        load_world_model(world_path)
