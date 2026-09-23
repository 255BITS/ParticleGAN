import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lib.lunar_training import (load_fast_policy, load_world_model,
                                train_fast_policy, train_world_model, FastPolicy)


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
