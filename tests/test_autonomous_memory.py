"""Runtime and optimization contracts for expert-free recurrent generation."""
import numpy as np
import torch

from experiments.autonomous_memory import SequenceCritic, frozen, rollout, trajectory_metrics
from experiments.memory_path import FastMemory, Generator, circles
from particlegan import get_recipe


def test_cold_rollout_uses_fixed_particle_and_independent_batch_states():
    writer, generator = FastMemory(), Generator("shared")
    z = torch.randn(4, 4)
    seen = []
    handle = generator.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    path, state = rollout(generator, writer, z, 8)
    handle.remove()
    assert len(seen) == 8
    for code in seen:
        torch.testing.assert_close(code, z)
    torch.testing.assert_close(path[:, 0], generator(z, torch.zeros(4, 32)))
    for i in range(4):
        single, single_state = rollout(generator, writer, z[i:i+1], 8)
        torch.testing.assert_close(path[i:i+1], single)
        torch.testing.assert_close(state[i:i+1], single_state)
    again, _ = rollout(generator, writer, z, 8)
    torch.testing.assert_close(path, again)  # No hidden persistent state across calls.


def test_g_backpropagates_through_time_without_training_writer():
    writer, generator = FastMemory(), Generator("shared")
    prior = get_recipe(num_particles=16).make_prior()
    z, _ = prior.sample(4)
    points = []
    def capture(module, args, output):
        output.retain_grad()
        points.append(output)
    handle = generator.register_forward_hook(capture)
    path, _ = rollout(generator, writer, z, 6)
    handle.remove()
    path[:, -1].square().sum().backward()
    assert points[0].grad is not None and points[0].grad.abs().sum() > 0
    assert all(p.grad is None for p in writer.parameters())
    assert all(p.requires_grad for p in writer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters())
    assert prior.z.grad.abs().sum() > 0


def test_discriminator_trains_writer_from_detached_sequences():
    critic, generator = SequenceCritic(6, "shared"), Generator("shared")
    path, _ = rollout(generator, critic.writer, torch.randn(4, 4), 6)
    critic(path.detach()).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.writer.parameters())
    assert all(p.grad is None for p in generator.parameters())


def test_frozen_writer_survives_nested_freezing_and_no_memory_is_static():
    critic = SequenceCritic(6, "frozen_writer")
    flags = [p.requires_grad for p in critic.parameters()]
    with frozen(critic):
        with frozen(critic.writer):
            assert not any(p.requires_grad for p in critic.parameters())
    assert [p.requires_grad for p in critic.parameters()] == flags
    critic(torch.randn(4, 6, 2)).sum().backward()
    assert all(p.grad is None for p in critic.writer.parameters())
    path, state = rollout(Generator("no_memory"), critic.writer, torch.randn(4, 4), 16, use_memory=False)
    torch.testing.assert_close(path, path[:, :1].expand_as(path))
    assert torch.count_nonzero(state) == 0


def test_dynamics_diagnostics_reject_stationary_and_spiraling_paths():
    _, clean = circles(8, 256, torch.Generator().manual_seed(73), "cpu")
    metrics = trajectory_metrics(clean.numpy())
    assert metrics["circle_like_fraction"] == 1
    assert metrics["relative_radial_rmse"] < 1e-5
    static = trajectory_metrics(np.ones((8, 256, 2)))
    assert static["circle_like_fraction"] == 0
    assert static["stationary_fraction"] == 1
    assert static["relative_radial_rmse"] is None
    theta = np.arange(256) * .2
    spiral = np.stack((np.cos(theta), np.sin(theta)), -1) * np.linspace(1, 4, 256)[:, None]
    assert trajectory_metrics(spiral[None])["circle_like_fraction"] == 0
