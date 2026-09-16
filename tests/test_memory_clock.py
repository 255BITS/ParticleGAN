import pytest
import torch
from torch import nn

from experiments import memory_scout as base
from experiments import memory_core_scout as core
from experiments import memory_handoff_scout as handoff
from experiments.memory_recent import clock_features
from experiments.diagnose_memory_clock import intervened_path


def test_clock_features_units_batching_and_required_time():
    reference = torch.zeros(3, 4)
    times = torch.tensor([0, 1, 1024])
    features = clock_features(times, reference, 2, .125)
    angles = times[:, None]*torch.tensor([.125, .25])
    torch.testing.assert_close(features, torch.cat((angles.sin(), angles.cos()), -1))
    torch.testing.assert_close(clock_features(times, reference, 2, .125, 0),
                               torch.tensor([[0., 0., 1., 1.]]).expand(3, -1))
    g, d, _, _ = handoff.build(handoff.Config(clock_bands=2), 'cpu')
    with pytest.raises(ValueError, match='explicit time_index'):
        g(reference, d.writer.initial(reference))


def test_cold_and_warm_evaluation_have_correct_absolute_time():
    class ClockReader(nn.Module):
        clock_bands = 1
        def forward(self, z, memory, hidden=None, *, time_index):
            return z.new_full((len(z), 2), time_index), hidden
    g, writer, z = ClockReader(), base.Writer('gru', 32), torch.zeros(2, 4)
    cold, _ = base.rollout(g, writer, z, 4)
    warm, _ = core.continuation(g, writer, z, torch.zeros(2, 8, 2), 4)
    torch.testing.assert_close(cold[0, :, 0], torch.arange(4).float())
    torch.testing.assert_close(warm[0, :, 0], torch.arange(8, 12).float())


def test_feedback_clock_uses_previous_position_and_shared_episode_origin():
    cfg = handoff.Config(clock_bands=3, clock_to_d=True, feedback_probability=1., max_prefix=8)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2)
    positions = torch.tensor([[0, 1, 4, 8], [2, 3, 5, 7]])
    origins = torch.tensor([[0], [1024]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    calls = []
    hook = g.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs['time_index'].clone()),
                                        with_kwargs=True)
    memory = handoff.training_memory(cfg, g, d, real, positions, z, torch.zeros(8, 8, 4),
        torch.ones_like(positions, dtype=torch.bool), 1, clock_origin=origins)
    torch.testing.assert_close(calls[0], ((positions-1).clamp_min(0)+origins).flatten())
    times = (positions+origins).flatten()
    with torch.no_grad():
        fake, _ = handoff.local_point(g, z, memory, times)
        d.handoff_head[-1].weight.mul_(1000)
    torch.testing.assert_close(calls[1], times)
    target = real[torch.arange(2)[:, None], positions].flatten(0, 1)
    view = handoff.CandidateView(d, memory, times)
    penalty = recipe.make_gradient_penalty()(view, target, fake, step=1)
    assert penalty.item() > 0
    (recipe.make_loss().d_loss(view(target), view(fake))+penalty).backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in d.writer.parameters())
    assert all(p.grad is None for p in g.parameters())
    hook.remove()


def test_clock_intervention_preserves_startup_and_warm_time_origin():
    class ClockReader(nn.Module):
        clock_bands = 1
        def forward(self, z, memory, *, time_index):
            return z.new_full((len(z), 2), time_index), None
    g, writer, z = ClockReader(), base.Writer('gru', 32), torch.zeros(2, 4)
    prefix = torch.zeros(2, 8, 2)
    normal = intervened_path(g, writer, z, prefix, 8, 'normal', switch=3)
    frozen = intervened_path(g, writer, z, prefix, 8, 'clock_frozen', switch=3)
    half = intervened_path(g, writer, z, prefix, 8, 'clock_half', switch=3)
    torch.testing.assert_close(normal[:, :3], frozen[:, :3])
    torch.testing.assert_close(normal[:, :3], half[:, :3])
    torch.testing.assert_close(frozen[0, 3:, 0], torch.full((5,), 11.))
    torch.testing.assert_close(half[0, 3:, 0], 11.+.5*torch.arange(5))
