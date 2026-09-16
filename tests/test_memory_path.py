"""Causal and gradient boundaries for the shared-memory experiment."""
import torch

from experiments.memory_path import Critic, FastMemory, Generator, circles, predictions
from particlegan import get_recipe


def test_fast_memory_streaming_matches_context_and_does_not_mutate():
    writer = FastMemory()
    context = torch.randn(5, 6, 2)
    state = torch.zeros(5, 8, 4)
    for point in context.unbind(1):
        before = state.clone()
        new_state = writer.write(state, point)
        torch.testing.assert_close(state, before)
        state = new_state
    torch.testing.assert_close(state.flatten(1), writer(context))


def test_generator_cannot_write_but_discriminator_trains_writer():
    recipe = get_recipe(num_particles=16)
    critic, generator, prior = Critic("shared"), Generator("shared"), recipe.make_prior()
    context, target = torch.randn(8, 4, 2), torch.randn(8, 2)
    z, _ = prior.sample(8)
    memory = critic.memory(context)
    fake = generator(z, memory.detach())
    recipe.make_loss().d_loss(critic(target, memory), critic(fake.detach(), memory)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.writer.parameters())
    assert all(p.grad is None for p in generator.parameters())
    assert all(p.grad is None for p in prior.parameters())
    critic.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    recipe.make_loss().g_loss(critic(fake, memory.detach()), critic(target, memory.detach()).detach()).backward()
    assert all(p.grad is None for p in critic.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters())
    assert prior.z.grad is not None and prior.z.grad.abs().sum() > 0


def test_future_observations_cannot_affect_earlier_predictions():
    critic, generator = Critic("shared"), Generator("shared")
    prior = get_recipe(num_particles=16).make_prior()
    observed = torch.randn(4, 7, 2)
    original, _ = predictions(generator, critic, prior, observed, particles=8)
    observed[:, 3:] += 100
    changed, _ = predictions(generator, critic, prior, observed, particles=8)
    torch.testing.assert_close(original[:, :3], changed[:, :3])
    assert not torch.allclose(original[:, 3:], changed[:, 3:])


def test_reversal_is_unobservable_until_reversed_point_arrives():
    _, normal = circles(8, 17, torch.Generator().manual_seed(123), "cpu")
    _, reversed_path = circles(8, 17, torch.Generator().manual_seed(123), "cpu", reversal=True)
    torch.testing.assert_close(normal[:, :9], reversed_path[:, :9])
    assert not torch.allclose(normal[:, 9], reversed_path[:, 9])
    torch.testing.assert_close(reversed_path[:, 9], normal[:, 7])
