import numpy as np
import torch

from experiments.memory_ddgan_scout import (
    Config, TransitionView, build, prefix_memory, rollout, train, transition_fake,
)


def rng(seed=777):
    return torch.Generator().manual_seed(seed)


def test_single_memory_fixed_particle_and_full_feedback_gradients():
    torch.set_num_threads(1)
    g, d, prior, _, diffusion = build(Config(), "cpu")
    z = prior(torch.arange(3))
    reads, particles, writes = [], [], []
    gh = g.register_forward_pre_hook(lambda module, args: (reads.append(args[1]), particles.append(args[0])) and None)
    original_write = d.writer.write
    def record_write(memory, point):
        writes.append(point)
        point.retain_grad()
        return original_write(memory, point)
    d.writer.write = record_write
    path, _ = rollout(g, d.writer, diffusion, z, 8, rng())
    gh.remove()
    assert len(reads) == 32 and len(writes) == 8
    assert all(particle is z for particle in particles)
    assert reads[0].count_nonzero() == 0
    assert all(all(read is reads[i] for read in reads[i:i+4]) for i in range(0, 32, 4))
    path[:, -1].square().sum().backward()
    assert writes[0].grad.abs().sum() > 0, "late output must backpropagate through earlier feedback"
    assert any(p.grad is not None and p.grad.abs().sum() for p in g.parameters())
    assert prior.z.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.writer.parameters())


def test_transition_memory_excludes_current_and_future_real_points():
    _, d, _, _, _ = build(Config(), "cpu")
    real = torch.randn(3, 8, 2)
    positions = torch.tensor([0, 2, 5])
    memory = prefix_memory(d.writer, real, positions)
    altered = real.clone()
    for row, position in enumerate(positions):
        altered[row, position:] += 100
    torch.testing.assert_close(memory, prefix_memory(d.writer, altered, positions))
    assert memory[0].count_nonzero() == 0
    assert memory[1:].abs().sum() > 0


def test_conditional_transition_exact_bcap_and_writer_ownership():
    g, d, prior, recipe, diffusion = build(Config(), "cpu")
    real = torch.randn(3, 64, 2)
    positions, times = torch.tensor([1, 12, 40]), torch.tensor([1, 2, 4])
    clean = real[torch.arange(3), positions]
    previous, noisy = diffusion.forward_pair(clean, times, generator=rng())
    z = prior(torch.arange(3))
    fake = transition_fake(g, d, diffusion, z, real, positions, noisy, times, torch.zeros_like(noisy))
    transition = TransitionView(d, real, noisy, positions, times)
    # Activate B-cap so its exact second backward is exercised, not only its zero branch.
    with torch.no_grad():
        d.transition_head[-1].weight.mul_(1000)
    penalty = recipe.make_gradient_penalty()(transition, previous, fake.detach(), step=1)
    assert penalty > 0
    (recipe.make_loss().d_loss(transition(previous), transition(fake.detach()))+penalty).backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in d.writer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert all(p.grad is None for p in g.parameters())


def test_rollout_noise_replay_and_final_reverse_is_clean():
    g, d, prior, _, diffusion = build(Config(), "cpu")
    z = prior(torch.arange(3))
    a, _ = rollout(g, d.writer, diffusion, z, 8, rng())
    b, _ = rollout(g, d.writer, diffusion, z, 8, rng())
    torch.testing.assert_close(a, b)
    clean, noisy = torch.randn(3, 2), torch.randn(3, 2)
    final = diffusion.reverse(clean, noisy, torch.ones(3, dtype=torch.long), torch.randn(3, 2))
    torch.testing.assert_close(final, clean)


def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    # Evaluation does not alter the saved training state; CUDA smoke covers full diagnostics.
    monkeypatch.setattr("experiments.memory_ddgan_scout.evaluate",
                        lambda *args: ({}, {"generated": np.zeros((4, 256, 2))}))
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = Config(name=name, steps=steps, schedule_steps=10, batch_size=4,
                     eval_batch=4, eval_steps=256, resume=resume)
        train(cfg, out, "cpu", lambda **kw: None)
        return torch.load(out/"model.pt", weights_only=False)
    full = run("full", 2)
    run("first", 1)
    resumed = run("resumed", 2, str(tmp_path/"first/model.pt"))
    for key in ("generator", "critic", "prior"):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
