import torch
import pytest

from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core
from experiments.autonomous_memory import frozen


def test_dense_context_is_causal_and_matches_individual_prefixes():
    cfg = handoff.Config(max_prefix=8)
    g, d, prior, _ = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2, requires_grad=True)
    positions = torch.tensor([[0, 2, 5, 8], [1, 3, 6, 7]])
    memory = handoff.selected_memories(d.writer, real, positions, 8)
    expected = torch.cat([core.context(d.writer, real[b:b+1, :p])
                          for b, row in enumerate(positions) for p in row])
    torch.testing.assert_close(memory, expected)
    memory[2].sum().backward()
    assert real.grad[0, :5].abs().sum() > 0
    assert real.grad[0, 5:].count_nonzero() == 0
    assert real.grad[1].count_nonzero() == 0
    assert memory[0].count_nonzero() == 0


@pytest.mark.parametrize('size,head', [(8, 'concat'), (32, 'concat'), (64, 'concat'), (32, 'interaction')])
def test_cached_memory_active_exact_penalty_and_ownership(size, head):
    cfg = handoff.Config(memory_dim=size, point_head=head, max_prefix=8)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    real = torch.randn(3, 64, 2)
    positions = torch.tensor([[0, 2, 5, 8]]*3)
    z = prior(torch.arange(3))[:, None].expand(-1, 4, -1).flatten(0, 1)
    memory = handoff.selected_memories(d.writer, real, positions, 8)
    with torch.no_grad():
        # Force an active cap so this checks second derivatives, not a zero penalty.
        d.handoff_head[-1].weight.mul_(1000)
        fake, _ = g(z, memory)
    target = real[torch.arange(3)[:, None], positions].flatten(0, 1)
    view = handoff.CandidateView(d, memory)
    snapshot = memory.detach().clone()
    loss = recipe.make_loss().d_loss(view(target), view(fake))
    penalty = recipe.make_gradient_penalty()(view, target, fake, step=1)
    assert penalty.item() > 0
    (loss+penalty).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.writer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert all(p.grad is None for p in g.parameters())
    torch.testing.assert_close(memory, snapshot)
    d.zero_grad(set_to_none=True)
    with frozen(d):
        memory = handoff.selected_memories(d.writer, real, positions, 8)
        fake, _ = g(z, memory)
        d.score_candidate(fake, memory).mean().backward()
    assert all(p.grad is None for p in d.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
    assert prior.z.grad.abs().sum() > 0


def test_cached_and_recomputed_context_give_same_discriminator_gradients():
    cfg = handoff.Config(samples_per_episode=1)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    _, original, _, _ = core.build(core.Config(handoff_weight=1, cold_weight=0), 'cpu')
    for key, value in d.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    real, fake = torch.randn(3, 64, 2), torch.randn(3, 2)
    positions = torch.tensor([0, 8, 32])
    target = real[torch.arange(3), positions]
    memory = handoff.selected_memories(d.writer, real, positions[:, None], 63)
    for model in (d, original):
        with torch.no_grad():
            model.handoff_head[-1].weight.mul_(1000)
    for view in (handoff.CandidateView(d, memory), core.HandoffView(original, real, positions)):
        loss = recipe.make_loss().d_loss(view(target), view(fake))
        (loss+recipe.make_gradient_penalty()(view, target, fake, step=1)).backward()
    for name, p in d.named_parameters():
        torch.testing.assert_close(p.grad, dict(original.named_parameters())[name].grad, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('mode', ['ordinary', 'auxiliary', 'feedback', 'recent_feedback', 'backprop_feedback',
                                  'clock', 'clock_shared', 'clock_feedback'])
def test_resume_and_bounded_generator_calls(tmp_path, monkeypatch, mode):
    torch.set_num_threads(1)
    monkeypatch.setattr(handoff, 'evaluate', lambda *a: ({'generated_256': {}}, {}))
    original_build = handoff.build
    counts = []
    def build(cfg, device):
        g, d, prior, recipe = original_build(cfg, device)
        calls = []
        def observe(module, args, result):
            assert len(args) == 2 and len(args[0]) == cfg.batch_size*cfg.samples_per_episode
            z = args[0].reshape(cfg.batch_size, cfg.samples_per_episode, -1)
            torch.testing.assert_close(z, z[:, :1].expand_as(z))
            calls.append(1)
        g.register_forward_hook(observe)
        counts.append(calls)
        return g, d, prior, recipe
    monkeypatch.setattr(handoff, 'build', build)
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = handoff.Config(name=name, steps=steps, schedule_steps=10, batch_size=4,
                             max_prefix=8, context_noise=.03, memory_noise=.02,
                             predict_weight=float(mode == 'auxiliary'), temporal_weight=float(mode == 'auxiliary'),
                             feedback_probability=.5 if 'feedback' in mode else 0., feedback_ramp_steps=2,
                             recent_points=4 if mode == 'recent_feedback' else 0,
                             residual_output=mode == 'recent_feedback', output_bound=3. if mode == 'recent_feedback' else 0.,
                             feedback_backprop=mode == 'backprop_feedback',
                             clock_bands=3 if 'clock' in mode else 0,
                             clock_to_d=mode == 'clock_shared',
                             clock_origin_max=1024 if 'clock' in mode else 0,
                             eval_batch=4, eval_steps=256, resume=resume)
        handoff.train(cfg, out, 'cpu', lambda **kw: None)
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 2)
    run('first', 1)
    resumed = run('resumed', 2, str(tmp_path/'first/model.pt'))
    assert [len(c) for c in counts] == ([8, 4, 4] if 'feedback' in mode else [4, 2, 2])
    for key in ('generator', 'critic', 'prior'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)


def test_auxiliary_trains_writer_without_changing_gan_head_or_generator():
    cfg = handoff.Config(predict_weight=1, temporal_weight=1)
    g, d, prior, _ = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 3, 63], [0, 2, 5, 62]])
    memory = handoff.selected_memories(d.writer, real, positions, 63)
    target = real[torch.arange(2)[:, None], positions].flatten(0, 1)
    loss, metrics = handoff.local_auxiliary(cfg, d, memory, real, positions, target)
    loss.backward()
    assert all(torch.isfinite(v) for v in metrics.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.writer.parameters())
    assert all(p.grad is None for p in d.handoff_head.parameters())
    assert all(p.grad is None for p in g.parameters())
    assert prior.z.grad is None
    # With no identifiable contexts, both tasks are exactly absent.
    memory = handoff.selected_memories(d.writer, real, positions*0, 63)
    loss, metrics = handoff.local_auxiliary(cfg, d, memory, real, positions*0, target)
    assert loss.item() == 0 and all(v.item() == 0 for v in metrics.values())


@pytest.mark.parametrize('recent_points', [0, 4])
def test_feedback_causality_detach_and_writer_gradients(recent_points):
    cfg = handoff.Config(max_prefix=8, feedback_probability=1., recent_points=recent_points,
                         residual_output=bool(recent_points), output_bound=3. if recent_points else 0.)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 4, 8], [2, 3, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    jitter = torch.zeros(8, 8, 4)
    memory = handoff.training_memory(cfg, g, d, real, positions, z, jitter,
                                    torch.ones_like(positions, dtype=torch.bool), 1)
    previous = core.context(d.writer, real[:1, :3])
    with torch.no_grad():
        replacement, _ = g(z[2:3], previous)
    expected = d.writer.write(previous, replacement)
    torch.testing.assert_close(memory[2:3], expected)
    assert memory[0].count_nonzero() == 0  # Never feed back into empty context.
    memory[2].sum().backward(retain_graph=True)
    assert real.grad[0, :3].abs().sum() > 0
    assert real.grad[0, 3:].count_nonzero() == 0  # Last real input replaced; no target leak.
    assert all(p.grad is None for p in g.parameters()) and prior.z.grad is None
    d.zero_grad(set_to_none=True)
    with torch.no_grad():
        d.handoff_head[-1].weight.mul_(1000)
        fake, _ = g(z, memory)
    target = real.detach()[torch.arange(2)[:, None], positions].flatten(0, 1)
    view = handoff.CandidateView(d, memory)
    penalty = recipe.make_gradient_penalty()(view, target, fake, step=1)
    assert penalty.item() > 0
    (recipe.make_loss().d_loss(view(target), view(fake))+penalty).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.writer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert all(p.grad is None for p in g.parameters()) and prior.z.grad is None
    d.zero_grad(set_to_none=True)
    with frozen(d):
        memory = handoff.training_memory(cfg, g, d, real.detach(), positions, z, jitter,
                                        torch.ones_like(positions, dtype=torch.bool), 1)
        fake, _ = g(z, memory)
        d.score_candidate(fake, memory).mean().backward()
    assert all(p.grad is None for p in d.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
    assert prior.z.grad.abs().sum() > 0


def test_zero_strength_feedback_equals_teacher_context_and_mask_is_respected():
    cfg = handoff.Config(max_prefix=8, feedback_probability=1., feedback_strength=0.)
    g, d, prior, _ = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2)
    positions = torch.tensor([[0, 1, 4, 8], [2, 3, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    jitter = torch.zeros(8, 8, 4)
    ordinary = handoff.selected_memories(d.writer, real, positions, 8)
    feedback = handoff.training_memory(cfg, g, d, real, positions, z, jitter,
                                      torch.ones_like(positions, dtype=torch.bool), 1)
    torch.testing.assert_close(feedback, ordinary)
    cfg.feedback_strength = 1.
    cfg.feedback_min_prefix = 4
    feedback = handoff.training_memory(cfg, g, d, real, positions, z, jitter,
                                      torch.ones_like(positions, dtype=torch.bool), 1)
    torch.testing.assert_close(feedback[positions.flatten() < 4], ordinary[positions.flatten() < 4])


def test_optional_feedback_gradient_reaches_proposal_but_not_frozen_writer():
    cfg = handoff.Config(max_prefix=8, feedback_probability=1., feedback_backprop=True)
    g, d, prior, _ = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2)
    positions = torch.tensor([[0, 1, 4, 8], [2, 3, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    jitter = torch.zeros(8, 8, 4)
    mask = torch.ones_like(positions, dtype=torch.bool)
    with frozen(d):
        detached = handoff.training_memory(cfg, g, d, real, positions, z, jitter, mask, 1)
        assert not detached.requires_grad
        connected = handoff.training_memory(cfg, g, d, real, positions, z, jitter, mask, 1, proposal_grad=True)
        connected.square().sum().backward()
    assert all(p.grad is None for p in d.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())
    assert prior.z.grad.abs().sum() > 0


def test_paired_generator_loss_cancels_shared_context_score_offsets():
    from particlegan.gan_loss import GANLoss
    # A context-only score bias must not create a spurious route for G to win.
    context_bias = torch.randn(5, requires_grad=True)
    fake = torch.randn(5, 2, requires_grad=True)
    real = torch.randn(5, 2)
    view = lambda point: point.sum(-1)+context_bias
    handoff.generator_loss(GANLoss(), view, real, fake).backward()
    torch.testing.assert_close(context_bias.grad, torch.zeros_like(context_bias))
    assert fake.grad.abs().sum() > 0
