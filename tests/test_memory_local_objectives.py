import json

import pytest
import torch

from experiments import memory_handoff_scout as h
from experiments import memory_local_objectives as o
from experiments.autonomous_memory import frozen


def config(**kw):
    return h.Config(**dict(dict(adversarial_only=True, clock_bands=3,
        max_prefix=8, feedback_probability=.5, feedback_strength=.25,
        feedback_backprop=True, feedback_judge_memory='mixed',
        local_pair_weight=.25, g_memory_adapter='proposal'), **kw))


def inputs(g, prior):
    real = torch.randn(3, 64, 2)
    pos = torch.tensor([[0, 1, 4, 8], [0, 2, 5, 7], [0, 3, 6, 8]])
    z = prior(torch.arange(3))[:, None].expand(-1, 4, -1).flatten(0, 1)
    return real, pos, z


@pytest.mark.parametrize('kind', ['nearest', 'shuffle'])
def test_mismatched_donors_different_episode_causal_and_writer_gradients(kind):
    torch.set_num_threads(1)
    cfg = config(mismatch_weight=.25, mismatch_kind=kind)
    g, d, p, recipe = h.build(cfg, 'cpu')
    real, pos, z = inputs(g, p)
    valid, donor = o.mismatch_indices(real, pos, kind)
    assert (valid//4 != donor//4).all()
    assert (pos.flatten()[valid] >= 4).all()
    # Targets never affect matching for a single selected position per episode.
    single = torch.full((3, 1), 4)
    a = o.mismatch_indices(real, single, kind)
    altered = real.clone()
    altered[:, 4:] += 1000
    b = o.mismatch_indices(altered, single, kind)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    with torch.no_grad():
        d.handoff_head[-1].weight.mul_(1000)
    loss, penalty, stats = o.mismatch_objective(cfg, d, real, real, pos, pos.flatten(),
        recipe.make_loss(), recipe.make_gradient_penalty(), 1)
    assert penalty.item() > 0
    (loss+penalty).backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in d.writer.parameters())
    assert all(v.grad is None for v in g.parameters())


def test_recovery_clean_judge_and_two_outputs_one_write():
    cfg = config(recovery_noise=.2)
    g, d, p, recipe = h.build(cfg, 'cpu')
    real, pos, z = inputs(g, p)
    noise = torch.randn_like(real)*.2
    captures = []
    def capture(module, args, output):
        output[0].retain_grad()
        captures.append((args[1].detach().clone(), output[0]))
    hook = g.register_forward_hook(capture)
    with frozen(d):
        view, actual, fake = h.local_pair_examples(cfg, g, d, real, real, pos, z,
            pos.flatten(), True, recovery_noise=noise)
        expected = h.selected_memories(d.writer, real, (pos-1).clamp_min(0), 8)
        dirty = h.selected_memories(d.writer, real+noise, (pos-1).clamp_min(0), 8)
        torch.testing.assert_close(view.memory, expected)
        torch.testing.assert_close(captures[0][0], dirty)
        torch.testing.assert_close(captures[1][0], d.writer.write(dirty, captures[0][1]))
        fake[:, 1].sum().backward()
    hook.remove()
    assert len(captures) == 2 and captures[0][1].grad.abs().sum() > 0
    assert all(v.grad is None for v in d.parameters())
    assert captures[0][0][0].count_nonzero() == 0


def test_future_queries_causal_independent_reads_bcap_and_runtime():
    cfg = config(future_weight=.25, future_query_bands=4, max_prefix=63)
    g, d, p, recipe = h.build(cfg, 'cpu')
    real, pos, z = inputs(g, p)
    pos[:, -1] = 63
    memories = []
    hook = g.register_forward_hook(lambda m, args, out: memories.append(args[1]))
    view, actual, fake = o.future_examples(cfg, g, d, real, real, pos, z, pos.flatten())
    hook.remove()
    assert len(memories) == 3 and all(v is memories[0] for v in memories)
    rows = torch.arange(3)[:, None]
    torch.testing.assert_close(actual[:, 2], real[rows, pos.clamp_max(51)+12].flatten(0, 1))
    assert not fake.requires_grad
    assert g.query_particle(z)[:, z.shape[1]:].count_nonzero() == 0
    normal = g(z, view.memory, time_index=view.times)[0]
    queried = g(z, view.memory, time_index=view.times, query_offset=0)[0]
    torch.testing.assert_close(normal, queried, rtol=0, atol=0)
    with torch.no_grad():
        d.future_head[-1].weight.mul_(1000)
    penalty = recipe.make_gradient_penalty()(view, actual, fake, step=1)
    assert penalty.item() > 0
    (recipe.make_loss().d_loss(view(actual), view(fake))+penalty).backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in d.writer.parameters())
    d.zero_grad(set_to_none=True)
    with frozen(d):
        view, actual, fake = o.future_examples(cfg, g, d, real, real, pos, z, pos.flatten(), True)
        h.generator_loss(recipe.make_loss(), view, actual, fake).backward()
    assert all(v.grad is None for v in d.parameters())
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in g.parameters())


def test_future_branch_cannot_read_its_targets():
    cfg = config(future_weight=.1, future_query_bands=4)
    g, d, p, _ = h.build(cfg, 'cpu')
    real, pos, z = inputs(g, p)
    pos.fill_(8)
    view, actual, fake = o.future_examples(cfg, g, d, real, real, pos, z, pos.flatten())
    altered = real.clone()
    altered[:, 8:] += 100
    changed_view, changed_actual, changed_fake = o.future_examples(
        cfg, g, d, altered, altered, pos, z, pos.flatten())
    torch.testing.assert_close(view.memory, changed_view.memory, rtol=0, atol=0)
    torch.testing.assert_close(fake, changed_fake, rtol=0, atol=0)
    torch.testing.assert_close(changed_actual-actual, torch.full_like(actual, 100))


def test_diagnostic_gradient_direction_and_no_parameter_accumulation():
    from experiments.diagnose_memory_local_signal import candidate_gradient
    class LinearScore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.direction = torch.nn.Parameter(torch.tensor([1., 0.]))

        def score_candidate(self, x, memory, times):
            return (x*self.direction).sum(-1)
    critic = LinearScore()
    with torch.no_grad():
        row = candidate_gradient(critic, torch.zeros(2, 8, 4), torch.zeros(2),
            torch.zeros(2, 2), torch.tensor([[2., 0.], [-2., 0.]]))
    assert row == dict(mean_cosine_toward_target=0., positive_alignment_fraction=.5,
                       mean_gradient_norm=1.)
    assert critic.direction.grad is None


def test_all_objectives_exact_resume_no_mse(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))
    def reject(*a, **kw):
        raise AssertionError('MSE objective forbidden')
    monkeypatch.setattr(h.F, 'mse_loss', reject)
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = config(name=name, steps=steps, schedule_steps=10, batch_size=4,
            eval_batch=4, eval_steps=256, resume=resume, mismatch_weight=.25,
            recovery_noise=.1, future_weight=.1, future_query_bands=4)
        h.train(cfg, out, 'cpu', lambda **kw: None)
        meta = json.loads((out/'config.json').read_text())
        assert meta['max_sequential_generated_writes'] == 1
        assert meta['reader_calls_g_phase'] == 14
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 4)
    run('first', 2)
    resumed = run('resumed', 4, str(tmp_path/'first/model.pt'))
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
