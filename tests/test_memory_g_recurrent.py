import json

import pytest
import torch

from experiments import memory_handoff_scout as h
from experiments import memory_g_recurrent as r
from experiments import memory_core_scout as core
from experiments import memory_scout as base
from experiments.autonomous_memory import frozen


def config(**kw):
    return h.Config(**dict(dict(g_state_dim=8, adversarial_only=True, clock_bands=3,
        max_prefix=8, feedback_probability=.5, feedback_strength=.25, feedback_backprop=True,
        feedback_judge_memory='mixed', local_pair_weight=.25, g_memory_adapter='proposal'), **kw))


@pytest.mark.parametrize('reads_d', [False, True])
def test_causal_states_observation_updates_and_no_read_mutation(reads_d):
    torch.set_num_threads(1)
    cfg = config(g_state_reads_d=reads_d)
    g, d, prior, _ = h.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 4, 8], [0, 2, 5, 7]])
    memory, state = r.selected_states(g, d.writer, real, positions, 8, True)
    state[2].sum().backward(retain_graph=True)
    assert real.grad[0, :4].abs().sum() > 0
    assert real.grad[0, 4:].count_nonzero() == 0
    assert all(p.grad is None for p in d.parameters())
    assert state[0].count_nonzero() == memory[0].count_nonzero() == 0
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    before = state.clone()
    calls = []
    hook = g.state_cell.register_forward_hook(lambda *a: calls.append(1))
    x, after = g(z, memory, state, time_index=positions.flatten())
    hook.remove()
    torch.testing.assert_close(before, after)
    assert not calls  # two proposal/final reads, zero state writes
    with pytest.raises(ValueError):
        g(z, memory, time_index=0)
    # A batched selected prefix is identical to an independently encoded prefix.
    m, s = d.writer.initial(real[:1]), g.initial_state(real[:1])
    for point in real[:1, :4].unbind(1):
        s = g.write_state(s, point, m)
        m = d.writer.write(m, point)
    torch.testing.assert_close(s[0], state[2])
    torch.testing.assert_close(m[0], memory[2])


@pytest.mark.parametrize('reads_d', [False, True])
def test_pair_gradients_ownership_and_exact_bcap(reads_d):
    cfg = config(g_state_reads_d=reads_d)
    g, d, prior, recipe = h.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2)
    positions = torch.tensor([[0, 1, 4, 8], [0, 2, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    view, actual, fake = h.local_pair_examples(cfg, g, d, real, real, positions, z, positions.flatten())
    assert not fake.requires_grad
    with torch.no_grad():
        d.pair_head[-1].weight.mul_(1000)
    pen = recipe.make_gradient_penalty()(view, actual, fake, step=1)
    assert pen.item() > 0
    (recipe.make_loss().d_loss(view(actual), view(fake))+pen).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.writer.parameters())
    assert all(p.grad is None for p in g.parameters())
    d.zero_grad(set_to_none=True)
    captured = []
    def capture(module, args, output):
        output[0].retain_grad()
        captured.append(output[0])
    hook = g.register_forward_hook(capture)
    with frozen(d):
        _, _, fake = h.local_pair_examples(cfg, g, d, real, real, positions, z, positions.flatten(), True)
        fake[:, 1].sum().backward()
    hook.remove()
    assert len(captured) == 2 and captured[0].grad.abs().sum() > 0
    assert all(p.grad is None for p in d.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.state_cell.parameters())
    assert sum(p.grad.abs().sum() for p in g.state_cell.parameters()) > 0
    assert prior.z.grad.abs().sum() > 0


def test_feedback_state_matches_written_observation_and_zero_prefix():
    cfg = config(feedback_probability=1., feedback_strength=1., feedback_min_prefix=1)
    g, d, prior, _ = h.build(cfg, 'cpu')
    real = torch.randn(1, 64, 2)
    positions = torch.tensor([[0, 1, 4, 8]])
    z = prior(torch.zeros(4, dtype=torch.long))
    jitter = torch.zeros(4, 8, 4)
    mask = torch.ones_like(positions, dtype=torch.bool)
    m, judge, s = h.training_contexts(cfg, g, d, real, positions, z, jitter, mask, 1,
        proposal_grad=True, return_state=True, state_grad=True)
    prev, prev_s = r.selected_states(g, d.writer, real, (positions-1).clamp_min(0), 8, True)
    proposal, _ = g(z, prev, prev_s, time_index=(positions-1).clamp_min(0).flatten())
    torch.testing.assert_close(m[1:], d.writer.write(prev, proposal)[1:])
    torch.testing.assert_close(s[1:], g.write_state(prev_s, proposal, prev)[1:])
    assert m[0].count_nonzero() == s[0].count_nonzero() == 0
    torch.testing.assert_close(judge, h.selected_memories(d.writer, real, positions, 8))


def test_runtime_matches_manual_loop_and_no_d_access():
    cfg = config(g_use_d_memory=False)
    g, d, prior, _ = h.build(cfg, 'cpu')
    z = prior(torch.arange(4))
    prefix = torch.randn(4, 5, 2)
    with torch.no_grad():
        m, s = d.writer.initial(z), g.initial_state(z)
        for point in prefix.unbind(1):
            s, m = g.write_state(s, point, m), d.writer.write(m, point)
        expected = []
        for t in range(3):
            x, _ = g(z, m, s, time_index=5+t)
            expected.append(x)
            s, m = g.write_state(s, x, m), d.writer.write(m, x)
        actual, _ = core.continuation(g, d.writer, z, prefix, 3)
        torch.testing.assert_close(actual, torch.stack(expected, 1))
        altered, _ = core.continuation(g, d.writer, z, prefix, 3, 'shuffle')
        torch.testing.assert_close(actual, altered, rtol=0, atol=0)
        cold, _ = base.rollout(g, d.writer, z, 3)
        empty, _ = core.continuation(g, d.writer, z, prefix[:, :0], 3)
        torch.testing.assert_close(cold, empty, rtol=0, atol=0)


def test_cold_second_point_gradient_reaches_first_through_g_state_alone():
    cfg = config(g_use_d_memory=False)
    g, d, prior, _ = h.build(cfg, 'cpu')
    real = torch.randn(1, 64, 2)
    positions = torch.zeros(1, 4, dtype=torch.long)
    z = prior(torch.zeros(4, dtype=torch.long))
    captured = []
    def capture(module, args, output):
        output[0].retain_grad()
        captured.append(output[0])
    hook = g.register_forward_hook(capture)
    with frozen(d):
        _, _, fake = h.local_pair_examples(cfg, g, d, real, real, positions,
            z, positions.flatten(), proposal_grad=True)
        fake[:, 1].sum().backward()
    hook.remove()
    assert len(captured) == 2 and captured[0].grad.abs().sum() > 0
    assert g.state_cell.weight_ih.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.parameters())


@pytest.mark.parametrize('reads_d', [False, True])
def test_exact_resume_no_mse_and_state_call_budget(tmp_path, monkeypatch, reads_d):
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({'generated_256': {}}, {}))
    def reject(*args, **kwargs):
        raise AssertionError('MSE training forbidden')
    monkeypatch.setattr(h.F, 'mse_loss', reject)
    original = h.build
    counts = []
    def build(cfg, device):
        g, d, p, recipe = original(cfg, device)
        calls = []
        g.state_cell.register_forward_hook(lambda *a: calls.append(1))
        counts.append(calls)
        return g, d, p, recipe
    monkeypatch.setattr(h, 'build', build)
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = config(name=name, steps=steps, schedule_steps=10, batch_size=4,
            eval_batch=4, eval_steps=256, resume=resume, g_state_reads_d=reads_d)
        h.train(cfg, out, 'cpu', lambda **kw: None)
        meta = json.loads((out/'config.json').read_text())
        assert meta['max_sequential_generated_writes'] == 1
        assert meta['reader_calls_g_phase'] == 8
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 4)
    run('first', 2)
    resumed = run('resumed', 4, str(tmp_path/'first/model.pt'))
    assert [len(c) for c in counts] == [4*2*(8*2+2), 2*2*(8*2+2), 2*2*(8*2+2)]
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
