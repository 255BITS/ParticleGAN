"""Causal future ranking shares the runtime head without training rollouts."""
import json

import pytest
import torch

from experiments import memory_handoff_scout as h
from experiments import memory_local_objectives as o


def config(**kw):
    return h.Config(**dict(dict(adversarial_only=True, clock_bands=3,
        max_prefix=8, feedback_probability=.5, feedback_strength=.25,
        feedback_backprop=True, feedback_judge_memory='mixed',
        local_pair_weight=.25, g_memory_adapter='proposal', mismatch_weight=.25,
        mismatch_kind='shuffle', future_rank_weight=.1, future_rank_bands=4), **kw))


def inputs(prior):
    real = torch.randn(3, 64, 2)
    positions = torch.full((3, 4), 8)
    z = prior(torch.arange(3))[:, None].expand(-1, 4, -1).flatten(0, 1)
    return real, positions, z


@pytest.mark.parametrize('kind', ['shuffle', 'nearest'])
def test_future_anchor_and_donor_bounds_and_explicit_target_offsets(kind):
    torch.set_num_threads(1)
    cfg = config(max_prefix=63, mismatch_kind=kind)
    g, d, p, recipe = h.build(cfg, 'cpu')
    real, positions, z = inputs(p)
    positions[:] = torch.tensor([[0, 4, 51, 52], [3, 12, 50, 63], [4, 11, 52, 62]])
    valid, donor = o.mismatch_indices(real, positions, kind, max_position=51)
    assert valid.tolist() == [1, 2, 5, 6, 8, 9]
    assert set(donor.tolist()) <= set(valid.tolist())
    assert (valid//4 != donor//4).all()
    memory = h.selected_memories(d.writer, real, positions, 63)[valid]
    rows = torch.arange(3)[:, None].expand_as(positions).flatten()
    flat = positions.flatten()
    expected_loss, expected_reg = 0., 0.
    for offset in cfg.future_rank_offsets:
        view = o.FutureRankView(d, memory, flat[valid], offset)
        actual, wrong = real[rows[valid], flat[valid]+offset], real[rows[donor], flat[donor]+offset]
        expected_loss += recipe.make_loss().d_loss(view(actual), view(wrong))/2
        expected_reg += recipe.make_gradient_penalty()(view, actual, wrong, step=1)/2
    loss, reg, _ = o.future_rank_objective(cfg, g, d, real, real, positions, flat, z,
        recipe.make_loss(), recipe.make_gradient_penalty(), 1)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(reg, expected_reg, rtol=0, atol=0)


@pytest.mark.parametrize('explored_grad', [True, False])
@pytest.mark.parametrize('context', ['clean', 'mixed', 'explored'])
def test_future_context_causal_clock_particle_one_write_and_gradients(context, explored_grad):
    cfg = config(future_rank_context=context, future_rank_explored_grad=explored_grad,
                 future_rank_strength=.8, future_rank_ramp_steps=4)
    g, d, p, recipe = h.build(cfg, 'cpu')
    real, positions, z = inputs(p)
    times = positions.flatten()+19
    valid, _ = o.mismatch_indices(real, positions, cfg.mismatch_kind, max_position=51)
    calls = []
    hook = g.register_forward_hook(lambda module, args, kwargs, output:
        calls.append((args[0].clone(), args[1].clone(), kwargs['time_index'].clone(), output[0])),
        with_kwargs=True)
    contexts = o.future_rank_contexts(cfg, g, d, real, positions, times, z, valid, 2)
    hook.remove()
    assert sum(weight for weight, _ in contexts) == 1
    assert len(contexts) == (2 if context == 'mixed' else 1)
    assert len(calls) == (0 if context == 'clean' else 1)
    if calls:
        torch.testing.assert_close(calls[0][0], z[valid])
        torch.testing.assert_close(calls[0][2], times[valid]-1)
        assert not calls[0][3].requires_grad
        previous = h.selected_memories(d.writer, real, positions-1, 8)[valid]
        torch.testing.assert_close(calls[0][1], previous)
        expected = d.writer.write(previous,
            .6*real[:, 7:8].expand(-1, 4, -1).flatten(0, 1)+.4*calls[0][3])
        torch.testing.assert_close(contexts[-1][1], expected)
        assert contexts[-1][1].requires_grad == explored_grad
    if context != 'explored':
        assert contexts[0][1].requires_grad  # Clean remains connected in the control.
    altered = real.clone()
    altered[:, 8:] += 100
    changed = o.future_rank_contexts(cfg, g, d, altered, positions, times, z, valid, 2)
    for (_, original), (_, modified) in zip(contexts, changed):
        torch.testing.assert_close(original, modified, rtol=0, atol=0)
    assert all(torch.equal(a, b) for a, b in zip(
        o.mismatch_indices(real, positions, cfg.mismatch_kind, max_position=51),
        o.mismatch_indices(altered, positions, cfg.mismatch_kind, max_position=51)))
    with torch.no_grad():
        d.handoff_head[-1].weight.mul_(1000)
    loss, reg, _ = o.future_rank_objective(cfg, g, d, real, real, positions, times, z,
        recipe.make_loss(), recipe.make_gradient_penalty(), 2)
    assert reg.item() > 0
    (loss+reg).backward()
    has_grad = lambda module: any(v.grad is not None and v.grad.abs().sum() > 0 for v in module.parameters())
    assert has_grad(d.writer) == (context != 'explored' or explored_grad)
    assert has_grad(d.handoff_head) and has_grad(d.horizon_projection)
    assert not has_grad(g) and not has_grad(p)
    # Even explored-only future detachment never alters the original clean mismatch.
    d.zero_grad(set_to_none=True)
    immediate, _, _ = o.mismatch_objective(cfg, d, real, real, positions, times,
        recipe.make_loss(), recipe.make_gradient_penalty(), 2, generator=g, z=z)
    immediate.backward()
    assert has_grad(d.writer)


def test_zero_horizon_ignores_learned_projection_and_uses_original_point_head():
    cfg = config()
    g, d, p, _ = h.build(cfg, 'cpu')
    real, positions, z = inputs(p)
    memory = h.selected_memories(d.writer, real, positions, 8)
    candidate = real[:, :4].flatten(0, 1)
    expected = d.handoff_head(torch.cat((candidate, memory.flatten(1)), -1)).squeeze(-1)
    with torch.no_grad():
        d.horizon_projection.weight.normal_()
    actual = d.score_candidate(candidate, memory, positions.flatten(), horizon=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(d.score_candidate(candidate, memory, horizon=4), expected)


def test_horizon_zero_architecture_four_updates_bitwise_with_active_bcap(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))
    original_build = h.build
    def active_build(cfg, device):
        modules = original_build(cfg, device)
        with torch.no_grad():
            modules[1].handoff_head[-1].weight.mul_(1000)
        return modules
    monkeypatch.setattr(h, 'build', active_build)
    def run(name, bands):
        out = tmp_path/name
        out.mkdir()
        cfg = config(name=name, steps=4, schedule_steps=10, batch_size=4,
            eval_batch=4, eval_steps=256, future_rank_weight=0., future_rank_bands=bands)
        events = []
        h.train(cfg, out, 'cpu', lambda **kw: events.append(kw))
        assert any(row.get('penalty', 0) > 0 for row in events)
        return torch.load(out/'model.pt', weights_only=False)
    original, enabled = run('original', 0), run('enabled', 4)
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in original[key].items():
            torch.testing.assert_close(value, enabled[key][name], rtol=0, atol=0)
    torch.testing.assert_close(original['torch_rng'], enabled['torch_rng'], rtol=0, atol=0)
    assert enabled['critic']['horizon_projection.weight'].count_nonzero() == 0


def test_future_rank_exact_resume_no_mse_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))
    def reject(*a, **kw):
        raise AssertionError('MSE objective forbidden')
    monkeypatch.setattr(h.F, 'mse_loss', reject)
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = config(name=name, steps=steps, schedule_steps=10, batch_size=4,
            eval_batch=4, eval_steps=256, resume=resume, future_rank_context='mixed',
            future_rank_strength=1., future_rank_ramp_steps=3)
        h.train(cfg, out, 'cpu', lambda **kw: None)
        meta = json.loads((out/'config.json').read_text())
        assert meta['max_sequential_generated_writes'] == 1
        assert meta['reader_calls_g_phase'] == 8 and meta['reader_calls_d_phase'] == 10
        assert meta['local_objectives']['mismatch']['context'] == 'clean'
        assert meta['local_objectives']['future_rank']['context_weights'] == [.5, .5]
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 4)
    run('first', 2)
    resumed = run('resumed', 4, str(tmp_path/'first/model.pt'))
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
    torch.testing.assert_close(full['torch_rng'], resumed['torch_rng'], rtol=0, atol=0)
