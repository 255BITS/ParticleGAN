"""Opposing inference/generation joints at one fixed physical event."""
import json

import pytest
import torch

from experiments import memory_handoff_scout as h
from experiments import memory_gibbs as gibbs
from experiments.autonomous_memory import frozen
from experiments.memory_recent import LocalReader
from particlegan import get_recipe


def config(**kw):
    return h.Config(**dict(dict(adversarial_only=True, clock_bands=3,
        max_prefix=8, feedback_probability=.5, feedback_strength=.25,
        feedback_backprop=True, feedback_judge_memory='mixed',
        local_pair_weight=.25, g_memory_adapter='proposal', mismatch_weight=.25,
        mismatch_kind='shuffle', gibbs_latent_dim=8, gibbs_steps=3,
        gibbs_weight=.1), **kw))


def inputs(prior):
    observed = torch.randn(3, 64, 2)
    positions = torch.tensor([[0, 4, 7, 8], [1, 5, 6, 8], [2, 4, 5, 8]])
    z = prior(torch.arange(3))[:, None].expand(-1, 4, -1).flatten(0, 1)
    times = positions.flatten()+19
    target = observed[torch.arange(3)[:, None], positions].flatten(0, 1)
    return observed, positions, z, times, target


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def assert_tree_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        assert a == b


def test_gibbs_default_off_retains_original_generator():
    cfg = config(gibbs_latent_dim=0, gibbs_weight=0)
    g, _, _, _ = h.build(cfg, 'cpu')
    assert type(g) is LocalReader
    torch.manual_seed(42)
    legacy = LocalReader(cfg)
    assert_tree_equal(g.state_dict(), legacy.state_dict())


def test_gibbs_critic_initialization_preserves_rng():
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    gibbs.GibbsCritic(config())
    torch.testing.assert_close(torch.get_rng_state(), cpu, rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda else [], cuda):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('steps', [1, 3])
def test_gibbs_inner_chain_fixed_context_producer_pair_and_truncation(steps, monkeypatch):
    torch.set_num_threads(1)
    cfg = config(gibbs_steps=steps)
    g, d, prior, _ = h.build(cfg, 'cpu')
    z = prior(torch.arange(3))
    memory = torch.randn(3, cfg.memory_dim//4, 4, requires_grad=True)
    original_memory = memory.detach().clone()
    times = torch.tensor([4, 17, 31])
    calls = []
    decode, infer = g.decode, g.infer

    def record_decode(z_arg, m_arg, latent, *, time_index=None):
        result = decode(z_arg, m_arg, latent, time_index=time_index)
        calls.append(('decode', z_arg, m_arg, time_index, latent, result[0], torch.is_grad_enabled()))
        return result

    def record_infer(m_arg, point, *, time_index=None):
        result = infer(m_arg, point, time_index=time_index)
        calls.append(('infer', None, m_arg, time_index, point, result, torch.is_grad_enabled()))
        return result

    monkeypatch.setattr(g, 'decode', record_decode)
    monkeypatch.setattr(g, 'infer', record_infer)
    point, latent = g.joint(z, memory, time_index=times)
    decodes = [call for call in calls if call[0] == 'decode']
    inferences = [call for call in calls if call[0] == 'infer']
    assert len(decodes) == steps and len(inferences) == steps-1
    for call in calls:
        assert call[2] is memory
        torch.testing.assert_close(call[3], times, rtol=0, atol=0)
    for call in decodes:
        assert call[1] is z
    assert decodes[-1][4] is latent  # before final decode, not inference after it
    assert decodes[-1][5] is point
    assert decodes[-1][-1] and point.requires_grad and latent.requires_grad
    assert all(not call[-1] for call in decodes[:-1])
    if inferences:
        assert inferences[-1][-1]
        assert not inferences[-1][4].requires_grad
        assert all(not call[-1] for call in inferences[:-1])
    torch.testing.assert_close(memory, original_memory, rtol=0, atol=0)
    point.square().sum().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert has_grad(g) and has_grad(prior) and not has_grad(d)
    with torch.no_grad():
        actual, _ = g(z, memory, time_index=times)
    torch.testing.assert_close(actual, point.detach(), rtol=0, atol=0)


@pytest.mark.parametrize('condition', [True, False])
def test_gibbs_examples_opposing_joints_and_no_future_leak(condition):
    cfg = config(gibbs_condition=condition)
    g, d, prior, _ = h.build(cfg, 'cpu')
    k = gibbs.GibbsCritic(cfg)
    observed, positions, z, times, target = inputs(prior)
    batch = gibbs.examples(cfg, g, d, k, observed, positions, z, times, target, 'critic')
    eligible = positions.flatten() >= 4
    memory = h.selected_memories(d.writer, observed, positions, cfg.max_prefix)[eligible]
    with torch.no_grad():
        inferred = g.infer(memory, target[eligible], time_index=times[eligible])
        point, producer = g.joint(z[eligible], memory, time_index=times[eligible])
    torch.testing.assert_close(batch.anchor, memory, rtol=0, atol=0)
    torch.testing.assert_close(batch.real, k.candidate(inferred, target[eligible]), rtol=0, atol=0)
    torch.testing.assert_close(batch.fake, k.candidate(producer, point), rtol=0, atol=0)
    assert batch.real.shape[1] == cfg.gibbs_latent_dim+2
    assert not any(x.requires_grad for x in (batch.anchor, batch.real, batch.fake))
    positions[:] = 4
    times[:] = 23
    original = gibbs.examples(cfg, g, d, k, observed, positions, z, times, target, 'critic')
    altered = observed.clone()
    altered[:, 4:] += 100
    modified = gibbs.examples(cfg, g, d, k, altered, positions, z, times, target, 'critic')
    for field in ('anchor', 'real', 'fake'):
        torch.testing.assert_close(getattr(original, field), getattr(modified, field), rtol=0, atol=0)


def test_gibbs_critic_default_active_bcap_only_trains_k():
    cfg = config()
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = gibbs.GibbsCritic(cfg)
    batch = gibbs.examples(cfg, g, d, k, *inputs(prior), 'critic')
    with torch.no_grad():
        k.head[-1].weight.mul_(1000)
    penalty, default = recipe.make_gradient_penalty(), get_recipe().make_gradient_penalty()
    for name in ('arm', 'coeff', 'kappa', 'lazy_k', 'method'):
        assert getattr(penalty, name) == getattr(default, name)
    adv, reg, _ = gibbs.critic_objective(recipe.make_loss(), penalty, k, batch, 1)
    assert reg > 0
    (adv+reg).backward()
    assert has_grad(k) and not has_grad(g) and not has_grad(d) and not has_grad(prior)
    assert all(torch.isfinite(p.grad).all() for p in k.parameters() if p.grad is not None)


def test_gibbs_cooperative_objective_keeps_both_joints_connected():
    cfg = config()
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = gibbs.GibbsCritic(cfg)
    observed, positions, z, times, target = inputs(prior)
    with frozen(d), frozen(k):
        batch = gibbs.examples(cfg, g, d, k, observed, positions, z, times, target, 'generator')
        assert not batch.anchor.requires_grad
        assert batch.real.requires_grad and batch.fake.requires_grad
        batch.real.retain_grad()
        batch.fake.retain_grad()
        loss = gibbs.generator_objective(recipe.make_loss(), k, batch)
        loss.backward()
    assert batch.real.grad is not None and batch.real.grad.abs().sum() > 0
    assert batch.fake.grad is not None and batch.fake.grad.abs().sum() > 0
    assert has_grad(g) and has_grad(prior) and not has_grad(d) and not has_grad(k)


def run_training(tmp_path, name, steps=4, **kw):
    out = tmp_path/name
    out.mkdir()
    cfg = config(name=name, steps=steps, schedule_steps=10, batch_size=4,
        eval_batch=4, eval_steps=256, **kw)
    events = []
    h.train(cfg, out, 'cpu', lambda **row: events.append(row))
    return torch.load(out/'model.pt', weights_only=False), json.loads((out/'config.json').read_text()), events


def test_gibbs_exact_resume_no_mse_and_metadata(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))

    def reject(*a, **kw):
        raise AssertionError('MSE training objective forbidden')

    monkeypatch.setattr(h.F, 'mse_loss', reject)
    full, meta, _ = run_training(tmp_path, 'full')
    run_training(tmp_path, 'first', steps=2)
    resumed, _, _ = run_training(tmp_path, 'resumed', resume=str(tmp_path/'first/model.pt'))
    for key in full:
        if key != 'config':
            assert_tree_equal(full[key], resumed[key])
    assert meta['max_sequential_generated_writes'] == 1
    assert meta['gradient_clipping'] is None and not meta['ema']
