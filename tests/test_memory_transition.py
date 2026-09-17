"""Local successor matching: causality, gradient ownership, and exact continuation."""
import json

import pytest
import torch

from experiments import memory_handoff_scout as h
from experiments import memory_transition as t
from experiments.autonomous_memory import frozen
from particlegan import get_recipe


def config(**kw):
    return h.Config(**dict(dict(adversarial_only=True, clock_bands=3,
        max_prefix=8, feedback_probability=.5, feedback_strength=.25,
        feedback_backprop=True, feedback_judge_memory='mixed',
        local_pair_weight=.25, g_memory_adapter='proposal', mismatch_weight=.25,
        mismatch_kind='shuffle', transition_weight=.1), **kw))


def inputs(prior):
    real = torch.randn(3, 64, 2)
    positions = torch.tensor([[0, 4, 7, 8], [1, 5, 6, 8], [2, 4, 5, 8]])
    z = prior(torch.arange(3))[:, None].expand(-1, 4, -1).flatten(0, 1)
    times = positions.flatten()+19
    target = real[torch.arange(3)[:, None], positions].flatten(0, 1)
    return real, positions, z, times, target


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_transition_initialization_preserves_cpu_and_cuda_rng():
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    t.TransitionCritic(config())
    torch.testing.assert_close(torch.get_rng_state(), cpu, rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda else [], cuda):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('condition', [True, False])
@pytest.mark.parametrize('include_x', [True, False])
def test_transition_causal_one_write_fixed_particle_and_target(condition, include_x):
    torch.set_num_threads(1)
    cfg = config(transition_condition=condition, transition_include_x=include_x)
    g, d, prior, _ = h.build(cfg, 'cpu')
    k = t.TransitionCritic(cfg)
    real, positions, z, times, target = inputs(prior)
    valid = positions.flatten() >= 4
    calls = []
    hook = g.register_forward_hook(lambda module, args, kwargs, output:
        calls.append((args[0].clone(), args[1].clone(), kwargs['time_index'].clone(), output[0])),
        with_kwargs=True)
    batch = t.examples(cfg, g, d, k, real, positions, z, times, target, 'critic')
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], z[valid])
    torch.testing.assert_close(calls[0][2], times[valid])
    expected = h.selected_memories(d.writer, real, positions, cfg.max_prefix)[valid]
    torch.testing.assert_close(calls[0][1], expected)
    torch.testing.assert_close(batch.anchor, expected)
    torch.testing.assert_close(batch.real, k.candidate(target[valid], d.writer.write(expected, target[valid])))
    torch.testing.assert_close(batch.fake, k.candidate(calls[0][3], d.writer.write(expected, calls[0][3])))
    assert batch.real.shape[1] == cfg.memory_dim+2*include_x
    assert not any(x.requires_grad for x in (batch.anchor, batch.real, batch.fake))
    # Alter target/future observations: the pre-write anchor and G proposal remain causal.
    altered = real.clone()
    # Use one common target index so no other selected anchor needs this observation.
    positions[:] = 4
    times[:] = 23
    original = t.examples(cfg, g, d, k, real, positions, z, times, target, 'critic')
    altered[:, 4:] += 100
    modified = t.examples(cfg, g, d, k, altered, positions, z, times, target, 'critic')
    torch.testing.assert_close(original.anchor, modified.anchor, rtol=0, atol=0)
    torch.testing.assert_close(original.fake, modified.fake, rtol=0, atol=0)
    assert t.examples(cfg, g, d, k, real, positions*0, z, times, target, 'critic') is None


def test_transition_critic_active_default_bcap_has_no_generator_or_writer_gradient():
    cfg = config()
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = t.TransitionCritic(cfg)
    real, positions, z, times, target = inputs(prior)
    batch = t.examples(cfg, g, d, k, real, positions, z, times, target, 'critic')
    with torch.no_grad():
        # Force a nonzero penalty, exercising its second derivative and ownership.
        list(k.modules())[-1].weight.mul_(1000)
    view = t.TransitionView(k, batch.anchor)
    penalty = recipe.make_gradient_penalty()
    default = get_recipe().make_gradient_penalty()
    for name in ('arm', 'coeff', 'kappa', 'lazy_k', 'method'):
        assert getattr(penalty, name) == getattr(default, name)
    regularizer = penalty(view, batch.real, batch.fake, step=1)
    assert regularizer > 0
    loss = recipe.make_loss().d_loss(view(batch.real), view(batch.fake))
    (loss+regularizer).backward()
    assert has_grad(k)
    assert not has_grad(d) and not has_grad(g) and not has_grad(prior)
    assert all(torch.isfinite(p.grad).all() for p in k.parameters() if p.grad is not None)


def test_transition_generator_receives_gradient_through_frozen_write():
    # Removing x from K prevents a direct sample path from hiding a detached write.
    cfg = config(transition_include_x=False)
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = t.TransitionCritic(cfg)
    real, positions, z, times, target = inputs(prior)
    with frozen(d), frozen(k):
        batch = t.examples(cfg, g, d, k, real, positions, z, times, target, 'generator')
        assert not batch.anchor.requires_grad and not batch.real.requires_grad
        assert batch.fake.requires_grad
        view = t.TransitionView(k, batch.anchor)
        recipe.make_loss().g_loss(view(batch.fake), view(batch.real)).backward()
    assert has_grad(g) and has_grad(prior)
    assert not has_grad(d) and not has_grad(k)


def test_transition_writer_alignment_only_trains_generated_write():
    cfg = config(transition_writer_weight=.1)
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = t.TransitionCritic(cfg)
    real, positions, z, times, target = inputs(prior)
    real.requires_grad_(True)
    target.requires_grad_(True)
    with frozen(k):
        batch = t.examples(cfg, g, d, k, real, positions, z, times, target, 'writer')
        assert not batch.anchor.requires_grad and not batch.real.requires_grad
        assert batch.fake.requires_grad
        view = t.TransitionView(k, batch.anchor)
        recipe.make_loss().g_loss(view(batch.fake), view(batch.real)).backward()
    assert has_grad(d.writer) and not has_grad(d.handoff_head)
    assert not has_grad(g) and not has_grad(prior) and not has_grad(k)
    assert real.grad is None and target.grad is None


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


def run_training(tmp_path, name, steps=4, **kw):
    out = tmp_path/name
    out.mkdir()
    cfg = config(name=name, steps=steps, schedule_steps=10, batch_size=4,
        eval_batch=4, eval_steps=256, **kw)
    events = []
    h.train(cfg, out, 'cpu', lambda **row: events.append(row))
    return torch.load(out/'model.pt', weights_only=False), json.loads((out/'config.json').read_text()), events


def test_transition_default_off_and_diagnostic_only_do_not_change_training(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))
    original, _, _ = run_training(tmp_path, 'off', transition_weight=0.)
    diagnostic, _, _ = run_training(tmp_path, 'diagnostic', transition_weight=0., transition_critic_only=True)
    for key in ('generator', 'critic', 'prior', 'opt_g', 'opt_d', 'rngs', 'torch_rng'):
        assert_tree_equal(original[key], diagnostic[key])


def test_transition_exact_resume_no_mse_and_metadata(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))
    def reject(*a, **kw):
        raise AssertionError('MSE training objective forbidden')
    monkeypatch.setattr(h.F, 'mse_loss', reject)
    settings = dict(transition_writer_weight=.1, transition_ramp_steps=3)
    full, meta, _ = run_training(tmp_path, 'full', **settings)
    run_training(tmp_path, 'first', steps=2, **settings)
    resumed, _, _ = run_training(tmp_path, 'resumed', resume=str(tmp_path/'first/model.pt'), **settings)
    for key in full:
        if key not in ('config',):
            assert_tree_equal(full[key], resumed[key])
    assert meta['max_sequential_generated_writes'] == 1
    assert meta['gradient_clipping'] is None and not meta['ema']


def test_transition_other_episode_donors_are_deterministic_and_exclude_same_episode():
    episodes = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3])
    donor = t.other_episode_donors(episodes)
    assert donor.shape == episodes.shape
    assert (episodes[donor] != episodes).all()
    assert torch.equal(donor, t.other_episode_donors(episodes.clone()))
    assert t.other_episode_donors(torch.zeros(4, dtype=torch.long)) is None
    assert t.other_episode_donors(torch.empty(0, dtype=torch.long)) is None


@pytest.mark.parametrize('space', ['memory', 'read'])
def test_transition_mismatch_exact_normalization_active_penalty_and_k_only(space):
    cfg = config(transition_space=space, transition_mismatch_weight=.25)
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = t.TransitionCritic(cfg)
    real, positions, z, times, target = inputs(prior)
    batch = t.examples(cfg, g, d, k, real, positions, z, times, target, 'critic')
    torch.testing.assert_close(batch.episodes, torch.arange(3)[:, None].expand_as(positions).flatten()[positions.flatten() >= 4])
    with torch.no_grad():
        k.head[-1].weight.mul_(1000)
    view = t.TransitionView(k, batch.anchor)
    gan, penalty = recipe.make_loss(), recipe.make_gradient_penalty()
    donor = t.other_episode_donors(batch.episodes)
    wrong = batch.real[donor]
    expected_adv = (gan.d_loss(view(batch.real), view(batch.fake))
        +.25*gan.d_loss(view(batch.real), view(wrong)))/1.25
    expected_reg = (penalty(view, batch.real, batch.fake, step=1)
        +.25*penalty(view, batch.real, wrong, step=1))/1.25
    adv, reg, _ = t.critic_objective(gan, penalty, k, batch, 1)
    torch.testing.assert_close(adv, expected_adv, rtol=0, atol=0)
    torch.testing.assert_close(reg, expected_reg, rtol=0, atol=0)
    assert reg > 0
    (adv+reg).backward()
    assert has_grad(k) and not has_grad(g) and not has_grad(d) and not has_grad(prior)


@pytest.mark.parametrize('mode', ['critic', 'generator', 'writer'])
def test_transition_read_is_causal_frozen_decoder_with_input_derivatives(mode):
    cfg = config(transition_space='read', transition_include_x=False)
    g, d, prior, recipe = h.build(cfg, 'cpu')
    k = t.TransitionCritic(cfg)
    real, positions, z, times, target = inputs(prior)
    calls = []
    def record(module, args, kwargs, output):
        calls.append({'z': args[0], 'memory': args[1], 'time': kwargs['time_index'],
                      'point': output[0], 'parameter_grad': any(p.requires_grad for p in module.parameters())})
    hook = g.register_forward_hook(record, with_kwargs=True)
    with frozen(k):
        if mode == 'generator':
            with frozen(d):
                batch = t.examples(cfg, g, d, k, real, positions, z, times, target, mode)
        else:
            batch = t.examples(cfg, g, d, k, real, positions, z, times, target, mode)
    hook.remove()
    assert len(calls) == 3  # One proposal; one read of each independent successor.
    valid = positions.flatten() >= 4
    proposal = [call for call in calls if torch.equal(call['time'], times[valid])]
    reads = [call for call in calls if torch.equal(call['time'], times[valid]+1)]
    assert len(proposal) == 1 and len(reads) == 2
    for call in calls:
        torch.testing.assert_close(call['z'], z[valid], rtol=0, atol=0)
    anchor = h.selected_memories(d.writer, real, positions, cfg.max_prefix)[valid]
    torch.testing.assert_close(proposal[0]['memory'], anchor)
    torch.testing.assert_close(reads[0]['memory'], d.writer.write(anchor, target[valid]))
    torch.testing.assert_close(reads[1]['memory'], d.writer.write(anchor, proposal[0]['point']))
    torch.testing.assert_close(batch.real, reads[0]['point'])
    torch.testing.assert_close(batch.fake, reads[1]['point'])
    assert batch.real.shape[1] == 2 and not batch.real.requires_grad
    assert not reads[1]['parameter_grad'] and not reads[1]['z'].requires_grad
    assert proposal[0]['point'].requires_grad == (mode == 'generator')
    assert batch.fake.requires_grad == (mode != 'critic')
    if mode != 'critic':
        with frozen(k):
            t.generator_objective(recipe.make_loss(), k, batch).backward()
        assert has_grad(g) == (mode == 'generator')
        assert has_grad(prior) == (mode == 'generator')
        assert has_grad(d.writer) == (mode == 'writer')
        assert not has_grad(k) and not has_grad(d.handoff_head)
    # No real t+1 observation enters the read target; it is G's own frozen read.
    positions[:] = 4
    times[:] = 23
    original = t.examples(cfg, g, d, k, real, positions, z, times, target, 'critic')
    altered = real.clone()
    altered[:, 4:] += 100
    changed = t.examples(cfg, g, d, k, altered, positions, z, times, target, 'critic')
    for attr in ('anchor', 'real', 'fake', 'episodes'):
        torch.testing.assert_close(getattr(original, attr), getattr(changed, attr), rtol=0, atol=0)


def test_transition_read_exact_resume_and_no_mse(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(h, 'evaluate', lambda *a: ({}, {}))
    def reject(*a, **kw):
        raise AssertionError('MSE training objective forbidden')
    monkeypatch.setattr(h.F, 'mse_loss', reject)
    settings = dict(transition_writer_weight=.03, transition_ramp_steps=3,
                    transition_space='read', transition_mismatch_weight=.25)
    full, meta, _ = run_training(tmp_path, 'full_read', **settings)
    run_training(tmp_path, 'first_read', steps=2, **settings)
    resumed, _, _ = run_training(tmp_path, 'resumed_read', resume=str(tmp_path/'first_read/model.pt'), **settings)
    for key in full:
        if key != 'config':
            assert_tree_equal(full[key], resumed[key])
    assert meta['max_sequential_generated_writes'] == 1
