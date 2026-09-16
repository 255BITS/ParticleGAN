import json

import numpy as np
import pytest
import torch

from experiments import memory_handoff_scout as handoff
from experiments.autonomous_memory import frozen
from experiments.memory_orbit_metrics import orbit_progress


def test_pair_causality_exact_penalty_and_gradient_ownership():
    torch.set_num_threads(1)
    cfg = handoff.Config(max_prefix=8, local_pair_weight=.5, adversarial_only=True)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 4, 8], [0, 2, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    view, actual, fake = handoff.local_pair_examples(cfg, g, d, real, real, positions, z, positions.flatten())
    assert actual.shape == fake.shape == (8, 2, 2)
    torch.testing.assert_close(actual[2], real[0, 3:5])
    # target4's pair is at 3,4, so its context must stop at real2.
    view.memory[2].sum().backward(retain_graph=True)
    assert real.grad[0, 2].abs().sum() > 0
    assert real.grad[0, 3:].count_nonzero() == 0
    assert not fake.requires_grad
    d.zero_grad(set_to_none=True)
    with torch.no_grad():
        d.pair_head[-1].weight.mul_(1000)
    pen = recipe.make_gradient_penalty()(view, actual, fake, step=1)
    assert pen.item() > 0
    (recipe.make_loss().d_loss(view(actual), view(fake))+pen).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.writer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert all(p.grad is None for p in g.parameters())
    d.zero_grad(set_to_none=True)
    calls = []
    def capture(module, args, output):
        output[0].retain_grad()
        calls.append(output[0])
    handle = g.register_forward_hook(capture)
    with frozen(d):
        view, actual, fake = handoff.local_pair_examples(cfg, g, d, real.detach(), real.detach(),
            positions, z, positions.flatten(), proposal_grad=True)
        # The second sample alone differentiates through first sample and frozen W.
        fake[:, 1].sum().backward()
    handle.remove()
    assert len(calls) == 2 and calls[0].grad.abs().sum() > 0
    assert all(p.grad is None for p in d.parameters())
    assert prior.z.grad.abs().sum() > 0
    zero_view, zero_real, zero_fake = handoff.local_pair_examples(cfg, g, d, real, real,
        torch.zeros_like(positions), z, torch.zeros_like(positions).flatten())
    assert zero_view.memory.count_nonzero() == 0
    torch.testing.assert_close(zero_real[0], real[0, :2])
    assert zero_fake.shape == (8, 2, 2)


def test_mixed_judging_and_random_strengths_use_expected_contexts():
    cfg = handoff.Config(max_prefix=8, feedback_probability=1, feedback_judge_memory='mixed',
                         feedback_strength_distribution='uniform', feedback_strength=.5)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2)
    positions = torch.tensor([[0, 1, 4, 8], [0, 2, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    jitter = torch.zeros(8, 8, 4)
    strengths = handoff.sample_feedback_strength(cfg, positions, torch.Generator().manual_seed(1))
    assert strengths.shape == (8, 1) and (strengths >= 0).all() and (strengths <= .5).all()
    memory, reference = handoff.training_contexts(cfg, g, d, real, positions, z, jitter,
        torch.ones_like(positions, dtype=torch.bool), 1, feedback_strength=strengths)
    expected = handoff.selected_memories(d.writer, real, positions, 8)
    torch.testing.assert_close(reference, expected)
    views = handoff.candidate_views(cfg, d, memory, reference, positions.flatten())
    assert [w for w, _ in views] == [.5, .5]
    assert views[0][1].memory is reference and views[1][1].memory is memory
    shared_grad = torch.autograd.grad(views[1][1](real[:, :4].flatten(0, 1)).sum(), d.writer.input.weight)[0]
    assert shared_grad.abs().sum() > 0
    mild = handoff.Config(feedback_probability=1, feedback_strength_distribution='mild_full', feedback_strength=.25)
    values = handoff.sample_feedback_strength(mild, torch.zeros(1000, 1), torch.Generator().manual_seed(3))
    assert set(values.flatten().tolist()) == {.25, 1.}


@pytest.mark.parametrize('distribution,pair_weight', [('uniform', 0.), ('mild_full', .5)])
def test_local_recovery_exact_resume_no_mse_and_call_budget(tmp_path, monkeypatch, distribution, pair_weight):
    torch.set_num_threads(1)
    monkeypatch.setattr(handoff, 'evaluate', lambda *a: ({'generated_256': {}}, {}))
    def reject(*args, **kwargs):
        raise AssertionError('MSE in training')
    monkeypatch.setattr(handoff.F, 'mse_loss', reject)
    original_build = handoff.build
    counts = []
    def build(cfg, device):
        g, d, prior, recipe = original_build(cfg, device)
        calls = []
        g.net.register_forward_hook(lambda *args: calls.append(1))
        counts.append(calls)
        return g, d, prior, recipe
    monkeypatch.setattr(handoff, 'build', build)
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = handoff.Config(name=name, steps=steps, schedule_steps=10, batch_size=4,
            max_prefix=8, clock_bands=3, g_memory_adapter='proposal', feedback_probability=.5,
            feedback_strength=.25, feedback_ramp_steps=2, feedback_backprop=True,
            feedback_judge_memory='mixed', feedback_strength_distribution=distribution,
            local_pair_weight=pair_weight, adversarial_only=True, eval_batch=4,
            eval_steps=256, resume=resume)
        handoff.train(cfg, out, 'cpu', lambda **kw: None)
        config = json.loads((out/'config.json').read_text())
        assert config['max_sequential_generated_writes'] == 1
        assert config['reader_calls_g_phase'] == (8 if pair_weight else 4)
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 4)  # includes the scheduled exact B-cap step
    run('first', 2)
    resumed = run('resumed', 4, str(tmp_path/'first/model.pt'))
    factor = 2 if pair_weight else 1
    assert [len(c) for c in counts] == [32*factor, 16*factor, 16*factor]
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)


def test_orbit_progress_rewards_correct_motion_and_detects_drift():
    t = np.arange(1100)
    center = np.array([.3, -.4])
    clean = np.stack((np.cos(.2*t), np.sin(.2*t)), -1)[None]+center
    perfect = orbit_progress(clean[:, 32:1056], clean, 32)
    assert perfect['quality'] == pytest.approx(1)
    assert perfect['good_step_fraction'] == 1
    assert perfect['initial_good_steps_mean'] == 1024
    assert perfect['longest_good_arc_turns_mean'] == pytest.approx(1024*.2/(2*np.pi))
    stopped = np.repeat(clean[:, 32:33], 1024, 1)
    assert orbit_progress(stopped, clean, 32)['quality'] < .03
    reversed_path = np.stack((np.cos(.2*(32-t[:1024])), np.sin(.2*(32-t[:1024]))), -1)[None]+center
    assert orbit_progress(reversed_path, clean, 32)['quality'] < .01
    radial = center+(clean[:, 32:1056]-center)*1.1
    assert orbit_progress(radial, clean, 32)['quality'] == pytest.approx(.5)
    drift = clean[:, 32:1056].copy()
    drift[:, 128:] += 2
    result = orbit_progress(drift, clean, 32)
    assert result['quality_first32'] == pytest.approx(1)
    assert result['quality_last256'] < .1
    assert result['initial_good_steps_mean'] == 128
    assert orbit_progress(clean[:, :1024])['quality'] == pytest.approx(1)
    assert orbit_progress(stopped)['quality'] < .01
    tiny = center+(clean[:, :1024]-center)*.01
    assert orbit_progress(tiny)['quality'] < .05
