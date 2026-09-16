import json

import pytest
import torch

from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core
from experiments.autonomous_memory import frozen
from experiments.diagnose_memory_dynamics import bypass_adapter


def test_clean_judging_history_is_causal_and_distinct_from_explored_read():
    cfg = handoff.Config(max_prefix=8, feedback_probability=1., feedback_judge_memory='clean',
                         feedback_backprop=True, clock_bands=3, adversarial_only=True)
    g, d, prior, recipe = handoff.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 4, 8], [2, 3, 5, 7]])
    z = prior(torch.arange(2))[:, None].expand(-1, 4, -1).flatten(0, 1)
    jitter = torch.zeros(8, 8, 4)
    mask = torch.ones_like(positions, dtype=torch.bool)
    memory, reference = handoff.training_contexts(cfg, g, d, real, positions, z, jitter, mask, 1)
    expected = handoff.selected_memories(d.writer, real, positions, 8)
    torch.testing.assert_close(reference, expected)
    assert not torch.allclose(memory[2], reference[2])
    assert memory[0].count_nonzero() == reference[0].count_nonzero() == 0
    # Reference for target4 includes real3, never real4 or anything later.
    reference[2].sum().backward()
    assert real.grad[0, 3].abs().sum() > 0
    assert real.grad[0, 4:].count_nonzero() == 0
    assert real.grad[1].count_nonzero() == 0
    assert all(p.grad is None for p in g.parameters())
    d.zero_grad(set_to_none=True)
    # The explored branch replaces real3. Its gradient reaches the earlier G
    # proposal through W, while reference is independent of that proposal.
    calls = []
    def capture(module, args, output):
        output[0].retain_grad()
        calls.append(output[0])
    handle = g.register_forward_hook(capture)
    with frozen(d):
        memory, reference = handoff.training_contexts(cfg, g, d, real.detach(), positions,
            z, jitter, mask, 1, proposal_grad=True)
        assert not reference.requires_grad
        fake, _ = handoff.local_point(g, z, memory, positions.flatten())
        target = real.detach()[torch.arange(2)[:, None], positions].flatten(0, 1)
        handoff.generator_loss(recipe.make_loss(), handoff.CandidateView(d, reference), target, fake).backward()
    handle.remove()
    assert len(calls) == 2 and calls[0].grad[2:].abs().sum() > 0
    assert all(p.grad is None for p in d.parameters())
    assert prior.z.grad.abs().sum() > 0


def test_proposal_adapter_identity_bypass_and_sample_conditioning():
    cfg = handoff.Config(clock_bands=3, g_memory_adapter='proposal', adversarial_only=True)
    g, d, _, _ = handoff.build(cfg, 'cpu')
    control, _, _, _ = handoff.build(handoff.Config(clock_bands=3), 'cpu')
    z, memory = torch.randn(4, 4), torch.randn(4, 8, 4)
    before = memory.clone()
    torch.testing.assert_close(g(z, memory, time_index=4)[0], control(z, memory, time_index=4)[0], rtol=0, atol=0)
    torch.testing.assert_close(g.readable_memory(z, memory, time_index=4), memory, rtol=0, atol=0)
    with pytest.raises(ValueError, match='Proposal adapter'):
        g.translate_memory(memory)
    with torch.no_grad():
        g.memory_adapter[-1].weight.normal_(std=.2)
    proposal = control(z, memory, time_index=4)[0]
    assert not torch.allclose(g.translate_memory(memory, proposal), g.translate_memory(memory, proposal+1))
    calls = []
    handle = g.net.register_forward_hook(lambda *args: calls.append(1))
    g(z, memory, time_index=4)[0].sum().backward()
    assert len(calls) == 2
    assert any(p.grad is not None and p.grad.abs().sum() for p in g.memory_adapter.parameters())
    with bypass_adapter(g):
        torch.testing.assert_close(g(z, memory, time_index=4)[0], control(z, memory, time_index=4)[0], rtol=0, atol=0)
    handle.remove()
    torch.testing.assert_close(memory, before, rtol=0, atol=0)
    assert all(p.grad is None for p in d.parameters())


@pytest.mark.parametrize('adapter', ['none', 'proposal'])
def test_clean_exploration_resume_counts_and_no_mse_objective(tmp_path, monkeypatch, adapter):
    torch.set_num_threads(1)
    monkeypatch.setattr(handoff, 'evaluate', lambda *a: ({'generated_256': {}}, {}))
    def reject(*args, **kwargs):
        raise AssertionError('MSE was called during adversarial-only training')
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
            max_prefix=8, clock_bands=3, clock_origin_max=100, g_memory_adapter=adapter,
            feedback_probability=.5, feedback_strength=.25, feedback_ramp_steps=2,
            feedback_backprop=True, feedback_judge_memory='clean', adversarial_only=True,
            eval_batch=4, eval_steps=256, resume=resume)
        handoff.train(cfg, out, 'cpu', lambda **kw: None)
        resolved = json.loads((out/'config.json').read_text())
        assert resolved['reader_calls_g_phase'] == (4 if adapter == 'proposal' else 2)
        assert resolved['fake_writes_g_phase'] == 1
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 2)
    run('first', 1)
    resumed = run('resumed', 2, str(tmp_path/'first/model.pt'))
    factor = 2 if adapter == 'proposal' else 1
    assert [len(c) for c in counts] == [8*factor, 4*factor, 4*factor]
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)


def test_adversarial_only_rejects_auxiliaries():
    for field in ('predict_weight', 'temporal_weight', 'repair_weight', 'stability_g_weight', 'stability_d_weight'):
        with pytest.raises(AssertionError):
            handoff.Config(adversarial_only=True, g_memory_adapter='residual', **{field: 1.})
