import json
import pytest
import torch
from torch import nn

from experiments import memory_handoff_scout as handoff
from experiments.autonomous_memory import frozen
from experiments.memory_recent import SlowFastWriter
from experiments.diagnose_memory_dynamics import feedback_jacobian_metrics


def test_slow_writer_preserves_fast_updates_and_original_parameters():
    _, ordinary, _, _ = handoff.build(handoff.Config(), 'cpu')
    _, slowed, _, _ = handoff.build(handoff.Config(slow_dim=16, slow_rate=.1), 'cpu')
    assert isinstance(slowed.writer, SlowFastWriter)
    for key, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, slowed.state_dict()[key], rtol=0, atol=0)
    memory, point = torch.randn(4, 8, 4), torch.randn(4, 2)
    original = ordinary.writer.write(memory, point).flatten(1)
    result = slowed.writer.write(memory, point).flatten(1)
    torch.testing.assert_close(result[:, 16:], original[:, 16:])
    torch.testing.assert_close(result[:, :16]-memory.flatten(1)[:, :16],
                               .1*(original[:, :16]-memory.flatten(1)[:, :16]))


def test_adapter_initially_matches_control_and_repair_trains_only_adapter():
    cfg = handoff.Config(clock_bands=3, g_memory_adapter='residual', repair_weight=10.)
    g, d, prior, _ = handoff.build(cfg, 'cpu')
    control, _, _, _ = handoff.build(handoff.Config(clock_bands=3), 'cpu')
    z, memory = torch.randn(4, 4), torch.randn(4, 8, 4, requires_grad=True)
    before = memory.detach().clone()
    torch.testing.assert_close(g(z, memory, time_index=4)[0], control(z, memory, time_index=4)[0], rtol=0, atol=0)
    loss = handoff.memory_repair(cfg, g, memory, torch.tensor([[0, 1, 4, 8]]), torch.randn_like(memory))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in g.memory_adapter.parameters())
    assert all(p.grad is None for p in g.net.parameters())
    assert all(p.grad is None for p in d.parameters()) and memory.grad is None
    assert prior.z.grad is None
    torch.testing.assert_close(memory.detach(), before, rtol=0, atol=0)
    assert handoff.memory_repair(cfg, g, memory, torch.zeros(1, 4).long(), torch.randn_like(memory)) == 0


def test_local_feedback_gain_includes_reader_and_has_correct_gradient_ownership():
    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.))
        def forward(self, z, memory):
            return self.scale*memory.flatten(1), None
    class Writer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.))
        def write(self, memory, point):
            return self.scale*point.reshape_as(memory)
    g, d = Reader(), nn.Module()
    d.writer = Writer()
    cfg = handoff.Config(stability_g_weight=1, stability_d_weight=1, stability_max_gain=1.)
    memory = torch.randn(4, 1, 2, requires_grad=True)
    z = torch.randn(4, 4, requires_grad=True)
    positions, times = torch.tensor([[0, 4, 6, 8]]), torch.tensor([0, 4, 6, 8])
    noise = torch.randn_like(memory)
    jacobian = feedback_jacobian_metrics(g, d.writer, z, memory, 4)
    assert jacobian['largest_singular_value_mean'] == 2.
    assert g.scale.grad is None and d.writer.scale.grad is None
    with frozen(d):
        loss, stats = handoff.local_stability(cfg, g, d, z, memory, positions, times, noise)
        torch.testing.assert_close(stats['gain_rms'], torch.tensor(2.))
        torch.testing.assert_close(loss, torch.tensor(3.))
        loss.backward()
    assert g.scale.grad > 0 and d.writer.scale.grad is None
    assert memory.grad is None and z.grad is None
    g.zero_grad(set_to_none=True)
    with frozen(g):
        loss, _ = handoff.local_stability(cfg, g, d, z, memory, positions, times, noise)
        loss.backward()
    assert d.writer.scale.grad > 0 and g.scale.grad is None


@pytest.mark.parametrize('repair_target', ['raw', 'translated'])
def test_dynamics_resume_is_exact_and_old_checkpoints_remain_loadable(tmp_path, monkeypatch, repair_target):
    torch.set_num_threads(1)
    monkeypatch.setattr(handoff, 'evaluate', lambda *a: ({'generated_256': {}}, {}))
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = handoff.Config(name=name, steps=steps, schedule_steps=10, batch_size=4,
            max_prefix=8, clock_bands=3, slow_dim=16, g_memory_adapter='residual',
            repair_weight=10., repair_target=repair_target,
            stability_g_weight=.1, stability_d_weight=.1, stability_max_gain=.1,
            eval_batch=4, eval_steps=256, resume=resume)
        handoff.train(cfg, out, 'cpu', lambda **kw: None)
        resolved = json.loads((out/'config.json').read_text())
        assert resolved['fake_writes_d_phase'] == resolved['fake_writes_g_phase'] == 2
        assert resolved['training_generator_unroll'] == 1
        return torch.load(out/'model.pt', weights_only=False)
    full = run('full', 2)
    run('first', 1)
    resumed = run('resumed', 2, str(tmp_path/'first/model.pt'))
    for key in ('generator', 'critic', 'prior', 'rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)


def test_legacy_checkpoint_without_new_config_or_rng_fields_resumes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(handoff, 'evaluate', lambda *a: ({'generated_256': {}}, {}))
    cfg = handoff.Config(steps=1, batch_size=4, max_prefix=8, eval_batch=4, eval_steps=256)
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(), second.mkdir()
    handoff.train(cfg, first, 'cpu', lambda **kw: None)
    saved = torch.load(first/'model.pt', weights_only=False)
    new_fields = {'slow_dim', 'slow_rate', 'g_memory_adapter', 'adapter_width', 'adapter_bottleneck',
                  'repair_weight', 'repair_noise', 'repair_target', 'stability_g_weight',
                  'stability_d_weight', 'stability_noise', 'stability_max_gain', 'dynamics_min_prefix'}
    saved['config'] = {k:v for k,v in saved['config'].items() if k not in new_fields}
    del saved['rngs']['stability'], saved['rngs']['repair']
    torch.save(saved, first/'legacy.pt')
    cfg.steps, cfg.resume = 2, str(first/'legacy.pt')
    handoff.train(cfg, second, 'cpu', lambda **kw: None)
