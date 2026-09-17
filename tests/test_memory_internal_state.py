import json

import pytest
import torch

from experiments import memory_handoff_scout as h
from experiments import memory_g_recurrent as r
from experiments import diagnose_memory_information as info
from experiments.autonomous_memory import frozen


def config(**kw):
    return h.Config(**dict(dict(g_state_dim=8, g_state_update='intent', g_use_d_memory=False,
        adversarial_only=True, clock_bands=3, max_prefix=8, feedback_probability=.5,
        feedback_strength=.25, feedback_backprop=True, feedback_judge_memory='mixed',
        local_pair_weight=.25, mismatch_weight=.25, mismatch_kind='shuffle',
        g_memory_adapter='proposal'), **kw))


@pytest.mark.parametrize('mode', ['embedded', 'intent', 'hybrid'])
def test_runtime_pure_read_final_features_and_no_d_access(mode):
    torch.set_num_threads(1)
    g, d, prior, _ = h.build(config(g_state_update=mode), 'cpu')
    z = prior(torch.arange(3))
    m, s = d.writer.initial(z), g.initial_state(z)
    calls = []
    hook = g.state_cell.register_forward_hook(lambda *args: calls.append(1))
    x, features = g(z, m, s, time_index=3)
    assert not calls
    assert features.shape == (3, 64)
    torch.testing.assert_close(g.net[-1](features), x)
    x2, f2 = g(z, torch.randn_like(m)*10, s, time_index=3)
    torch.testing.assert_close(x, x2, rtol=0, atol=0)
    torch.testing.assert_close(features, f2, rtol=0, atol=0)
    nxt = g.generated_state(s, x, m, features)
    assert len(calls) == 1
    hook.remove()
    # Emitted-sample perturbations cannot affect intent state; they must affect
    # the observation and hybrid variants. This is the experimental distinction.
    perturbed = g.generated_state(s, x+3, m, features)
    if mode == 'intent':
        torch.testing.assert_close(nxt, perturbed, rtol=0, atol=0)
    else:
        assert not torch.allclose(nxt, perturbed)
    prefix = torch.randn(3, 4, 2)
    with torch.no_grad():
        expected = []
        for point in prefix.unbind(1):
            s, m = g.write_state(s, point, m), d.writer.write(m, point)
        for t in range(3):
            x, f = g(z, m, s, time_index=4+t)
            expected.append(x)
            s, m = g.generated_state(s, x, m, f), d.writer.write(m, x)
        actual, _ = r.continuation(g, d.writer, z, prefix, 3)
        torch.testing.assert_close(actual, torch.stack(expected, 1))
        shuffled, _ = r.continuation(g, d.writer, z, prefix, 3, 'shuffle')
        torch.testing.assert_close(actual, shuffled, rtol=0, atol=0)
        # A deployment loop needs no D calls at all, including at handoff.
        dummy, s = z.new_zeros(len(z), 8, 4), g.initial_state(z)
        for point in prefix.unbind(1):
            s = g.write_state(s, point, dummy)
        d_free = []
        for t in range(3):
            x, f = g(z, dummy, s, time_index=4+t)
            s = g.generated_state(s, x, dummy, f)
            d_free.append(x)
        torch.testing.assert_close(actual, torch.stack(d_free, 1), rtol=0, atol=0)


def test_matched_initialization_and_mixed_feedback_endpoints():
    models = [h.build(config(g_state_update=mode), 'cpu')[0] for mode in ('embedded','intent','hybrid')]
    for other in models[1:]:
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
    g = models[1]
    s, m = torch.randn(3, 8), torch.randn(3, 8, 4)
    x, actual, features = torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 64)
    torch.testing.assert_close(g.generated_state(s,x,m,features,actual=actual,strength=0),
                               g.write_state(s,actual,m), rtol=0, atol=0)
    torch.testing.assert_close(g.generated_state(s,x,m,features,actual=actual,strength=1),
                               g.generated_state(s,x,m,features), rtol=0, atol=0)


def test_intent_second_point_trains_features_without_point_bottleneck():
    cfg = config(clock_to_d=True)
    g, d, prior, recipe = h.build(cfg, 'cpu')
    real = torch.randn(2, 64, 2)
    positions = torch.zeros(2, 4, dtype=torch.long)
    z = prior(torch.arange(2))[:, None].expand(-1,4,-1).flatten(0,1)
    captured = []
    def capture(module, args, output):
        output[0].retain_grad(); output[1].retain_grad()
        captured.append(output)
    hook = g.register_forward_hook(capture)
    with frozen(d):
        view, actual, fake = h.local_pair_examples(cfg,g,d,real,real,positions,z,positions.flatten(),True)
        fake[:,1].square().sum().backward()
    hook.remove()
    assert captured[0][1].grad.abs().sum() > 0
    assert captured[0][0].grad is None or captured[0][0].grad.count_nonzero() == 0
    assert g.state_cell.weight_ih.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.parameters())
    g.zero_grad(set_to_none=True); prior.zero_grad(set_to_none=True)
    view, actual, fake = h.local_pair_examples(cfg,g,d,real,real,positions,z,positions.flatten())
    with torch.no_grad():
        d.pair_head[-1].weight.mul_(1000)
    penalty = recipe.make_gradient_penalty()(view,actual,fake,step=1)
    assert penalty.item() > 0
    (recipe.make_loss().d_loss(view(actual),view(fake))+penalty).backward()
    assert all(p.grad is None for p in g.parameters())


@pytest.mark.parametrize('mode', ['embedded', 'intent', 'hybrid'])
def test_exact_resume_and_no_mse(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(h,'evaluate',lambda *a: ({'generated_256':{}},{}))
    def reject(*a, **k):
        raise AssertionError('No MSE training')
    monkeypatch.setattr(h.F,'mse_loss',reject)
    def run(name, steps, resume=None):
        out = tmp_path/name; out.mkdir()
        cfg = config(name=name,steps=steps,schedule_steps=10,batch_size=4,eval_batch=4,
                     eval_steps=256,resume=resume,g_state_update=mode,clock_to_d=True)
        h.train(cfg,out,'cpu',lambda **kw: None)
        metadata = json.loads((out/'config.json').read_text())
        assert metadata['max_sequential_generated_writes'] == 1
        assert metadata['g_recurrence']['d_memory_access'] is False
        return torch.load(out/'model.pt',weights_only=False)
    full = run('full',4)
    run('first',2)
    resumed = run('resumed',4,str(tmp_path/'first/model.pt'))
    for key in ('generator','critic','prior','rngs'):
        for name, value in full[key].items():
            torch.testing.assert_close(value,resumed[key][name],rtol=0,atol=0)


def test_information_collect_tracks_actual_g_state():
    g,d,p,_ = h.build(config(), 'cpu')
    x = torch.randn(3, 7, 2); ids = torch.arange(3)
    states,z = info.collect(g,d,p,x,ids,[0,1,3],prefix=4,memory_kind='Mg')
    altered = x.clone()
    altered[:,4:] += 100
    changed,_ = info.collect(g,d,p,altered,ids,[0,1,3],prefix=4,memory_kind='Mg')
    for depth in (0,1,3):
        torch.testing.assert_close(states['generated'][depth],changed['generated'][depth],rtol=0,atol=0)
    assert not torch.allclose(states['real'][3],changed['real'][3])
    with torch.no_grad():
        m,s = d.writer.initial(z),g.initial_state(z)
        for point in x[:,:4].unbind(1):
            s,m = g.write_state(s,point,m),d.writer.write(m,point)
        torch.testing.assert_close(states['generated'][0],s)
        for n in range(3):
            point,f = g(z,m,s,time_index=4+n)
            s,m = g.generated_state(s,point,m,f),d.writer.write(m,point)
        torch.testing.assert_close(states['generated'][3],s)
