import json

import pytest
import torch

from experiments.memory_scout import Config, build, rollout, train


@pytest.mark.parametrize("writer", ["trace", "gru", "delay"])
def test_writer_ownership_and_exact_bcap(writer):
    cfg = Config(writer=writer, critic="multiscale", differences=True)
    g, d, prior, recipe = build(cfg, "cpu")
    z = prior(torch.arange(3))
    path, _ = rollout(g, d.writer, z, 64)
    path[:, -1].square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in g.parameters())
    assert prior.z.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.writer.parameters())
    g.zero_grad(set_to_none=True)
    real = torch.randn_like(path)
    (d(path.detach()).mean()+recipe.make_gradient_penalty()(d, real, path.detach(), step=1)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in d.writer.parameters())
    assert all(p.grad is None for p in g.parameters())


def test_isolated_initialization_and_cold_rollout():
    a = build(Config(), "cpu")
    b = build(Config(critic="multiscale"), "cpu")
    for left, right in ((a[0], b[0]), (a[1].writer, b[1].writer), (a[2], b[2])):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key])
    g, d, prior, _ = a
    z = prior(torch.arange(3))
    path, _ = rollout(g, d.writer, z, 8)
    again, _ = rollout(g, d.writer, z, 8)
    torch.testing.assert_close(path, again)
    for i in range(3):
        single, _ = rollout(g, d.writer, z[i:i+1], 8)
        torch.testing.assert_close(path[i:i+1], single)
    static, _ = rollout(g, d.writer, z, 8, intervention="zero")
    torch.testing.assert_close(static, static[:, :1].expand_as(static))


@pytest.mark.parametrize("writer_lr_mult", [1.0, .1])
def test_resume_matches_uninterrupted(tmp_path, writer_lr_mult):
    torch.set_num_threads(1)
    def run(name, steps, resume=None):
        out = tmp_path/name
        out.mkdir()
        cfg = Config(name=name, steps=steps, schedule_steps=10, batch_size=4,
                     eval_batch=4, eval_steps=256, resume=resume, writer_lr_mult=writer_lr_mult)
        train(cfg, out, "cpu", lambda **kw: None)
        return torch.load(out/"model.pt", weights_only=False)
    full = run("full", 2)
    run("first", 1)
    resumed = run("resumed", 2, str(tmp_path/"first/model.pt"))
    for key in ("generator", "critic", "prior"):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)


def test_private_recurrence_control_can_move_without_memory():
    g, d, prior, _ = build(Config(g_private=True, read_memory=False, recipe={"z_dim": 8}), "cpu")
    z = prior(torch.arange(3))
    path, _ = rollout(g, d.writer, z, 8)
    zeroed, _ = rollout(g, d.writer, z, 8, intervention="zero")
    torch.testing.assert_close(path, zeroed)
    assert (path[:, 1:]-path[:, :-1]).abs().sum() > 0
    path[:, -1].square().sum().backward()
    assert g.cell.weight_hh.grad.abs().sum() > 0
    assert all(p.grad is None for p in d.writer.parameters())


def test_particle_modulation_preserves_initial_function_and_gets_gradients():
    ordinary = build(Config(), "cpu")
    modulated = build(Config(g_film=True), "cpu")
    z = ordinary[2](torch.arange(3))
    path, _ = rollout(ordinary[0], ordinary[1].writer, z, 8)
    other, _ = rollout(modulated[0], modulated[1].writer, z, 8)
    torch.testing.assert_close(path, other)
    other[:, -1].square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in modulated[0].film.parameters())
    assert all(p.grad is None for p in modulated[1].writer.parameters())


@pytest.mark.parametrize("dimension", [8, 64])
def test_memory_size_supports_rollout_and_exact_bcap(dimension):
    g, d, prior, recipe = build(Config(writer="gru", memory_dim=dimension), "cpu")
    path, states = rollout(g, d.writer, prior(torch.arange(3)), 64, states=True)
    assert states.shape == (3, 64, dimension)
    path[:, -1].square().sum().backward()
    assert all(p.grad is None for p in d.writer.parameters())
    loss = d(path.detach()).mean()+recipe.make_gradient_penalty()(d, torch.randn_like(path), path.detach(), step=1)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in d.writer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)


@pytest.mark.parametrize("concat", [True, False])
def test_memory_film_reads_memory_without_writer_parameter_gradients(concat):
    g, d, prior, _ = build(Config(writer="gru", g_film=True, g_film_source="memory", g_memory_concat=concat), "cpu")
    z = prior(torch.arange(3))
    path, _ = rollout(g, d.writer, z, 8)
    path[:, -1].square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() for p in g.film.parameters())
    assert all(p.grad is None for p in d.writer.parameters())
    # FiLM starts as identity. Activate its weights to check the actual read path.
    with torch.no_grad():
        for layer in g.film:
            layer.weight.fill_(.1)
    memory = torch.randn(3, 8, 4, requires_grad=True)
    g(z.detach(), memory)[0].sum().backward()
    assert memory.grad.abs().sum() > 0
    zeroed, _ = rollout(g, d.writer, z, 8, intervention="zero")
    torch.testing.assert_close(zeroed, zeroed[:, :1].expand_as(zeroed))


@pytest.mark.parametrize("mapping,frequencies", [(True, 0), (False, 2), (True, 2)])
def test_mapped_fourier_film_preserves_start_and_transmits_memory_gradients(mapping, frequencies):
    baseline = build(Config(writer="gru"), "cpu")
    g, d, prior, _ = build(Config(writer="gru", g_film=True, g_film_source="memory",
                                 g_film_mapping=mapping, g_film_fourier=frequencies), "cpu")
    z = prior(torch.arange(3)).detach()
    expected, _ = rollout(baseline[0], baseline[1].writer, z, 8)
    actual, _ = rollout(g, d.writer, z, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        for layer in g.film:
            layer.weight.fill_(.01)
    memory = torch.randn(3, 8, 4, requires_grad=True)
    g(z, memory)[0].square().sum().backward()
    assert memory.grad.abs().sum() > 0 and torch.isfinite(memory.grad).all()
    if mapping:
        assert all(p.grad is not None and p.grad.abs().sum() for p in g.film_mapping.parameters())
    assert all(p.grad is None for p in d.writer.parameters())


def test_geometry_head_supports_exact_bcap_and_preserves_writer_gradients():
    g, d, prior, recipe = build(Config(writer="gru", geometry="oriented"), "cpu")
    fake, _ = rollout(g, d.writer, prior(torch.arange(3)), 64)
    penalty = recipe.make_gradient_penalty()(d, torch.randn_like(fake), fake.detach(), step=1)
    (d(fake.detach()).mean()+penalty).backward()
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() for p in d.writer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() for p in d.geometry_head.parameters())


def test_freezing_learned_writer_keeps_its_parameters_fixed(tmp_path):
    cfg = Config(steps=1, schedule_steps=10, batch_size=4, eval_batch=4, eval_steps=256)
    first = tmp_path/"first"
    first.mkdir()
    train(cfg, first, "cpu", lambda **kw: None)
    before = torch.load(first/"model.pt", weights_only=False)
    cfg.steps, cfg.freeze_writer_at, cfg.resume = 2, 1, str(first/"model.pt")
    after_dir = tmp_path/"after"
    after_dir.mkdir()
    train(cfg, after_dir, "cpu", lambda **kw: None)
    after = torch.load(after_dir/"model.pt", weights_only=False)
    for key, value in before["writer"].items():
        torch.testing.assert_close(value, after["writer"][key], rtol=0, atol=0)
    assert any(not torch.equal(v, after["generator"][k]) for k, v in before["generator"].items())
