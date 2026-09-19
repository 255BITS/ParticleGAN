"""Checkpoint expansion preserves learned functions, RNG and existing Adam state."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments import train_cifar_ae_sagan as old
from experiments import train_cifar_ae_sagan_depth as new
from tests.test_cifar_ae_sagan import compare_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def parent():
    record = json.loads((ROOT / 'reports/cifar-particle-ae/sagan_gd_16k_200k/CHECKPOINTS.json').read_text())['final']
    cfg = json.loads((ROOT / 'reports/cifar-particle-ae/sagan_gd_16k_200k/results.json').read_text())[0]['config']
    cfg = {**cfg, 'g_attention_depth': 2, 'd_attention_depth': 2, 'steps': 240000, 'out_dir': 'runs/cifar_particle_ae/depth_test',
           'resume_checkpoint': record['path'], 'resume_sha256': record['sha256']}
    ck, audit = new.load_resume(cfg)
    assert audit['interventions'] == {k: {'before': 1, 'after': 2} for k in ('g_attention_depth', 'd_attention_depth')}
    return cfg, ck


def test_base_initialization_and_rng_unchanged(parent):
    cfg, _ = parent
    torch.manual_seed(cfg['seed'])
    before = old.build_models({k: v for k, v in cfg.items() if k not in ('g_attention_depth', 'd_attention_depth')})
    state = torch.get_rng_state().clone()
    torch.manual_seed(cfg['seed'])
    after = new.build_models(cfg)
    assert torch.equal(state, torch.get_rng_state())
    for a, b in zip(before, after):
        bs = b.state_dict()
        compare_state(a.state_dict(), {k: bs[k] for k in a.state_dict()})
    assert sum(p.numel() for p in after[0].parameters()) == 936003


def test_real_checkpoint_identity_live_and_ema_and_new_block_learns(parent):
    cfg, ck = parent
    g, _, _ = new.build_models(cfg)
    base = copy.deepcopy(g)
    del base.attention_extra
    for key in ('G', 'ema_G'):
        base.load_state_dict(ck[key])
        result = g.load_state_dict(ck[key], strict=False)
        assert len(result.missing_keys) == 4 and not result.unexpected_keys
        z = torch.randn(2, cfg['z_dim'], requires_grad=True)
        a, b = base(z), g(z)
        assert torch.equal(a, b)
        ga = torch.autograd.grad(a.square().mean(), z)[0]
        gb = torch.autograd.grad(b.square().mean(), z)[0]
        assert torch.equal(ga, gb)
    optimizer = torch.optim.Adam(g.attention_extra.parameters(), lr=.0003)
    for step in range(2):
        optimizer.zero_grad()
        g(torch.randn(2, cfg['z_dim'])).square().mean().backward()
        for name, p in g.attention_extra.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            if step == 1 or name == 'project.weight':
                assert p.grad.abs().sum() > 0, name
        optimizer.step()


def test_adam_migration_preserves_all_existing_slots_and_next_updates(parent):
    cfg, ck = parent
    g, _, e = new.build_models(cfg)
    base = copy.deepcopy(g)
    del base.attention_extra
    old_e = copy.deepcopy(e)
    prior = torch.nn.Parameter(torch.empty_like(ck['prior']['z']))
    old_prior = torch.nn.Parameter(torch.empty_like(prior))
    def optim(gen, enc, particle):
        return torch.optim.Adam([{'params': gen.parameters()},
                                 {'params': [p for p in enc.parameters() if p.requires_grad]},
                                 {'params': [particle]}], lr=.0003)
    before = optim(base, old_e, old_prior)
    before.load_state_dict(copy.deepcopy(ck['optimizer_g']))
    after = optim(g, e, prior)
    snapshot = copy.deepcopy(ck)
    audit = new.grow_checkpoint(ck, g, after)
    assert audit['added_parameter_count'] == 5120
    after.load_state_dict(ck['optimizer_g'])
    for name in snapshot:
        if name not in ('G', 'ema_G', 'optimizer_g'):
            compare_state(snapshot[name], ck[name])
    for name in ('G', 'ema_G'):
        compare_state(snapshot[name], {k: ck[name][k] for k in snapshot[name]})
    pairs = list(zip(base.parameters(), list(g.parameters())[:-4]))
    pairs += list(zip([p for p in old_e.parameters() if p.requires_grad],
                      [p for p in e.parameters() if p.requires_grad]))
    pairs += [(old_prior, prior)]
    for a, b in pairs:
        compare_state(before.state[a], after.state[b])
        with torch.no_grad(): a.zero_(); b.zero_()
        a.grad = torch.full_like(a, .001)
        b.grad = a.grad.clone()
    assert all(not after.state[p] for p in g.attention_extra.parameters())
    for optimizer in (before, after):
        for group in optimizer.param_groups: group['fused'] = False
        optimizer.step()
    for a, b in pairs:
        assert torch.equal(a, b)
        compare_state(before.state[a], after.state[b])


def test_reject_depth_removal_and_other_recipe_changes(parent, tmp_path):
    cfg, ck = parent
    new.validate(cfg)
    with pytest.raises(ValueError, match='resume cannot change'):
        new.load_resume({**cfg, 'fixed_sigma': .3})
    ck['config']['g_attention_depth'] = 2
    path = tmp_path / 'grown.pt'
    torch.save(ck, path)
    with pytest.raises(ValueError, match='cannot remove'):
        new.load_resume({**cfg, 'g_attention_depth': 1, 'resume_checkpoint': str(path),
                         'resume_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


def test_real_discriminator_identity_input_gradient_and_adam(parent):
    cfg, ck = parent
    _, d, _ = new.build_models(cfg)
    base = copy.deepcopy(d)
    del base.critic.pixel.attention_extra
    base.load_state_dict(ck['D'])
    result = d.load_state_dict(ck['D'], strict=False)
    assert len(result.missing_keys) == 4 and not result.unexpected_keys
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    a, b = base(x), d(x)
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a.sum(), x)[0]
    gb = torch.autograd.grad(b.sum(), x)[0]
    assert torch.equal(ga, gb)
    assert all(not p.requires_grad for p in d.critic.features.parameters())
    before = torch.optim.Adam([p for p in base.parameters() if p.requires_grad])
    before.load_state_dict(copy.deepcopy(ck['optimizer_d']))
    after = torch.optim.Adam([p for p in d.parameters() if p.requires_grad])
    original = copy.deepcopy(ck)
    audit = new.grow_checkpoint(ck, d, after, 'D')
    assert audit['added_parameter_count'] == 5120
    after.load_state_dict(ck['optimizer_d'])
    for key in original:
        if key not in ('D', 'optimizer_d'):
            compare_state(original[key], ck[key])
    params = dict(d.named_parameters())
    for name, p in base.named_parameters():
        if p.requires_grad:
            compare_state(before.state[p], after.state[params[name]])
    assert all(not after.state[p] for p in d.critic.pixel.attention_extra.parameters())


@pytest.mark.skipif(__import__('os').environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='CUDA double backward')
def test_grown_discriminator_active_double_backward():
    cfg = {**new.DEFAULTS, 'width': 8, 'z_dim': 8}
    _, d, _ = new.build_models(cfg)
    d = d.cuda().requires_grad_(True)
    with torch.no_grad():
        d.critic.pixel.output.weight.mul_(100.)
        # A learned nonzero added projection must support second derivatives too.
        d.critic.pixel.attention_extra.project.weight.normal_(0, .02)
    real, fake = [torch.randn(2, 3, 32, 32, device='cuda') for _ in range(2)]
    penalty = new.GradientPenalty(coeff=1., lazy_k=8)(d, real, fake, step=8)
    assert torch.isfinite(penalty) and penalty > 0
    penalty.backward()
    for p in d.critic.pixel.attention_extra.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
