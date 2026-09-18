#!/usr/bin/env python
"""Read-only gradient diagnosis respecting selective G-growth routing."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import train_cifar_ae_features as trainer
from particlegan import GANLoss, MoGParticlePrior
from torchvision.datasets import CIFAR10


def probe(path, images, batches=16):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    for name, expected in ck['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    cfg = {**trainer.DEFAULTS, **ck['config']}
    g, d, e = [m.cuda() for m in trainer.build_models(cfg)]
    eg = copy.deepcopy(g).eval().requires_grad_(False)
    trainer.install_backbone(d, cfg['d_backbone'])
    trainer.install_growth(cfg, g, d, eg)
    prior = MoGParticlePrior(cfg['num_particles'], cfg['z_dim'], sigma_rel=cfg['sigma_rel']).cuda()
    for module, key in ((g, 'G'), (d, 'D'), (e, 'E'), (prior, 'prior')):
        module.load_state_dict(ck[key], strict=True)
        module.eval()
    d.requires_grad_(False); e.requires_grad_(False); prior.requires_grad_(False)
    g.requires_grad_(True)
    params = list(g.parameters())
    stream = trainer.rng(cfg['seed'] + 40000)
    rows = []
    for _ in range(batches):
        ids = torch.randint(len(images), (64,), device='cuda', generator=stream)
        real = images[ids].float() / 127.5 - 1
        z, _ = prior.sample(64, stream)
        fake = g(z)
        with torch.no_grad():
            dr = d(real)
        df = d(fake)
        adv = GANLoss().g_loss(df, dr)
        ag = torch.autograd.grad(adv, params, retain_graph=True)
        input_grad, = torch.autograd.grad(df.sum(), fake)
        rec = trainer.reconstruction_loss(g, e, prior, real, cfg) * cfg['recon_weight']
        rg = torch.autograd.grad(rec, params, allow_unused=True)
        av = torch.cat([v.flatten() for v in ag])
        rv = torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p, v in zip(params, rg)])
        rows.append({'adv_G_norm': float(av.norm()), 'applied_rec_G_norm': float(rv.norm()),
                     'rec_over_adv': float(rv.norm() / av.norm().clamp_min(1e-20)),
                     'rec_adv_cosine': float(torch.nn.functional.cosine_similarity(av, rv, dim=0)),
                     'D_fake_input_grad_norm': float(input_grad.flatten(1).norm(dim=1).mean()),
                     'paired_real_above_fake': float((dr > df).float().mean()),
                     'real_minus_fake_mean': float((dr.mean() - df.mean()).detach()),
                     'g_loss': float(adv.detach())})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return {'checkpoint': str(path), 'sha256': digest, 'step': ck['step'],
            'weights': 'live G/D/E/prior, not EMA', 'batches': rows,
            'mean': {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
            'routing': {'new_G_reconstruction': cfg['recon_growth_grad']},
            'note': 'Applied G gradients respect routing. Reconstructions and prior gradients are not FID measurements.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    data = CIFAR10(str(ROOT / 'data'), train=True, download=False).data
    images = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().cuda()
    results = []
    for path in args.checkpoint:
        result = probe(path, images)
        print(path.parent.name, result['step'], json.dumps(result['mean']), flush=True)
        results.append(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_json(args.out, results)
