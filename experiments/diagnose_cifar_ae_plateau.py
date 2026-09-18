#!/usr/bin/env python
"""Frozen-checkpoint sampling and training-gradient diagnostics. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar_ae_capacity import build_models, rng, generation, write_json
from lib.cifar_metrics import FIDEvaluator, PROTOCOL, uint8_images
from particlegan import GANLoss, MoGParticlePrior


def gradient_probe(g, d, e, prior, images, cfg, batches=16):
    g.requires_grad_(True)
    d.requires_grad_(False)
    e.requires_grad_(False)
    stream = rng(cfg['seed'] + 41000)
    rows = []
    params = list(g.parameters())
    for i in range(batches):
        x = images[torch.randint(len(images), (64,), generator=stream, device='cuda')].float() / 127.5 - 1
        z, _ = prior.sample(64, stream)
        fake = g(z)
        dr, df = d(x), d(fake)
        adv = GANLoss().g_loss(df, dr.detach())
        encoded = e(x, prior.means(), prior.sigma, cfg['temperature'])[0]
        rec = (g(encoded) - x).square().mean() * cfg['recon_weight']
        a = torch.cat([v.flatten() for v in torch.autograd.grad(adv, params)])
        r = torch.cat([v.flatten() for v in torch.autograd.grad(rec, params)])
        row = {'adv_g_norm': float(a.norm()), 'recon_g_norm': float(r.norm()),
               'recon_over_adv': float(r.norm() / a.norm().clamp_min(1e-20)),
               'gradient_cosine': float(torch.nn.functional.cosine_similarity(a, r, dim=0)),
               'real_score': float(dr.mean()), 'fake_score': float(df.mean()),
               'paired_accuracy': float((dr > df).float().mean()),
               'real_score_std': float(dr.std()), 'fake_score_std': float(df.std())}
        for label, data in [('real', x), ('fake', fake.detach())]:
            data = data.detach().requires_grad_(True)
            norm = torch.autograd.grad(d(data).sum(), data)[0].flatten(1).norm(dim=1)
            row[label + '_input_grad_norm'] = float(norm.mean())
            row[label + '_over_cap_fraction'] = float((norm > 1).float().mean())
        rows.append(row)
    g.requires_grad_(False)
    return {'batches': rows, 'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}


@torch.no_grad()
def encode(e, prior, images, cfg):
    ids, offsets = [], []
    for raw in images.split(cfg['eval_batch_size']):
        _, ix, u, _ = e(raw.cuda().float() / 127.5 - 1, prior.means(), prior.sigma, cfg['temperature'])
        ids.append(ix); offsets.append(u)
    return torch.cat(ids), torch.cat(offsets)


def main(args):
    from torchvision.datasets import CIFAR10
    from torchvision.utils import save_image
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    start = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for name, expected in ck['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    cfg = ck['config']
    g, d, e = [m.cuda().eval() for m in build_models(cfg)]
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
                             sigma_rel=cfg['sigma_rel'], generator=rng(cfg['seed']+1, 'cpu')).cuda()
    images = torch.from_numpy(CIFAR10(cfg['data_dir'], train=True).data).permute(0,3,1,2).contiguous()
    test = torch.from_numpy(CIFAR10(cfg['data_dir'], train=False).data).permute(0,3,1,2).contiguous()
    evaluator = FIDEvaluator(images, cfg['fid_cache'], cfg['eval_batch_size'])
    for m, name in [(g,'G'), (d,'D'), (e,'E'), (prior,'prior')]:
        m.load_state_dict(ck[name]); m.requires_grad_(False)
    print(f'START step={ck["step"]} checkpoint={args.checkpoint} samples={args.samples}', flush=True)
    probe = gradient_probe(g, d, e, prior, images.cuda(), cfg)
    write_json(out / 'gradients.json', probe)
    print('GRADIENTS ' + json.dumps(probe['mean']), flush=True)
    for m, name in [(g,'ema_G'), (e,'ema_E'), (prior,'ema_prior')]:
        m.load_state_dict(ck[name]); m.requires_grad_(False)
    ids, offsets = encode(e, prior, images, cfg)
    with torch.no_grad():
        count = torch.bincount(ids, minlength=prior.num_particles).float()
        p = count/count.sum()
        mean = offsets.mean(0)
        cov = torch.cov(offsets.T)
        chol = torch.linalg.cholesky(cov + torch.eye(cfg['z_dim'], device='cuda')*1e-5)
        sums = torch.zeros_like(prior.means()).index_add_(0, ids, offsets)
        squares = torch.zeros_like(sums).index_add_(0, ids, offsets.square())
        # Shrink sparse components toward global moments using 32 pseudo-observations.
        cm = (sums + 32*mean) / (count[:,None]+32)
        cv = ((squares + 32*offsets.square().mean(0)) / (count[:,None]+32) - cm.square()).clamp_min(1e-5)
        latent = {'train_count': len(images), 'used_particles': int((count>0).sum()),
                  'usage_tv': float((p-1/len(p)).abs().sum()/2),
                  'offset_rms': float(offsets.square().mean().sqrt()),
                  'offset_mean_rms': float(mean.square().mean().sqrt()),
                  'offset_cov_eigenvalues': torch.linalg.eigvalsh(cov).cpu().tolist()}
    write_json(out / 'latent.json', latent)
    results = []
    variants = ['prior', 'frequency_only', 'global_offsets_uniform', 'global_offsets_frequency',
                'conditional_diag_frequency', 'train_code_replay', 'test_reconstruction']
    with (out / 'metrics.jsonl').open('w', buffering=1) as log, torch.no_grad():
        for variant in variants:
            t = time.perf_counter()
            stream = rng(cfg['seed']+10000)
            chunks = []
            n = len(test) if variant == 'test_reconstruction' else args.samples
            if variant == 'test_reconstruction':
                ti, tu = encode(e, prior, test, cfg)
            for lo in range(0,n,cfg['eval_batch_size']):
                b = min(cfg['eval_batch_size'], n-lo)
                if variant == 'prior':
                    z, _ = prior.sample(b, stream)
                elif variant == 'train_code_replay':
                    ix = torch.randint(len(ids), (b,), device='cuda', generator=stream)
                    z = prior.means()[ids[ix]] + prior.sigma * offsets[ix]
                elif variant == 'test_reconstruction':
                    z = prior.means()[ti[lo:lo+b]] + prior.sigma * tu[lo:lo+b]
                else:
                    ix = (torch.randint(len(p),(b,),device='cuda',generator=stream)
                          if variant == 'global_offsets_uniform'
                          else torch.multinomial(p,b,replacement=True,generator=stream))
                    eps = torch.randn((b,cfg['z_dim']),device='cuda',generator=stream)
                    if variant.startswith('global_offsets'):
                        eps = mean + eps @ chol.T
                    elif variant == 'conditional_diag_frequency':
                        eps = cm[ix] + cv[ix].sqrt()*eps
                    z = prior.means()[ix] + prior.sigma*eps
                chunks.append(uint8_images(g(z)).cpu())
            generated = torch.cat(chunks)
            save_image(generated[:100].float()/255, out / f'{variant}.png', nrow=10)
            fid = evaluator(generated)
            row = {'variant': variant, 'fid': fid, 'samples': n, 'seconds': time.perf_counter()-t,
                   'diagnostic_only': variant != 'prior'}
            results.append(row); log.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == digest
    write_json(out / 'summary.json', {'checkpoint': str(args.checkpoint), 'sha256': digest,
               'checkpoint_unchanged': True, 'step': ck['step'], 'protocol': PROTOCOL,
               'results': results, 'gradients': probe['mean'], 'latent': latent,
               'seconds': time.perf_counter()-start,
               'interpretation': 'All altered samplers are diagnostics; replay/reconstruction are not unconditional benchmarks. Gaussian fits use train encodings only. Test reconstruction has 10k samples and is not directly comparable to FID50k.'})
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=50000)
    main(p.parse_args())
