#!/usr/bin/env python
"""Read-only live-weight gradient probe for completed intervention checkpoints."""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.diagnose_cifar_ae_plateau import gradient_probe
from experiments.train_cifar_ae_capacity import build_models, rng, write_json
from particlegan import MoGParticlePrior, GANLoss, ParticleRegularizer
from torchvision.datasets import CIFAR10

def prior_probe(g,d,e,prior,images,cfg):
    g.requires_grad_(False);d.requires_grad_(False);e.requires_grad_(False)
    prior.requires_grad_(True)
    stream=rng(cfg['seed']+41000)
    rows=[]
    regularizer=ParticleRegularizer()
    sg=torch.autograd.grad(regularizer(prior.z),prior.z)[0]
    for _ in range(16):
        x=images[torch.randint(len(images),(64,),device='cuda',generator=stream)].float()/127.5-1
        z,_=prior.sample(64,stream)
        with torch.no_grad():dr=d(x)
        adv=GANLoss().g_loss(d(g(z)),dr)
        # Measure the hypothetical reconstruction force on the prior, even if
        # this checkpoint's training config has since detached that path.
        code=e(x,prior.means(),prior.sigma,cfg['temperature'])[0]
        rec=(g(code)-x).square().mean()*cfg['recon_weight']
        ag=torch.autograd.grad(adv,prior.z)[0]
        rg=torch.autograd.grad(rec,prior.z)[0]
        rows.append({'adversarial_norm':float(ag.norm()),'reconstruction_norm':float(rg.norm()),
                     'spread_norm':float(sg.norm()),'recon_over_adv':float(rg.norm()/ag.norm().clamp_min(1e-20)),
                     'recon_adv_cosine':float(torch.nn.functional.cosine_similarity(rg.flatten(),ag.flatten(),dim=0)),
                     'recon_nonzero_rows':int((rg.norm(dim=1)>0).sum())})
    return {'mean':{k:sum(r[k] for r in rows)/len(rows) for k in rows[0]},'batches':rows,
            'note':'Potential reconstruction gradient before detaching the prior; not necessarily an applied training gradient.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for name,expected in ck['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    cfg=ck['config']
    g,d,e=[m.cuda().eval() for m in build_models(cfg)]
    prior=MoGParticlePrior(num_particles=cfg['num_particles'],z_dim=cfg['z_dim'],
                           sigma_rel=cfg['sigma_rel'],generator=rng(cfg['seed']+1,'cpu')).cuda()
    for m,name in [(g,'G'),(d,'D'),(e,'E'),(prior,'prior')]:
        m.load_state_dict(ck[name]);m.requires_grad_(False)
    x=torch.from_numpy(CIFAR10(cfg['data_dir'],train=True).data).permute(0,3,1,2).contiguous().cuda()
    result=gradient_probe(g,d,e,prior,x,cfg)
    result['prior_probe']=prior_probe(g,d,e,prior,x,cfg)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==digest
    result.update(checkpoint=str(a.checkpoint),sha256=digest,step=ck['step'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.out,result)
    print(result['mean'],flush=True)
    print('PRIOR',result['prior_probe']['mean'],flush=True)
