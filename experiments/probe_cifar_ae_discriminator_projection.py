#!/usr/bin/env python
"""Read-only decomposition of actual GAN gradients after projection through G.

Each branch is weighted by the derivative of the combined logistic GAN loss,
not by a separate branch loss. Sum of contributions must recover total G/prior
parameter gradients. This distinguishes image-space from parameter-space cancellation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import diagnose_cifar_ae_discriminator as diag
from particlegan import GANLoss


def probe(path, batches):
    expected=diag.digest(path)
    ck,cfg,g,d,e,prior=diag.load_parent(path,expected)
    before=diag.base.state_hash([g,d,e,prior])
    array=CIFAR10(cfg['data_dir'],train=False,download=False).data
    images=torch.from_numpy(array).permute(0,3,1,2).contiguous().cuda()
    stream=diag.base.rng(cfg['seed']+80000)
    ids=torch.randperm(len(images),device='cuda',generator=stream)[:64*batches]
    g.requires_grad_(True);prior.requires_grad_(True)
    gp=list(g.parameters());pp=list(prior.parameters());rows=[]
    for lo in range(0,len(ids),64):
        real=images[ids[lo:lo+64]].float()/127.5-1
        z,_=prior.sample(len(real),stream)
        fake=g(z);scores=diag.branch_scores(d,fake)
        with torch.no_grad():dr=d(real)
        loss=GANLoss().g_loss(scores['total'],dr)
        weight,=torch.autograd.grad(loss,scores['total'],retain_graph=True)
        vectors={}
        for branch in ['total','pixel','features']:
            grad=torch.autograd.grad(scores[branch],gp+pp,grad_outputs=weight,retain_graph=True)
            vectors[branch]={'G':torch.cat([v.flatten() for v in grad[:len(gp)]]),
                             'prior':torch.cat([v.flatten() for v in grad[len(gp):]])}
        row={}
        for target in ['G','prior']:
            total,pixel,features=[vectors[b][target] for b in ['total','pixel','features']]
            error=(total-pixel-features).norm()/total.norm().clamp_min(1e-20)
            assert error<1e-4, float(error)
            row[target]={'total_norm':float(total.norm()),'pixel_norm':float(pixel.norm()),'features_norm':float(features.norm()),
                         'pixel_features_cosine':float(F.cosine_similarity(pixel,features,dim=0)),
                         'total_over_sum_norm':float(total.norm()/(pixel.norm()+features.norm()).clamp_min(1e-20)),
                         'decomposition_relative_error':float(error)}
        rows.append(row)
    assert before==diag.base.state_hash([g,d,e,prior])
    assert expected==diag.digest(path)
    return {'checkpoint':str(path),'sha256':expected,'step':ck['step'],'batches':batches,
            'mean':{target:{key:sum(r[target][key] for r in rows)/len(rows) for key in rows[0][target]} for target in ['G','prior']},
            'batches_values':rows,'models_and_parent_unchanged':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--batches',type=int,default=16)
    a=p.parse_args();assert 0<a.batches<=156
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    result=[]
    for path in a.checkpoint:
        r=probe(path,a.batches);result.append(r);print(path,r['mean'],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    diag.base.write_json(a.out,{'probes':result,'script_sha256':diag.digest(__file__),
                              'diagnostic_dependency_sha256':diag.digest(diag.__file__),
                              'precision':'FP32, TF32 disabled for additive decomposition check',
                              'note':'Combined GAN loss gradient decomposed into weighted pixel and pretrained-feature contributions. Live weights, no optimizer steps.'})
