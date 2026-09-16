#!/usr/bin/env python
"""Audit cached/NHWC numerical agreement on one saved CIFAR discriminator."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar_ddgan import DEFAULTS, load_cifar
from lib.image_moonshots import build_models
from particlegan import DDGAN


def relative(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return {'relative_error':float((a-b).norm()/a.norm().clamp_min(1e-12)),
            'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
            'max_abs_error':float((a-b).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg={**DEFAULTS,**ck['config']}
    _, model=build_models(cfg); model.load_state_dict(ck['D']); model.cuda().eval()
    images, labels=load_cifar(cfg)
    c=labels[:8].cuda(); t=torch.arange(8,device='cuda')%4+1
    rng=torch.Generator(device='cuda').manual_seed(542)
    real,xt=DDGAN(cfg['alpha_bar'], validate_args=False).cuda().forward_pair(images[:8].cuda().float()/127.5-1,t,rng)
    results={}
    for tf32 in (False,True):
        torch.backends.cuda.matmul.allow_tf32=tf32
        torch.backends.cudnn.allow_tf32=tf32
        outputs={}
        for layout,cache in [('nchw',False),('nchw',True),('nhwc',False),('nhwc',True)]:
            d=copy.deepcopy(model)
            if layout=='nhwc': d.to(memory_format=torch.channels_last)
            candidate=real.detach().clone().requires_grad_()
            condition=xt
            if layout=='nhwc':
                candidate=candidate.detach().contiguous(memory_format=torch.channels_last).requires_grad_()
                condition=condition.contiguous(memory_format=torch.channels_last)
            features=d.condition_features(condition) if cache else None
            score=d(candidate,c,condition,t,condition_features=features)[0]
            grad=torch.autograd.grad(score.sum(),candidate,create_graph=True)[0]
            # Squared norm = kappa-zero bcap, activates all samples.
            loss=grad.square().flatten(1).sum(1).mean()
            params=[p for p in d.parameters() if p.requires_grad]
            values=torch.autograd.grad(loss,params,allow_unused=True)
            parameter_grad=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,values)])
            outputs[f'{layout}_cache{cache}']=[x.detach().cpu() for x in (score,grad,parameter_grad)]
        reference=outputs['nchw_cacheFalse']
        results[str(tf32)]={name:{metric:relative(a,b) for metric,a,b in zip(('score','candidate_grad','bcap_parameter_grad'),reference,output)} for name,output in outputs.items()}
    Path(args.output).write_text(json.dumps({'checkpoint':args.checkpoint,'comparisons_by_tf32':results},indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
