#!/usr/bin/env python
"""Read-only checkpoint diagnostics; critic accuracy is not an independent metric."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.train_cifar_ddgan import load_cifar
from lib.image_moonshots import build_models
from lib.denoising_toy import DrawSource,DiffusionSchedule


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cuda',weights_only=False);cfg=ck['config']
    g,d=build_models(cfg)
    g,d=g.cuda().eval(),d.cuda().eval()
    g.load_state_dict(ck['G']);d.load_state_dict(ck['D'])
    prior=DrawSource(cfg['prior'],cfg['num_particles'],cfg['z_dim'],cfg['seed']+101,'cuda');prior.load_state_dict(ck['prior'])
    schedule=DiffusionSchedule(cfg['alpha_bar']).cuda()
    x,c=load_cifar(cfg);x,c=x[:256].cuda().float()/127.5-1,c[:256].cuda()
    rng=torch.Generator('cuda').manual_seed(876)
    result={'step':ck['step'],'config':cfg,'weights':'non_ema','prior_std':float(prior.table.detach().std(0).mean()),'timesteps':[]}
    for k in range(1,schedule.steps+1):
        t=torch.full_like(c,k)
        with torch.no_grad():
            real,xt=schedule.forward_pair(x,t,rng)
            clean=g(prior.sample(len(c),rng)[0],c,xt,t)
            clean2=g(prior.sample(len(c),rng)[0],c,xt,t)
            fake=schedule.reverse(clean,xt,t,torch.randn(xt.shape,device='cuda',generator=rng))
            dr,cr=d(real,c,xt,t);df,cf=d(fake,c,xt,t)
        candidate=real.detach().requires_grad_()
        grad=torch.autograd.grad(d(candidate,c,xt,t)[0].sum(),candidate)[0]
        result['timesteps'].append({'t':k,'clean_mse_to_real':float((clean-x).square().mean()),'clean_mse_when_changing_z':float((clean-clean2).square().mean()),'D_real_minus_fake':float((dr-df).mean()),'D_real_candidate_grad_norm':float(grad.flatten(1).norm(dim=1).mean()),'D_real_class_acc_training_head':float((cr.argmax(1)%cfg['classes']==c).float().mean()),'D_fake_class_acc_training_head':float((cf.argmax(1)%cfg['classes']==c).float().mean()),'D_real_target_acc_training_head':float((cr.argmax(1)==d.ucd_labels(c,t)).float().mean()),'D_fake_target_acc_training_head':float((cf.argmax(1)==d.ucd_labels(c,t)).float().mean())})
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['timesteps'],indent=2))


if __name__=='__main__':main()
