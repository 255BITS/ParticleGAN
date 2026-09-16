#!/usr/bin/env python
"""Compare actual pretrained-D bcap parameter gradients with central differences."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar_ddgan import DEFAULTS, load_cifar
from lib.image_moonshots import build_models
from particlegan import DDGAN
from particlegan import GradientPenalty
from lib.cifar_speed import cifar_penalty, finite_difference_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = {**DEFAULTS, **ck['config']}
    torch.backends.cuda.matmul.allow_tf32 = cfg['tf32']
    torch.backends.cudnn.allow_tf32 = cfg['tf32']
    torch.manual_seed(451)
    _, d = build_models(cfg)
    d.load_state_dict(ck['D']); d.to(device).train()
    images, labels = load_cifar(cfg)
    c = labels[:16].to(device)
    x0 = images[:16].to(device).float() / 127.5 - 1
    t = (torch.arange(16, device=device) % 4) + 1
    rng = torch.Generator(device=device).manual_seed(542)
    real, xt = DDGAN(cfg['alpha_bar'], validate_args=False).to(device).forward_pair(x0, t, rng)
    features = d.condition_features(xt)
    critic = lambda x: d(x, c, xt, t, condition_features=features)[0]
    params = [p for p in d.parameters() if p.requires_grad]
    def flattened(loss):
        values = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p, v in zip(params, values)])
    reg = GradientPenalty('b_cap', 1)
    norm = reg._grad_norm(critic, real)
    exact = torch.relu(norm - 1).square().mean()
    reference = flattened(exact)
    result = {'checkpoint':args.checkpoint, 'step':ck['step'], 'device':str(device), 'tf32':cfg['tf32'],
              'samples':16, 'exact_penalty':float(exact.detach()), 'norm_mean':float(norm.detach().mean()),
              'active_fraction':float((norm > 1).float().mean()), 'comparisons':[]}
    for eps in (.005, .01, .05, .1):
        fd = finite_difference_norm(critic, real, eps)
        loss = torch.relu(fd - 1).square().mean()
        gradient = flattened(loss)
        result['comparisons'].append({'eps':eps, 'penalty':float(loss.detach()),
            'norm_relative_error':float((fd.detach()-norm.detach()).norm()/norm.detach().norm()),
            'gradient_relative_error':float((gradient-reference).norm()/reference.norm().clamp_min(1e-12)),
            'gradient_cosine':float(torch.nn.functional.cosine_similarity(gradient,reference,dim=0))})
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
