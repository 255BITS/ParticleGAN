#!/usr/bin/env python
"""Compare bcap backward phases before/after restricting gradient targets."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.profile_cifar_ae_gpu1 import torch, yaml, DEFAULTS, CIFAR10, Harness
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
cfg = dict(DEFAULTS, **yaml.safe_load(Path('configs/cifar_particle_ae/lazy_long/n08.yaml').read_text()))
images = torch.from_numpy(CIFAR10(cfg['data_dir'], train=True, download=False).data).permute(0,3,1,2).contiguous().cuda()
h = Harness(cfg, images, 'baseline')
for s in range(1,65):
    h.step(s)
result = {}
for variant in ('baseline', 'pruned_d', 'reuse_pruned_d'):
    h.variant = variant
    h.instrument = True
    h.events = []
    for s in range(1,33):
        h.step(s)
    torch.cuda.synchronize()
    groups = {}
    for regularized, name, start, end, host in h.events:
        key = ('regularized' if regularized else 'ordinary') + '/' + name
        groups.setdefault(key, []).append(start.elapsed_time(end))
    result[variant] = {k:sum(v)/len(v) for k,v in groups.items()}
    print('PRUNED_PHASES', variant, json.dumps(result[variant]), flush=True)
Path('runs/cifar_particle_ae/performance_gpu1/pruned_phases.json').write_text(json.dumps(result, indent=2)+'\n')
