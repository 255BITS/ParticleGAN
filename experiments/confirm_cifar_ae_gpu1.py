#!/usr/bin/env python
"""Paired warmed confirmation of the promising GPU-1 candidates."""
import json
from pathlib import Path
import sys
import time
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
for s in range(1, 161):
    h.step(s)
# Same model and optimizer continue throughout: fixed-size blocks, no reseeding.
# Reverse-order second pass brackets candidate measurements against drift.
variants = ['baseline', 'batched_d', 'pruned_d', 'reuse_pruned_d', 'baseline',
            'reuse_pruned_d', 'pruned_d', 'batched_d', 'baseline']
rows = []
for variant in variants:
    h.variant = variant
    for s in range(1,17):
        h.step(s)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for s in range(1,161):
        last = h.step(s)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    row = {'variant': variant, 'seconds': elapsed, 'steps_per_second':160/elapsed, 'ms_per_step':elapsed/160*1000,
           'last_losses':last.cpu().tolist()}
    rows.append(row)
    print('PAIRED', json.dumps(row), flush=True)
    Path('runs/cifar_particle_ae/performance_gpu1/confirmation.json').write_text(json.dumps(rows, indent=2)+'\n')
print('PAIRED_COMPLETE', flush=True)
