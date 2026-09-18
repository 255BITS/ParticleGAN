#!/usr/bin/env python
"""Read-only, matched-count generation audit of late toy training regression."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary
from lib.mog_metrics import evaluate
from lib.toy_metrics import sliced_w1
from lib.toy_models import SimpleMLPGenerator
from particlegan import MoGParticlePrior


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    provenance = code_provenance(str(ROOT / 'experiments/train_mog_vae.py'), sys.executable)
    rows = {}
    for name in ['00_gan', '01_ae_gan', '02_categorical_no_kl_gan', 'cat_sharp_obs003_gan']:
        out = ROOT / 'runs/mog_vae/scout' / name
        s = json.loads((out / 'summary.json').read_text()); cfg = s['config']
        assert has_valid_summary(str(out), cfg, provenance)
        path = out / 'checkpoint_004000.pt'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        assert before == s['checkpoints'][path.name]
        checkpoint = torch.load(path, map_location='cuda', weights_only=False)
        g = SimpleMLPGenerator(2, cfg['width']).cuda()
        p = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=2, sigma_rel=cfg['sigma_rel'], device='cuda').cuda()
        g.load_state_dict(checkpoint['G']); p.load_state_dict(checkpoint['prior'])
        with torch.no_grad():
            m, _, fake, real = evaluate(g, p, 100000, cfg['seed'])
            m = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in m.items()}
            m['sample_sw1'] = sliced_w1(fake[:8192], real[:8192], seed=cfg['seed'] + 10001)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        rows[name] = dict(checkpoint_sha256=before, checkpoint_unchanged=True, samples=100000,
                          step4000=m, step6000={k: s['final'][k] for k in ['modes', 'hq', 'width_ratio', 'sample_sw1']})
        print(name, '4k', m['modes'], m['hq'], '6k', s['final']['modes'], s['final']['hq'], flush=True)
    result = dict(protocol='4k read-only checkpoint generation, same 100k samples and RNG as final 6k evaluation',
                  total_seconds=time.perf_counter()-started, rows=rows)
    (ROOT / 'reports/mog-vae/late_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
