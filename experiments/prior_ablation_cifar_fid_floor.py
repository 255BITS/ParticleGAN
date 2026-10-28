#!/usr/bin/env python
"""FID evaluation noise floor for a finished CIFAR particle DDGAN run.

Reloads the final EMA generator and prior of one run and recomputes FID50k with
``--draws`` extra evaluation-stream seeds (``seed + 10000 + k``, k = 1..draws).
Weights, training trajectory and the real-statistics cache are unchanged, so the
spread is sampling noise of the FID estimate itself (not a training-seed
experiment). Writes ``fid_floor.json`` next to the run's ``summary.json``.

    python experiments/prior_ablation_cifar_fid_floor.py --run /path/to/out_dir --draws 2
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import train_cifar_particle_ddgan as host  # noqa: E402
from particlegan import init  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2)
    args = parser.parse_args()
    from lib.cifar_metrics import FIDEvaluator
    summary = json.loads((args.run / 'summary.json').read_text())
    cfg = summary['config']
    if cfg['model'] != 'ddgan':
        raise ValueError('only the DDGAN host is supported')
    state = torch.load(args.run / f"checkpoint_{cfg['steps']:06d}.pt", map_location='cuda', weights_only=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    g = host.ParticleDDGenerator(cfg['z_dim'], cfg['width'], cfg['alpha_bar']).cuda()
    recipe = host.training_recipe(cfg)
    table_recipe = recipe.replace(prior_update='learned', prior_code_law='table')
    prior = init.deterministic_orthogonal_(table_recipe.make_prior(generator=host.rng(cfg['seed'] + 1, 'cpu'))).cuda()
    if recipe.prior_update != 'learned':
        recipe.apply_prior_policy(prior)
    g.load_state_dict(state['ema_G'])
    prior.load_state_dict(state['ema_prior'])
    g.eval().requires_grad_(False)
    prior.eval().requires_grad_(False)
    evaluator = FIDEvaluator(None, cfg['fid_cache'], cfg['eval_batch_size'])
    scratch = args.run / 'fid_floor_samples'
    scratch.mkdir(exist_ok=True)
    rows = [{'eval_seed_offset': 0, 'fid': summary['final']['fid'], 'fid5k': summary['final']['fid5k'],
             'source': 'summary.json'}]
    for k in range(1, args.draws + 1):
        draw = dict(cfg, seed=cfg['seed'] + k)  # generation() reads only seed + 10000 from cfg['seed']
        with torch.no_grad():
            result = host.generation(g, prior, evaluator, draw, cfg['final_samples'], scratch, cfg['steps'])
        rows.append({'eval_seed_offset': k, 'fid': result['fid'], 'fid5k': result['fid5k'], 'source': 'recomputed'})
        print(json.dumps(rows[-1]), flush=True)
    values = [row['fid'] for row in rows]
    report = {'schema': 'cifar_fid_eval_floor_v1', 'run': str(args.run), 'step': state['step'],
              'samples': cfg['final_samples'], 'rows': rows, 'mean': statistics.fmean(values),
              'std': statistics.stdev(values) if len(values) > 1 else None,
              'range': max(values) - min(values)}
    host.write_json(args.run / 'fid_floor.json', report)
    print(f"FLOOR fid_mean={report['mean']:.3f} std={report['std']} range={report['range']:.3f}", flush=True)


if __name__ == '__main__':
    main()
