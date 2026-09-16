"""Completed-only independent D/G memory interventions; no training objectives."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core
from experiments.memory_orbit_metrics import orbit_progress


@torch.no_grad()
def diagnose(path, device='cpu'):
    json.loads((path/'summary.json').read_text())  # completed checkpoint required
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    for key, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[key])
        module.eval()
    z = prior(torch.arange(cfg.eval_batch, device=device))
    with np.load(path/'trajectories.npz') as arrays:
        clean = arrays['continuation_reference'].copy()
        prefixes = {n: arrays[f'observed_prefix{n}'].copy() for n in (8, 32)}
    modes = [None, 'zero', 'shuffle']
    if cfg.g_state_dim:
        modes += ['g_zero', 'g_shuffle']
    result = {}
    for n, observed in prefixes.items():
        prefix = torch.as_tensor(observed, device=device)
        result[f'prefix{n}'] = {}
        normal = None
        for mode in modes:
            generated, _ = core.continuation(g, d.writer, z, prefix, cfg.eval_steps, mode)
            array = generated.cpu().numpy()
            if mode is None:
                normal = array
            result[f'prefix{n}'][mode or 'normal'] = {
                'quality': orbit_progress(array, clean, n),
                'fidelity': core.fidelity(array, clean, n),
                'mean_distance_from_normal': float(np.linalg.norm(array-normal, axis=-1).mean()),
                'bitwise_equal_normal': bool(np.array_equal(array, normal)),
            }
    return {'name': cfg.name, 'source': str(path), 'steps': cfg.steps,
            'g_state_dim': cfg.g_state_dim, 'g_state_reads_d': cfg.g_state_reads_d,
            'g_use_d_memory': cfg.g_use_d_memory, 'device': device, 'results': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for path in args.runs:
        row = diagnose(path, args.device)
        rows.append(row)
        print(json.dumps({'event': 'diagnostic_complete', 'name': row['name']}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'evaluation_only': True,
        'definition': 'Warm1024 after real prefixes8/32. Normal and interventions rerun on the same device. D zero/shuffle alters every autonomous read and any G state updater input, not stored D state. G zero/shuffle alters the state used for reading and the subsequent G update every step. Real prefixes are unmodified. D is still written from generated samples. Persistent intervention dependence is distinct from benefit over a separately trained no-D model.',
        'results': rows}, indent=2, allow_nan=False)+'\n')
