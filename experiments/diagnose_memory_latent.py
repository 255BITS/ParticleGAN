"""Evaluation-only latent/context intervention; never a fixed-particle solution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments.memory_scout import diagnostics


@torch.no_grad()
def diagnose(path, device):
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval()
    z = prior(torch.arange(cfg.eval_batch, device=device))
    variants = {}
    for switch in (None, 1, 4):
        memory = d.writer.initial(z)
        path_points = []
        for step in range(cfg.eval_steps):
            read_z = z if switch is None or step < switch else z.roll(1, 0)
            point, _ = handoff.local_point(g, read_z, memory, step)
            memory = d.writer.write(memory, point)
            path_points.append(point)
        generated = torch.stack(path_points, 1).cpu().numpy()
        if switch is None:
            with np.load(path/'trajectories.npz') as original:
                np.testing.assert_allclose(generated, original['generated'], atol=1e-5, rtol=1e-5)
        variants['fixed' if switch is None else f'swap_after_{switch}'] = {
            'full256': diagnostics(generated[:, :256]), 'full1024': diagnostics(generated)}
    return {'name': cfg.name, 'evaluation_only': True,
            'note': 'Swapped variants violate fixed-particle policy; diagnostic only, excluded from leaderboard.',
            'metrics': variants}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [diagnose(path, args.device) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(row['name'], {name: item['full1024']['circle_like_fraction'] for name, item in row['metrics'].items()})
