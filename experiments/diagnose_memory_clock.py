"""Completed-checkpoint clock/memory interventions, excluded from training/ranking."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core
from experiments.memory_scout import diagnostics


def intervened_path(g, writer, z, prefix, steps, variant, switch=32):
    """Preserve the first switch generated points, then alter only G's inputs."""
    memory = core.context(writer, prefix)
    captured = None
    points = []
    for step in range(steps):
        if step == switch:
            captured = memory.clone()
        read = memory
        time_index = prefix.shape[1]+step
        if step >= switch:
            if variant == 'clock_frozen':
                time_index = prefix.shape[1]+switch
            elif variant == 'clock_half':
                time_index = prefix.shape[1]+switch+(step-switch)*.5
            elif variant == 'memory_zero':
                read = torch.zeros_like(memory)
            elif variant == 'memory_shuffle':
                read = memory.roll(1, 0)
            elif variant == 'memory_frozen':
                read = captured
        point, _ = handoff.local_point(g, z, read, time_index)
        points.append(point)
        memory = writer.write(memory, point)
    return torch.stack(points, 1)


@torch.no_grad()
def diagnose(path, device):
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval()
    z = prior(torch.arange(cfg.eval_batch, device=device))
    # Use the saved panel: CPU/CUDA RNGs differ even with identical seeds.
    with np.load(path/'trajectories.npz') as original:
        observed = torch.as_tensor(original[f'observed_prefix{max(cfg.eval_prefixes)}'], device=device)
        clean_np = original['continuation_reference'].copy()
    clean = torch.as_tensor(clean_np, device=device)
    variants = ('normal', 'clock_frozen', 'clock_half', 'memory_zero', 'memory_shuffle', 'memory_frozen')
    results = {}
    with np.load(path/'trajectories.npz') as original:
        for prefix_length in (0, *cfg.eval_prefixes):
            prefix = observed[:, :prefix_length]
            rows = {}
            normal = None
            for variant in variants:
                generated = intervened_path(g, d.writer, z, prefix, cfg.eval_steps, variant).cpu().numpy()
                if variant == 'normal':
                    normal = generated
                    key = f'prefix{prefix_length}' if prefix_length else 'generated'
                    np.testing.assert_allclose(generated, original[key], atol=1e-5, rtol=1e-5)
                else:
                    np.testing.assert_array_equal(generated[:, :32], normal[:, :32])
                rows[variant] = {'full256': diagnostics(generated[:, :256]),
                    'full_long': diagnostics(generated),
                    'mean_position_change_after_intervention': float(np.linalg.norm(generated[:, 32:]-normal[:, 32:], axis=-1).mean())}
                if prefix_length:
                    rows[variant]['fidelity256'] = core.fidelity(generated[:, :256], clean_np, prefix_length)
                    rows[variant]['fidelity_long'] = core.fidelity(generated, clean_np, prefix_length)
            results[f'prefix{prefix_length}'] = rows
    # Paired local probes hold the actual memory fixed while changing one input.
    local = {}
    for n in cfg.eval_prefixes:
        memory = core.context(d.writer, observed[:, :n])
        normal, _ = handoff.local_point(g, z, memory, n)
        row = {'normal_target_mse': float((normal-clean[:, n]).square().mean())}
        for variant, read, time_index in (
            ('clock_zero', memory, 0), ('clock_previous', memory, n-1),
            ('memory_zero', torch.zeros_like(memory), n), ('memory_shuffle', memory.roll(1, 0), n)):
            altered, _ = handoff.local_point(g, z, read, time_index)
            row[variant] = {'output_change_mse': float((altered-normal).square().mean()),
                            'target_mse': float((altered-clean[:, n]).square().mean())}
        local[f'prefix{n}'] = row
    return {'name': cfg.name, 'evaluation_only': True, 'switch_after_generated_points': 32,
            'note': 'Interventions alter G reads only; writer continues consuming outputs. Dependence is not proof of beneficial use.',
            'rollouts': results, 'real_context_local_probes': local}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [diagnose(path, args.device) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    for result in results:
        print(result['name'], {key: row['full_long']['circle_like_fraction']
                              for key, row in result['rollouts']['prefix0'].items()})
