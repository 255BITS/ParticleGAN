"""Read-only matched-memory and wrong-time probes of completed pointwise models."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments.memory_path import circles


@torch.no_grad()
def diagnose(path, device):
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    for key, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[key])
        module.eval()
    # One fixed, fresh panel shared across models; no training or seed sweep.
    rng = torch.Generator(device=device).manual_seed(20260918)
    real, clean = circles(512, 64, rng, device, noise=cfg.noise)
    z = prior(torch.arange(512, device=device))
    results = {}
    for position in (8, 32):
        positions = torch.full((512, 1), position, device=device)
        memory = handoff.selected_memories(d.writer, real, positions, position)
        target = real[:, position]
        scores = d.score_candidate(target, memory, position)
        fake, _ = handoff.local_point(g, z, memory, position)
        row = {'real_minus_fake_score': (scores-d.score_candidate(fake, memory, position)).mean().item()}
        for name, candidate, state in [
            ('previous_point', real[:, position-1], memory),
            ('future3_point', real[:, position+3], memory),
            ('shuffled_memory', target, memory.roll(1, 0)),
            ('zero_memory', target, torch.zeros_like(memory)),
        ]:
            delta = scores-d.score_candidate(candidate, state, position)
            row[name] = {'correct_preferred_fraction': (delta > 0).float().mean().item(),
                         'mean_score_margin': delta.mean().item()}
        row['g_clean_target_mse'] = ((fake-clean[:, position])**2).mean().item()
        # Two valid real histories end at the identical observed point but move
        # in opposite directions. This probes motion beyond current position.
        reverse_history = real[:, position-1:2*position-1].flip(1)
        reverse_memory = handoff.selected_memories(d.writer, reverse_history, positions, position)
        reverse_fake, _ = handoff.local_point(g, z, reverse_memory, position)
        opposite = real[:, position-2]
        forward_right = scores > d.score_candidate(opposite, memory, position)
        reverse_right = d.score_candidate(opposite, reverse_memory, position) > d.score_candidate(target, reverse_memory, position)
        row['opposite_direction_histories'] = {
            'forward_preferred_fraction': forward_right.float().mean().item(),
            'reverse_preferred_fraction': reverse_right.float().mean().item(),
            'both_preferred_fraction': (forward_right & reverse_right).float().mean().item(),
            'g_reverse_clean_target_mse': ((reverse_fake-clean[:, position-2])**2).mean().item(),
            'g_reverse_uses_forward_target_mse': ((reverse_fake-clean[:, position])**2).mean().item(),
        }
        for name, state in [('shuffled', memory.roll(1, 0)), ('zero', torch.zeros_like(memory))]:
            alternate, _ = handoff.local_point(g, z, state, position)
            row[f'g_{name}_target_mse'] = ((alternate-clean[:, position])**2).mean().item()
            row[f'g_{name}_output_mse_change'] = ((alternate-fake)**2).mean().item()
        results[f'prefix{position}'] = row
    return {'name': cfg.name, 'source': str(path), 'panel_size': 512, 'metrics': results}


def main():
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
        print(json.dumps(row))


if __name__ == '__main__':
    main()
