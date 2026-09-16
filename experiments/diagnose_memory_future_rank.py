"""Completed-model future identity ranking before/after one generated write."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as h
from experiments.diagnose_memory_local_signal import encode
from experiments.memory_local_objectives import mismatch_indices


@torch.no_grad()
def ranking(d, g, z, observed, prefix=32, horizons=(0, 4, 12)):
    """Cache strictly causal states and donor IDs once for every future query."""
    if len(z) < 2 or prefix < 4 or prefix+max(horizons) >= observed.shape[1]:
        raise ValueError('Need two episodes, prefix>=4 and observed future targets')
    if any(n < 0 for n in horizons):
        raise ValueError('Future horizons must be nonnegative')
    times = torch.full((len(z),), prefix, device=z.device)
    previous = encode(d.writer, observed[:, :prefix-1])
    clean = d.writer.write(previous, observed[:, prefix-1])
    proposal, _ = h.local_point(g, z, previous, times-1)
    explored = d.writer.write(previous, proposal)
    donors, endpoint_distances = {}, {}
    for kind in ('nearest', 'shuffle'):
        valid, donor = mismatch_indices(observed, times[:, None], kind)
        assert torch.equal(valid, torch.arange(len(z), device=z.device))
        donors[kind] = donor
        endpoint_distances[kind] = float((observed[:, prefix-1]-observed[donor, prefix-1]).norm(dim=-1).mean())
    result = {'donor_endpoint_distance': endpoint_distances, 'contexts': {}}
    for name, memory in (('clean', clean), ('generated_write', explored)):
        queries = {}
        for horizon in horizons:
            actual = observed[:, prefix+horizon]
            actual_score = d.score_candidate(actual, memory, times, horizon=horizon)
            row = {'target_index': prefix+horizon, 'actual_score_mean': float(actual_score.mean()),
                   'actual_score_std': float(actual_score.std(unbiased=False)), 'donors': {}}
            for kind, donor in donors.items():
                negative_score = d.score_candidate(actual[donor], memory, times, horizon=horizon)
                gap = actual_score-negative_score
                row['donors'][kind] = {'correct_rank_fraction': float((gap > 0).float().mean()),
                    'mean_margin': float(gap.mean()), 'median_margin': float(gap.median()),
                    'negative_score_mean': float(negative_score.mean()),
                    'negative_score_std': float(negative_score.std(unbiased=False))}
            queries[str(horizon)] = row
        result['contexts'][name] = queries
    return result


@torch.no_grad()
def diagnose(path, device='cpu', prefixes=(8, 32), horizons=(0, 4, 12)):
    # Refuse partial training runs: the final summary is written on completion.
    json.loads((path/'summary.json').read_text())
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = h.Config(**saved['config'])
    if cfg.g_state_dim:
        raise ValueError('This diagnostic requires stateless G')
    g, d, prior, _ = h.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval().requires_grad_(False)
    with np.load(path/'trajectories.npz') as arrays:
        clean = arrays['continuation_reference'].copy()
        initial = arrays['observed_prefix32'].copy()
    # Exactly match diagnose_memory_local_signal's saved panel/noise convention.
    rng = np.random.default_rng(12012)
    observed = clean+rng.normal(0, cfg.noise, clean.shape).astype(np.float32)
    observed[:, :32] = initial
    observed = torch.as_tensor(observed, device=device)
    z = prior(torch.arange(len(clean), device=device))
    return {'name': cfg.name, 'source': str(path), 'steps': cfg.steps, 'device': device,
            'horizon_conditioned': bool(cfg.future_rank_bands),
            'horizon_bands': cfg.future_rank_bands,
            'prefixes': {str(n): ranking(d, g, z, observed, n, horizons) for n in prefixes}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for path in args.runs:
        rows.append(diagnose(path, args.device))
        print(json.dumps({'event': 'future_ranking_complete', 'name': rows[-1]['name']}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'evaluation_only': True,
        'definition': 'Completed models only. Saved original continuation-reference panel and observed prefix32; '
        'fixed NumPy noise seed12012 elsewhere, matching local-signal diagnostic. '
        'At prefixes8/32 rank actual x[t+h] against nearest/shuffled other-episode x[t+h], h=0/4/12. '
        'Donors are selected once by original last-prefix observations, never targets or generated proposals. '
        'All horizons and donors share each cached memory: either clean real prefix, or replace only its '
        'last observation with a full model-generated sample made at t-1 using the same fixed particle. '
        'The scorer receives current t and explicit horizon h. Models without horizon conditioning ignore h. '
        'Each model supplies its own generated corruption, so after-write results are not a comparison '
        'of discriminators on identical corrupted memories. Rankings are relative, not calibrated '
        'probabilities; mismatched futures may be plausible. No optimizer, loss, or generated trajectory '
        'training is used. Absolute score offsets across models are not comparable.',
        'results': rows}, indent=2, allow_nan=False)+'\n')
