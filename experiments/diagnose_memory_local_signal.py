"""Completed-model history ranking and late state restoration; evaluation only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as h
from experiments.memory_local_objectives import mismatch_indices
from experiments.memory_core_scout import fidelity
from experiments.memory_orbit_metrics import orbit_progress


def encode(writer, observations):
    memory = writer.initial(observations)
    for x in observations.unbind(1):
        memory = writer.write(memory, x)
    return memory


def generate(g, writer, z, memory, clock, length):
    points = []
    for step in range(length):
        x, _ = h.local_point(g, z, memory, clock+step)
        points.append(x)
        memory = writer.write(memory, x)
    return torch.stack(points, 1), memory


def candidate_gradient(d, memory, times, candidate, target):
    """Directional probe only; no optimizer or training backward pass."""
    with torch.enable_grad():
        x = candidate.detach().requires_grad_(True)
        score = d.score_candidate(x, memory.detach(), times)
        gradient, = torch.autograd.grad(score.sum(), x)
    correction = target-candidate
    norm = gradient.norm(dim=-1)
    alignment = (gradient*correction).sum(-1)/(norm*correction.norm(dim=-1)).clamp_min(1e-12)
    return dict(mean_cosine_toward_target=float(alignment.mean()),
        positive_alignment_fraction=float((alignment > 0).float().mean()),
        mean_gradient_norm=float(norm.mean()))


def ranking(d, g, z, observed, prefix, write_strength=None):
    times = torch.full((len(z),), prefix, device=z.device)
    if write_strength is None:
        memory = encode(d.writer, observed[:, :prefix])
    else:
        previous = encode(d.writer, observed[:, :prefix-1])
        proposal, _ = h.local_point(g, z, previous, times-1)
        memory = d.writer.write(previous,
            (1-write_strength)*observed[:, prefix-1]+write_strength*proposal)
    actual = observed[:, prefix]
    real_score = d.score_candidate(actual, memory, times)
    positions = times[:, None]
    negatives, endpoint_distances = {}, {}
    for kind in ('nearest', 'shuffle'):
        valid, donor = mismatch_indices(observed, positions, kind)
        assert torch.equal(valid, torch.arange(len(z), device=z.device))
        negatives[kind] = actual[donor]
        endpoint_distances[kind] = float((observed[:, prefix-1]-observed[donor, prefix-1]).norm(dim=-1).mean())
    negatives['earlier_real'] = observed[:, prefix-2]
    negatives['later_real'] = observed[:, prefix+4]
    generated, _ = h.local_point(g, z, memory, times)
    negatives['generated'] = generated
    result = {}
    for name, negative in negatives.items():
        gap = real_score-d.score_candidate(negative, memory, times)
        result[name] = dict(correct_rank_fraction=float((gap > 0).float().mean()),
            mean_margin=float(gap.mean()), median_margin=float(gap.median()))
    result['donor_endpoint_distance'] = endpoint_distances
    # Evaluation error only, never an objective.
    result['generated_next_point_mse'] = float((generated-actual).square().mean())
    result['generated_gradient'] = candidate_gradient(d, memory, times, generated, actual)
    return result


@torch.no_grad()
def diagnose(path, device='cpu', restore=True):
    json.loads((path/'summary.json').read_text())
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = h.Config(**saved['config'])
    if cfg.g_state_dim:
        raise ValueError('This diagnostic isolates D state and currently requires stateless G')
    g, d, prior, _ = h.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval()
    with np.load(path/'trajectories.npz') as arrays:
        clean = arrays['continuation_reference'].copy()
        initial = arrays['observed_prefix32'].copy()
    reference = torch.as_tensor(clean, device=device)
    # Same fixed noise on all models/devices. Preserve saved initial history.
    rng = np.random.default_rng(12012)
    observed = clean+rng.normal(0, cfg.noise, clean.shape).astype(np.float32)
    observed[:, :32] = initial
    observed = torch.as_tensor(observed, device=device)
    z = prior(torch.arange(len(clean), device=device))
    result = dict(name=cfg.name, source=str(path), device=device, steps=cfg.steps,
        ranking={f'prefix{n}': ranking(d, g, z, observed, n) for n in (8, 32, 48)})
    result['ranking_after_write'] = {str(strength): {
        f'prefix{n}': ranking(d, g, z, observed, n, strength) for n in (8, 32, 48)}
        for strength in (.25, 1.)}
    if restore:
        start, duration = 288, 128
        initial_memory = encode(d.writer, observed[:, :32])
        _, autonomous = generate(g, d.writer, z, initial_memory, 32, start-32)
        full_real = encode(d.writer, observed[:, :start])
        recent_real = encode(d.writer, observed[:, start-32:start])
        states = dict(autonomous=(autonomous, start), real_full=(full_real, start),
            real_recent32=(recent_real, start), real_recent32_clock32=(recent_real, 32))
        result['restoration'] = {}
        for name, (memory, clock) in states.items():
            generated, _ = generate(g, d.writer, z, memory, clock, duration)
            array = generated.cpu().numpy()
            q = orbit_progress(array, clean, start)
            q = {key: value for key, value in q.items() if not key.startswith('per_particle')}
            result['restoration'][name] = dict(quality=q, fidelity=fidelity(array, clean, start),
                next_point_mse=float((generated[:, 0]-reference[:, start]).square().mean()),
                memory_rms=float(memory.square().mean().sqrt()), clock=clock,
                next_point_gradient=candidate_gradient(d, memory,
                    torch.full((len(z),), clock, device=device), generated[:, 0], reference[:, start]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ranking-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for path in args.runs:
        result = diagnose(path, args.device, not args.ranking_only)
        results.append(result)
        print(json.dumps(dict(event='diagnostic_complete', name=result['name'])), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(evaluation_only=True,
        definition='Fixed saved real-reference panel, fresh fixed observation noise after saved prefix32. '
        'Point head ranks actual noisy next point against nearest/shuffled other-episode continuations, '
        'earlier/later real points and generated next point. Nearest donors chosen by prior observation only. '
        'These are relative rankings, not calibrated probabilities or guaranteed impossible negatives. '
        'Ranking-after-write replaces only the last prefix observation with a blend of observed and '
        'generated point at strength .25 or1; same original targets and causal donors. '
        'Late restoration: after256 autonomous updates from prefix32, clock288, compare autonomous state '
        'with true observed prefix288 state or true most-recent32 observations encoded from zero. '
        'Recent32 clock32 is a clock control. Same particle/device and reference position; generate128 '
        'points after intervention. Full-real versus recent-real distinguishes history length. '
        'State restoration jumps position and state; improvement does not by itself prove information '
        'erasure or a unique mechanism. Sample-gradient probes report alignment of ascending D score '
        'with the target-minus-generated direction; this is an evaluation-only local Euclidean probe, '
        'not a training objective or proof of correct recovery. Ranking uses noisy targets; restoration '
        'uses clean targets. These short diagnostic continuations are not1024 pass metrics.',
        results=results), indent=2, allow_nan=False)+'\n')
