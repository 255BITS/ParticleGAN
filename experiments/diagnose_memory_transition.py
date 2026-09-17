"""Completed-model one-write transition and read-response diagnostics; evaluation only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as h
from experiments.diagnose_memory_information import episodes
from experiments.diagnose_memory_local_signal import encode


def paired_scores(positive, negative):
    gap = positive-negative
    return {'positive_mean': float(positive.mean()), 'negative_mean': float(negative.mean()),
            'mean_margin': float(gap.mean()), 'median_margin': float(gap.median()),
            'positive_rank_fraction': float((gap > 0).float().mean())}


@torch.no_grad()
def transition_panel(g, d, z, observed, prefix=32, critic=None):
    """One autonomous write, then independent reads with the same particle and next clock."""
    if len(z) < 2 or prefix < 1 or prefix+1 >= observed.shape[1]:
        raise ValueError('Need two histories, positive prefix, and two continuation observations')
    times = torch.full((len(z),), prefix, device=z.device)
    anchor = encode(d.writer, observed[:, :prefix])
    real_x = observed[:, prefix]
    fake_x, _ = h.local_point(g, z, anchor, times)
    real_m = d.writer.write(anchor, real_x)
    fake_m = d.writer.write(anchor, fake_x)
    donor = torch.arange(len(z), device=z.device).roll(1)
    successors = {'real': real_m, 'generated': fake_m, 'shuffled_real': real_m[donor]}
    reads = {name: h.local_point(g, z, memory, times+1)[0]
             for name, memory in successors.items()}
    target = observed[:, prefix+1]
    spread = (real_m-real_m.mean(0)).square().mean()
    state_mse = (real_m-fake_m).square().mean()
    result = {
        'generated_point_mse': float((fake_x-real_x).square().mean()),
        'successor_memory_mse': float(state_mse),
        'real_successor_memory_variance': float(spread),
        'normalized_successor_memory_mse': float(state_mse/spread.clamp_min(1e-12)),
        'next_read_mse': {name: float((point-target).square().mean()) for name, point in reads.items()},
        'next_read_change_mse': {
            'generated_vs_real': float((reads['generated']-reads['real']).square().mean()),
            'shuffled_vs_real': float((reads['shuffled_real']-reads['real']).square().mean())},
        'transition_candidate_space': getattr(critic, 'space', 'memory') if critic is not None else None,
        'transition_critic': None}
    if critic is not None:
        space = getattr(critic, 'space', 'memory')
        real_feature, fake_feature = ((reads['real'], reads['generated']) if space == 'read'
                                      else (real_m, fake_m))
        real_candidate = critic.candidate(real_x, real_feature)
        fake_candidate = critic.candidate(fake_x, fake_feature)
        real_score = critic.score_candidate(real_candidate, anchor)
        fake_score = critic.score_candidate(fake_candidate, anchor)
        shuffled_real_score = critic.score_candidate(real_candidate, anchor[donor])
        shuffled_fake_score = critic.score_candidate(fake_candidate, anchor[donor])
        scores = {
            'real_vs_generated': paired_scores(real_score, fake_score),
            'real_vs_other_episode_successor': paired_scores(real_score,
                critic.score_candidate(real_candidate[donor], anchor)),
            'real_vs_generated_shuffled_anchor': paired_scores(shuffled_real_score, shuffled_fake_score),
            'paired_margin_correct_vs_shuffled_anchor': paired_scores(
                real_score-fake_score, shuffled_real_score-shuffled_fake_score),
            'real_anchor_vs_shuffled_anchor': paired_scores(real_score,
                shuffled_real_score),
            'generated_anchor_vs_shuffled_anchor': paired_scores(fake_score,
                shuffled_fake_score)}
        # Deliberately inconsistent hybrids identify score dependence; not valid transitions.
        if getattr(critic, 'include_x', True):
            hybrid_real_x = critic.score_candidate(critic.candidate(real_x, fake_feature), anchor)
            hybrid_real_feature = critic.score_candidate(critic.candidate(fake_x, real_feature), anchor)
            scores[f'real_vs_real_x_generated_{space}'] = paired_scores(real_score, hybrid_real_x)
            scores[f'real_vs_generated_x_real_{space}'] = paired_scores(real_score, hybrid_real_feature)
            scores[f'generated_vs_real_x_generated_{space}'] = paired_scores(fake_score, hybrid_real_x)
            scores[f'generated_vs_generated_x_real_{space}'] = paired_scores(fake_score, hybrid_real_feature)
        result['transition_critic'] = scores
    return result


@torch.no_grad()
def diagnose(path, observed, ids, device='cpu', prefixes=(8, 32)):
    path = Path(path)
    json.loads((path/'summary.json').read_text())  # A completed run is mandatory.
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = h.Config(**saved['config'])
    if cfg.g_state_dim:
        raise ValueError('This diagnostic requires stateless G')
    g, d, prior, _ = h.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval().requires_grad_(False)
    critic = None
    if saved.get('transition_critic') is not None:
        from experiments.memory_transition import TransitionCritic
        critic = TransitionCritic(cfg).to(device)
        critic.load_state_dict(saved['transition_critic'])
        critic.eval().requires_grad_(False)
    observed, ids = observed.to(device), ids.to(device)
    z = prior(ids)
    return {'name': cfg.name, 'source': str(path), 'steps': cfg.steps,
            'checkpoint_sha256': hashlib.sha256((path/'model.pt').read_bytes()).hexdigest(),
            'transition_critic_enabled': critic is not None,
            'transition_config': {key: value for key, value in vars(cfg).items()
                                  if key.startswith('transition_')},
            'prefixes': {str(n): transition_panel(g, d, z, observed, n, critic) for n in prefixes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--prefixes', nargs='+', type=int, default=[8, 32])
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(7179381)
    observed, _, ids = episodes(args.count, max(args.prefixes)+2, rng)
    rows = []
    for path in args.runs:
        rows.append(diagnose(path, observed, ids, args.device, args.prefixes))
        print(json.dumps({'event': 'transition_diagnostic_complete', 'name': rows[-1]['name']}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'evaluation_only': True, 'count': args.count,
        'definition': 'Fresh held-out noisy circle histories from the training process distribution, '
        'shared exactly across models. Learned particles are sampled from the existing 512 table '
        'and are not held out. A fixed particle generates at t, then one writer update gives the '
        'generated successor. Compare with the observed successor and independently read both '
        'at t+1 using the same particle and clock; also shuffle observed successor across episodes. '
        'All MSE quantities are evaluation only, never GAN objectives. Memory distance is in '
        'model-specific coordinates; normalized distance is descriptive, not a universal quality '
        'score. Next-read errors use noisy observed targets and do not establish long-run success. '
        'The transition candidate space is reported per model. Memory-space K sees successor M; '
        'read-space K sees current G(z, successor M, t+1), with optional first-point x in either '
        'space. In read space the real candidate is a model read after an observed write, not '
        'ground-truth next observation. The same fixed particle and next clock are used on both '
        'sides, and none of these reads are written back. '
        'K scores compare joint real/generated transitions and shuffled anchors; optional hybrids '
        'deliberately violate sample/write consistency to probe score dependence. Their margins '
        'do not establish causal feature importance. Absolute correct-versus-shuffled anchor '
        'scores can change through arbitrary anchor-only offsets unconstrained by paired GAN '
        'losses. The paired-margin correct-versus-shuffled comparison cancels such offsets '
        'and measures how conditioning changes real/fake separation, not process retention. '
        'The real-versus-other-episode-successor task compares complete real candidates from '
        'different episodes at the same anchor; this mirrors the history-negative critic '
        'objective and also cancels arbitrary anchor-only offsets. '
        'K scores across independently trained critics '
        'are not calibrated/comparable. No K values are reported for the original baseline. '
        'Use existing information diagnostics separately for retention over autonomous execution. '
        'Completed runs only; no optimization or trajectory training.', 'results': rows},
        indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
