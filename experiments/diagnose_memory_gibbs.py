"""Completed-model opposing-joint and same-event refinement diagnostics; evaluation only."""
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
from experiments.diagnose_memory_transition import paired_scores


def mse(first, second):
    return float((first-second).square().mean())


@torch.no_grad()
def gibbs_panel(g, d, z, observed, prefix=32, critic=None, refinement_steps=(1, 2, 3, 7)):
    """All refinements concern one event: no memory writes or clock advances."""
    if len(z) < 2 or not 0 < prefix < observed.shape[1]:
        raise ValueError('Need two histories and a positive prefix before the target')
    if any(n < 1 or int(n) != n for n in refinement_steps):
        raise ValueError('Refinement counts must be positive integers')
    times = torch.full((len(z),), prefix, device=z.device)
    memory = encode(d.writer, observed[:, :prefix])
    target = observed[:, prefix]
    donor = torch.arange(len(z), device=z.device).roll(1)
    fake, _ = h.local_point(g, z, memory, times)
    interventions = {'zero': torch.zeros_like(memory), 'shuffle': memory[donor]}
    memory_reads = {name: h.local_point(g, z, altered, times)[0]
                    for name, altered in interventions.items()}
    result = {'point_mse': mse(fake, target),
              'memory_interventions': {name: {'point_mse': mse(point, target),
                  'output_change_mse': mse(point, fake)} for name, point in memory_reads.items()},
              'refinement': None, 'latent': None, 'joint_critic': None}
    if not hasattr(g, 'joint'):
        if critic is not None:
            raise ValueError('A Gibbs critic requires a Gibbs generator')
        return result
    generated, producer = g.joint(z, memory, time_index=times)
    torch.testing.assert_close(generated, fake, rtol=0, atol=0)
    inferred_real = g.infer(memory, target, time_index=times)
    inferred_generated = g.infer(memory, fake, time_index=times)
    real_decode = g.decode(z, memory, inferred_real, time_index=times)[0]
    latent_reads = {name: g.decode(z, memory, altered, time_index=times)[0]
                    for name, altered in {'zero': torch.zeros_like(producer),
                                          'shuffle': producer[donor]}.items()}
    result['latent'] = {
        'real_inferred_variance': float(inferred_real.var(0, unbiased=False).mean()),
        'producer_variance': float(producer.var(0, unbiased=False).mean()),
        'producer_vs_inferred_generated_mse': mse(producer, inferred_generated),
        'real_decode_mse': mse(real_decode, target),
        'interventions': {name: {'point_mse': mse(point, target),
              'output_change_mse': mse(point, fake)} for name, point in latent_reads.items()}}
    result['refinement'] = {}
    for steps in sorted(set(refinement_steps)):
        point, latent = g.joint(z, memory, time_index=times, steps=steps)
        inferred = g.infer(memory, point, time_index=times)
        result['refinement'][str(steps)] = {'point_mse': mse(point, target),
            'output_change_from_configured_mse': mse(point, fake),
            'producer_vs_inferred_generated_mse': mse(latent, inferred)}
    if critic is not None:
        def score(latent, point, anchor=memory):
            return critic.score_candidate(torch.cat((latent, point), -1), anchor, time_index=times)

        real_score = score(inferred_real, target)
        fake_score = score(producer, fake)
        shuffled_real = score(inferred_real, target, memory[donor])
        shuffled_fake = score(producer, fake, memory[donor])
        result['joint_critic'] = {
            'real_vs_generated': paired_scores(real_score, fake_score),
            'real_vs_shuffled_latent': paired_scores(real_score, score(inferred_real[donor], target)),
            'generated_vs_shuffled_latent': paired_scores(fake_score, score(producer[donor], fake)),
            'generated_producer_vs_reencoded_latent': paired_scores(fake_score,
                score(inferred_generated, fake)),
            'real_vs_other_episode_pair': paired_scores(real_score,
                score(inferred_real[donor], target[donor])),
            'real_vs_generated_shuffled_anchor': paired_scores(shuffled_real, shuffled_fake),
            'paired_margin_correct_vs_shuffled_anchor': paired_scores(real_score-fake_score,
                shuffled_real-shuffled_fake)}
    return result


@torch.no_grad()
def diagnose(path, observed, ids, device='cpu', prefixes=(8, 32), refinement_steps=(1, 2, 3, 7)):
    path = Path(path)
    json.loads((path/'summary.json').read_text())  # Never inspect an unfinished run.
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = h.Config(**saved['config'])
    if cfg.g_state_dim:
        raise ValueError('This diagnostic requires stateless G')
    g, d, prior, _ = h.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval().requires_grad_(False)
    critic = None
    if saved.get('gibbs_critic') is not None:
        from experiments.memory_gibbs import GibbsCritic
        critic = GibbsCritic(cfg).to(device)
        critic.load_state_dict(saved['gibbs_critic'])
        critic.eval().requires_grad_(False)
    observed, ids = observed.to(device), ids.to(device)
    z = prior(ids)
    return {'name': cfg.name, 'source': str(path), 'steps': cfg.steps,
            'checkpoint_sha256': hashlib.sha256((path/'model.pt').read_bytes()).hexdigest(),
            'gibbs_config': {key: value for key, value in vars(cfg).items() if key.startswith('gibbs_')},
            'prefixes': {str(n): gibbs_panel(g, d, z, observed, n, critic, refinement_steps)
                         for n in prefixes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--prefixes', nargs='+', type=int, default=[8, 32])
    parser.add_argument('--refinement-steps', nargs='+', type=int, default=[1, 2, 3, 7])
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(7179381)
    observed, _, ids = episodes(args.count, max(args.prefixes)+2, rng)
    rows = []
    for path in args.runs:
        rows.append(diagnose(path, observed, ids, args.device, args.prefixes, args.refinement_steps))
        print(json.dumps({'event': 'gibbs_diagnostic_complete', 'name': rows[-1]['name']}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'evaluation_only': True, 'count': args.count,
        'definition': 'Fresh noisy circle histories from the training process distribution; identical '
        'histories and learned particle IDs across models. Histories are held out, particles are not. '
        'Each panel fixes real-prefix memory, particle and clock. Refinement counts are inference '
        'interventions inside one event, counted as total decoder calls (one means no refinement), '
        'without memory writes or time advances. Extra refinement '
        'counts can be out of training support. All MSE values are evaluation diagnostics, never '
        'training objectives. Point errors use noisy observations. Real-decode error uses the actual '
        'target inside its encoder and is a reconstruction probe, not a forecast. Latent distances '
        'and variances are model-specific and cannot establish quality across representations. '
        'Zero/shuffle interventions measure output sensitivity and local error, not long-run memory '
        'benefit. K compares (E(M,x_real),x_real) with (h_producer,x_fake), never a post-write memory '
        'against a pre-write memory. Shuffling latent breaks pair compatibility; this is descriptive '
        'score dependence, not a causal attribution. Critic scores are uncalibrated across models. '
        'Comparisons hold anchor fixed; paired-margin conditioning differences cancel anchor-only '
        'offsets. A baseline without Gibbs machinery reports ordinary G errors and memory '
        'interventions. Completed runs only; no optimization or generated trajectory training.',
        'results': rows}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
