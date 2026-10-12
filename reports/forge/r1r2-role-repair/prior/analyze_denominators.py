"""Prospective epsilon readout from archived terminal native Adam moments.

No candidate declaration, construction, sampling, gradient or update occurs.
Thresholds describe saved scales; they are not a registered parameter grid.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
PUBLICATIONS = ('reports/forge/r1r2-tier1-search/publication/results.json',
                'reports/forge/r1r2-five-track-repair/publication/results.json')
CANDIDATES = {
    'r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9': 'constant',
    'r1r2-five-track-momentum-v1': 'momentum',
    'r1r2-amsgrad-repair-v1': 'amsgrad',
}
TASKS = ('ring16_acquisition', 'gaussian1d_smoke', 'five_word_joint_smoke', 'two_pole')
EPSILONS = (1e-6, 1e-5, 1e-4, 1e-3)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(value):
    if value.numel() == 0:
        return None
    return torch.quantile(value, value.new_tensor([0., .1, .5, .9, 1.])).tolist()


def roles(saved, task_id):
    """Map actual public optimizer groups, not the host bookkeeping labels."""
    if 'trainer' in saved:
        g, d = saved['trainer']['optimizers']
        assert len(g['param_groups']) == 2 and len(d['param_groups']) == 1
        return [('G', g, g['param_groups'][0], 'eps'),
                ('P', g, g['param_groups'][1], 'prior_eps_or_eps'),
                ('D', d, d['param_groups'][0], 'd_eps_or_eps')]
    if task_id == 'five_word_joint_smoke':
        g, d = saved['fixture']['api_state']['optimizers']
        assert len(g['param_groups']) == 3 and len(d['param_groups']) == 1
        return [('G', g, g['param_groups'][0], 'eps'),
                ('E', g, g['param_groups'][1], 'eps'),
                ('P', g, g['param_groups'][2], 'eps_only_prior_eps_not_consumed'),
                ('D', d, d['param_groups'][0], 'd_eps_or_eps')]
    assert task_id == 'two_pole'
    g_list = saved['optimizers']['generator']
    d = saved['optimizers']['discriminator']
    assert len(g_list) == 1 and len(g_list[0]['param_groups']) == 1
    return [('G_direct', g_list[0], g_list[0]['param_groups'][0], 'eps'),
            ('D', d, d['param_groups'][0], 'd_eps_or_eps')]


def group_stats(role, optimizer, group, epsilon_binding):
    denominators, moments, updates, steps = [], [], [], []
    assert group['weight_decay'] == 0 and not group.get('maximize', False)
    b1, b2 = group['betas']
    for index in group['params']:
        state = optimizer['state'][index]
        t = int(state['step'])
        variance = state['max_exp_avg_sq' if group['amsgrad'] else 'exp_avg_sq'].double()
        denominator = (variance / (1 - b2 ** t)).sqrt()
        moment = state['exp_avg'].double() / (1 - b1 ** t)
        update = -group['lr'] * moment / (denominator + group['eps'])
        denominators.append(denominator.flatten())
        moments.append(moment.flatten())
        updates.append(update.flatten())
        steps.append(t)
    denominator, moment, update = map(torch.cat, (denominators, moments, updates))
    assert torch.isfinite(denominator).all() and torch.isfinite(update).all()
    active = moment != 0
    energy = update.square()
    total_energy = float(energy.sum())
    epsilon_effects = {}
    for epsilon in EPSILONS:
        # Exact elementwise multiplier if terminal moment, variance, LR stay fixed.
        ratio = (denominator + group['eps']) / (denominator + epsilon)
        energy_share = float(energy[denominator < epsilon].sum()) / total_energy if total_energy else None
        epsilon_effects[str(epsilon)] = {
            'denominator_below_epsilon_fraction_all': float((denominator < epsilon).double().mean()),
            'denominator_below_epsilon_fraction_active': float((denominator[active] < epsilon).double().mean()) if active.any() else None,
            'baseline_update_energy_below_epsilon_fraction': energy_share,
            'same_moment_update_l2_ratio': float((energy * ratio.square()).sum().sqrt()) / total_energy ** .5 if total_energy else None,
            'same_moment_update_l1_ratio': float((update.abs() * ratio).sum() / update.abs().sum()) if total_energy else None,
            'same_moment_coordinate_step_ratio_min_p10_median_p90_max_active': quantiles(ratio[active]),
        }
    return {'role': role, 'epsilon_binding': epsilon_binding,
            'betas': list(group['betas']), 'lr': group['lr'], 'amsgrad': group['amsgrad'],
            'current_eps': group['eps'], 'state_steps_min_max': [min(steps), max(steps)],
            'coordinates': denominator.numel(), 'active_moment_coordinates': int(active.sum()),
            'denominator_min_p10_median_p90_max_all': quantiles(denominator),
            'denominator_min_p10_median_p90_max_active': quantiles(denominator[active]),
            'same_moment_baseline_update_l2': total_energy ** .5,
            'same_moment_baseline_update_max_abs': float(update.abs().max()),
            'same_moment_baseline_update_median_abs_active': float(update[active].abs().median()) if active.any() else None,
            'epsilon_effects': epsilon_effects}


def main():
    torch.set_num_threads(1)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    results = []
    for publication in PUBLICATIONS:
        for certificate in read(ROOT / publication)['certified_results']:
            candidate, task_id = certificate['candidate_id'], certificate['task_id']
            if candidate not in CANDIDATES or task_id not in TASKS:
                continue
            state_path = Path(certificate['checkpoint']['artifact_root']) / 'provenance-state.pt'
            state_sha = sha(state_path)
            assert state_sha == certificate['checkpoint']['sha256']
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            groups = roles(saved, task_id)
            present = [role for role, _, _, _ in groups]
            results.append({
                'recipe': CANDIDATES[candidate], 'task_id': task_id,
                'candidate_id': candidate, 'candidate_revision': certificate['candidate_revision'],
                'attempt_id': certificate['attempt_id'], 'source_digest': certificate['source_digest'],
                'checkpoint_completed_steps': certificate['checkpoint']['completed_steps'],
                'artifact': str(state_path), 'artifact_sha256': state_sha,
                'source_publication': publication, 'gate_status': certificate['gate_status'],
                'reported_gate_metrics': certificate['metrics'],
                'roles_present': present,
                'roles_absent': {'E': 'no encoder in this task'} if 'E' not in present else {},
                'optimizer_groups': [group_stats(*args) for args in groups],
            })
    assert len(results) == 12
    request_path = Path(results[0]['artifact']).parent.parent / 'request.json'
    frozen_source = read(request_path)['request']['source']
    binding_sources = {}
    for filename in ('particlegan/recipes.py', 'experiments/forge/behavior_adapters.py',
                     'benchmarks/toy_audit/api_images.py'):
        digest = sha(ROOT / filename)
        assert digest == frozen_source['files'][filename]
        binding_sources[filename] = digest
    for result in results:
        if result['task_id'] == 'two_pole':
            result['roles_absent'].update(G='direct generated coordinates replace a generator network',
                                         P='no sampled latent prior; host prior label refers to direct G coordinates')
    report = {
        'schema_version': 1, 'scope': 'prospective_saved_terminal_denominator_analysis',
        'reproducer_sha256': sha(Path(__file__)), 'diagnostic_epsilon_thresholds': list(EPSILONS),
        'quantile_order': ['minimum', 'p10', 'median', 'p90', 'maximum'],
        'denominator_definition': 'sqrt(v_t/(1-beta2**t)); use max_exp_avg_sq for AMSGrad; epsilon excluded',
        'same_moment_ratio_definition': '(denominator+current_eps)/(denominator+proposed_eps)',
        'epsilon_binding_source_sha256': binding_sources,
        'results': results, 'models_constructed': 0, 'gradient_calls': 0,
        'training_updates': 0, 'sampling_draws': 0,
        'python_rng_unchanged': python_rng == random.getstate(),
        'torch_cpu_rng_unchanged': torch.equal(torch_rng, torch.get_rng_state()),
        'limitations': [
            'Terminal moments only: no early or per-update denominator history is archived in these states.',
            'Same-moment ratios are algebraically exact counterfactual last-step scalings, not trained outcomes.',
            'A small denominator need not carry appreciable update energy or be a harmful direction.',
            'Changing epsilon alters subsequent gradients and moments; final pole margin cannot certify all80updates.',
            'WordFixture does not consume prior_eps in its manually constructed prior group; global eps and d_eps are consumed.',
        ],
    }
    assert report['python_rng_unchanged'] and report['torch_cpu_rng_unchanged']
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
