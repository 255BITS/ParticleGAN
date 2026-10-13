"""Read saved Adam state; no model construction, gradients, sampling or updates.

The algebraic last displacement is recovered from terminal Adam moments. Its
inverse is an approximate pre-update snapshot, not an archived checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.boundaries import prior_control_binding
from experiments.forge.planning import load_idea

ROOT = Path(__file__).resolve().parents[4]
QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012/queue')
ATTEMPTS = {
    'momentum': ('r1r2-five-track-momentum-v1', 'a3991f4b7aeb452cb6a7caa299ad8ff2'),
    'amsgrad': ('r1r2-five-track-optimizer-v1', '1dda7bf5c88942368359f1ac207e9e93'),
}
TASKS = ('gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
         'ring16_acquisition', 'five_word_joint_smoke', 'clockfree_audit_measurement_v1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_delta(state, group):
    """Native Adam terminal moment algebra, including bias correction/epsilon."""
    assert group['weight_decay'] == 0 and not group.get('maximize', False)
    step = int(state['step'])
    b1, b2 = group['betas']
    moment = state['exp_avg'].double()
    variance = state['max_exp_avg_sq' if group['amsgrad'] else 'exp_avg_sq'].double()
    return -group['lr'] * (moment / (1 - b1 ** step)) / (
        (variance / (1 - b2 ** step)).sqrt() + group['eps'])


def role_stats(optimizer, group):
    deltas = [last_delta(optimizer['state'][i], group) for i in group['params']]
    flat = torch.cat([x.flatten() for x in deltas])
    result = {
        'betas': list(group['betas']), 'lr': group['lr'], 'amsgrad': group['amsgrad'],
        'coordinates': flat.numel(), 'moving_coordinates': int((flat != 0).sum()),
        'last_displacement_l2': float(flat.norm()),
    }
    if group['amsgrad']:
        ratios = []
        for index in group['params']:
            state = optimizer['state'][index]
            active = state['exp_avg'] != 0
            step = int(state['step'])
            correction = 1 - group['betas'][1] ** step
            v = state['exp_avg_sq'].double() / correction
            vmax = state['max_exp_avg_sq'].double() / correction
            ratios.append(((v.sqrt() + group['eps']) / (vmax.sqrt() + group['eps']))[active])
        ratio = torch.cat(ratios)
        result['same_moment_amsgrad_step_ratio_min_median_p90_max'] = torch.quantile(
            ratio, ratio.new_tensor([0., .5, .9, 1.])).tolist()
    return result, deltas


def forward_and_jacobian(params, z):
    w0, b0, w1, b1, w2, b2 = params
    a = z @ w0.T + b0
    b = torch.nn.functional.leaky_relu(a, .2) @ w1.T + b1
    output = torch.nn.functional.leaky_relu(b, .2) @ w2.T + b2
    slope_a = torch.where(a >= 0, 1., .2)
    slope_b = torch.where(b >= 0, 1., .2)
    jacobian = torch.einsum('oi,bi,ij,bj,jk->bok', w2, slope_b, w1, slope_a, w0)
    return output, jacobian, torch.cat([a >= 0, b >= 0], dim=1)


@torch.no_grad()
def analyze(label, study, attempt):
    directory = QUEUE / study / attempt
    request = read(directory / 'request.json')['request']
    task = request['tasks']['ring16_acquisition']
    saved_path = directory / 'provenance/provenance-state.pt'
    trainer = torch.load(saved_path, map_location='cpu', weights_only=False)['trainer']
    assert trainer['completed_steps'] == 1600
    g_opt, d_opt = trainer['optimizers']
    g_stats, g_delta = role_stats(g_opt, g_opt['param_groups'][0])
    p_stats, p_delta = role_stats(g_opt, g_opt['param_groups'][1])
    d_stats, _ = role_stats(d_opt, d_opt['param_groups'][0])
    params = [value.double() for value in trainer['models']['G'].values()]
    z = trainer['models']['prior']['z'].double()
    assert z.shape == (256, 4) and len(params) == len(g_delta) == 6 and len(p_delta) == 1
    row_motion = p_delta[0].norm(dim=1)
    p_stats.update(rows=256, moving_rows=int((row_motion != 0).sum()),
                   row_displacement_min_median_p90_max=torch.quantile(
                       row_motion, row_motion.new_tensor([0., .5, .9, 1.])).tolist())
    z_before = z - p_delta[0]
    g_before = [parameter - delta for parameter, delta in zip(params, g_delta)]
    before = forward_and_jacobian(g_before, z_before)
    after_g = forward_and_jacobian(params, z_before)
    after_prior = forward_and_jacobian(params, z)
    g_motion, prior_motion = after_g[0] - before[0], after_prior[0] - after_g[0]
    active = (g_motion.norm(dim=1) > 0) & (prior_motion.norm(dim=1) > 0)
    cosine = (g_motion[active] * prior_motion[active]).sum(1) / (
        g_motion[active].norm(dim=1) * prior_motion[active].norm(dim=1))
    means = z.new_tensor(task['execution']['host_definition']['means'])
    labels = torch.cdist(after_prior[0], means).argmin(1)
    radial = means[labels] / means[labels].norm(dim=1, keepdim=True)
    tangent = torch.stack([-radial[:, 1], radial[:, 0]], dim=1)
    shape = {}
    for name, panel in [('before_algebraic', before), ('after_g', after_g), ('after_prior', after_prior)]:
        # sigma_prior=.1 and sigma_target=.1 cancel in the variance ratio.
        shape[name] = {
            'mean_radial_local_variance_ratio': float(torch.einsum('bo,bok->bk', radial, panel[1]).square().sum(1).mean()),
            'mean_tangent_local_variance_ratio': float(torch.einsum('bo,bok->bk', tangent, panel[1]).square().sum(1).mean()),
        }
    batch = task['execution']['host_definition']['batch']
    return {
        'label': label, 'attempt_id': attempt, 'candidate_id': request['candidate']['id'],
        'candidate_revision': request['candidate_revision'],
        'source_digest': request['source']['digest'], 'execution_commit': request['source']['origin_commit'],
        'artifact': str(saved_path), 'artifact_sha256': sha(saved_path),
        'request_sha256': sha(directory / 'request.json'), 'completed_steps': 1600,
        'optimizer_roles': {'G': g_stats, 'prior': p_stats, 'D': d_stats},
        'prior_covariance_eigenvalues': torch.linalg.eigvalsh(torch.cov(z.T)).tolist(),
        'prior_rows_moved_outside_last_g_batch_lower_bound': max(0, p_stats['moving_rows'] - batch),
        'algebraic_last_step': {
            'g_center_displacement_mean_norm': float(g_motion.norm(dim=1).mean()),
            'prior_center_displacement_mean_norm': float(prior_motion.norm(dim=1).mean()),
            'total_center_displacement_mean_norm': float((after_prior[0] - before[0]).norm(dim=1).mean()),
            'cosine_valid_rows': int(active.sum()), 'opposing_center_displacement_rows': int((cosine < 0).sum()),
            'mean_center_displacement_cosine': float(cosine.mean()),
            'prior_only_activation_pattern_changed_rows': int((after_g[2] != after_prior[2]).any(dim=1).sum()),
            'local_shape_256_row_weighted': shape,
        },
    }, request


def bindings(request):
    result = {}
    for candidate_id in ('r1r2-role-m001-v1', 'r1r2-role-m110-v1'):
        candidate = load_idea(ROOT, candidate_id)
        tasks = {}
        for task_id in TASKS:
            task = request['tasks'][task_id]
            context = task_formulation_context(candidate, task, request['protocol'], root=ROOT)
            recipe = context.recipe
            tasks[task_id] = {
                'G_E_betas': list(recipe.betas), 'D_betas': list(recipe.d_betas or recipe.betas),
                'sampled_prior_betas': list(recipe.prior_betas or recipe.betas),
                'prior_binding': prior_control_binding(task), 'amsgrad': recipe.amsgrad,
            }
        result[candidate_id] = tasks
    return result


def main():
    torch.set_num_threads(1)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    results = []
    request = None
    for label, (study, attempt) in ATTEMPTS.items():
        result, request = analyze(label, study, attempt)
        results.append(result)
    report = {'schema_version': 1, 'scope': 'saved_terminal_state_only',
              'reproducer_sha256': sha(Path(__file__)), 'results': results,
              'registered_role_bindings': bindings(request),
              'training_updates': 0, 'sampling_draws': 0,
              'python_rng_unchanged': python_rng == random.getstate(),
              'torch_cpu_rng_unchanged': torch.equal(torch_rng, torch.get_rng_state()),
              'limitations': [
                  'Algebraic inverse of a float32 Adam displacement is approximate, not a saved pre-update checkpoint.',
                  'Terminal moments reveal the last applied displacement, not a future gradient or step.',
                  'Local Jacobians at row centers omit finite-noise activation crossings and sample-to-mode reassignment.',
                  'State trajectories differ; comparisons establish descriptive associations, not a causal role effect.',
                  'No consumed row indices are reconstructed; off-batch motion uses batch128 as a strict upper bound.',
              ]}
    assert report['python_rng_unchanged'] and report['torch_cpu_rng_unchanged']
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
