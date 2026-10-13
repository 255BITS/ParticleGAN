"""Saved Ring16 table/decoder geometry; no models, draws, gradients or updates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import torch

ROOT = Path(__file__).resolve().parents[4]
QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-spectral-repair-20261012/queue')
HELPER = ROOT / 'reports/forge/r1r2-five-track-repair/density_readout.py'
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
RING = 'ring16_acquisition'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covariance(rows):
    centered = rows - rows.mean(0)
    return centered.T @ centered / len(rows)


def aggregate(values):
    return {'count': len(values), 'mean': sum(values) / len(values),
            'min': min(values), 'max': max(values)} if values else {'count': 0, 'mean': None, 'min': None, 'max': None}


def forward_jacobian(models, locations):
    w = lambda i: models['G'][f'net.{i}.weight'].double()
    b = lambda i: models['G'][f'net.{i}.bias'].double()
    first = locations @ w(0).T + b(0)
    second = torch.nn.functional.leaky_relu(first, .2) @ w(2).T + b(2)
    centers = torch.nn.functional.leaky_relu(second, .2) @ w(4).T + b(4)
    jacobian = torch.einsum('oi,bi,ij,bj,jk->bok', w(4), torch.where(second >= 0, 1., .2),
                            w(2), torch.where(first >= 0, 1., .2), w(0))
    return centers, jacobian


def find_attempts():
    found = []
    publication = read(ROOT / 'reports/forge/r1r2-role-repair/publication/results.json')
    for role in ('m000', 'm110'):
        matches = [r for r in publication['certified_results'] if r['role'] == role and r['task_id'] == RING]
        assert len(matches) == 1
        record = matches[0]
        found.append((role, Path(record['checkpoint']['artifact_root']).parent,
                      record['checkpoint']['sha256']))
    for role, candidate in [('spectral', 'r1r2-dualnorm-repair-v1'),
                            ('critic_spectral', 'r1r2-dualnorm-critic-only-v1')]:
        matches = [p.parent for p in (QUEUE / candidate).glob('*/graded-result.json')
                   if RING in read(p)['grades']]
        assert len(matches) == 1
        found.append((role, matches[0], None))
    return found


@torch.no_grad()
def analyze(role, attempt, expected_state_sha, scorer):
    request = read(attempt / 'request.json')['request']
    grade = read(attempt / 'graded-result.json')['grades'][RING]
    assert request['source']['digest'] == SOURCE and request['protocol']['seed'] == 0
    task = request['tasks'][RING]
    spec, prior = task['execution']['host_definition'], task['execution']['prior']
    assert prior == {'kind': 'mog', 'sigma': .1, 'standardize': False}
    assert (spec['particles'], spec['z_dim'], spec['layers'], spec['hidden']) == (256, 4, 2, 64)
    state_path, sample_path = attempt / 'provenance/provenance-state.pt', attempt / 'observed-samples.pt'
    state_sha = sha(state_path)
    if expected_state_sha:
        assert state_sha == expected_state_sha
    saved = torch.load(state_path, map_location='cpu', weights_only=False)
    models = saved['trainer']['models']
    assert saved['trainer']['completed_steps'] == 1600
    panel = torch.load(sample_path, map_location='cpu', weights_only=True)[-1]
    assert panel['step'] == grade['metrics']['step'] == 1600
    points = panel['samples']
    assert points.shape == (4096, 2) and points.dtype == torch.float32
    metrics = scorer.gate_metrics(points, spec)
    for key, value in metrics.items():
        assert math.isclose(value, grade['metrics'][key], abs_tol=1e-6, rel_tol=1e-6)
    z = models['prior']['z'].double()
    assert z.shape == (256, 4) and math.isclose(float(models['prior']['sigma']), .1, abs_tol=1e-7)
    centers, jacobian = forward_jacobian(models, z)
    means = z.new_tensor(spec['means'])
    target_covariance = z.new_tensor(spec['covariances'])
    target_variance = float(target_covariance[0, 0, 0])
    assert torch.allclose(target_covariance, .01 * torch.eye(2).double().expand(16, 2, 2))
    row_labels = torch.cdist(centers, means).argmin(1)
    sample_labels = torch.cdist(points, points.new_tensor(spec['means'])).argmin(1)
    local_covariances = prior['sigma'] ** 2 * (jacobian @ jacobian.transpose(1, 2))
    components = []
    for component in range(16):
        row_mask, sample_mask = row_labels == component, sample_labels == component
        latent_rows, output_rows, members = z[row_mask], centers[row_mask], points[sample_mask].double()
        record = {'component': component, 'rows': len(latent_rows), 'samples': len(members),
                  'geometry_status': 'AVAILABLE' if len(latent_rows) >= 2 and len(members) >= 10 else 'UNAVAILABLE'}
        if record['geometry_status'] == 'AVAILABLE':
            radial = means[component] / means[component].norm()
            tangent = torch.stack([-radial[1], radial[0]])
            empirical, between = covariance(members), covariance(output_rows)
            noise = local_covariances[row_mask].mean(0)
            record['latent_within_component_trace'] = float(covariance(latent_rows).trace())
            record['noise_only_linear_proxy_covariance_error'] = float((noise - target_covariance[component]).norm() / target_covariance[component].norm())
            record['noise_only_linear_proxy_min_eigen_ratio'] = float(torch.linalg.eigvalsh(noise)[0] / target_variance)
            for name, direction in [('radial', radial), ('tangent', tangent)]:
                record[f'observed_{name}_variance_ratio'] = float(direction @ empirical @ direction / target_variance)
                record[f'mapped_center_{name}_variance_ratio'] = float(direction @ between @ direction / target_variance)
                record[f'local_noise_{name}_variance_ratio'] = float(direction @ noise @ direction / target_variance)
                record[f'linear_proxy_residual_{name}_variance_ratio'] = record[f'observed_{name}_variance_ratio'] - record[f'mapped_center_{name}_variance_ratio'] - record[f'local_noise_{name}_variance_ratio']
        components.append(record)
    available = [r for r in components if r['geometry_status'] == 'AVAILABLE']
    fields = ('latent_within_component_trace',
              'noise_only_linear_proxy_covariance_error', 'noise_only_linear_proxy_min_eigen_ratio',
              'observed_radial_variance_ratio', 'observed_tangent_variance_ratio',
              'mapped_center_radial_variance_ratio', 'mapped_center_tangent_variance_ratio',
              'local_noise_radial_variance_ratio', 'local_noise_tangent_variance_ratio',
              'linear_proxy_residual_radial_variance_ratio', 'linear_proxy_residual_tangent_variance_ratio')
    nearest = torch.cdist(z, z)
    nearest.fill_diagonal_(float('inf'))
    nearest = nearest.min(1).values
    prior_controls = ('optimizer_family', 'prior_update', 'prior_regularizer', 'prior_reg',
                      'prior_l2', 'prior_lr_mult', 'prior_betas', 'prior_proximal_strength')
    result = {
        'role': role, 'candidate_id': request['candidate']['id'],
        'candidate_revision': request['candidate_revision'], 'attempt_id': attempt.name,
        'source_digest': SOURCE, 'execution_commit': request['source']['origin_commit'],
        'artifact_root': str(attempt), 'artifact_sha256': {
            name: sha(attempt / name) for name in ('request.json', 'graded-result.json', 'observed-samples.pt', 'provenance/provenance-state.pt')},
        'actual_initial_prior_receipt': saved['initialization']['prior'],
        'initial_table_values': {'status': 'UNAVAILABLE', 'reason': 'Only initial hashes and parameter seeds are archived; no replay/draws authorized.'},
        'recipe_prior_controls': {k: request['candidate']['resolved_recipe'][k] for k in prior_controls},
        'gate_status': grade['gate_status'], 'endpoint_metrics': metrics,
        'terminal_passing_suffix': grade['evaluator_result']['convergence']['passing_suffix'],
        'prior_table': {'global_population_covariance_eigenvalues': torch.linalg.eigvalsh(covariance(z)).tolist(),
                        'global_population_covariance_trace': float(covariance(z).trace()),
                        'mean_squared_radius': float(z.square().sum(1).mean()),
                        'nearest_row_distance_min_median_max': [float(nearest.min()), float(nearest.median()), float(nearest.max())],
                        'within_component_covariance_trace_row_weighted': sum(r['rows'] * r['latent_within_component_trace'] for r in available) / sum(r['rows'] for r in available)},
        'generator': {'weight_operator_norms': {f'net.{i}.weight': float(torch.linalg.svdvals(models['G'][f'net.{i}.weight'].double())[0]) for i in (0, 2, 4)},
                      'mean_jacobian_frobenius_squared': float(jacobian.square().sum((1, 2)).mean()),
                      'jacobian_singular_values_mean': torch.linalg.svdvals(jacobian).mean(0).tolist()},
        'component_unweighted': {field: aggregate([r[field] for r in available]) for field in fields},
        'components': components,
    }
    return result, models, z, means


def main():
    torch.set_num_threads(1)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    module_spec = importlib.util.spec_from_file_location('r1r2_saved_density_helper', HELPER)
    scorer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(scorer)
    results, states = [], {}
    for role, attempt, expected_sha in find_attempts():
        result, models, z, means = analyze(role, attempt, expected_sha, scorer)
        results.append(result)
        states[role] = (models, z, means)
    crossed = []
    for g_role, (models, _, _) in states.items():
        for p_role, (_, z, means) in states.items():
            centers, jacobian = forward_jacobian(models, z)
            labels = torch.cdist(centers, means).argmin(1)
            crossed.append({'generator_endpoint': g_role, 'table_endpoint': p_role,
                            'mean_jacobian_frobenius_squared': float(jacobian.square().sum((1, 2)).mean()),
                            'mapped_centers_outside_target_3sigma': int(((centers - means[labels]).norm(dim=1) > .3).sum())})
    report = {'schema_version': 1, 'scope': 'saved_endpoint_table_and_generator_diagnosis',
              'reproducer_sha256': sha(Path(__file__)), 'gate_helper_sha256': sha(HELPER),
              'target_component_sigma': .1, 'fixed_prior_component_sigma': .1,
              'results': results, 'crossed_saved_state_jacobian_diagnostics': crossed,
              'models_constructed': 0, 'gradient_calls': 0, 'sampling_draws': 0, 'optimizer_updates': 0,
              'python_rng_unchanged': python_rng == random.getstate(),
              'torch_cpu_rng_unchanged': torch.equal(torch_rng, torch.get_rng_state()),
              'limitations': [
                  'Initial table values are unavailable; endpoint spread differences are not measured per-run expansion from initialization.',
                  'Local noise covariance is first-order at each saved row, not an exact finite-noise variance decomposition.',
                  'Saved sample row identities are unavailable; row weights and nearest-mode assignments need not match empirical sample weights.',
                  'Crossed saved G/table evaluations are functional diagnostics of untrained combinations, not qualified model samples or gate results.',
                  'Endpoint differences mix trained feedback trajectories; they cannot identify a unique causal prior policy.',
              ]}
    slow = [r for r in read(ROOT / 'reports/forge/r1r2-five-track-repair/publication/results.json')['certified_results']
            if r['candidate_id'] == 'r1r2-prior-slow-v1' and r['task_id'] == RING]
    assert len(slow) == 1
    old = slow[0]
    report['retained_prior_rate_negative'] = {
        key: old[key] for key in ('candidate_id', 'candidate_revision', 'attempt_id', 'source_digest', 'task_id', 'gate_status')}
    report['retained_prior_rate_negative'].update(
        source_report='reports/forge/r1r2-five-track-repair/publication/results.json',
        delta='prior_lr_mult 1 -> .25',
        checkpoint_sha256=old['checkpoint']['sha256'],
        endpoint_metrics={key: old['metrics'][key] for key in scorer.METRICS})
    assert report['python_rng_unchanged'] and report['torch_cpu_rng_unchanged']
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
