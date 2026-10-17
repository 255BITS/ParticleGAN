"""Read-only direct-Adam bound and certified earlier-observation analysis.

No optimizer update, model construction or sample draw is performed. Reproduce
with --evidence-root pointing to the hydrated v2 archive's original receipts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from particlegan.recipes import learning_rate_scale


ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def displacement_bound(*, lr, beta2, steps, horizon, start, floor):
    """Zero-origin per-coordinate bound: beta1=0, zero moments, gain <=2.

    v_t >= (1-beta2)*g_t**2 gives |g_t|/sqrt(vhat_t) <=
    sqrt((1-beta2**t)/(1-beta2)). Positive epsilon or AMSGrad can only
    decrease the step. Triangle inequality bounds every coordinate and hence
    the mean absolute movement. The first direct-response gain is exactly 1.
    """
    if not 0 <= beta2 < 1 or not 0 <= steps <= horizon or lr <= 0:
        raise ValueError('requires positive rate, constant beta2 in [0,1), and a bounded horizon')
    return sum(lr * learning_rate_scale(t - 1, horizon, start, floor)
               * (1. if t == 1 else 2.) * math.sqrt((1. - beta2 ** t) / (1. - beta2))
               for t in range(1, steps + 1))


def analyze(root, evidence_root):
    word_path = root / 'reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json'
    word = json.loads(word_path.read_text())
    task_path = root / 'configs/forge/tasks/two_pole.json'
    task = json.loads(task_path.read_text())
    recipe = word['recipe']
    horizon = task['execution']['steps']
    threshold = next(value for name, op, value in task['evaluation']['thresholds'] if name == 'mean_abs' and op == '>=')
    inputs = {str(path.relative_to(root)): sha256(path) for path in (word_path, task_path)}
    measured = []
    for relative in ('reports/forge/k3p-global-tier1-v2/receipts',
                     'reports/forge/k3p-global-tier1-v2/input-noise/receipts'):
        for path in sorted((root / relative).glob('*.json')):
            receipt = json.loads(path.read_text())
            inputs[str(path.relative_to(root))] = sha256(path)
            for item in receipt['tasks']:
                if item['task_id'] != 'two_pole':
                    continue
                row = {'candidate_id': receipt['candidate_id'], 'attempt_id': item['attempt_id'],
                       'source_digest': receipt['source_digest'], 'settings': {key: item['effective_recipe'][key]
                           for key in ('lr', 'd_lr_mult', 'reg_coeff', 'input_noise_std')},
                       'status': item['status'], 'final': item['metrics'],
                       'direct_gain_applications': item['guards']['mechanism_audit']['mechanisms']['direct_particle_gain']['applied'],
                       'critic_anchor_applications': item['guards']['mechanism_audit']['mechanisms']['critic_anchor']['applied']}
                if evidence_root is not None:
                    original = next(a for a in item['durable_certificate']['artifacts'] if a['path'].endswith('/result.json'))
                    result_path = evidence_root / original['path']
                    if sha256(result_path) != original['sha256']:
                        raise ValueError(f'original result byte identity differs: {original["path"]}')
                    result = json.loads(result_path.read_text())
                    observations = next(t['evidence']['observations'] for t in result['task_results'] if t['task_id'] == 'two_pole')
                    if len(observations) != 24:
                        raise ValueError('expected the unchanged complete 24-check curve')
                    peak = max(observations, key=lambda p: p['mean_abs'])
                    row.update(original_result=original, peak_movement=peak['mean_abs'], peak_step=peak['step'],
                               minimum_observed_critic_median=min(p['grad_med'] for p in observations),
                               maximum_observed_critic_median=max(p['grad_med'] for p in observations))
                measured.append(row)
    sources = {relative: sha256(root / relative) for relative in (
        'particlegan/k3p.py', 'particlegan/recipes.py', 'particlegan/grad_regularizers.py',
        'experiments/forge/behavior_adapters.py', 'benchmarks/locked_shared/two_pole.py',
        'reports/forge/k3p-global-tier1-v3/direct-rate-analysis.py')}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    bounds = []
    for beta2 in (.9, .95, .99, .999):
        maximum = displacement_bound(lr=recipe['lr'], beta2=beta2, steps=horizon, horizon=horizon,
                                     start=recipe['lr_anneal_start'], floor=recipe['lr_floor'])
        bounds.append({'lr': recipe['lr'], 'direct_particle_betas': [0., beta2],
                       'mean_absolute_displacement_upper_bound': maximum,
                       'excluded_by_bound': maximum < threshold,
                       'passing_is_not_implied': True})
    dimension = word['host']['definition']['joint_critic_widths'][0]
    return {'schema_version': 1, 'id': 'k3p-direct-rate-analysis-v1', 'scope': 'read_only_analytic_and_saved_receipt_diagnostic',
            'qualification_input': False, 'optimizer_updates': 0, 'sampling_draws': 0,
            'source_digest': digest, 'source_digest_kind': 'analysis_source_sha256_map_digest',
            'source_sha256': sources, 'input_sha256': inputs,
            'word_reference': {'id': word['id'], 'source_digest': word['source_digest'],
                               'candidate_revision': word['candidate_revision'], 'gate_status': word['grade']['gate_status']},
            'bound_assumptions': {'initial_coordinates': 'all zero', 'initial_adam_moments': 'all zero',
                'adam_beta1': 0., 'constant_beta2': True, 'weight_decay': 0., 'epsilon_nonnegative': True,
                'gain_first_step': 1., 'gain_maximum_later_steps': 2., 'optimizer_updates': horizon,
                'base_lr': recipe['lr'], 'lr_anneal_start': recipe['lr_anneal_start'], 'lr_floor': recipe['lr_floor'],
                'schedule_horizon': horizon, 'schedule_role': 'prior', 'prior_lr_mult_consumed': False},
            'proof': 'v_t >= (1-beta2)*g_t^2; |g_t|/sqrt(v_t/(1-beta2^t)) <= sqrt((1-beta2^t)/(1-beta2)); sum gain_t*lr_t times this bound. Triangle inequality bounds each zero-origin coordinate and mean absolute movement.',
            'unchanged_movement_threshold': threshold, 'direct_beta2_bounds': bounds,
            'early_k3p_real_gradient_units': {'formula': 'reg_coeff/(2*input_dimension) * mean(||grad_real D||^2)',
                'reg_coeff': recipe['reg_coeff'], 'two_pole_dimension': 1, 'word_joint_dimension': dimension,
                'two_pole_prefactor': recipe['reg_coeff'] / 2., 'word_prefactor': recipe['reg_coeff'] / (2. * dimension),
                'scope': 'early phase only; the late cap/proximity terms use their existing separate units'},
            'measured_two_pole': measured,
            'limits': ['The bound excludes a recipe; a larger bound does not establish a passing trajectory.',
                'Higher-rate c170 runs also plateau below .3 despite ample possible displacement, so the low-rate step bound is not their explanation.',
                'Critic gradient median covers real and fake inputs; it is not the exact per-particle generator force.',
                'Behavioral receipts retain numerical curves and optimizer/mechanism bindings, not reconstructible critic/particle checkpoints.',
                'Larger beta2 can slow responses to falling gradients; this numeric contrast has no promised gain.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(ROOT, args.evidence_root), indent=2, sort_keys=True) + '\n')
