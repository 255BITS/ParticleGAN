"""Three fixed-state counterfactuals; no training, cold gate or promotion."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator
from reports.toy100.chamfer_pullback import chamfer_pullback


STEPS = (1133, 1148, 1186)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def paired_grade(support, step):
    """Exactly the fixed seed9 prior draws and seed402+step legacy output noise."""
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(402 + step)
        idx = torch.randint(0, len(support), (mode_hold.EVAL_N,),
                            generator=torch.Generator().manual_seed(9))
        samples = support[idx] + .029 * torch.randn_like(support[idx])
        return mode_hold.diversity(samples, mode_hold.ring_means(), detailed=True)


def qualifies(row):
    return row['modes'] == mode_hold.N_MODES and row['hq'] >= mode_hold.PASS_HQ


def run_filter(states_path, diagnosis_path, output):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite evidence directory: {output}')
    output.mkdir(parents=True)
    state_bytes, diagnosis_bytes = states_path.read_bytes(), diagnosis_path.read_bytes()
    # Freeze read bytes before evaluation, so a concurrently archived input cannot
    # change this filter's reference or input provenance.
    (output / 'states.pt').write_bytes(state_bytes)
    (output / 'diagnosis.json.gz').write_bytes(gzip.compress(diagnosis_bytes, mtime=0))
    diagnosis = json.loads(diagnosis_bytes)
    if sha(state_bytes) != diagnosis['post_gan_counterfactual_sha256']:
        raise RuntimeError('tensor sidecar does not match exact-replay diagnosis')
    references = {row['step']: row for row in diagnosis['rows']}
    source_paths = [Path(__file__), ROOT / 'reports/toy100/chamfer_pullback.py',
                    ROOT / 'tests/test_chamfer_pullback.py',
                    ROOT / 'benchmarks/locked_shared/mlp.py',
                    ROOT / 'benchmarks/locked_shared/mode_hold.py']
    sources = {}
    for path in source_paths:
        data = path.read_bytes()
        destination = output / 'sources' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        sources[str(path.relative_to(ROOT))] = sha(data)
    declaration = {
        'scope': 'isolated_same_post_GAN_state_counterfactual_no_training',
        'shared_gate_eligible': False,
        'scratch_optimizer_policy': 'unit_mean_bidirectional_chamfer_prior_pullback',
        'steps': list(STEPS), 'forward_coefficient': 1., 'backward_coefficient': 1.,
        'max_halves': 8, 'pinv_rtol': 1e-6,
        'advance_rule': 'all three same-noise checks strictly improve over one-sided projection and meet unchanged 8 modes/HQ>=.9; then warm training is still required',
        'evaluation': {'samples': 4096, 'prior_index_seed': 9,
                       'output_noise_seed': '402+step', 'output_sigma': .029,
                       'center_access': 'diagnostic scoring only, absent from correction helper'},
        'moments': 'metric supplied from already-advanced Adam; no optimizer instantiated or stepped',
        'states_sha256': sha(state_bytes), 'diagnosis_sha256': sha(diagnosis_bytes),
        'warm_state_sha256': diagnosis['warm_state_sha256'],
        'original_final_state_sha256': diagnosis['final_state_sha256'],
        'source_sha256': sources,
        'runtime': {'torch': str(torch.__version__), 'python': platform.python_version(),
                    'threads': torch.get_num_threads(), 'cpu_capability': torch.backends.cpu.get_cpu_capability()},
    }
    write_json(output / 'declaration.json', declaration)
    states = torch.load(output / 'states.pt', weights_only=True, map_location='cpu')
    results = []
    for step in STEPS:
        state = states[str(step)]
        reference = references[step]
        with torch.random.fork_rng(devices=[]):
            clean = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                       mode_hold.N_HIDDEN, 2)
        clean.load_state_dict(state['generator_state'])
        z = torch.nn.Parameter(state['prior_z'].clone())
        metric, real = state['prior_adam_metric'].clone(), state['d_real_batch'].clone()
        with torch.no_grad():
            before = clean(z).detach()
        if not torch.equal(before, state['post_gan_clean_support']):
            raise RuntimeError(f'clean model reconstruction is not exact at {step}')
        original_support = torch.tensor(reference['post_projection']['points'], dtype=before.dtype)
        original_grade = paired_grade(original_support, step)
        if any(original_grade[key] != reference['observed_live'][key] for key in ('modes', 'hq')):
            raise RuntimeError(f'original paired noisy score does not reproduce at {step}')
        network_before = deepcopy(clean.state_dict())
        metric_before, real_before = metric.clone(), real.clone()
        rng = torch.get_rng_state().clone()
        row = chamfer_pullback(clean, z, real, metric, max_halves=8, pinv_rtol=1e-6)
        if not torch.equal(rng, torch.get_rng_state()):
            raise RuntimeError('correction consumed global RNG')
        if any(not torch.equal(value, clean.state_dict()[key]) for key, value in network_before.items()):
            raise RuntimeError('correction changed generator state')
        if not torch.equal(metric_before, metric) or not torch.equal(real_before, real):
            raise RuntimeError('correction changed its metric or captured minibatch')
        if z.grad is not None or any(p.grad is not None for p in clean.parameters()):
            raise RuntimeError('correction wrote gradient buffers')
        with torch.no_grad():
            after = clean(z).detach()
        after_grade = paired_grade(after, step)
        target = torch.tensor(row['target_points'], dtype=before.dtype)
        result = {
            'step': step, 'original_one_sided_noisy': original_grade,
            'post_gan_noisy': paired_grade(before, step),
            'symmetric_ideal_noisy': paired_grade(target, step),
            'symmetric_actual_noisy': after_grade,
            'clean_before': mode_hold.diversity(before, mode_hold.ring_means(), detailed=True),
            'clean_after': mode_hold.diversity(after, mode_hold.ring_means(), detailed=True),
            'strictly_improves': after_grade['hq'] > original_grade['hq'],
            'passes_unchanged_check': qualifies(after_grade),
            'exact_clean_model_reconstruction': True,
            'exact_original_noisy_score': True,
            'network_metric_real_rng_and_grad_buffers_unchanged': True,
            'optimizer_calls': 0, 'moment_updates': 0,
            'correction': row,
        }
        results.append(result)
        print(json.dumps({'step': step, 'original_hq': original_grade['hq'],
                          'symmetric_hq': after_grade['hq'], 'modes': after_grade['modes'],
                          'alpha': row['alpha'], 'objective_before': row['objective_before'],
                          'objective_after': row['objective_after'],
                          'qualifies': result['passes_unchanged_check']}, allow_nan=False), flush=True)
    eligible = all(r['strictly_improves'] and r['passes_unchanged_check'] for r in results)
    result = {'shared_gate_eligible': False, 'scope': declaration['scope'],
              'isolated_filter_status': 'SURVIVES_FOR_WARM_TEST' if eligible else 'FAIL_STOP',
              'training_performed': False, 'rows': results,
              'declaration_sha256': sha((output / 'declaration.json').read_bytes())}
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--diagnosis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_filter(args.states, args.diagnosis, args.output)


if __name__ == '__main__':
    main()
