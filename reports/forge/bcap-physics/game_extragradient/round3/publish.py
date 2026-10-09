"""Export this completed frozen campaign, using original receipts and saved outputs.

No training, model calls, random draws, live regrading or leaderboard generation.
Original bulk envelopes are mirrored byte-for-byte into the ignored local tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import render
from experiments.forge.artifacts import verify_artifacts
import torch

CAMPAIGN = 'game-extragradient-round3-v1'
STUDIES = {'candidate': 'game-extragradient-round3-candidate-v1',
           'control': 'game-extragradient-round3-control-v1'}


def variation(points, key):
    values = [float(p[key]) for p in points if key in p]
    return {'count': len(values), 'minimum': min(values), 'maximum': max(values),
            'total_variation': sum(abs(a - b) for a, b in zip(values, values[1:]))} if values else None


def numeric_pass(task, point):
    for key, op, bound in task['evaluation']['thresholds']:
        value = point.get(key)
        if value is None or not {'>=': lambda: value >= bound, '<=': lambda: value <= bound,
                                 '==': lambda: value == bound}[op]():
            return False
    return True


def summarize(task, row):
    evidence = row.get('evidence', {})
    points = evidence.get('observations', [])
    passes = [numeric_pass(task, p) for p in points]
    suffix = 0
    for passed in reversed(passes):
        if not passed:
            break
        suffix += 1
    result = {'gate_status': row['gate_status'], 'reason': row.get('reason'),
              'final_metrics': row.get('metrics', {}), 'evaluator_result': row.get('evaluator_result', {}),
              'observed_count': len(points), 'full_pass_count': sum(passes),
              'terminal_passing_suffix': suffix,
              'terminal_observation': points[-1] if points else None,
              'cost': row.get('cost', {})}
    if task['id'].startswith('gaussian1d'):
        result['stability_proxies'] = {k: variation(points, k) for k in ('std_ratio', 'cdf_ks', 'mean_error_sigma')}
        if task['id'] == 'gaussian1d_stability':
            result['stationary_passes'] = sum(numeric_pass(task, p) for p in points if p['step'] <= 4000)
            result['shift_hold_passes'] = sum(numeric_pass(task, p) for p in points if p['step'] > 5000)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--certificates', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    state = read_json(args.queue / 'queue/state.json')
    entries = {arm: [(rid, entry) for rid, entry in state['submissions'].items()
                     if entry['request'].get('study', {}).get('id') == sid]
               for arm, sid in STUDIES.items()}
    if any(len(v) != 1 for v in entries.values()):
        raise ValueError('exactly one frozen submission per study is required')
    entries = {arm: values[0] for arm, values in entries.items()}
    if any(entry['status'] in {'queued', 'running', 'paused'} for _, entry in entries.values()):
        raise ValueError('campaign is active; publish only after all bounded jobs end')
    output = args.output.resolve()
    results, provenance, media = {}, {}, []
    matched = {}
    for arm, (rid, entry) in entries.items():
        request = entry['request']
        if request['campaign_id'] != CAMPAIGN or request['protocol']['seed'] != 0:
            raise ValueError('wrong campaign/seed')
        if stable_hash(request['source']['files']) != request['source']['digest']:
            raise ValueError('frozen source map digest differs')
        results[arm] = {'request_id': rid, 'candidate_id': request['candidate']['id'],
                        'candidate_revision': request['candidate_revision'], 'tasks': {}}
        provenance[arm] = {'source': {k: request['source'][k] for k in ('origin_commit', 'digest', 'snapshot_path')},
                           'runtime': request['runtime'], 'compute_profiles': request['compute_profiles'],
                           'protocol_sha256': stable_hash(request['protocol']),
                           'study_sha256': stable_hash(request['study']), 'attempts': [], 'task_contracts': {}}
        matched[arm] = {}
        for jobdef in request['jobs']:
            task_id = jobdef['task_id']
            task = request['tasks'][task_id]
            job = state['jobs'][jobdef['compatibility_key']]
            provenance[arm]['task_contracts'][task_id] = {
                'task_sha256': stable_hash(task), 'initializer': task['execution']['initializer'],
                'prior': task['execution'].get('prior'), 'updates': task['execution']['steps'],
                'evaluation': task['evaluation'], 'dependencies': task.get('dependencies', [])}
            result = job.get('result')
            if not result:
                dependencies = task.get('dependencies', [])
                deps = [d['task'] if isinstance(d, dict) else d for d in dependencies]
                produced = [r for d in request['jobs']
                            for r in (state['jobs'][d['compatibility_key']].get('result') or {}).get('task_results', [])]
                gates = {r['task_id']: r['gate_status'] for r in produced}
                if not task.get('preflight_blockers') and (task_id != 'gaussian1d_stability' or not deps
                        or all(gates.get(d) == 'PASS' for d in deps)):
                    raise ValueError('unexplained missing task evidence: ' + task_id)
                results[arm]['tasks'][task_id] = {'gate_status': 'BLOCKED',
                    'reason': '; '.join(task.get('preflight_blockers', [])) or job.get('reason', job.get('blocked_reason', 'own smoke checkpoint prerequisite did not pass')),
                    'job_status': job['status'], 'paid_seconds': 0}
                continue
            attempt = result['attempt_id']
            original = args.certificates / attempt
            envelope, stored, certificate = [read_json(original / (name + '.json'))
                                             for name in ('request', 'result', 'evidence')]
            producer = envelope.get('request', envelope)
            local = Path(certificate['local_artifact_root'])
            if (stored != result or read_json(local / 'result.json') != result
                    or certificate['result_hash'] != stable_hash(result)
                    or certificate['source'] != producer['source']
                    or certificate['runtime'] != producer['runtime']
                    or result['candidate_revision'] != producer['candidate_revision']
                    or producer['source']['digest'] != request['source']['digest']
                    or producer['candidate_revision'] != request['candidate_revision']):
                raise ValueError('source/result/runtime receipt differs for ' + attempt)
            mirror = ROOT / 'reports/forge/attempts' / attempt
            mirror.mkdir(parents=True, exist_ok=True)
            for name in ('request', 'result', 'evidence'):
                source, destination = original / (name + '.json'), mirror / (name + '.json')
                if source.resolve() != destination.resolve():
                    shutil.copyfile(source, destination)
            row = next(r for r in result['task_results'] if r['task_id'] == task_id)
            if row['compatibility_key'] != jobdef['compatibility_key']:
                raise ValueError('task compatibility differs')
            results[arm]['tasks'][task_id] = summarize(task, row)
            evidence = row.get('evidence', {})
            matched[arm][task_id] = {k: v for k, v in evidence.items()
                if any(token in k for token in ('initial', 'batch_sequence', 'stream')) or k == 'data_sha256'}
            audits = evidence.get('rng_audits', [])
            matched[arm][task_id]['rng_audit_summary'] = {'count': len(audits), 'sha256': stable_hash(audits),
                'unintended_rng_deviations': sum(a.get('unintended_rng_deviations', 0) for a in audits)}
            checkpoint = evidence.get('provenance_checkpoint')
            if checkpoint:
                artifact_root = Path(checkpoint['artifact_root'])
                verify_artifacts(artifact_root, checkpoint['artifact_manifest'])
                cp_path = artifact_root / checkpoint['path']
                if file_hash(cp_path) != checkpoint['sha256']:
                    raise ValueError('checkpoint bytes differ')
                saved = torch.load(cp_path, map_location='cpu', weights_only=True)
                applied = saved.get('applied', saved)
                initialization = applied.get('initialization', {})
                matched[arm][task_id]['initialization'] = {name: {k: v for k, v in proof.items()
                    if k in ('initializer', 'initial_state_sha256')} if isinstance(proof, dict) else proof
                    for name, proof in initialization.items()}
                matched[arm][task_id]['final_named_stream_states'] = checkpoint.get('named_stream_state_sha256')
                recipe = applied.get('recipe', {})
                matched[arm][task_id]['actual_recipe_sha256'] = stable_hash(recipe)
                matched[arm][task_id]['actual_recipe'] = {k: recipe[k] for k in (
                    'loss', 'critic_formulation', 'optimizer_family', 'optimizer_momentum',
                    'optimizer_smoothing', 'optimizer_convolution', 'same_batch_extragradient',
                    'lr', 'd_lr_mult', 'prior_lr_mult', 'lr_floor', 'network_lr_floor',
                    'lr_schedule', 'total_steps', 'reg_arm', 'reg_coeff', 'reg_kappa', 'reg_every',
                    'input_noise_std', 'output_noise_std', 'ema_decay', 'prior_reg',
                    'latent_damping_max_rate', 'direct_particle_gain', 'reg_anchor_weight', 'd_guard_ratio') if k in recipe}
                matched[arm][task_id]['optimizer_group_bindings'] = applied.get('optimizer_group_bindings')
                optimizer_packets = saved.get('optimizers', {})
                if 'trainer' in saved:
                    optimizer_packets = dict(enumerate(saved['trainer']['optimizers']))
                completed = saved.get('trainer', {}).get('completed_steps')
                steps = [v['step'] for opt in optimizer_packets.values() if isinstance(opt, dict)
                         for v in opt.get('state', {}).values() if 'step' in v]
                matched[arm][task_id]['logical_optimizer_clock'] = {'completed_steps': completed,
                    'minimum_parameter_tick': min(steps) if steps else None,
                    'maximum_parameter_tick': max(steps) if steps else None}
                if arm == 'candidate' and not recipe.get('same_batch_extragradient'):
                    raise ValueError('candidate did not checkpoint the actual correction recipe')
                if completed is not None and steps and max(steps) != completed:
                    raise ValueError('optimizer clock advanced inconsistently')
            receipt = {'attempt_id': attempt, 'result_hash': stable_hash(result),
                       'envelope_hashes': {name: file_hash(original / (name + '.json')) for name in ('request', 'result', 'evidence')},
                       'local_artifact_root': str(local), 'task_id': task_id,
                       'compatibility_key': jobdef['compatibility_key'],
                       'artifact_manifest': evidence.get('artifact_manifest'),
                       'provenance_checkpoint': evidence.get('provenance_checkpoint')}
            provenance[arm]['attempts'].append(receipt)
            if row.get('evidence') and row['gate_status'] in {'PASS', 'FAIL'}:
                gif = output / 'media' / arm / (task_id + '.gif')
                cached_receipt = gif.with_suffix('.json')
                if gif.exists() and cached_receipt.exists():
                    rendered = read_json(cached_receipt)
                    if rendered['gif_sha256'] != file_hash(gif) or rendered['observations_sha256'] != stable_hash(evidence['observations']):
                        raise ValueError('existing media differs from the original observations')
                    for path, digest in rendered['source_inputs'].items():
                        if file_hash(Path(path)) != digest:
                            raise ValueError('media source bytes differ')
                else:
                    rendered = render(task, deepcopy(row), local, gif)
                media.append({'arm': arm, 'attempt_id': attempt, **rendered})
            print(json.dumps({'arm': arm, 'task': task_id, 'gate_status': row['gate_status'],
                              'attempt': attempt}), flush=True)
        results[arm]['counts'] = dict(Counter(t['gate_status'] for t in results[arm]['tasks'].values()))
    source_equal = provenance['candidate']['source'] == provenance['control']['source']
    runtime_equal = provenance['candidate']['runtime'] == provenance['control']['runtime']
    if not source_equal or not runtime_equal:
        raise ValueError('candidate/control source or runtime differs')
    checks = {}
    for task_id in matched['candidate'].keys() & matched['control'].keys():
        a, b = matched['candidate'][task_id], matched['control'][task_id]
        checks[task_id] = {'initialization_equal': a.get('initialization') == b.get('initialization'),
                          'final_named_stream_states_equal': a.get('final_named_stream_states') == b.get('final_named_stream_states'),
                          'note': 'Final stream parity is a state audit, not direct bytewise consumed-batch proof.'}
        if not checks[task_id]['initialization_equal']:
            raise ValueError('matched initializer differs on ' + task_id)
    provenance['matched_checks'] = {'source_equal': source_equal, 'runtime_equal': runtime_equal,
                                    'task_checks': checks, 'evidence_stream_projections': matched}
    source_files = next(iter(entries.values()))[1]['request']['source']['files']
    changed = [path for path, digest in source_files.items() if file_hash(ROOT / path) != digest]
    if changed:
        raise ValueError('measured scientific source differs at publication: ' + str(changed))
    provenance['publication_source_verification'] = {'verified_files': len(source_files),
        'changed_measured_files': changed, 'publication_scripts_are_training_evidence': False}
    results['campaign'] = state['campaigns'][CAMPAIGN]
    results['qualification_input'] = False
    atomic_json(output / 'results.json', results)
    atomic_json(output / 'provenance.json', provenance)
    atomic_json(output / 'media-index.json', {'qualification_input': False,
        'optimizer_updates_added': 0, 'sampling_draws_added': 0, 'receipts': media})


if __name__ == '__main__':
    main()
