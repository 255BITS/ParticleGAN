"""Validate and publish compact results from frozen Forge attempts; never train."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import export_attempt, render
from reports.forge.regenerate_technique_inventory import project_receipt

CANDIDATE = 'bispf-throughput-v1'
CONTROL = 'bispf-optimized-quintic-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(root, queue_root, destination, archive, *, partial=False):
    state = read_json(queue_root / 'queue/state.json')
    rows = []
    for job in state['jobs'].values():
        if not job.get('attempts'):
            continue
        if partial and job['status'] != 'terminal':
            continue
        require(job['status'] == 'terminal', 'A started job is still running')
        require(len(job['attempts']) == 1, 'Unexpected paid retry')
        attempt = job['attempts'][0]
        local = Path(attempt['path'])
        durable = root / 'reports/forge/attempts' / attempt['attempt_id']
        summary = project_receipt(root, attempt['attempt_id'])
        atomic_json(destination / 'receipts' / (attempt['attempt_id'] + '.json'), summary)
        stored = archive / 'attempts' / attempt['attempt_id']
        stored.mkdir(parents=True, exist_ok=True)
        for name in ('request.json', 'evidence.json', 'result.json'):
            shutil.copyfile(durable / name, stored / name)
        envelope = read_json(durable / 'request.json')
        request = envelope.get('request', envelope)
        result = read_json(durable / 'result.json')
        candidate = request['candidate']['id']
        require(candidate in (CANDIDATE, CONTROL), 'Unexpected candidate')
        require(request['protocol']['seed'] == 0, 'Protocol seed differs')
        for row, compact in zip(result['task_results'], summary['task_results']):
            evidence = row.get('evidence', {})
            provenance = evidence.get('provenance_checkpoint', {})
            item = dict(candidate=candidate, task=row['task_id'], status=row['gate_status'],
                        metrics=compact.get('metrics'), cost=compact.get('cost'),
                        evaluator=compact.get('evaluator_summary'), attempt=attempt['attempt_id'],
                        request=request['request_id'], study=request.get('study', {}).get('id'),
                        view=request['view']['id'], view_revision=request['view'].get('revision'),
                        source_digest=request['source']['digest'],
                        source_commit=request['source']['origin_commit'],
                        recipe_sha256=stable_hash(row.get('recipe', request['candidate'].get('resolved_recipe'))),
                        sampling=compact.get('sampling'), initialization_sha256=stable_hash(row.get('initialization')),
                        initial_rng_sha256=stable_hash(row.get('rng')),
                        final_named_streams=provenance.get('named_stream_state_sha256'),
                        provenance_checkpoint={k: provenance[k] for k in ('path', 'sha256', 'state_sha256', 'completed_steps') if k in provenance},
                        receipt=dict(path='receipts/' + attempt['attempt_id'] + '.json',
                                     sha256=file_hash(destination / 'receipts' / (attempt['attempt_id'] + '.json'))),
                        original_receipts={name: dict(path=str(stored / name), sha256=file_hash(stored / name))
                                           for name in ('request.json', 'evidence.json', 'result.json')},
                        local_artifact_root=str(local))
            if request['view']['id'] in ('bispf_fastpath_quality', 'bispf_throughput_small'):
                media_dir = destination / 'media' / candidate
                media_dir.mkdir(parents=True, exist_ok=True)
                export_attempt(durable, media_dir)
                gif = media_dir / (row['task_id'] + '.gif')
                item['media'] = dict(path=gif.relative_to(destination).as_posix(), sha256=file_hash(gif),
                                     optimizer_updates_added=0, sampling_draws_added=0)
            if request['view']['id'] == 'bispf_throughput_endurance' and evidence.get('dense'):
                gif = destination / 'media' / candidate / (row['task_id'] + '.gif')
                gif.parent.mkdir(parents=True, exist_ok=True)
                render(request['tasks'][row['task_id']],
                       {**row, 'evidence': {**evidence, 'observations': evidence['dense']}}, local, gif)
                item['media'] = dict(path=gif.relative_to(destination).as_posix(), sha256=file_hash(gif),
                                     optimizer_updates_added=0, sampling_draws_added=0)
            if row['task_id'].startswith('dcgan_performance_'):
                raw = read_json(local / 'raw-result.json')
                item['performance'] = {key: deepcopy(raw[key]) for key in
                                       ('metrics', 'identity', 'hardware', 'runtime', 'validity', 'spec', 'prior_policy')}
                # Source hashes already bind the immutable full source map once.
                item['performance']['identity']['source'].pop('files_sha256', None)
                gif = local / 'training.gif'
                if gif.is_file():
                    target = destination / 'media' / candidate / (row['task_id'] + '.gif')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(gif, target)
                    item['media'] = dict(path=target.relative_to(destination).as_posix(), sha256=file_hash(target),
                                         optimizer_updates_added=0, sampling_draws_added=0)
            rows.append(item)
    return rows, state


def matched(rows, tasks, *, performance=False):
    result = []
    for task in sorted(tasks):
        cells = {c: next(r for r in rows if r['candidate'] == c and r['task'] == task) for c in (CONTROL, CANDIDATE)}
        a, b = cells[CONTROL], cells[CANDIDATE]
        checks = {key: a[key] == b[key] for key in ('source_digest', 'initialization_sha256', 'initial_rng_sha256', 'sampling')}
        if performance:
            x, y = a['performance'], b['performance']
            checks.update({key: x['identity'][key] == y['identity'][key] for key in
                           ('initial_model_sha256', 'initial_parameter_sha256', 'initialization_sha256', 'batch_sequence_sha256')})
            checks['runtime'] = x['runtime'] == y['runtime']
            checks['physical_gpu'] = x['hardware']['physical_uuid'] == y['hardware']['physical_uuid']
            checks['spec'] = x['spec'] == y['spec']
            old, new = x['metrics']['update_ms_median'], y['metrics']['update_ms_median']
        else:
            x, y = a['final_named_streams'], b['final_named_streams']
            data = {k for k in x if '"data"' in k}
            checks['consumed_data_streams'] = bool(data) and {k: x[k] for k in data} == {k: y[k] for k in data}
            checks['physical_slot'] = a['cost'].get('device') == b['cost'].get('device')
            old, new = (r['cost']['phase_timing']['phases']['training_updates'] for r in (a, b))
            old, new = old['seconds'] * 1000 / old['calls'], new['seconds'] * 1000 / new['calls']
        require(all(checks.values()), f'Matched conditions differ for {task}: {checks}')
        result.append(dict(task=task, control_ms=old, candidate_ms=new, speedup=old/new,
                           control_steps_per_second=1000/old, candidate_steps_per_second=1000/new,
                           control_status=a['status'], candidate_status=b['status'], checks=checks,
                           control_attempt=a['attempt'], candidate_attempt=b['attempt']))
    return result


def publish(root, artifact_root, *, partial=False):
    destination = root / 'reports/forge/bispf-throughput'
    rows, states = [], []
    for name in ('queue', 'performance-queue'):
        collected, state = collect(root, artifact_root/name, destination, artifact_root/'archive', partial=partial)
        rows += collected
        states.append(state)
    ordinary = [r for r in rows if r['view'] == 'discriminator_stability' and r['candidate'] == CANDIDATE]
    archived = read_json(root / 'reports/forge/bispf-optimized/quality.json')
    old_passes = {r['task'] for r in archived['rows'] if r['status'] == 'PASS'}
    current = {r['task']: r for r in ordinary}
    # Jobs shared with the explicit small diagnostic may have its original view.
    for row in rows:
        if row['candidate'] == CANDIDATE and row['view'] in ('bispf_throughput_small', 'bispf_throughput_endurance'):
            current.setdefault(row['task'], row)
    require(partial or set(current) == {r['task'] for r in archived['rows']}, 'Ordinary suite incomplete')
    regressions = sorted(t for t in old_passes if t in current and current[t]['status'] != 'PASS')
    ordinary_submission = next(v for v in states[0]['submissions'].values()
                               if v['request']['view']['id'] == 'discriminator_stability')
    # Use the repository's existing verified-parent bridge for word holds,
    # matching the current inventory publisher rather than its older generic
    # uninterrupted-only continuation aggregation.
    from reports.forge.word_checkpoint_publication import install
    install()
    from experiments.forge.views import qualify
    frozen = ordinary_submission['request']
    ordinary_raw = [{**row, '_attempt_id': item['attempt']} for item in ordinary
                    for row in read_json(Path(item['local_artifact_root']) / 'result.json')['task_results']]
    qualified = qualify(frozen['view'], frozen['tasks'], ordinary_raw, candidate=frozen['candidate'])
    atomic_json(artifact_root / 'current-qualification.json', qualified)
    archived_rows = {r['task']: r for r in archived['rows']}
    for task, row in current.items():
        row['archived_status'] = archived_rows[task]['status']
        # Hash the full actual final metrics, including arrays, before compact projection.
        raw = read_json(Path(row['local_artifact_root']) / 'result.json')
        actual = next(r for r in raw['task_results'] if r['task_id'] == task)
        row['archived_final_metrics_equal'] = stable_hash(actual['metrics']) == archived_rows[task]['final_metrics_sha256']
    performance = matched(rows, {r['task'] for r in rows if 'performance' in r}, performance=True)
    image_rows = [r for r in rows if r['view'] in ('bispf_fastpath_quality', 'bispf_throughput_small')]
    images = matched(image_rows, {r['task'] for r in image_rows})
    value = dict(schema='bispf_throughput_readout_v1', candidate=CANDIDATE, control=CONTROL, seed=0,
                 partial=partial, numerical_evidence_scope='source-bound Forge measured gates; no natural-image quality or public-default promotion',
                 quality=dict(counts=dict(Counter(r['status'] for r in current.values())),
                              scope='All 30 incumbent numerical task contracts; current ordinary run plus explicitly nonqualifying endurance preservation diagnostic. This combined count is not current-policy qualification.',
                              ordinary_policy=dict(view_revision=ordinary_submission['request']['view']['revision'],
                                                   submission_status=ordinary_submission['status'],
                                                   reason=ordinary_submission.get('reason'),
                                                   qualification={k: qualified[k] for k in
                                                       ('status', 'eligible', 'qualified_tier', 'required_passed', 'required_total', 'held_passed', 'held_total', 'policy_fingerprint')},
                                                   checkpoint_publication_adapter='reports/forge/word_checkpoint_publication.py',
                                                   measured_counts=dict(Counter(r['status'] for r in ordinary)),
                                                   measured_tasks=len(ordinary)),
                              tasks=len(current), archived_passes=len(old_passes), preserved_passes=sum(t in current and current[t]['status']=='PASS' for t in old_passes),
                              regressions=regressions, rows=sorted(current.values(), key=lambda r:r['task'])),
                 performance=performance, images=images,
                 primary_throughput_gate=next(r['speedup'] for r in performance if r['task']=='dcgan_performance_r256') >= 2,
                 campaigns={k: v for state in states for k,v in state['campaigns'].items()},
                 attempts=rows, raw_artifact_root=str(artifact_root), publisher_sha256=file_hash(Path(__file__)))
    atomic_json(destination / 'results.json', value)
    print(json.dumps(dict(quality=value['quality']['counts'], preserved=value['quality']['preserved_passes'],
                          regressions=regressions, performance=[(r['task'],r['speedup']) for r in performance],
                          images=[(r['task'],r['speedup']) for r in images]), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    publish(args.root.resolve(),args.artifacts.resolve(),partial=args.partial)
