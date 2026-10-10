"""Read certified Phase3 results/checkpoints and export compact scalar counters.

No trainer, model constructor, sampler or evaluator is called. This companion
never regrades timeouts/interrupted jobs and never launches a scientific retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch


FROZEN_COMMIT = 'ad29d3b475ec724a7f4b3c668ad54b814a00736f'
SOURCE_DIGEST = '883c2cde79f5565b8449cc3571004e50f73bfc565f2bb86d6c032c234f7917a1'
PUBLICATION_ADAPTER = dict(commit='f90591b85383052e803903088b775f4e9e95d8e1',
    sha256='70302ffba5fbaf673375d3e4877363edf8ac80026b67cd9f872beb5ebe9c88e5',
    scope='Read-only annotation of unavailable own-checkpoint dependencies; no training or regrading.')
TIERS = {'gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
         'ring16_acquisition', 'five_word_joint_smoke'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def geometry_records(saved):
    found = {}
    def visit(value, path='state'):
        if isinstance(value, dict):
            geometry = value.get('anisotropic_geometry')
            if isinstance(geometry, dict):
                for key, item in geometry.items():
                    assert type(item) in (int, float) and math.isfinite(item) and item >= 0, (path, key)
                assert geometry['compressed_calls'] <= geometry['calls']
                assert geometry['zero_covariance_anchors'] <= geometry['anchors']
                assert geometry['minimum_ridge'] <= geometry['maximum_ridge']
                if 'active_calls' in value:
                    assert geometry['calls'] == value['active_calls']
                if 'completed_steps' in value:
                    assert geometry['calls'] == value['completed_steps']
                found[path + '.anisotropic_geometry'] = geometry
            for name, child in value.items():
                if name not in ('models', 'role_parameters', 'streams', 'initialization', 'anisotropic_geometry'):
                    visit(child, f'{path}.{name}')
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f'{path}[{index}]')
    visit(saved)
    return found


def compact_metrics(row):
    return {key: value for key, value in row.get('metrics', {}).items()
            if type(value) in (int, float, bool) or value is None}


def compact_history(histories):
    """Keep identity/cost facts here; full certified history lives in results."""
    return [dict(attempt_id=item['attempt_id'], candidate_id=item['candidate_id'],
        final_selected=item['final_selected'], attempt_status=item['attempt_status'],
        cost_owner=item['cost_owner'], retry_of=item.get('retry_of'),
        canonical_result_hash=item['provenance']['canonical_result_hash'],
        source_digest=item['provenance']['source_digest'],
        source_commit=item['provenance']['source_origin_commit'],
        task_results=[{key: row[key] for key in
            ('task_id', 'gate_status', 'raw_status', 'reason', 'cost') if key in row}
            for row in item['task_results']]) for item in histories]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result_path = options.publication / 'phase3-results.json'
    result = read(result_path)
    assert result['source_commit'] == FROZEN_COMMIT and result['source_digest'] == SOURCE_DIGEST
    assert result['qualification_input'] is False and result['protocol_seed'] == 0
    assert result['optimizer_updates_added'] == result['sampling_draws_added'] == 0
    cells = {(c['role'], c['task_id']): c for c in result['task_cells']}
    assert len(cells) == 32 and {c[1] for c in cells} >= TIERS
    assert not any(c['gate_status'] in ('PENDING', 'RUNNING', 'NOT_RUN') for c in cells.values())
    media_path = options.publication / 'media/index.json'
    media = read(media_path)['media']
    assert len(media) == result['actual_training_gifs']
    assert {(m['role'], m['task_id']) for m in media} == {
        key for key, cell in cells.items() if cell['gate_status'] in ('PASS', 'FAIL')}
    for item in media:
        assert digest(options.publication / item['gif']) == item['gif_sha256']
        assert item['optimizer_updates_added'] == item['sampling_draws_added'] == 0
        assert item['frames'] >= 2
    rows = {(r['role'], r['task_id']): r for r in result['task_results']}
    progress = read(options.archive / 'phase3-progress.json')
    queue = read(options.archive / 'phase3-queue/queue/state.json')
    contracts = {}
    for role, request_id in progress['requests'].items():
        request = queue['submissions'][request_id]['request']
        contracts[role] = {task_id: dict(adapter=task['adapter'],
            execution_steps=task['execution'].get('steps'),
            initializer=task['execution'].get('initializer'),
            prior=task['execution'].get('prior'),
            timeout_seconds=task['resources'].get('timeout_seconds'),
            sampling_law=task['evaluation'].get('sampling_law'),
            evaluation_kind=task['evaluation'].get('kind'),
            observations=task['evaluation'].get('observations'),
            minimum_stable_checks=task['evaluation'].get('minimum_stable_checks'),
            thresholds=task['evaluation'].get('thresholds'),
            dependencies=task.get('dependencies', []))
            for task_id, task in request['tasks'].items()}
    assert contracts['baseline'] == contracts['candidate']
    paid_by_role = {}
    for role, request_id in progress['requests'].items():
        attempts = [item for item in result['paid_attempt_history']
                    if item['cost_owner']['request'] == request_id]
        paid_by_role[role] = dict(attempts=len(attempts),
            wall_seconds=sum(row.get('cost', {}).get('wall_seconds', 0)
                             for item in attempts for row in item['task_results']),
            linked_execution_repairs=sum(bool(item.get('retry_of')) for item in attempts))
    tasks = []
    counters = []
    for (role, task), cell in sorted(cells.items()):
        row = rows.get((role, task), {})
        descriptor = row.get('provenance_checkpoint')
        geometry = {}
        if descriptor:
            checkpoint = Path(descriptor['artifact_root']) / descriptor['path']
            assert checkpoint.is_file() and digest(checkpoint) == descriptor['sha256']
            assert checkpoint.stat().st_size == descriptor['bytes']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            from experiments.forge.state import state_digest
            assert state_digest(saved) == descriptor['state_sha256']
            geometry = geometry_records(saved)
        # Inactive controls must not silently claim an active geometry mechanism.
        if role == 'baseline':
            assert not geometry
        scalar = dict(role=role, task=task, original_tier=1 if task in TIERS else 2,
                      status=cell['gate_status'], raw_status=row.get('raw_status'),
                      reason=cell.get('reason', row.get('reason')),
                      attempt_id=cell.get('attempt_id'), metrics=compact_metrics(row),
                      cost=row.get('cost'), evaluator_summary=row.get('evaluator_summary', {}))
        tasks.append(scalar)
        if descriptor:
            counters.append(dict(role=role, task=task, attempt_id=cell.get('attempt_id'),
                                 checkpoint=descriptor, geometry=geometry,
                                 transport_diagnostics=row.get('mechanism_stats', {})))
    repaired, regressed, retained, unresolved = [], [], [], []
    for task in sorted({task for _, task in cells}):
        pair = [cells[(role, task)]['gate_status'] for role in ('baseline', 'candidate')]
        if pair == ['FAIL', 'PASS']:
            repaired.append(task)
        elif pair == ['PASS', 'FAIL']:
            regressed.append(task)
        elif pair == ['PASS', 'PASS']:
            retained.append(task)
        elif any(status not in ('PASS', 'FAIL') for status in pair):
            unresolved.append(dict(task=task, baseline=pair[0], candidate=pair[1]))
    summary = dict(schema_version=1, scope='phase3_paired_research_diagnostic', qualification_input=False,
                   source_commit=FROZEN_COMMIT, source_digest=SOURCE_DIGEST, protocol_seed=0,
                   source_results_sha256=digest(result_path), task_matrix=tasks,
                   publication_adapter=PUBLICATION_ADAPTER,
                   media_index_sha256=digest(media_path),
                   complete_measured_repairs=repaired, complete_measured_regressions=regressed,
                   retained_passes=retained, unresolved_pairs=unresolved,
                   original_tier1={role:dict(Counter(c['gate_status'] for (r,t),c in cells.items()
                                                  if r==role and t in TIERS)) for role in ('baseline','candidate')},
                   outcomes=result['outcomes'], accounting=result['accounting'],
                   paid_attempt_history=compact_history(result['paid_attempt_history']),
                   paired_task_contracts=contracts['baseline'],
                   paid_totals_by_role=paid_by_role,
                   actual_training_gifs=result['actual_training_gifs'],
                   optimizer_updates_added=0, sampling_draws_added=0,
                   interpretation='Only complete paired PASS/FAIL results establish repairs/regressions. Execution gaps remain unresolved; no ordinary qualification or default promotion.')
    counter_receipt = dict(schema_version=1, qualification_input=False,
                           source_commit=FROZEN_COMMIT, source_digest=SOURCE_DIGEST,
                           source_results_sha256=digest(result_path),
                           checkpoint_counters=counters, optimizer_updates_added=0, sampling_draws_added=0,
                           interpretation='Analytic condition bounds and cumulative ridge/anchor counters are read from exact certified checkpoint bytes, never used to drive training. Mirrored applied/consumer paths represent the same cumulative state and must not be summed as independent calls.')
    recovery_root = options.archive.parents[1] / 'restart-20261010'
    recovery = recovery_root / 'after.json'
    if recovery.exists():
        recovered = read(recovery)
        summary['reboot_recovery_receipt'] = dict(path=str(recovery), sha256=digest(recovery),
            charges_include_downtime=recovered['charges_include_downtime'],
            track=recovered['tracks']['anisotropic'])
    authorization = recovery_root / 'authorized-retries.json'
    if authorization.exists():
        authorized = read(authorization)
        repairs = [entry for entry in authorized['retries'] if entry['track'] == 'anisotropic']
        assert len(repairs) == 6 and all(entry['source_commit'] == FROZEN_COMMIT for entry in repairs)
        summary['execution_repair_authorization'] = dict(path=str(authorization),
            sha256=digest(authorization), scope=authorized['scope'],
            user_instruction=authorized['user_instruction'], authorized_at=authorized['authorized_at'],
            original_paid_ceilings_unchanged=authorized['original_paid_ceilings_unchanged'],
            max_additional_attempts_per_job=authorized['max_additional_attempts_per_job'],
            predecessor_bindings=repairs)
    write(options.output / 'analysis.json', summary)
    write(options.output / 'geometry-counters.json', counter_receipt)
    print(json.dumps(dict(outcomes=result['outcomes'], repairs=repaired, regressions=regressed,
                          unresolved=len(unresolved), counters=len(counters))))


if __name__ == '__main__':
    main()
