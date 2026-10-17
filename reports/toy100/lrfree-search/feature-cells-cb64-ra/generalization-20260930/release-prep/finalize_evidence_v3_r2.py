"""Join actual original gates and explicit source bridges without numerical work.

The parent invokes this sealed helper in a fresh release-prep output directory.
Missing latest replay/suite/gates remain pending. Execution labels stay original.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
from pathlib import Path
import types

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'
V2_SHA = '9b3e52d3caf82795d6c296c41d98b6ce381972b952a0cb98e130be437dc4d684'
V2_PATH = HERE / 'finalize_evidence_v2.py'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


if digest(V2_PATH) != V2_SHA:
    raise RuntimeError('Sealed v2 helper changed')
v2 = types.ModuleType('sealed_evidence_finalizer_v2')
v2.__file__ = str(V2_PATH)
exec(compile(V2_PATH.read_bytes(), str(V2_PATH), 'exec'), v2.__dict__)
v1 = v2.v1
sha = v2.sha


class Evidence(v1.Evidence):
    def raw(self, path, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                self.pending.append('Missing ' + str(path))
            return None
        raw = path.read_bytes()
        key, value = str(path), sha(raw)
        self.check(key not in self.inputs or self.inputs[key] == value,
                   'Input changed during collection: ' + key)
        self.inputs[key] = value
        return raw

    def bind(self, path, expected):
        path = Path(path)
        if not path.is_file():
            self.pending.append('Missing ' + str(path))
            return
        value, key = digest(path), str(path)
        self.check(key not in self.inputs or self.inputs[key] == value,
                   'Input changed during collection: ' + key)
        self.inputs[key] = value
        self.check(value == expected, 'Pinned input changed: ' + key)

    def integrity(self, record, where):
        super().integrity(record, where)
        if isinstance(record, dict) and getattr(self, 'expected_file_count', None) is not None:
            self.check(record.get('files') == self.expected_file_count,
                       where + ': frozen file count mismatch')


@contextmanager
def route(e, prepared, lane):
    lane = Path(lane)
    selected = prepared['lanes'][str(lane)]
    previous = (v1.LANE, v1.RUN_SHA, v1.MANIFEST_SHA, v1.SOURCE_FREEZE_SHA,
                getattr(e, 'expected_file_count', None))
    v1.LANE = lane
    v1.RUN_SHA = selected['package_sha256']
    v1.MANIFEST_SHA = selected['package_manifest_sha256']
    v1.SOURCE_FREEZE_SHA = selected['source_freeze_sha256']
    e.expected_file_count = selected['frozen_file_count']
    try:
        yield
    finally:
        (v1.LANE, v1.RUN_SHA, v1.MANIFEST_SHA, v1.SOURCE_FREEZE_SHA,
         e.expected_file_count) = previous


def frozen(e, prepared, lane):
    selected = prepared['lanes'][str(lane)]
    record = e.read(lane / 'SOURCE-FREEZE.json')
    if record is None:
        return None
    e.check(e.inputs[str(lane / 'SOURCE-FREEZE.json')] == selected['source_freeze_sha256'],
            lane.name + ': source freeze changed')
    e.check(record.get('package_sha256') == selected['package_sha256'] and
            record.get('config_sha256') == v1.CONFIG_SHA, lane.name + ': package/config mismatch')
    e.check(len(record.get('hashes', {})) == selected['frozen_file_count'],
            lane.name + ': declared frozen count mismatch')
    for path, value in record.get('hashes', {}).items():
        e.bind(path, value)
    return record


def original_queue(e, lane, name, group, tasks, interrupted=False):
    # Reuse sealed authority and original aggregate requirements unchanged.
    return v2.queue_rows(e, lane, name, group, tasks, interrupted)


def fresh_queue(e, prepared, lane):
    board = e.read(lane / 'scoreboard-moving.json')
    if board is None:
        return [], None
    wanted = {('moving', task) for task in v1.NATIVE}
    listed = board.get('results', [])
    identities = [(item.get('kind'), item.get('task')) for item in listed]
    e.check(board.get('group') == 'moving', 'Fresh moving queue group changed')
    e.check(len(set(identities)) == len(identities) and set(identities) <= wanted,
            'Fresh moving queue duplicate/nonoriginal tasks')
    closed = board.get('status') in ('PASS', 'FAIL') and 'completed' in board and set(identities) == wanted
    if not closed:
        e.pending.append('Fresh original three-moving queue not complete')
    rows = []
    with route(e, prepared, lane):
        e.integrity(board.get('integrity'), 'Fresh moving queue initial')
        if closed:
            e.integrity(board.get('integrity_after'), 'Fresh moving queue final')
            e.check(board['status'] == ('PASS' if all(item.get('status') == 'PASS' for item in listed) else 'FAIL'),
                    'Fresh moving aggregate verdict mismatch')
        for task in v1.NATIVE:
            if ('moving', task) not in identities:
                e.pending.append('Fresh moving/' + task + ' lacks a closed result')
                continue
            row = v1.moving(e, task)
            if row is not None:
                item = listed[identities.index(('moving', task))]
                e.check(row['quality_status'] == item.get('status'), task + ': fresh scoreboard mismatch')
                row.update(source_lane=str(lane), source_freeze_sha256=v1.SOURCE_FREEZE_SHA,
                           execution_candidate='RA15-partial-recovery', original_fresh_execution=True,
                           fresh_latest_candidate_execution=False,
                           latest_source_equivalence='Explicit RA15→latest defaultCPU/valid-checkpoint bridge')
                rows.append(row)
    return rows, dict(receipt=str(lane / 'scoreboard-moving.json'), status=board.get('status'),
                      closed_for_scope=closed, required_count=3, recorded_count=len(listed))


def bridges(e, prepared):
    records = {}
    for name in ('no_fire', 'portability'):
        specification = prepared['bridges'][name]
        record = e.read(specification['path'])
        if record is None:
            continue
        e.check(e.inputs[specification['path']] == specification['sha256'], name + ': bridge changed')
        for key, value in specification['required_fields'].items():
            e.check(record.get(key) == value, name + ': bridge field mismatch: ' + key)
        for path, value in record.get('read_only_file_sha256', {}).items():
            e.bind(path, value)
        records[name] = record
    return records


def latest_replay(e, prepared):
    lane = Path(prepared['latest_replay']['lane'])
    for name, value in prepared['latest_replay']['prepared_file_sha256'].items():
        e.bind(lane / name, value)
    closed = e.read(lane / 'CLOSED.json')
    if closed is None:
        return None
    e.check(closed.get('status') == prepared['latest_replay']['required_closed_status'],
            'Latest original CUDA replay closure status mismatch')
    e.check(closed.get('package_sha256') == prepared['latest_package']['package_sha256'],
            'Latest replay package identity mismatch')
    e.check(closed.get('fresh_training_updates') == 0 and closed.get('fresh_replay_updates_total') == 40,
            'Latest replay scope changed')
    e.check(closed.get('replay_status') == {'toy': 'PASS', 'mnist': 'PASS'},
            'Latest strict CUDA replay not qualified')
    for field in ('native_and_CPU_map_loss_state_sample_bits_identical',
                  'native_matches_pinned_original_control', 'checkpoint_alias_protocol_preserved'):
        e.check(closed.get(field) is True, 'Latest replay missing strict proof: ' + field)
    for name, value in closed.get('file_sha256', {}).items():
        e.bind(lane / name, value)
    completion = e.read(lane / 'COMPLETION-replay.json')
    aggregate = e.read(lane / ('replay-' + prepared['latest_candidate'] + '.json'))
    if completion is None or aggregate is None:
        return None
    e.check(completion.get('status') == 'COMPLETE' and completion.get('returncode') == 0,
            'Latest replay process did not complete')
    e.check(completion.get('fresh_replay_updates_total') == 40 and completion.get('fresh_training_updates') == 0,
            'Latest replay process budget mismatch')
    e.check(set(aggregate) == {'toy', 'mnist'}, 'Latest replay fixture set changed')
    for task, result in aggregate.items():
        e.check(result.get('status') == 'PASS' and result.get('start_step') == 1000 and result.get('steps_replayed') == 10,
                task + ': actual latest original10 replay scope mismatch')
        e.check(len(result.get('branches', [])) == 2 and
                [row.get('step') for row in result.get('per_update_comparison', [])] == list(range(1001, 1011)),
                task + ': actual two10 branches incomplete')
        e.check(result.get('native_continuation_control', {}).get('status') == 'PASS',
                task + ': latest pinned native control mismatch')
    return dict(status='PASS', candidate=prepared['latest_candidate'], fresh_training_updates=0,
                fresh_replay_updates_total=40, fixtures={key: value['status'] for key, value in aggregate.items()},
                receipt=str(lane / 'CLOSED.json'), closed_sha256=e.inputs[str(lane / 'CLOSED.json')],
                original_fresh_learned_training_source='RA13-settled', original_updates_per_fixture=2000)


def portability_GPU(e, prepared):
    specification = prepared['portable_GPU_regression']
    lane = Path(specification['lane'])
    for name, value in specification['prepared_file_sha256'].items():
        e.bind(lane / name, value)
    completion = e.read(lane / 'attempt-1/COMPLETION.json')
    result = e.read(lane / 'attempt-1/TEST-RESULT.json')
    if completion is None or result is None:
        return None
    e.check(completion.get('status') == result.get('status') == 'PASS' and
            completion.get('returncode') == result.get('returncode') == 0,
            'Actual CUDA-default portability regression did not pass')
    e.check(result.get('package_sha256') == prepared['latest_package']['package_sha256'],
            'CUDA-default regression latest package mismatch')
    e.check(len(result.get('actual_tests', [])) == 1 and result['actual_tests'][0]['outcome'] == 'passed',
            'CUDA-default regression selected test did not execute exactly once')
    e.check(result.get('portable_diagnostic_actual_step_calls') == 40 and result.get('original_seed') == 1234,
            'CUDA-default regression original recipe changed')
    e.check(result.get('deterministic_algorithms') is True and
            result.get('TF32_matmul') is False and result.get('TF32_cudnn') is False,
            'CUDA-default regression runtime policy changed')
    for path, value in [(lane / 'attempt-1/run.log', completion.get('log_sha256')),
                        (lane / 'attempt-1/TEST-RESULT.json', completion.get('result_sha256'))]:
        e.bind(path, value)
    return dict(status=result['status'], candidate=prepared['latest_candidate'],
                receipt=str(lane / 'attempt-1/COMPLETION.json'), completed=completion.get('completed'),
                actual_tests=result['actual_tests'], actual_step_calls=40, quality_scorer_calls=0)


def append_inventory(e, inventory, complete):
    original = {item['archive_relative_path']: item for item in (inventory or {}).get('artifacts', [])}
    paths = set()
    folders = ['validation-ra14', 'validation-ra14-r2', 'validation-ra14-moving-r2', 'validation-ra15',
               'validation-ra16', 'diagnostics/moving-rotated-controller', 'diagnostics/moving-rotated-recovery',
               'diagnostics/moving-rotated-window-ra15', 'diagnostics/moving-rotated-window-ra15-r2',
               'portability/partial-recovery', 'portability/ra16-portability', 'mnist/ra15-replay',
               'mnist/ra16-replay', 'diagnostics/ra16-portable-gpu', 'integration-prep',
               'pkg-RA15-partial-recovery', 'pkg-RA16-portability']
    for folder in folders:
        if (ROOT / folder).exists():
            paths.update(path for path in (ROOT / folder).rglob('*') if path.is_file() and
                         path.suffix in ('.json', '.jsonl', '.log', '.py', '.md', '.patch'))
    paths.update(path for path in HERE.rglob('*') if path.is_file() and
                 path.suffix in ('.json', '.jsonl', '.log', '.py', '.md', '.patch'))
    paths.update(ROOT / 'configs' / name for name in ('RA15-partial-recovery.json', 'RA16-portability.json'))
    rows = []
    for path in sorted(paths):
        relative = str(path.relative_to(ROOT))
        if relative in original:
            e.bind(path, original[relative]['sha256'])
            continue
        raw = e.raw(path)
        rows.append(dict(local_path=str(path), archive_relative_path=relative, bytes=len(raw), sha256=sha(raw),
                         category='SOURCE_OR_PROTOCOL' if path.suffix in ('.py', '.md', '.patch') else 'SMALL_EVIDENCE',
                         scope='FINALIZED_SMALL_ARTIFACT' if complete else 'SNAPSHOT_REBIND_AFTER_COMPLETE'))
    return dict(status='READY_TO_APPEND' if complete else 'PENDING_FINAL_REBIND',
                original_inventory_sha256=v1.INVENTORY_SHA, files=rows, count=len(rows),
                bytes=sum(row['bytes'] for row in rows),
                excluded='PT/PTH/checkpoints, datasets, NPZ/NPY clouds, images and GIFs stay local.',
                rule='Append final small source/JSON/JSONL/log evidence absent from the original510-file inventory; preserve every attempt.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if HERE.resolve() not in output.parents or output.exists():
        parser.error('--output must be a new child directory under release-prep')
    e = Evidence()
    prepared = e.read(HERE / 'FINALIZER-V3-R2-PREPARATION.json')
    if prepared is None:
        parser.error('Latest-source finalizer preparation must be sealed before invocation')
    for name, value in prepared['helper_file_sha256'].items():
        e.bind(HERE / name, value)
    e.bind(V2_PATH, V2_SHA)
    with route(e, prepared, v2.ORIGINAL):
        inventory, learned = v1.source_and_bridge(e)
    old_adapter = v2.source_bridges(e)
    retained = v2.retained_attempts(e)
    latest_bridges = bridges(e, prepared)
    quality_lane = Path(prepared['affected_quality_lane'])
    suite_lane = Path(prepared['latest_suite_lane'])
    frozen(e, prepared, quality_lane)
    frozen(e, prepared, suite_lane)
    rows, boards = [], []
    old_ports, board = original_queue(e, v2.ORIGINAL, 'scoreboard-all.json', 'all',
                                    [('screen', task) for task in v1.PORTS], True)
    boards.append(board)
    for row in old_ports:
        if row['task'] == 'ring_shift':
            continue
        row.update(execution_candidate='RA14-replay', original_fresh_execution=True,
                   fresh_latest_candidate_execution=False, latest_source_equivalence='Explicit zero-fire/KNN then defaultCPU/valid-checkpoint bridges')
        rows.append(row)
    old_native, board = original_queue(e, v2.CORRECTED, 'scoreboard-native.json', 'native',
                                     [('screen', task) for task in v1.NATIVE])
    boards.append(board)
    for row in old_native:
        row.update(execution_candidate='RA14-replay', original_fresh_execution=True,
                   fresh_latest_candidate_execution=False, latest_source_equivalence='Explicit zero-fire then defaultCPU/valid-checkpoint bridges')
        rows.append(row)
    fresh_moving, board = fresh_queue(e, prepared, quality_lane)
    rows.extend(fresh_moving)
    boards.append(board)
    with route(e, prepared, quality_lane):
        ring = v1.screen(e, 'ring_shift')
        if ring is not None:
            ring.update(source_lane=str(quality_lane), source_freeze_sha256=v1.SOURCE_FREEZE_SHA,
                        execution_candidate='RA15-partial-recovery', original_fresh_execution=True,
                        fresh_latest_candidate_execution=False,
                        latest_source_equivalence='Explicit RA15→latest defaultCPU/valid-checkpoint bridge')
            rows.append(ring)
    original_moving, old_moving_board = original_queue(e, v2.CORRECTED, 'scoreboard-moving.json', 'moving',
                                                     [('moving', task) for task in v1.NATIVE])
    package = prepared['latest_package']
    actual_mapping = {}
    h = hashlib.sha256()
    for path in sorted((Path(package['path']) / 'particlegan').rglob('*.py')):
        raw = e.raw(path)
        relative = str(path.relative_to(Path(package['path']) / 'particlegan'))
        actual_mapping[relative] = sha(raw)
        h.update(relative.encode() + b'\0' + raw + b'\0')
    e.check(actual_mapping == package['source_sha256'] and h.hexdigest() == package['package_sha256'],
            'Actual latest package source identity changed')
    e.package_map = actual_mapping
    with route(e, prepared, suite_lane):
        suite = v1.full_suite(e)
    if suite is not None:
        receipt = e.read(suite_lane / 'FULL-TESTS.json')
        for path, value in {**receipt.get('sources', {}), **receipt.get('tests', {})}.items():
            e.bind(path, value)
    portable_GPU = portability_GPU(e, prepared)
    if suite is not None and portable_GPU is not None:
        receipt = e.read(suite_lane / 'FULL-TESTS.json')
        e.check(receipt.get('numerical_started', 0) >= (portable_GPU['completed'] or float('inf')),
                'Latest full suite ran before the actual CUDA-default regression closed')
    replay = latest_replay(e, prepared)
    historical = v2.historical_suite(e, inventory)
    cancelled = e.read(quality_lane / 'FULL-SUITE-CANCELLED.json')
    cancelled_receipt = e.read(quality_lane / 'FULL-TESTS.json')
    e.check(cancelled_receipt is not None and cancelled_receipt.get('status') == 'CANCELLED_BEFORE_TEST_EXECUTION' and
            not (quality_lane / 'full-pytest.log').exists(), 'RA15 cancelled suite history changed')
    e.check(cancelled is not None and cancelled.get('status') == 'CANCELLED_BEFORE_TEST_EXECUTION' and
            cancelled.get('test_updates') == 0, 'RA15 cancelled suite recorded numerical execution')
    for name, value in prepared['retained_RA15_unlaunched_replay_file_sha256'].items():
        e.bind(ROOT / 'mnist/ra15-replay' / name, value)
    e.check(not (ROOT / 'mnist/ra15-replay/LAUNCH-replay.json').exists(), 'Retained RA15 prepared replay was launched')
    if len(rows) != 19:
        e.pending.append(f'Original finalized gate receipts incomplete: {len(rows)}/19')
    else:
        e.check(Counter(row['kind'] for row in rows) == {'portability': 13, 'moving': 3, 'native': 3},
                'Original13+3+3 balance mismatch')
        e.check({(row['queue_kind'], row['task']) for row in rows} == set(v1.PLAN), 'Original19 task set changed')
    complete = not e.pending and len(rows) == 19 and suite is not None and replay is not None and portable_GPU is not None
    quality = 'PENDING' if not complete else ('PASS' if all(row['quality_status'] == 'PASS' for row in rows)
              and suite['status'] == replay['status'] == 'PASS' and learned['toy_original_gate'] == 'PASS' else 'FAIL')
    append = append_inventory(e, inventory, complete)
    if complete and e.pending:
        complete, quality = False, 'PENDING'
        append['status'] = 'PENDING_FINAL_REBIND'
        for row in append['files']:
            row['scope'] = 'SNAPSHOT_REBIND_AFTER_COMPLETE'
    validity = 'INVALID' if e.defects else ('VALID' if complete else 'PENDING')
    learned.update(latest_candidate=prepared['latest_candidate'], latest_replay=replay,
                   original_fresh_training_source='RA13-settled', fresh_latest_training_updates=0,
                   latest_scope='Explicit source equivalence plus actual latest native/CPU-map CUDA40 replay')
    report = dict(status='COMPLETE' if complete else 'PENDING', finalizer_version=3, finalizer_revision=2,
                  candidate=prepared['latest_candidate'], evidence_validity=validity,
                  qualification_base=prepared['qualification_base'],
                  quality_qualification=quality, overall_qualification='INVALID' if e.defects else quality,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), required_gate_count=19,
                  finalized_receipt_count=len(rows), gate_counts={kind: dict(
                      total=sum(row['kind'] == kind for row in rows),
                      passed=sum(row['kind'] == kind and row['quality_status'] == 'PASS' for row in rows),
                      failed=sum(row['kind'] == kind and row['quality_status'] == 'FAIL' for row in rows))
                      for kind in ('portability', 'moving', 'native')},
                  gates=rows, scoreboard_routes=boards, full_suite=suite, learned=learned,
                  actual_CUDA_default_portability_regression=portable_GPU,
                  historical_RA14_moving_gates=original_moving, historical_RA14_moving_queue=old_moving_board,
                  historical_RA13_full_suite=historical, original_adapter_bridge=old_adapter,
                  latest_source_bridges={key: dict(path=prepared['bridges'][key]['path'],
                                                 sha256=prepared['bridges'][key]['sha256'], status=record.get('status'))
                                         for key, record in latest_bridges.items()},
                  retained_attempts=retained, retained_RA15_cancelled_suite=cancelled,
                  retained_RA15_replay='FROZEN_PREPARED_UNLAUNCHED', source_identities=prepared['lanes'],
                  retained_failures=['RA12 static false fires and learned regressions',
                      'RA13 strict learned native/CPU-map replay FAIL',
                      'RA14 first zero-update bridge normalization failure',
                      'RA14 original ALL moving diagnostic ERROR after500 and zero-update native import ERROR',
                      'RA14 original moving rotated100 qualityFAIL at1500',
                      'First RA15 paired window NaN comparison ERROR before updates',
                      'Separate moving-r2 prepared lane and RA15 learned replay prepared but never launched',
                      'RA15 full suite cancelled before execution',
                      'RA15 CPU factory meta mismatch and malformed shape restored before reaction error',
                      'All source preparation failures retained without quality inference'],
                  pending=sorted(set(e.pending)), defects=e.defects,
                  numerical_labels='15 original RA14 quality executions,4 fresh RA15 affected gates, RA13 fresh learned training; latest source qualifies through explicit bridges and actual latest CUDA replay/full suite.',
                  tensor_loads=0, model_calls=0, scorer_calls=0, GPU_operations=0, repository_mutations=0)
    output.mkdir()
    def write(name, value):
        with (output / name).open('x') as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')
    write('QUALIFICATION.json' if complete else 'PENDING.json', report)
    write('ARCHIVE-APPEND.json', append)
    write('INPUTS.json', dict(helper_sha256=digest(Path(__file__)), file_sha256=e.inputs))
    if complete:
        lines = ['# Original qualification for ' + prepared['latest_candidate'], '',
                 f'Evidence: **{validity}**. Original quality qualification: **{quality}**.', '',
                 '| Scope | Task | Quality | Executed source |', '|---|---|---|---|']
        lines.extend(f'| {row["kind"]} | {row["task"]} | {row["quality_status"]} | {row["execution_candidate"]} |' for row in rows)
        lines += ['', report['numerical_labels'], '',
                  'Qualification base: PR155 ' + prepared['qualification_base']['original_reference_commit'] + '.',
                  'The observed PR155 upstream head ' + prepared['qualification_base']['observed_upstream_commit'] + ' requires its separate source/execution bridge and suite; this closure does not qualify that source.',
                  'Moving quality uses COMPLETION.verdict.periods and the original scorers. Native gates retain34 observations, five20k terminal draws and the100k holdout.',
                  'RA14 moving rotated100 qualityFAIL remains in history. Earlier adapter errors, NaN comparison failure, cancelled RA15 suite and unlaunched RA15 replay remain retained.',
                  'Toy original gate: ' + str(learned['toy_original_gate']) + '; all9 postupdate metric/LR records match RA11. MNIST all10 metric/LR records match corrected E22; no numerical MNIST gate was added.',
                  'Inherited final learned metrics: ' + json.dumps(learned['inherited_final_metrics'], sort_keys=True) + '.',
                  'Latest original learned CUDA replay:40 actual updates, two10 branches for each fixture; fresh learned training remains2000 RA13 updates per fixture.',
                  'Actual latest full-suite counts: ' + json.dumps(suite['counts'], sort_keys=True) + '.',
                  'Actual pytest summary: ' + str(suite['summary_line']) + '.', '', 'Actual skip reasons:', '']
        lines.extend('- ' + reason for reason in suite['skip_reasons'])
        lines += ['', 'Raw checkpoints, datasets, clouds and images stay local. This helper performed no numerical work.', '']
        write('QUALIFICATION.md', '\n'.join(lines))
    files = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    write('FROZEN.json', dict(status='CLOSED_FINAL_QUALIFICATION' if complete else 'CLOSED_PENDING_SNAPSHOT',
                              file_sha256=files, evidence_validity=validity, quality_qualification=quality))
    print(json.dumps(dict(status=report['status'], evidence_validity=validity, quality=quality,
                          receipts=len(rows), pending=len(set(e.pending)), defects=len(e.defects), output=str(output)), sort_keys=True))
    return 2 if e.defects else 0


if __name__ == '__main__':
    raise SystemExit(main())
