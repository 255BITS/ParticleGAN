"""Self-contained publication controls; no archive, GPU, model or queue required."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'reports/forge/continuous-baseline-20261003/publish_results.py'
spec = importlib.util.spec_from_file_location('continuous_publication_controls', PATH)
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def identity(path):
    return {'path': str(path.resolve()), 'sha256': pub.sha(path), 'bytes': path.stat().st_size}


def media(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.new('RGB', (12, 12), (i * 17, i * 5, i * 9)) for i in range(count)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return {**identity(path), 'frames': count}


def source(tmp, kind, monkeypatch, extra=None):
    root = tmp / f'{kind}-snapshot'
    commit, _, relative, _ = pub.PINS[kind]
    file = root / relative; file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text('# synthetic source-bound software fixture\n')
    files = {relative: pub.sha(file)}
    for name, text in (extra or {}).items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        files[name] = pub.sha(path)
    digest = pub.stable(files)
    monkeypatch.setitem(pub.PINS, kind, (commit, digest, relative, files[relative]))
    manifest = {'files': files, 'digest': digest, 'origin_commit': commit,
                'schema_version': 1, 'snapshot_path': str(root)}
    write(root / 'forge-source.json', manifest)
    return {'source': {'commit': commit, 'execution_digest': digest, 'files_sha256': {relative: files[relative]}},
            'execution_source': manifest}


def attempt(packet, row, tmp, paid=10.):
    row.update(attempt_key=pub.stable(row['id']), command=['original-child', row['id']],
               log_path=str(tmp / row['id'] / 'run.log'), child_returncode=0,
               paid_wall_seconds=paid, charged_seconds=paid, new_paid_seconds=paid)
    folder = Path(packet['queue_root']) / 'policy/attempts' / row['attempt_key']
    write(folder / 'supervisor-request.json', {'source': packet['execution_source'], 'command': row['command'],
          'log_path': row['log_path'], 'token': row['id'], 'started_monotonic': 100.,
          'deadline_monotonic': 100. + row['allowance_seconds']})
    write(folder / 'supervisor-terminal.json', {'token': row['id'], 'attempt_status': 'completed',
          'paid_wall_seconds': paid, 'child_returncode': 0})


@pytest.fixture
def evidence_fixture(tmp_path, monkeypatch):
    baseline = {**source(tmp_path, 'baseline', monkeypatch, {pub.BASELINE_CONFIG: '{"synthetic_software_config":true}'}), 'queue_root': str(tmp_path / 'queue'),
                'family': 'atlas', 'executed_family': 'atlas', 'lane_runtime': pub.RUNTIME,
                'spec': {'total_paid_cap_seconds': 10800., 'export_grace_seconds': 60.},
                'required': 19, 'rows': [], 'case_definitions': {}, 'qualification_input': False}
    baseline['snapshot_locations'] = {'package': baseline['execution_source']['snapshot_path']}
    for case_id in pub.BASELINE_IDS:
        group, task = case_id.removeprefix('atlas-original19-').split('-', 1)
        steps = 7000 if group == 'native' else 1500 if group == 'moving' else 600
        definition = {'id': case_id, 'group': group, 'task': task, 'original_host': {'steps': steps},
                      'observation_steps': [0, steps // 2, steps], 'original_requirements': [['score', '<=', .1]]}
        baseline['case_definitions'][case_id] = definition
        cap = 2400. if group == 'native' else 1800.
        row = {'id': case_id, 'group': group, 'task': task, 'status': 'PASS', 'full_protocol_complete': True,
               'case_sha256': pub.stable(definition), 'timeout_seconds': cap, 'allowance_seconds': cap + 60.,
               'original_gate': 'PASS', 'completed_steps': steps, 'native_gates': {}, 'reported_original_status': 'PASS'}
        directory = tmp_path / 'baseline' / group / task
        result = directory / ('frames.npz.verdict.json' if group == 'moving' else 'result.json')
        write(result, {'status': 'PASS', 'seconds': 1., 'task': task,
                       'header': {'torch': pub.RUNTIME['torch'], 'cuda': pub.RUNTIME['cuda'],
                                  'gpu': pub.RUNTIME['cuda_device_model'], 'python': 'original-child'}})
        row.update(result_path=str(result), result_sha256=pub.sha(result))
        attempt(baseline, row, tmp_path)
        write(directory / 'request.json', {'packet': deepcopy(baseline), 'row': deepcopy(row), 'target': str(directory)})
        (directory / 'final-state.pt').write_bytes(b'software checkpoint')
        row['artifacts'] = {p.name: {k: v for k, v in identity(p).items() if k != 'path'}
                            for p in directory.iterdir()}
        row['media'] = {**media(directory / 'goal-metrics.gif', 3), 'actual_steps': definition['observation_steps'],
                        'metric_only': True, 'new_draws': False, 'training_updates': 0}
        write(directory / 'media-receipt.json', row['media'])
        baseline['rows'].append(row)
    # Real requests include the full immutable registry before the first child.
    for row in baseline['rows']:
        request = Path(row['result_path']).parent / 'request.json'
        d = pub.read(request); d['packet']['case_definitions'] = deepcopy(baseline['case_definitions']); write(request, d)
        row['artifacts']['request.json'] = {k: v for k, v in identity(request).items() if k != 'path'}
    baseline.update(completed=19, media_completed=19, required_evidence_complete=True,
                    spent_seconds=190., new_paid_seconds=190.)
    baseline_path = tmp_path / 'baseline/study.json'; write(baseline_path, baseline)
    monkeypatch.setattr(pub, '_expected_definitions', lambda packet, root: deepcopy(baseline['case_definitions']))
    hold_source = source(tmp_path, 'hold', monkeypatch, {'hold-original-source/benchmarks/proof.py': '# old scientific source\n'})
    old_source = {'commit': pub.ORIGINAL_COMMIT, 'files_sha256': {
        'benchmarks/proof.py': hold_source['execution_source']['files']['hold-original-source/benchmarks/proof.py']}}
    startup_source = source(tmp_path, 'startup', monkeypatch)
    history = {'reason': 'synthetic no-update startup', 'records': [], 'paid_seconds': 4.,
               'ordinary_updates': 0, 'automatic_retry': False}
    startup_root = tmp_path / 'startup'
    for family in ('atlas', 'e22'):
        row = {'id': f'c6-{family}-broad-hold-1200-to-1350', 'status': 'INCOMPLETE',
               'timeout_seconds': 180., 'allowance_seconds': 240., 'full_protocol_complete': False}
        old = {**deepcopy(startup_source), 'rows': [row], 'queue_root': baseline['queue_root']}
        attempt(old, row, tmp_path, paid=2.); row['child_returncode'] = 1
        terminal = Path(old['queue_root']) / 'policy/attempts' / row['attempt_key'] / 'supervisor-terminal.json'
        t = pub.read(terminal); t['child_returncode'] = 1; write(terminal, t)
        request = startup_root / family / row['id'] / 'request.json'; write(request, {'software_startup': True})
        log = request.parent / 'run.log'; log.write_text('namespace guard rejects before model construction')
        path = startup_root / family / 'study.json'; write(path, old)
        history['records'].append({'family': family, 'study_path': str(path), 'study_sha256': pub.sha(path),
                                  'source': old['source'], 'status': 'INCOMPLETE', 'scientific_updates': 0,
                                  'paid_wall_seconds': 2., 'artifacts': {'request.json': identity(request), 'run.log': identity(log)}})
    parents = {}
    for family in ('atlas', 'e22'):
        path = tmp_path / 'parents' / family / 'receipt.json'; write(path, {'original1200': 'PASS'})
        parents[family] = {'path': str(path), 'receipt_sha256': pub.sha(path), 'artifacts': {},
                           'receipt': {'observations': []}, 'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE'}
    hold_defs = {f'c6-{family}-broad-hold-1200-to-1350': {'id': f'c6-{family}-broad-hold-1200-to-1350',
                 'family': family, 'original_receipt_sha256': parents[family]['receipt_sha256'],
                 'new_observation_steps': [1250, 1300, 1350], 'new_updates': 150, 'total_execution_limit': 1350}
                 for family in ('atlas', 'e22')}
    lanes, paths = [], []
    for family in ('atlas', 'e22'):
        case_id = f'c6-{family}-broad-hold-1200-to-1350'
        lane = {**deepcopy(hold_source), 'executed_family': family, 'queue_root': str(tmp_path / 'hold-queue'),
                'lane_runtime': pub.RUNTIME, 'original_source': old_source, 'parents': deepcopy(parents),
                'case_definitions': deepcopy(hold_defs), 'scientific_snapshot': str(Path(hold_source['execution_source']['snapshot_path']) / 'hold-original-source'),
                'spec': {'total_paid_cap_seconds': 480., 'export_grace_seconds': 60., 'engineering_recovery': history,
                         'previous_paid_seconds': 4., 'remaining_paid_cap_seconds': 476.},
                'baseline_control': {'path': baseline['rows'][0]['result_path'], 'sha256': baseline['rows'][0]['result_sha256'],
                                     'maintained_source': baseline['source']}}
        row = {'id': case_id, 'family': family, 'case_sha256': pub.stable(hold_defs[case_id]),
               'timeout_seconds': 180., 'allowance_seconds': 240., 'status': 'FAIL', 'study_gate': 'FAIL',
               'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE', 'full_protocol_complete': True,
               'completed_steps': 1350, 'new_updates': 150, 'compound_hold': {'status': 'FAIL'},
               'acquisition_seconds': 1., 'export_seconds': 1.}
        directory = tmp_path / 'holds' / family / case_id
        write(directory / 'result.json', {'status': 'COMPLETE', 'acquisition_seconds': 1., 'export_seconds': 1.,
              'observations': [{'step': s, 'passed': False} for s in (1250, 1300, 1350)],
              'original_observation_flags': [{'step': 1200, 'passed': True}]})
        for name in ('appended-observations.npz', 'continued-state.pt'):
            (directory / name).write_bytes(b'self-contained software artifact')
        row['media'] = media(directory / 'goal.gif', 12)
        row.update(result_path=str(directory / 'result.json'), result_sha256=pub.sha(directory / 'result.json'),
                   artifacts={name: identity(directory / name) for name in ('appended-observations.npz', 'continued-state.pt', 'goal.gif')})
        attempt(lane, row, tmp_path)
        lane.update(rows=[row], spent_seconds=10., completed=1)
        path = tmp_path / 'holds' / family / 'study.json'; write(path, lane); paths.append(path); lanes.append(lane)
    summary_path = tmp_path / 'holds/hold-continuation.json'
    write(summary_path, {'schema': 'c6_hold_continuation_summary_v2', 'family_studies': lanes,
                        'engineering_recovery': history, 'total_paid_cap_seconds': 480.,
                        'previous_paid_seconds': 4., 'new_paid_seconds': 20., 'spent_seconds': 24.})
    fake_driver = SimpleNamespace(engineering_history=lambda root: deepcopy(history),
                                  original=lambda family: deepcopy(parents[family]), PARENTS=parents,
                                  verify_cost=lambda row: row['charged_seconds'])
    monkeypatch.setattr(pub, '_load_driver', lambda *args: fake_driver)

    def certifier(kind, path, row):
        raw = pub.read(row['result_path'])
        verified = {k: deepcopy(v) for k, v in row.items() if k not in {'command', 'media', 'cost'}}
        verified['full_protocol_complete'] = True
        if kind == 'baseline':
            verified['status'] = verified['original_gate'] = raw['status']
            verified['reported_original_status'] = raw['status']
        else:
            verified['status'] = verified['study_gate'] = 'FAIL'
        return verified

    monkeypatch.setattr(pub, 'original_certify', certifier)
    monkeypatch.setattr(pub, 'publisher_identity', lambda: {'commit': 'a' * 40, 'sha256': pub.sha(PATH)})
    return SimpleNamespace(baseline=baseline, baseline_path=baseline_path, hold_paths=paths,
                           summary_path=summary_path, startup_root=startup_root, tmp=tmp_path, history=history)


def collect(f):
    return pub.collect(f.baseline_path, f.hold_paths, f.summary_path, f.startup_root)


def update_baseline(f, callback):
    d = pub.read(f.baseline_path); callback(d); write(f.baseline_path, d)


def test_verified_export_is_shareable_and_never_updates_raw_or_queue(evidence_fixture):
    f = evidence_fixture
    before = {str(p): pub.sha(p) for p in f.tmp.rglob('*') if p.is_file()}
    output = f.tmp / 'fresh-publication'
    result = pub.publish(f.baseline_path, f.hold_paths, f.summary_path, f.startup_root, output)
    assert result['baseline']['completed'] == result['baseline']['media_completed'] == 19
    assert result['hold_continuations']['scientific_counts']['FAIL'] == 2
    assert result['cost']['paid_seconds'] == 214.  # H1 counted exactly once.
    assert result['hold_continuations']['engineering_startup']['records'][0]['status'] == 'INCOMPLETE'
    assert not result['qualification_input'] and not result['default_adoption'] and not result['speed_ranking']
    assert len(list((output / 'media').glob('*.gif'))) == 21
    saved = pub.read(output / 'results.json')
    assert len(saved['baseline']['rows']) == 19
    for row in saved['baseline']['rows'] + saved['hold_continuations']['rows']:
        assert pub.sha(output / row['media']['path']) == row['media']['sha256']
    assert all(pub.sha(p) == expected for p, expected in before.items())
    assert 'independent 100k' in (output / 'README.md').read_text()
    assert len(saved['provenance']['inputs']) < saved['provenance']['verified_file_count_including_snapshots']


@pytest.mark.parametrize('mutation,match', [
    (lambda d: d['rows'].pop(), 'denominator'),
    (lambda d: d['rows'].__setitem__(1, deepcopy(d['rows'][0])), 'denominator'),
    (lambda d: d['source'].__setitem__('commit', 'f' * 40), 'source identity'),
    (lambda d: d['lane_runtime'].__setitem__('torch', 'different'), 'runtime'),
    (lambda d: d.__setitem__('completed', 18), 'counts'),
    (lambda d: d.__setitem__('required_evidence_complete', False), 'completeness'),
    (lambda d: d.__setitem__('qualification_input', True), 'qualification'),
    (lambda d: d['rows'][0].__setitem__('timeout_seconds', 900.), 'budget'),
    (lambda d: d['rows'][0].__setitem__('status', 'RUNNING'), 'stable boundary'),
    (lambda d: d['rows'][0].__setitem__('status', 'FAIL'), 'original certifier'),
    (lambda d: d['rows'][0].__setitem__('full_protocol_complete', False), 'original certifier'),
])
def test_forged_metadata_rejected(evidence_fixture, mutation, match):
    f = evidence_fixture
    update_baseline(f, mutation)
    with pytest.raises(ValueError, match=match):
        collect(f)


@pytest.mark.parametrize('name', ['final-state.pt', 'result.json', 'goal-metrics.gif'])
def test_missing_required_artifact_rejected(evidence_fixture, name):
    f = evidence_fixture
    (Path(f.baseline['rows'][0]['result_path']).parent / name).unlink()
    with pytest.raises(ValueError, match='unavailable'):
        collect(f)


@pytest.mark.parametrize('kind', ['artifact', 'source', 'supervisor_source', 'cost', 'exit', 'frame'])
def test_bound_artifact_source_supervisor_and_cost_rejected(evidence_fixture, kind):
    f = evidence_fixture; row = f.baseline['rows'][0]
    if kind == 'artifact':
        (Path(row['result_path']).parent / 'final-state.pt').write_bytes(b'changed')
    elif kind == 'source':
        (Path(f.baseline['execution_source']['snapshot_path']) / pub.BASELINE_DRIVER).write_text('changed source')
    elif kind in {'supervisor_source', 'exit'}:
        root = Path(f.baseline['queue_root']) / 'policy/attempts' / row['attempt_key']
        path = root / ('supervisor-request.json' if kind == 'supervisor_source' else 'supervisor-terminal.json')
        d = pub.read(path)
        if kind == 'exit': d['child_returncode'] = 1
        else: d['source']['origin_commit'] = 'd' * 40
        write(path, d)
    elif kind == 'cost':
        def zero(d):
            d['rows'][0].update(paid_wall_seconds=0., charged_seconds=0., new_paid_seconds=0.)
            d.update(spent_seconds=180., new_paid_seconds=180.)
        update_baseline(f, zero)
    else:
        update_baseline(f, lambda d: d['rows'][0]['media'].__setitem__('frames', 2))
    with pytest.raises(ValueError):
        collect(f)


@pytest.mark.parametrize('status', ['UNKNOWN', 'BLOCKED', 'INCOMPLETE', 'ERROR'])
def test_unavailable_is_unassessed_never_scientific_failure(evidence_fixture, status):
    f = evidence_fixture
    def unavailable(d):
        row = d['rows'][-1]
        row.update(status=status, full_protocol_complete=False)
        for k in ('result_path', 'result_sha256', 'media', 'artifacts', 'original_gate', 'reported_original_status', 'completed_steps', 'native_gates'):
            row.pop(k, None)
        if status in {'UNKNOWN', 'BLOCKED'}:
            for k in ('attempt_key', 'command', 'log_path', 'child_returncode', 'paid_wall_seconds', 'charged_seconds', 'new_paid_seconds'):
                row.pop(k, None)
            d.update(spent_seconds=180., new_paid_seconds=180.)
        d.update(completed=18, media_completed=18, required_evidence_complete=False)
    update_baseline(f, unavailable)
    result, _ = collect(f)
    row = result['baseline']['rows'][-1]
    assert row['execution_status'] == status and row['scientific_status'] is None
    assert result['baseline']['scientific_counts']['FAIL'] == 0
    assert result['baseline']['scientific_counts']['UNASSESSED'] == 1


def test_binary_pass_without_retained_result_is_rejected(evidence_fixture):
    f = evidence_fixture
    update_baseline(f, lambda d: d['rows'][-1].pop('result_path'))
    with pytest.raises((ValueError, KeyError)):
        collect(f)


def test_raw_change_detected_between_verification_and_publication(evidence_fixture, monkeypatch):
    f = evidence_fixture
    original = pub.readme
    def mutate(result):
        Path(f.baseline['rows'][0]['result_path']).write_text('{}')
        return original(result)
    monkeypatch.setattr(pub, 'readme', mutate)
    output = f.tmp / 'fresh-publication'
    with pytest.raises(ValueError, match='hash/size changed'):
        pub.publish(f.baseline_path, f.hold_paths, f.summary_path, f.startup_root, output)
    assert not output.exists()


def test_existing_or_raw_output_is_refused(evidence_fixture):
    f = evidence_fixture
    for output in (f.tmp, f.tmp / 'baseline' / 'publication', f.tmp / 'queue' / 'publication',
                   f.tmp / 'parents' / 'atlas' / 'publication'):
        with pytest.raises(ValueError, match='NEW|overlaps'):
            pub.publish(f.baseline_path, f.hold_paths, f.summary_path, f.startup_root, output)


def test_wrong_hold_grade_and_coherent_cost_reduction_rejected(evidence_fixture):
    f = evidence_fixture; path = f.hold_paths[0]
    lane = pub.read(path); lane['rows'][0]['status'] = lane['rows'][0]['study_gate'] = 'PASS'; write(path, lane)
    with pytest.raises(ValueError, match='original certifier'):
        collect(f)
    lane['rows'][0].update(status='FAIL', study_gate='FAIL', paid_wall_seconds=0., charged_seconds=0.)
    lane['spent_seconds'] = 0.; write(path, lane)
    with pytest.raises(ValueError, match='supervised paid cost'):
        collect(f)


def test_h1_changed_log_and_summary_double_count_rejected(evidence_fixture):
    f = evidence_fixture
    summary = pub.read(f.summary_path); summary['spent_seconds'] += 4.; write(f.summary_path, summary)
    with pytest.raises(ValueError, match='combined hold charged'):
        collect(f)
    summary['spent_seconds'] -= 4.; write(f.summary_path, summary)
    Path(f.history['records'][0]['artifacts']['run.log']['path']).write_text('forged early success')
    with pytest.raises(ValueError, match='hash/size changed'):
        collect(f)


def test_original_certifier_launch_is_cpu_verification_only(monkeypatch, tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs)); return SimpleNamespace(returncode=0, stdout='{"status":"FAIL"}', stderr='')
    monkeypatch.setattr(pub.subprocess, 'run', run)
    assert pub.original_certify('hold', tmp_path / 'study.json', {'id': 'case'})['status'] == 'FAIL'
    command, options = calls[0]
    assert '--_certify' in command and '--child' not in command
    assert options['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert options['timeout'] == 60


def actual_baseline_driver():
    # Exercise the real original certifier's native structural guard directly.
    path = ROOT / pub.BASELINE_DRIVER
    spec = importlib.util.spec_from_file_location('publication_native_certifier_control', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_partial_native_100k_scope_rejected_by_original_certifier(tmp_path):
    driver = actual_baseline_driver(); target = tmp_path / 'native'; target.mkdir()
    package = tmp_path / 'package'; (package / 'particlegan').mkdir(parents=True)
    (package / 'particlegan' / 'empty.py').write_text('# synthetic package\n')
    write(package / driver.CONFIG, {'config': 'software fixture'})
    steps = [6000, 6250, 6500, 6750, 7000]
    row = {'id': 'native-case', 'task': 'grid100', 'group': 'native'}
    packet = {'case_definitions': {'native-case': {'original_host': {'steps': 7000, 'terminal_steps': steps},
              'observation_steps': steps, 'sampling': 'noisy primary; clean separate'}},
              'snapshot_locations': {'package': str(package)}}
    header = {'options': driver.OPTIONS, 'device': 'cuda:0', 'cuda_visible_devices': '1',
              'package_root': str(package.resolve()), 'package_sha256': driver.package_digest(package),
              'overrides': {'config': 'software fixture', 'initialization': 'batch_feature_zero'}}
    write(target / 'result.json', {'task': 'grid100', 'status': 'PASS', 'completed_steps': 7000,
                                 'stream_deviations': 0, 'header': header})
    (target / 'metrics.jsonl').write_text(''.join(json.dumps({'step': s}) + '\n' for s in steps))
    (target / 'final-state.pt').write_bytes(b'software state')
    write(target / 'native-noisy/summary.json', {'completed_steps': 7000, 'eval_steps': steps,
                                               'accuracy': {'holdout_samples': 20000}})
    write(target / 'native-noisy/verdict.json', {'coverage': {'status': 'PASS'}, 'accuracy': {'status': 'PASS'}})
    with pytest.raises(ValueError, match='100k protocol incomplete'):
        driver.certify(packet, row, target, 0)


def test_native_noisy_coverage_fail_cannot_be_overridden_by_accuracy_pass(tmp_path):
    import numpy as np
    driver = actual_baseline_driver(); target = tmp_path / 'native'; target.mkdir()
    package = tmp_path / 'package'; (package / 'particlegan').mkdir(parents=True)
    (package / 'particlegan/empty.py').write_text('# fixture\n'); write(package / driver.CONFIG, {})
    steps = [6000, 6250, 6500, 6750, 7000]
    row = {'id': 'native-case', 'task': 'grid100', 'group': 'native'}
    packet = {'case_definitions': {'native-case': {'original_host': {'steps': 7000, 'terminal_steps': steps},
              'observation_steps': steps, 'sampling': 'noisy primary; clean separate'}}, 'snapshot_locations': {'package': str(package)}}
    write(target / 'result.json', {'task': 'grid100', 'status': 'PASS', 'completed_steps': 7000, 'stream_deviations': 0,
              'header': {'options': driver.OPTIONS, 'device': 'cuda:0', 'cuda_visible_devices': '1',
                         'package_root': str(package.resolve()), 'package_sha256': driver.package_digest(package),
                         'overrides': {'initialization': 'batch_feature_zero'}}})
    (target / 'metrics.jsonl').write_text(''.join(json.dumps({'step': s}) + '\n' for s in steps))
    (target / 'final-state.pt').write_bytes(b'software state')
    for law in ('noisy', 'clean'):
        folder = target / f'native-{law}'
        write(folder / 'summary.json', {'completed_steps': 7000, 'eval_steps': steps, 'accuracy': {'holdout_samples': 100000}})
        write(folder / 'verdict.json', {'coverage': {'status': 'FAIL' if law == 'noisy' else 'PASS'}, 'accuracy': {'status': 'PASS'}})
        for name, n in [('final_samples.npz', 20000), ('holdout_samples.npz', 100000)] + [(f'quality_checks/step_{s:06d}.npz', 20000) for s in steps]:
            path = folder / name; path.parent.mkdir(parents=True, exist_ok=True)
            cloud = np.zeros((n, 2), dtype=np.float32)
            np.savez_compressed(path, live=cloud, ema=cloud, target=cloud)
    result = driver.certify(packet, row, target, 0)
    assert result['status'] == result['original_gate'] == 'FAIL'
    assert result['reported_original_status'] == 'PASS'
    assert result['native_gates']['clean'] == {'coverage': 'PASS', 'accuracy': 'PASS'}


@pytest.fixture
def native_supplement(tmp_path, monkeypatch):
    steps = [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
    case = 'atlas-original19-native-grid100'
    definition = {'id': case, 'group': 'native', 'task': 'grid100', 'observation_steps': steps}
    row = {'id': case, 'group': 'native', 'full_protocol_complete': True, 'definition': definition,
           'case_sha256': pub.stable(definition), 'scientific_status': 'FAIL', 'reported_original_status': 'PASS',
           'native_gates': {'noisy': {'coverage': 'FAIL', 'accuracy': 'PASS'},
                            'clean': {'coverage': 'PASS', 'accuracy': 'PASS'}}, 'artifacts': {}, 'media': None}
    required = {'request.json', 'result.json', 'metrics.jsonl', 'final-state.pt', 'adapter-relocation.json', 'native-score-wrapper.py'}
    for law in ('noisy', 'clean'):
        required.update(f'native-{law}/{name}' for name in ('config.json', 'events.jsonl', 'summary.json', 'verdict.json', 'final_samples.npz', 'holdout_samples.npz'))
        required.update(f'native-{law}/quality_checks/step_{s:06d}.npz' for s in (6000, 6250, 6500, 6750, 7000))
    required.update(f'native-noisy/snapshots/step_{s:06d}.npz' for s in steps)
    for name in required:
        path = tmp_path / 'raw' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(name.encode())
        row['artifacts'][name] = identity(path)
    src = {'commit': pub.PINS['baseline'][0], 'execution_digest': pub.PINS['baseline'][1]}
    renderer = {'commit': 'a' * 40, 'files': {relative: identity(ROOT / relative) for relative in
                ('reports/forge/continuous-baseline-20261003/export_moving_goal.py',
                 'reports/forge/continuous-baseline-20261003/export_native_goal.py')}}
    def blob(commit, relative):
        return (ROOT / relative).read_bytes() if commit == 'a' * 40 else b'changed commit identity'
    monkeypatch.setattr(pub, 'git_blob', blob)
    receipt = {'schema': 'original_atlas_native_goal_media_v1', 'case_id': case, 'source': src,
               'execution_source_digest': src['execution_digest'], 'source_bound_case_sha256': row['case_sha256'],
               'original_gate': 'FAIL', 'original_gates': row['native_gates'], 'reported_original_status': 'PASS',
               'full_budget_complete': True, 'complete_execution_snapshot_verified': True,
               'posthoc_media_only': True, 'rescoring': False, 'training_updates': 0,
               'model_forwards': 0, 'new_draws': 0, 'interpolated_frames': 0,
               'original_observation_steps': steps, 'original_gate_samples': {'independent_holdout': 100000},
               'displayed_steps': [0, 50, 750, 1750, 2750, 3750, 4750, 5750, 7000],
               'raw_inputs': deepcopy(row['artifacts']), 'renderer_source': renderer,
               'gif': media(tmp_path / 'native-goal.gif', 9)}
    path = tmp_path / 'native-media-receipt.json'; write(path, receipt)
    return SimpleNamespace(path=path, receipt=receipt, baseline={'rows': [row], 'source': src})


def test_native_supplement_keeps_joint_gate_and_accuracy_status_separate(native_supplement):
    f = native_supplement
    result = pub.supplemental(f.path, f.baseline, pub.Evidence())
    assert result['original_gate'] == 'FAIL' and result['gif']['frames'] == 9
    assert len(result['raw_inputs']) == 62


@pytest.mark.parametrize('mutation,match', [
    (lambda d: d['renderer_source'].__setitem__('commit', 'b' * 40), 'commit does not contain'),
    (lambda d: d['displayed_steps'].__setitem__(-1, 6999), 'nine inclusive'),
    (lambda d: d['raw_inputs'].pop('native-clean/holdout_samples.npz'), 'inventory'),
    (lambda d: d['raw_inputs'].pop('native-noisy/snapshots/step_000000.npz'), 'inventory'),
    (lambda d: d.__setitem__('original_gate', 'PASS'), 'source/verdict'),
    (lambda d: d['original_gate_samples'].__setitem__('independent_holdout', 20000), '100k scope'),
    (lambda d: d.__setitem__('training_updates', 1), 'source/verdict'),
    (lambda d: d['gif'].__setitem__('frames', 8), 'frame count'),
    (lambda d: d['source'].__setitem__('execution_digest', 'c' * 64), 'source/verdict'),
    (lambda d: d['renderer_source']['files'].pop('reports/forge/continuous-baseline-20261003/export_moving_goal.py'), 'dependency'),
])
def test_supplement_source_shape_commit_and_case_inventory_fail_closed(native_supplement, mutation, match):
    f = native_supplement; mutation(f.receipt); write(f.path, f.receipt)
    with pytest.raises(ValueError, match=match):
        pub.supplemental(f.path, f.baseline, pub.Evidence())


def test_supplement_cannot_borrow_other_case_artifacts(native_supplement, tmp_path):
    f = native_supplement
    path = tmp_path / 'different-case' / 'final-state.pt'; path.parent.mkdir(); path.write_bytes(b'final-state.pt')
    f.receipt['raw_inputs']['final-state.pt'] = identity(path); write(f.path, f.receipt)
    with pytest.raises(ValueError, match='does not belong'):
        pub.supplemental(f.path, f.baseline, pub.Evidence())


def test_uncommitted_publisher_cannot_export(monkeypatch):
    def output(command, **kwargs):
        return 'a' * 40 if command[:3] == ['git', 'rev-parse', 'HEAD'] else b'different source bytes'
    monkeypatch.setattr(pub.subprocess, 'check_output', output)
    with pytest.raises(ValueError, match='commit the exact'):
        pub.publisher_identity()


def test_frozen_module_loading_does_not_write_bytecode(monkeypatch, tmp_path):
    (tmp_path / 'publication_control_dependency.py').write_text('VALUE = 7\n')
    (tmp_path / 'driver.py').write_text('import publication_control_dependency\nVALUE = publication_control_dependency.VALUE\n')
    monkeypatch.setattr(pub.sys, 'dont_write_bytecode', False)
    module = pub._load_driver(tmp_path, 'driver.py')
    assert module.VALUE == 7
    assert pub.sys.dont_write_bytecode is False
    assert not list(tmp_path.rglob('__pycache__'))
    pub.sys.modules.pop('publication_control_dependency', None)


def test_coherently_rebound_wrong_raw_runtime_is_rejected(evidence_fixture):
    f = evidence_fixture; packet = pub.read(f.baseline_path); row = packet['rows'][0]
    path = Path(row['result_path']); raw = pub.read(path); raw['header']['gpu'] = 'Different GPU'; write(path, raw)
    row['result_sha256'] = pub.sha(path)
    row['artifacts'][path.name] = {k: v for k, v in identity(path).items() if k != 'path'}
    write(f.baseline_path, packet)
    with pytest.raises(ValueError, match='raw runtime differs'):
        collect(f)
