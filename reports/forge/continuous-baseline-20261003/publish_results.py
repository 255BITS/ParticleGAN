"""Offline, fail-closed publication of the frozen Atlas19 and C6 hold studies.

This reads existing evidence and copies existing GIFs. It does not run a learner,
sample a model, render frames, initialize a queue, rank configurations or promote
defaults. Complete rows are checked by their original frozen certifiers; the hold
certifier's fresh CPU process only validates retained arrays and checkpoint data.
Use a new --output directory. Raw archives and source snapshots remain necessary
for independent rechecking; the compact publication does not replace them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
SELF = 'reports/forge/continuous-baseline-20261003/publish_results.py'
BASELINE_DRIVER = 'reports/forge/continuous-baseline-20261003/run_atlas_baseline.py'
BASELINE_CONFIG = 'configs/100gaussians/atlas.json'
HOLD_DRIVER = 'reports/forge/continuous-baseline-20261003/run_hold_continuation.py'
PINS = {
    'baseline': ('a0d6d89fb470f551b3f790016a237c40a377e1e8',
                 'e2b6c0e675c5da26663ac7a6181025df1d964dfe43e9a70591374ee94688d985',
                 BASELINE_DRIVER, '8dd658294dbbb2027e69a0e46097e2d2c0f5382078d407ade25e06c6c22ff1c1'),
    'hold': ('82c85cc32c43746074cb2809ae0f2aed271484bd',
             '429fc7d9a90a710346637da72ebe611a0a73515c05aede32c00640824fdfa760',
             HOLD_DRIVER, 'adcfbd92b39e3affa2626965bf32101bd4ade8a0ad00c9a4e4bbd4a0df4a0b58'),
    'startup': ('e97cae6d897369354588c29b65b02893ab066484',
                '29069307aaaefe61b6ab5c2c9b5b26a8b536323570db4054f2abea8f64dd7b29',
                HOLD_DRIVER, '93342f094a40d1e36df7161b0d250cef90ff7a9ffb965e0041c8365f13e85128'),
}
ORIGINAL_COMMIT = '8021a1c50c4aff90ddea5010d368cffdc857b2f6'
RUNTIME = {'python': '3.12.13', 'torch': '2.13.0+cu126', 'cuda': '12.6',
           'device': 'cuda:0', 'cuda_device_model': 'NVIDIA RTX A6000', 'torch_threads': 1}
STATUSES = ('PASS', 'FAIL', 'INCOMPLETE', 'ERROR', 'BLOCKED', 'UNKNOWN')
PORTS = ('img_intensity2', 'mode_hold', 'img_blobs4', 'img_bars4', 'img_stripes2',
         'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width',
         'vector_anisotropic', 'vector_overlap', 'vector_spiral', 'stationary', 'ring_shift')
NATIVE = ('grid100', 'rotated100', 'staggered100')
BASELINE_IDS = tuple(f'atlas-original19-{g}-{t}' for g, ts in
                     (('portability', PORTS), ('moving', NATIVE), ('native', NATIVE)) for t in ts)
HOLD_IDS = tuple(f'c6-{f}-broad-hold-1200-to-1350' for f in ('atlas', 'e22'))
SOURCE_SUFFIXES = {'.py', '.json', '.toml', '.yaml', '.yml', '.sh'}
FALSE_FLAGS = ('qualification_input', 'default_adoption', 'speed_ranking', 'speed_eligible',
               'ordinary_eight_case_qualification', 'current_forge_mog_clean_qualification')


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'invalid nonnegative finite {label}')
    return value


def equal_number(actual, expected, label):
    if not math.isclose(number(actual, label), number(expected, label), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f'{label} differs from retained evidence')


class Evidence:
    """Hash every input before use and recheck it before publication is installed."""
    def __init__(self):
        self.files = {}
        self.snapshot_roots = set()

    def file(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f'required artifact unavailable: {path}')
        record = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
        if expected is not None:
            if record['sha256'] != expected.get('sha256') or ('bytes' in expected and record['bytes'] != expected['bytes']):
                raise ValueError(f'artifact hash/size changed: {path}')
        if str(path) in self.files and self.files[str(path)] != record:
            raise ValueError(f'input changed while being read: {path}')
        self.files[str(path)] = record
        return record

    def json(self, path, expected=None):
        record = self.file(path, expected)
        value = read(path)
        self.file(path, record)
        return value

    def unchanged(self):
        for record in tuple(self.files.values()):
            self.file(record['path'], record)


def no_qualification(value):
    if any(value.get(name) is not None and value.get(name) is not False for name in FALSE_FLAGS):
        raise ValueError('this publication cannot claim family/default/speed qualification')


def snapshot(packet, kind, evidence):
    commit, digest, driver, driver_sha = PINS[kind]
    source, manifest = packet['source'], packet['execution_source']
    if (source.get('commit') != commit or source.get('execution_digest') != digest
            or manifest.get('origin_commit') != commit or manifest.get('digest') != digest
            or source.get('files_sha256', {}).get(driver) != driver_sha
            or manifest.get('files', {}).get(driver) != driver_sha
            or stable(manifest.get('files')) != digest):
        raise ValueError(f'{kind} frozen source identity differs')
    root = Path(manifest['snapshot_path']).resolve()
    evidence.snapshot_roots.add(root)
    disk = evidence.json(root / 'forge-source.json')
    # snapshot_path is registration metadata, not part of the manifest digest.
    if any(disk.get(k) != manifest.get(k) for k in ('digest', 'files', 'origin_commit', 'schema_version')):
        raise ValueError('on-disk frozen manifest differs')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts and p.name != 'forge-source.json'}
    if actual - set(manifest['files']):
        raise ValueError('undeclared executable/config files in frozen snapshot')
    for relative, expected in manifest['files'].items():
        member = (root / relative).resolve()
        if not member.is_relative_to(root):
            raise ValueError('source manifest path escapes its snapshot')
        evidence.file(member, {'sha256': expected})
    if any(manifest['files'].get(k) != v for k, v in source['files_sha256'].items()):
        raise ValueError('protected source bindings differ from complete snapshot')
    return root


def _load_driver(root, relative):
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('_offline_original_certifier_' + stable(str(root))[:12], root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def _certify_worker(kind, study_path, case_id):
    """Fresh namespace; original certifiers read retained evidence only."""
    packet = read(study_path)
    root = snapshot(packet, kind, Evidence())
    module = _load_driver(root, PINS[kind][2])
    row = next(r for r in packet['rows'] if r['id'] == case_id)
    target = Path(row['result_path']).parent
    return module.certify(packet, row, target, row['child_returncode'])


def original_certify(kind, study_path, row):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--_certify', kind,
               '--study', str(study_path), '--case', row['id']]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    if done.returncode:
        raise ValueError('original frozen certifier rejected retained evidence: ' + done.stderr[-3000:])
    return json.loads(done.stdout)


def rows_by_id(packet, ids):
    rows = packet.get('rows', [])
    if len(rows) != len(ids) or {r.get('id') for r in rows} != set(ids):
        raise ValueError('required case denominator missing or duplicated')
    if any(r.get('status') not in STATUSES for r in rows):
        raise ValueError('publication requires a stable boundary; RUNNING/unknown status is not a verdict')
    return {r['id']: r for r in rows}


def artifacts(row, target, evidence):
    result = {}
    for relative, declaration in row.get('artifacts', {}).items():
        path = (target / relative).resolve()
        if not path.is_relative_to(target.resolve()) or declaration.get('path', str(path)) != str(path):
            raise ValueError('artifact path escapes its declared case')
        result[relative] = evidence.file(path, declaration)
    return result


def gif(declaration, evidence, *, steps=None, frames=None):
    from PIL import Image
    actual = evidence.file(declaration['path'], declaration)
    with Image.open(actual['path']) as image:
        if image.format != 'GIF':
            raise ValueError('goal media must be an actual GIF')
        count = image.n_frames
    if declaration.get('frames', count) != count or (frames is not None and frames != count):
        raise ValueError('GIF frame count differs from actual bytes')
    if steps is not None and (declaration.get('actual_steps') != steps or len(steps) != count):
        raise ValueError('GIF does not match the declared actual observation selection')
    return {**actual, 'frames': count, 'actual_steps': steps}


def verify_cost(packet, row, evidence):
    paid = number(row.get('paid_wall_seconds', 0.), 'paid cost')
    reserve = number(row.get('unmeasured_interrupt_reserved_seconds', 0.), 'interrupt reservation')
    charged = number(row.get('charged_seconds', paid + reserve), 'charged cost')
    equal_number(charged, paid + reserve, 'charged cost')
    if row['status'] in {'UNKNOWN', 'BLOCKED'}:
        if paid or reserve or charged or row.get('full_protocol_complete') or row.get('result_path'):
            raise ValueError('unavailable row cannot carry scientific credit or paid acquisition')
        return {'paid_seconds': paid, 'reserved_seconds': reserve, 'charged_seconds': charged}
    if not re.fullmatch('[0-9a-f]{64}', row.get('attempt_key', '')):
        raise ValueError('measured row lacks its durable attempt identity')
    directory = Path(packet['queue_root']) / 'policy/attempts' / row['attempt_key']
    request = evidence.json(directory / 'supervisor-request.json')
    terminal_path = directory / 'supervisor-terminal.json'
    if (request.get('source') != packet['execution_source'] or request.get('command') != row.get('command')
            or request.get('log_path') != row.get('log_path')):
        raise ValueError('durable supervisor source/command identity differs')
    equal_number(request['deadline_monotonic'] - request['started_monotonic'], row['allowance_seconds'], 'supervisor allowance')
    if terminal_path.is_file():
        terminal = evidence.json(terminal_path)
        if terminal.get('token') != request.get('token'):
            raise ValueError('fenced supervisor terminal receipt')
        equal_number(paid, terminal['paid_wall_seconds'], 'supervised paid cost')
        if row.get('child_returncode') is not None and terminal.get('child_returncode') != row['child_returncode']:
            raise ValueError('recorded child exit differs from durable receipt')
        if row.get('full_protocol_complete') and terminal.get('attempt_status') != 'completed':
            raise ValueError('partial supervisor attempt cannot qualify a complete row')
    elif not reserve:
        raise ValueError('measured attempt has no terminal receipt or conservative reservation')
    if reserve and (row['status'] not in {'INCOMPLETE', 'ERROR'} or charged < row['allowance_seconds']):
        raise ValueError('unmeasured interruption lost its full reservation')
    if row.get('full_protocol_complete'):
        raw = evidence.json(row['result_path'], {'sha256': row['result_sha256']})
        minimum = raw.get('acquisition_seconds', raw.get('seconds', 0.)) + raw.get('export_seconds', 0.)
        if paid + 1e-6 < number(minimum, 'actual child elapsed cost'):
            raise ValueError('paid cost is below complete child elapsed time')
    return {'paid_seconds': paid, 'reserved_seconds': reserve, 'charged_seconds': charged}


def _check_verified(row, verified, fields):
    for field in fields:
        if row.get(field) != verified.get(field):
            raise ValueError(f'recorded {field} differs from original certifier')
    if row['status'] in {'PASS', 'FAIL'} and not verified.get('full_protocol_complete'):
        raise ValueError('partial evidence cannot supply a binary scientific verdict')


def _expected_definitions(packet, root):
    driver = _load_driver(root, BASELINE_DRIVER)
    inputs = deepcopy(packet['external_inputs'])
    original_files = inputs['files']
    inputs['files'] = {str(root / driver._external_relative(name, packet)): value for name, value in original_files.items()}
    inputs['harness'] = packet['snapshot_locations']['harness']
    inputs['reference'] = read(root / driver.REFERENCE)
    expected = {d['id']: d for d in (driver.task_definition(g, t, inputs) for g, t in driver.ordered_cases())}
    # Only relocate reads; the definition retains the registered original path/hash map.
    for definition in expected.values():
        definition['external_inputs_sha256'] = stable(original_files)
    return expected


def question(definition):
    group, task = definition['group'], definition['task']
    if group == 'native':
        return 'Does the original noisy served sampler recover all 100 components and their Gaussian fidelity, including independent 100k accuracy?'
    if group == 'moving':
        return 'Does coverage follow two 30-degree target turns while keeping at least 90% of its observed pre-turn quality?'
    if task.startswith('img_'):
        return 'Does the original small image host reproduce every declared template mode with the original noisy-output quality and mass gates?'
    if task == 'mode_hold':
        return 'Can the original 12-row host retain the declared ring modes under its original noisy serving law?'
    if task == 'stationary':
        return 'Can the original stationary ring host satisfy its complete noisy coverage protocol?'
    if task == 'ring_shift':
        return 'Can the original ring host preserve coverage through the declared target shift?'
    return 'Does the original host recover the declared vector distribution under the original noisy-serving quality and mass gates?'


def verify_baseline(study_path, evidence):
    study_path = Path(study_path).resolve()
    packet = evidence.json(study_path)
    no_qualification(packet)
    root = snapshot(packet, 'baseline', evidence)
    rows = rows_by_id(packet, BASELINE_IDS)
    if (packet.get('required') != 19 or packet.get('family') != 'atlas' or packet.get('executed_family') != 'atlas'
            or packet.get('lane_runtime') != RUNTIME or packet['spec'].get('total_paid_cap_seconds') != 10800.
            or packet['spec'].get('export_grace_seconds') != 60.
            or Path(packet['snapshot_locations']['package']).resolve() != root):
        raise ValueError('original Atlas19 runtime/family/resources changed')
    expected = _expected_definitions(packet, root)
    if set(expected) != set(BASELINE_IDS) or packet.get('case_definitions') != expected:
        raise ValueError('original19 task law/requirements changed')
    result_rows = []
    for case_id in BASELINE_IDS:
        row = rows[case_id]; definition = expected[case_id]
        no_qualification(row)
        cap = 2400. if definition['group'] == 'native' else 1800.
        if (row.get('case_sha256') != stable(definition) or row.get('timeout_seconds') != cap
                or row.get('allowance_seconds') != cap + 60.
                or (row.get('group'), row.get('task')) != (definition['group'], definition['task'])):
            raise ValueError('row case/source-bound budget changed')
        cost = verify_cost(packet, row, evidence)
        record = {'id': case_id, 'group': definition['group'], 'task': definition['task'],
                  'question': question(definition), 'definition': definition, 'case_sha256': row['case_sha256'],
                  'execution_status': row['status'], 'scientific_status': None,
                  'full_protocol_complete': False, 'media': None, 'cost': cost,
                  'reason': row.get('reason'), 'qualification_input': False}
        if row.get('result_path'):
            evidence.file(row['result_path'], {'sha256': row['result_sha256']})
            target = Path(row['result_path']).parent
            record['artifacts'] = artifacts(row, target, evidence)
            request = evidence.json(target / 'request.json')
            if (request.get('packet', {}).get('source') != packet['source']
                    or request['packet'].get('execution_source') != packet['execution_source']
                    or request['packet'].get('case_definitions') != packet['case_definitions']
                    or request['packet'].get('lane_runtime') != packet['lane_runtime']
                    or request.get('row', {}).get('case_sha256') != row['case_sha256']
                    or request['row'].get('timeout_seconds') != cap
                    or request.get('target') != str(target)):
                raise ValueError('original immutable request/source/runtime differs')
            verified = original_certify('baseline', study_path, row)
            _check_verified(row, verified, ('status', 'full_protocol_complete', 'original_gate', 'completed_steps', 'native_gates', 'reported_original_status'))
            if verified.get('full_protocol_complete'):
                if definition['group'] != 'moving':
                    header = evidence.json(row['result_path']).get('header', {})
                    if (header.get('torch') != packet['lane_runtime']['torch']
                            or header.get('cuda') != packet['lane_runtime']['cuda']
                            or header.get('gpu') != packet['lane_runtime']['cuda_device_model']
                            or header.get('python') != row['command'][0]):
                        raise ValueError('original raw runtime differs from registered scientific runtime')
                # Saved artifact declarations must include every scientific artifact.
                extras = set(verified['artifacts']) - set(row.get('artifacts', {}))
                if extras - {'goal-metrics.gif', 'media-receipt.json'}:
                    raise ValueError('scientific artifact absent from declared hash inventory')
                record.update(scientific_status=verified['status'], full_protocol_complete=True,
                              native_gates=verified.get('native_gates'), reported_original_status=verified.get('reported_original_status'),
                              clean_diagnostic_status=verified.get('clean_diagnostic_status'),
                              final_metrics=verified.get('final_metrics'), result=evidence.file(row['result_path']))
        elif row['status'] in {'PASS', 'FAIL'} or row.get('full_protocol_complete'):
            raise ValueError('a scientific verdict requires complete retained evidence')
        if row.get('media'):
            if not record['full_protocol_complete']:
                raise ValueError('original goal media cannot replace complete scientific evidence')
            media = row['media']; steps = definition['observation_steps']
            n = min(9, len(steps))
            chosen = [steps[(i * (len(steps) - 1)) // (n - 1)] for i in range(n)] if n > 1 else steps
            if media.get('metric_only') is not True or media.get('new_draws') is not False or media.get('training_updates') != 0:
                raise ValueError('original metric media must be observation-only')
            record['media'] = gif(media, evidence, steps=chosen)
            media_receipt = Path(media['path']).parent / 'media-receipt.json'
            receipt = evidence.json(media_receipt)
            if any(receipt.get(k) != v for k, v in media.items()):
                raise ValueError('raw media receipt differs from study declaration')
            record['media_receipt'] = evidence.file(media_receipt)
        result_rows.append(record)
    completed = sum(r['full_protocol_complete'] for r in result_rows)
    media_completed = sum(r['media'] is not None for r in result_rows)
    if packet.get('completed') != completed or packet.get('media_completed') != media_completed:
        raise ValueError('declared Atlas19 completion/media counts differ')
    if packet.get('required_evidence_complete') is not (completed == media_completed == 19):
        raise ValueError('declared original19 completeness differs')
    cost = _sum_costs(result_rows)
    equal_number(packet['spent_seconds'], cost['charged_seconds'], 'Atlas19 ledger charged cost')
    equal_number(packet['new_paid_seconds'], sum(r.get('new_paid_seconds', 0.) for r in packet['rows']), 'Atlas19 new paid ledger')
    if cost['charged_seconds'] > 10800.:
        raise ValueError('Atlas19 paid/reservation cap exceeded')
    return {'source': _source_record(packet), 'runtime': packet['lane_runtime'], 'study': evidence.file(study_path),
            'configuration': {'path_in_snapshot': BASELINE_CONFIG, 'sha256': packet['execution_source']['files'][BASELINE_CONFIG],
                              'overrides': evidence.json(root / BASELINE_CONFIG)},
            'required_questions': 19, 'declared_updates': 48800, 'completed': completed, 'media_completed': media_completed,
            'required_evidence_complete': completed == media_completed == 19,
            'execution_counts': _counts(result_rows, 'execution_status'), 'scientific_counts': _counts(result_rows, 'scientific_status'),
            'cost': cost, 'rows': result_rows, 'qualification_input': False, 'default_adoption': False, 'speed_ranking': False}


def _source_record(packet):
    manifest = packet['execution_source']
    return {'commit': packet['source']['commit'], 'execution_digest': manifest['digest'],
            'snapshot_path': manifest['snapshot_path'], 'snapshot_file_count': len(manifest['files']),
            'protected_files_sha256': packet['source']['files_sha256'], 'manifest_sha256': sha(Path(manifest['snapshot_path']) / 'forge-source.json')}


def compact_original_source(source):
    return {'commit': source['commit'], 'files_sha256_digest': stable(source['files_sha256']),
            'file_count': len(source['files_sha256']), 'full_manifest_location': 'hash-bound original raw C6 receipt and maintained snapshot'}


def _counts(rows, field):
    counts = {s: sum(r.get(field) == s for r in rows) for s in STATUSES}
    counts['UNASSESSED'] = sum(r.get(field) is None for r in rows)
    return counts


def _sum_costs(rows):
    return {key: math.fsum(r['cost'][key] for r in rows) for key in ('paid_seconds', 'reserved_seconds', 'charged_seconds')}


def verify_holds(study_paths, summary_path, startup_root, baseline, evidence):
    if len(study_paths) != 2:
        raise ValueError('both named hold studies are required')
    lanes = [evidence.json(p) for p in study_paths]
    if {p.get('executed_family') for p in lanes} != {'atlas', 'e22'}:
        raise ValueError('hold families missing or duplicated')
    hroot = snapshot(lanes[0], 'hold', evidence)
    driver = _load_driver(hroot, HOLD_DRIVER)
    history = driver.engineering_history(Path(startup_root))
    for record in history['records']:
        old = evidence.json(record['study_path'], {'sha256': record['study_sha256']})
        snapshot(old, 'startup', evidence)
        verify_cost(old, old['rows'][0], evidence)
        for artifact in record['artifacts'].values():
            evidence.file(artifact['path'], artifact)
    records = []
    for study_path, lane in zip(study_paths, lanes):
        no_qualification(lane)
        root = snapshot(lane, 'hold', evidence)
        family = lane['executed_family']; case_id = f'c6-{family}-broad-hold-1200-to-1350'
        if (lane.get('lane_runtime') != RUNTIME or lane.get('original_source', {}).get('commit') != ORIGINAL_COMMIT
                or lane['spec'].get('total_paid_cap_seconds') != 480. or lane['spec'].get('export_grace_seconds') != 60.
                or lane['spec'].get('engineering_recovery') != history
                or lane['spec'].get('previous_paid_seconds') != history['paid_seconds']
                or lane['spec'].get('remaining_paid_cap_seconds') != 480. - history['paid_seconds']):
            raise ValueError('hold runtime/source/recovery/budget differs')
        science = Path(lane['scientific_snapshot']).resolve()
        if science != root / 'hold-original-source':
            raise ValueError('hold original scientific namespace differs')
        for relative, expected in lane['original_source']['files_sha256'].items():
            if lane['execution_source']['files'].get('hold-original-source/' + relative) != expected:
                raise ValueError('old8021 namespace absent from maintained manifest')
        for f in ('atlas', 'e22'):
            parent = driver.original(f)
            if lane['parents'][f] != parent:
                raise ValueError('hold original source/recipe/flags/checkpoint bindings differ')
            evidence.file(parent['path'], {'sha256': parent['receipt_sha256']})
            for artifact in parent['artifacts'].values():
                evidence.file(artifact['path'], artifact)
        control = lane['baseline_control']
        first = baseline['rows'][0]
        if (first['scientific_status'] != 'PASS' or not first['full_protocol_complete']
                or control.get('path') != first['result']['path'] or control.get('sha256') != first['result']['sha256']
                or control.get('maintained_source', {}).get('execution_digest') != baseline['source']['execution_digest']):
            raise ValueError('one-host baseline prerequisite lacks source-bound PASS')
        row = rows_by_id(lane, (case_id,))[case_id]
        no_qualification(row)
        definition = lane['case_definitions'].get(case_id)
        if (set(lane['case_definitions']) != set(HOLD_IDS) or row.get('case_sha256') != stable(definition)
                or row.get('timeout_seconds') != 180. or row.get('allowance_seconds') != 240.
                or definition.get('original_receipt_sha256') != driver.PARENTS[family]['receipt_sha256']
                or definition.get('new_observation_steps') != [1250, 1300, 1350]
                or definition.get('new_updates') != 150 or definition.get('total_execution_limit') != 1350):
            raise ValueError('named hold case/150-update scope differs')
        compact_definition = deepcopy(definition)
        if 'original_source' in compact_definition:
            compact_definition['original_source'] = compact_original_source(lane['original_source'])
        record = {'id': case_id, 'family': family, 'execution_status': row['status'], 'scientific_status': None,
                  'original_gate': 'PASS', 'original_study_gate': 'INCOMPLETE', 'full_protocol_complete': False,
                  'question': 'Do three additional saved-cloud checks sustain the original 1200-update result, first confirmed at 1100, for five post-confirmation checks?',
                  'definition': compact_definition, 'case_sha256': row['case_sha256'],
                  'definition_projection': 'Repeated source file map replaced by its digest/count; full declaration remains hash-bound in raw study/request.',
                  'source': _source_record(lane), 'original_source': compact_original_source(lane['original_source']),
                  'runtime': lane['lane_runtime'], 'study': evidence.file(study_path), 'media': None,
                  'baseline_control': control, 'cost': verify_cost(lane, row, evidence), 'reason': row.get('reason')}
        if row.get('result_path'):
            evidence.file(row['result_path'], {'sha256': row['result_sha256']})
            verified = original_certify('hold', Path(study_path).resolve(), row)
            _check_verified(row, verified, ('status', 'original_gate', 'original_study_gate', 'study_gate', 'full_protocol_complete', 'compound_hold', 'completed_steps', 'new_updates'))
            driver.verify_cost(row)
            if verified.get('full_protocol_complete'):
                raw = evidence.json(row['result_path'])
                record.update(scientific_status=verified['status'], full_protocol_complete=True,
                              compound_hold=verified['compound_hold'], new_updates=150, completed_steps=1350,
                              appended_observations=raw['observations'], original_observation_flags=raw['original_observation_flags'],
                              result=evidence.file(row['result_path']), artifacts=artifacts(row, Path(row['result_path']).parent, evidence),
                              media=gif(row['media'], evidence, frames=12))
        elif row['status'] in {'PASS', 'FAIL'} or row.get('full_protocol_complete'):
            raise ValueError('binary hold verdict lacks full retained continuation')
        equal_number(lane['spent_seconds'], record['cost']['charged_seconds'], 'hold lane charged ledger')
        if lane.get('completed') != int(record['full_protocol_complete']):
            raise ValueError('hold completion count differs')
        records.append(record)
    records.sort(key=lambda r: r['family'])
    summary = evidence.json(summary_path)
    no_qualification(summary)
    saved_lanes = summary.get('family_studies', [])
    if (summary.get('schema') != 'c6_hold_continuation_summary_v2' or len(saved_lanes) != 2
            or {p.get('executed_family'): p for p in saved_lanes} != {p['executed_family']: p for p in lanes}
            or summary.get('engineering_recovery') != history or summary.get('total_paid_cap_seconds') != 480.):
        raise ValueError('hold summary disagrees with its immutable family studies')
    costs = _sum_costs(records)
    equal_number(summary['previous_paid_seconds'], history['paid_seconds'], 'startup paid debit')
    # The frozen source calls summed lane charged_seconds "new_paid_seconds";
    # its interrupt reservations must stay explicit in this publication.
    equal_number(summary['new_paid_seconds'], costs['charged_seconds'], 'hold new charged debit')
    equal_number(summary['spent_seconds'], costs['charged_seconds'] + history['paid_seconds'], 'combined hold charged ledger')
    if summary['spent_seconds'] > 480.:
        raise ValueError('combined startup/continuation allowance exceeded')
    return {'required_questions': 2, 'completed': sum(r['full_protocol_complete'] for r in records),
            'new_updates': sum(r.get('new_updates', 0) for r in records), 'rows': records,
            'execution_counts': _counts(records, 'execution_status'), 'scientific_counts': _counts(records, 'scientific_status'),
            'engineering_startup': history, 'cost': {**costs, 'previous_paid_seconds': history['paid_seconds'],
                'combined_charged_seconds': costs['charged_seconds'] + history['paid_seconds']},
            'summary': evidence.file(summary_path), 'qualification_input': False, 'default_adoption': False, 'speed_ranking': False}


def supplemental(path, baseline, evidence):
    receipt = evidence.json(path)
    record = next((r for r in baseline['rows'] if r['id'] == receipt.get('case_id')), None)
    if record is None or not record['full_protocol_complete']:
        raise ValueError('supplemental media needs its complete declared original case')
    kind = record['group']
    if receipt.get('schema') != f'original_atlas_{kind}_goal_media_v1' or kind not in {'moving', 'native'}:
        raise ValueError('unsupported supplemental media contract')
    no_qualification(receipt)
    if (receipt.get('source', {}).get('execution_digest') != baseline['source']['execution_digest']
            or receipt.get('source', {}).get('commit') != baseline['source']['commit']
            or receipt.get('execution_source_digest') != baseline['source']['execution_digest']
            or receipt.get('source_bound_case_sha256') != record['case_sha256']
            or receipt.get('original_gate') != record['scientific_status']
            or receipt.get('full_budget_complete') is not True
            or receipt.get('complete_execution_snapshot_verified') is not True
            or receipt.get('posthoc_media_only') is not True or receipt.get('rescoring') is not False
            or any(receipt.get(k) != 0 for k in ('training_updates', 'model_forwards', 'new_draws', 'interpolated_frames'))):
        raise ValueError('supplemental source/verdict/pure-observation scope differs')
    if kind == 'native':
        if (receipt.get('original_gates') != record['native_gates']
                or receipt.get('reported_original_status') != record['reported_original_status']
                or receipt.get('original_observation_steps') != record['definition']['observation_steps']
                or receipt.get('original_gate_samples', {}).get('independent_holdout') != 100000):
            raise ValueError('native supplemental media conflates primary/diagnostic/100k scope')
        steps = receipt.get('displayed_steps'); expected_frames = 9
        if steps != [0, 50, 750, 1750, 2750, 3750, 4750, 5750, 7000]:
            raise ValueError('native media must use the exact nine inclusive retained observations')
        required = {'request.json', 'result.json', 'metrics.jsonl', 'final-state.pt',
                    'adapter-relocation.json', 'native-score-wrapper.py'}
        for law in ('noisy', 'clean'):
            required.update(f'native-{law}/{name}' for name in ('config.json', 'events.jsonl', 'summary.json', 'verdict.json',
                            'final_samples.npz', 'holdout_samples.npz'))
            required.update(f'native-{law}/quality_checks/step_{s:06d}.npz' for s in (6000, 6250, 6500, 6750, 7000))
        required.update(f'native-noisy/snapshots/step_{s:06d}.npz' for s in record['definition']['observation_steps'])
    else:
        steps = receipt.get('actual_steps'); expected_frames = 4
        if steps != [0, 500, 1000, 1500] or receipt.get('gate_samples') != 20000:
            raise ValueError('moving supplemental observation/sample scope changed')
        required = {'frames.npz', 'frames.npz.verdict.json', 'request.json', 'runner.py'}
        required.update(f'frames.npz.checkpoint-{s:06d}.pt' for s in (500, 1000, 1500))
    expected_artifacts = deepcopy(record.get('artifacts', {}))
    if record.get('media'):
        required.update(('goal-metrics.gif', 'media-receipt.json'))
        expected_artifacts['goal-metrics.gif'] = {k: v for k, v in record['media'].items() if k in ('path', 'sha256', 'bytes')}
        expected_artifacts['media-receipt.json'] = record['media_receipt']
    if set(receipt.get('raw_inputs', {})) != required:
        raise ValueError('supplemental complete raw artifact inventory differs')
    for name, artifact in receipt['raw_inputs'].items():
        if expected_artifacts.get(name) != artifact:
            raise ValueError('supplemental raw input does not belong to its certified case')
        evidence.file(artifact['path'], artifact)
    renderer = receipt['renderer_source']
    declarations = renderer.get('files', {renderer.get('git_path'): renderer})
    if not re.fullmatch('[0-9a-f]{40}', renderer.get('commit', '')):
        raise ValueError('supplemental exporter commit is not frozen')
    expected_paths = {'reports/forge/continuous-baseline-20261003/export_moving_goal.py'}
    if kind == 'native':
        expected_paths.add('reports/forge/continuous-baseline-20261003/export_native_goal.py')
    if set(declarations) != expected_paths:
        raise ValueError('supplemental exporter dependency inventory differs')
    for relative, artifact in declarations.items():
        if Path(artifact['path']).resolve() != ROOT / relative:
            raise ValueError('supplemental exporter path differs from its declared checkout')
        evidence.file(artifact['path'], artifact)
        blob = git_blob(renderer['commit'], relative)
        if hashlib.sha256(blob).hexdigest() != artifact['sha256']:
            raise ValueError('supplemental exporter commit does not contain the declared bytes')
    return {'case_id': record['id'], 'kind': kind, 'receipt': evidence.file(path),
            'gif': gif(receipt['gif'], evidence, frames=expected_frames), 'actual_steps': steps,
            'renderer_source': renderer, 'raw_inputs': receipt['raw_inputs'], 'original_gate': record['scientific_status']}


def git_blob(commit, relative):
    return subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT)


def collect(baseline_study, hold_studies, hold_summary, startup_root, media_receipts=()):
    evidence = Evidence()
    baseline = verify_baseline(baseline_study, evidence)
    holds = verify_holds(hold_studies, hold_summary, startup_root, baseline, evidence)
    media = [supplemental(path, baseline, evidence) for path in media_receipts]
    if len({m['case_id'] for m in media}) != len(media):
        raise ValueError('duplicate supplemental media case')
    evidence.unchanged()
    return {'schema': 'continuous_original_atlas19_and_c6_hold_publication_v1',
            'baseline': baseline, 'hold_continuations': holds, 'supplemental_media': media,
            'cost': {'paid_seconds': baseline['cost']['paid_seconds'] + holds['cost']['paid_seconds'] + holds['cost']['previous_paid_seconds'],
                     'reserved_seconds': baseline['cost']['reserved_seconds'] + holds['cost']['reserved_seconds'],
                     'charged_seconds': baseline['cost']['charged_seconds'] + holds['cost']['combined_charged_seconds'],
                     'declared_combined_paid_cap_seconds': 11280.,
                     'scope': 'Summed supervisory attempt cost, including H1 exactly once; not global elapsed time, FLOPs or speed ranking.'},
            'qualification_input': False, 'default_adoption': False, 'speed_ranking': False,
            'raw_inputs_unchanged': True, 'new_training_updates': 0, 'new_sampler_calls': 0,
            'provenance': {'inputs': sorted((r for r in evidence.files.values()
                           if Path(r['path']).name == 'forge-source.json'
                           or not any(Path(r['path']).is_relative_to(root) for root in evidence.snapshot_roots)), key=lambda r: r['path']),
                           'independent_recheck_requires_raw_archives_and_exact_snapshots': True}}, evidence


def readme(result):
    baseline = result['baseline']; holds = result['hold_continuations']
    lines = ['# Frozen Atlas19 and C6 persistence evidence', '',
             f"Atlas19: {baseline['completed']}/19 full protocols and {baseline['media_completed']}/19 original goal GIFs.", '',
             'This publication keeps the original noisy serving cohort separate from clean diagnostics and the C6 persistence extension. '
             'It supplies no current MoG qualification, family winner, shipped default or speed ranking. Missing or blocked evidence is unassessed, never a numerical failure.', '',
             '| Original question | Updates | Execution | Scientific | Goal GIF |', '|---|---:|---|---|---|']
    for row in baseline['rows']:
        link = f"[actual observations]({row['media']['path']})" if row['media'] else 'unavailable'
        lines.append(f"| `{row['group']}/{row['task']}` | {row['definition']['original_host']['steps']} | {row['execution_status']} | {row['scientific_status'] or 'UNASSESSED'} | {link} |")
    lines += ['', 'Each question, full host/seed/budget, exact observation cadence and unchanged numerical requirements are retained in `results.json`. '
              'Native scientific PASS requires both original noisy coverage and Gaussian accuracy, including the final five 20k checks and the independent 100k holdout. Clean and EMA results remain diagnostic.', '',
              '| C6 continuation | Original 1200 gate | Original study | New hold gate | Added updates | Goal GIF |', '|---|---|---|---|---:|---|']
    for row in holds['rows']:
        link = f"[actual states]({row['media']['path']})" if row['media'] else 'unavailable'
        lines.append(f"| {row['family']} | PASS | INCOMPLETE | {row['scientific_status'] or 'UNASSESSED'} | {row.get('new_updates', 0)} | {link} |")
    lines += ['', 'The named hold variant restores each complete checkpoint from source `8021a1c5` and adds only 150 updates. '
              'It checks 1250/1300/1350 with the original scorer and retains the original 1200 PASS and study INCOMPLETE. '
              'Its compound requirement is five post-confirmation passing checks; it grants no ordinary eight-case credit.', '',
              f"Paid attempt cost: {result['cost']['paid_seconds']:.6f} seconds; unmeasured interrupt reservations: {result['cost']['reserved_seconds']:.6f} seconds. "
              f"The two preserved H1 startup INCOMPLETE attempts contribute {holds['cost']['previous_paid_seconds']:.6f} paid seconds exactly once. "
              'Summed paid cost is not elapsed study time or a cross-hardware speed comparison.', '',
              'This directory is shareable with its copied GIFs and hash-bound JSON. Independent rechecking additionally needs the listed raw archives, '
              'durable supervisor receipts and exact frozen source snapshots; bulk traces and checkpoints were not copied.', '']
    if result['supplemental_media']:
        lines += ['Supplemental retained-cloud views (original grades unchanged):', '']
        lines += [f"- `{m['case_id']}`: [actual retained clouds]({m['gif']['path']})." for m in result['supplemental_media']]
        lines.append('')
    return '\n'.join(lines)


def publisher_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    blob = subprocess.check_output(['git', 'show', f'{commit}:{SELF}'], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != sha(__file__):
        raise ValueError('commit the exact offline publisher before exporting actual evidence')
    return {'commit': commit, 'git_path': SELF, 'sha256': sha(__file__), 'python': platform.python_version()}


def publish(baseline_study, hold_studies, hold_summary, startup_root, output, media_receipts=()):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('publication output must be a NEW directory')
    exporter = publisher_identity()
    result, evidence = collect(baseline_study, hold_studies, hold_summary, startup_root, media_receipts)
    protected_roots = [Path(baseline_study).resolve().parent, Path(startup_root).resolve()]
    protected_roots += [Path(p).resolve().parent for p in hold_studies]
    protected_roots += [Path(result['baseline']['source']['snapshot_path'])]
    protected_roots += [Path(r['source']['snapshot_path']) for r in result['hold_continuations']['rows']]
    protected_roots += [Path(record['path']).parent for record in evidence.files.values()]
    protected_roots += [Path(read(p)['queue_root']).resolve() for p in [baseline_study, *hold_studies]]
    if output.is_relative_to(ROOT) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected_roots):
        raise ValueError('publication output overlaps a worktree, raw study or source snapshot')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.continuous-publication-', dir=output.parent))
    try:
        copied = []
        for row in result['baseline']['rows'] + result['hold_continuations']['rows']:
            if row.get('media'):
                copied.append((row['media'], f"media/{row['id']}.gif"))
        for item in result['supplemental_media']:
            copied.append((item['gif'], f"media/{item['case_id']}-retained-clouds.gif"))
        for declaration, relative in copied:
            source = evidence.file(declaration['path'], declaration)
            destination = temporary / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source['path'], destination)
            if sha(destination) != source['sha256']:
                raise ValueError('copied goal GIF bytes differ')
            declaration['raw_path'] = source['path']; declaration['path'] = relative
        result['publisher_source'] = exporter
        result['provenance']['input_file_count'] = len(result['provenance']['inputs'])
        result['provenance']['verified_file_count_including_snapshots'] = len(evidence.files)
        result['provenance']['input_manifest_sha256'] = stable(result['provenance']['inputs'])
        (temporary / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        (temporary / 'README.md').write_text(readme(result))
        evidence.unchanged()
        # rename fails if another publisher has populated the requested destination.
        if output.exists():
            raise ValueError('publication destination appeared during verification')
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-study', type=Path)
    parser.add_argument('--hold-study', type=Path, action='append', default=[])
    parser.add_argument('--hold-summary', type=Path)
    parser.add_argument('--startup-root', type=Path)
    parser.add_argument('--media-receipt', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--_certify', choices=('baseline', 'hold'), help=argparse.SUPPRESS)
    parser.add_argument('--study', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--case', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._certify:
        print(json.dumps(_certify_worker(args._certify, args.study, args.case), allow_nan=False))
        return 0
    if any(v is None for v in (args.baseline_study, args.hold_summary, args.startup_root, args.output)):
        parser.error('--baseline-study, two --hold-study paths, --hold-summary, --startup-root and --output are required')
    try:
        result = publish(args.baseline_study, args.hold_study, args.hold_summary, args.startup_root, args.output, args.media_receipt)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f'Publication rejected: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'publication': str(args.output), 'baseline_completed': result['baseline']['completed'],
                      'baseline_scientific_counts': result['baseline']['scientific_counts'],
                      'hold_scientific_counts': result['hold_continuations']['scientific_counts'],
                      'qualification_input': False}))
    return 0  # Verified publication, independent of scientific PASS/FAIL.


if __name__ == '__main__':
    raise SystemExit(main())
