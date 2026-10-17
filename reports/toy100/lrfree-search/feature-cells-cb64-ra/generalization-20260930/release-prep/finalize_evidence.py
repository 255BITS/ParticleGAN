"""Read finalized JSON/log evidence only; write qualification files inside release-prep.

Root invokes once final scopes close, or in a new output directory for a pending
snapshot. No candidate imports, tensor reads, scorers, numerical jobs or repo writes.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'
LANE = ROOT / 'validation-ra14'
PORTS = ('mode_hold', 'img_intensity2', 'img_blobs4', 'img_bars4', 'vector_unequal_mass',
         'img_stripes2', 'vector_two_broad', 'vector_unequal_width', 'vector_anisotropic',
         'vector_overlap', 'vector_spiral', 'ring_shift', 'stationary')
NATIVE = ('grid100', 'rotated100', 'staggered100')
PLAN = [('screen', task) for task in PORTS] + [('moving', task) for task in NATIVE]
PLAN += [('screen', task) for task in NATIVE]
OBSERVATIONS = [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
TERMINALS = [6000, 6250, 6500, 6750, 7000]
SOURCE_FREEZE_SHA = 'b998f0760458bf1b39ec8a8ed6ff00af0b3eab2c95e299c3154391949351c0de'
INVENTORY_SHA = '484bb263f164473054c05d15ac5f428d5b47cda635700b17ed352f0f97087177'
RUN_SHA = '68f5706590683a44348cb04b3798917411bfaa54e5d45b17d57c345a4da33c15'
MANIFEST_SHA = 'cfd04148d34257e90a4ba734515167c38937870f98e571541cc36f7f7e536bc6'
CONFIG_SHA = 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
RA13_CLOSED_SHA = 'b1fdad84be2a18918f25220fd8118a9a4181d56fe90ec90997d7a7be4522d971'
RA14_CLOSED_SHA = '07ac938ad49588b320d2bc83943bf691d85aa97db55d937c081267a8c9df3d97'
GPU_UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Evidence:
    def __init__(self):
        self.inputs = {}
        self.pending = []
        self.defects = []
        self.package_map = {}

    def raw(self, path, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                self.pending.append('Missing ' + str(path.relative_to(ROOT)))
            return None
        raw = path.read_bytes()
        self.inputs[str(path)] = sha(raw)
        return raw

    def read(self, path, required=True):
        raw = self.raw(path, required)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as error:
            self.defects.append(f'Unreadable JSON {path}: {error}')
            return None

    def check(self, condition, message):
        if not condition:
            self.defects.append(message)

    def integrity(self, record, where):
        self.check(isinstance(record, dict), where + ': integrity record absent')
        if not isinstance(record, dict):
            return
        for key, wanted in [('status', 'VALID'), ('package_sha256', RUN_SHA),
                            ('config_sha256', CONFIG_SHA), ('source_freeze_sha256', SOURCE_FREEZE_SHA)]:
            self.check(record.get(key) == wanted, where + ': ' + key + ' mismatch')


def source_and_bridge(e):
    frozen = e.read(LANE / 'SOURCE-FREEZE.json')
    e.check(e.inputs.get(str(LANE / 'SOURCE-FREEZE.json')) == SOURCE_FREEZE_SHA,
            'Final lane source freeze differs from the reviewed lane')
    inventory = e.read(HERE / 'INVENTORY.json')
    e.check(e.inputs.get(str(HERE / 'INVENTORY.json')) == INVENTORY_SHA,
            'Original archive inventory changed')
    closure = e.read(ROOT / 'RA14-SOURCE-CLOSURE.json')
    if frozen and closure:
        e.check(frozen.get('package_sha256') == RUN_SHA and frozen.get('config_sha256') == CONFIG_SHA,
                'Frozen package/config identity mismatch')
        e.check(closure.get('package_sha256') == MANIFEST_SHA and closure.get('config_sha256') == CONFIG_SHA,
                'Restoration-only source closure identity mismatch')
        package = Path(frozen['package'])
        mapping = {}
        run_digest = hashlib.sha256()
        for path in sorted((package / 'particlegan').rglob('*.py')):
            relative = str(path.relative_to(package / 'particlegan'))
            raw = e.raw(path)
            mapping[relative] = sha(raw)
            run_digest.update(relative.encode() + b'\0' + raw + b'\0')
        e.check(mapping == closure.get('source_sha256'), 'Actual final package source file map changed')
        e.package_map = mapping
        e.check(sha(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()) == MANIFEST_SHA,
                'Actual package manifest digest mismatch')
        e.check(run_digest.hexdigest() == RUN_SHA, 'Actual package run digest mismatch')
        for name in ['RA13-settled.json', 'RA14-replay.json']:
            raw = e.raw(ROOT / 'configs' / name)
            e.check(raw is not None and sha(raw) == CONFIG_SHA, name + ' differs from the shared config')
        # Bind only the finalization authorities; original collectors already checked
        # the complete source/input freeze before and after numerical execution.
        for path in [LANE / name for name in ('run_all.py', 'run_full_suite.py', 'collect.py',
                                              'run_screen.py', 'run_moving.py', 'freeze.py')]:
            raw = e.raw(path)
            e.check(raw is not None and sha(raw) == frozen['hashes'].get(str(path)),
                    'Frozen finalization authority changed: ' + path.name)
    old = e.read(ROOT / 'mnist/ra13-settled/CLOSED.json')
    new = e.read(ROOT / 'mnist/ra14-replay-r2/CLOSED.json')
    summary = e.read(ROOT / 'mnist/ra14-replay-r2/summary.json')
    e.check(e.inputs.get(str(ROOT / 'mnist/ra13-settled/CLOSED.json')) == RA13_CLOSED_SHA,
            'Original RA13 training/failure closure changed')
    e.check(e.inputs.get(str(ROOT / 'mnist/ra14-replay-r2/CLOSED.json')) == RA14_CLOSED_SHA,
            'Corrected RA14 replay closure changed')
    if old and new and summary:
        e.check(new.get('file_sha256', {}).get('summary.json') ==
                e.inputs.get(str(ROOT / 'mnist/ra14-replay-r2/summary.json')),
                'Closed replay summary hash mismatch')
        e.check(old.get('source_validity') == 'PASS', 'Original RA13 training source not valid')
        e.check(old.get('original_training_updates_per_fixture') == 2000, 'Original training budget mismatch')
        for key in ['mnist_all_checkpoint_metrics_equal_corrected_E22',
                    'mnist_all_checkpoint_lrs_equal_corrected_E22',
                    'toy_all_postupdate_metrics_and_lrs_equal_original_RA11']:
            e.check(old.get(key) is True, 'Original learned parity not qualified: ' + key)
        e.check(old.get('mnist_quality_gate') is None, 'A new MNIST gate has been introduced')
        e.check(old.get('replay_status') == {'toy': 'FAIL', 'mnist': 'FAIL'},
                'Original strict RA13 replay failures were not preserved')
        e.check(new.get('replay_status') == {'toy': 'PASS', 'mnist': 'PASS'}, 'RA14 replay is not PASS')
        e.check(new.get('original_training_equivalence_proven') is True and
                new.get('fresh_training_updates') == 0 and
                new.get('original_quality_receipt_sha256') == RA13_CLOSED_SHA,
                'RA13 fresh-training inheritance lacks the explicit helper-only bridge')
        e.check(summary.get('package_sha256') == RUN_SHA and summary.get('config_sha256') == CONFIG_SHA,
                'Closed replay package/config identity mismatch')
        for task in ['toy', 'mnist']:
            control = summary.get('fixtures', {}).get(task, {}).get('native_continuation_vs_sealed_RA13', {})
            e.check(control.get('status') == 'PASS' and all(value is True for value in control.get('checks', {}).values())
                    and len(control.get('per_update', [])) == 10,
                    task + ': native RA13 continuation bridge not complete')
    return inventory, dict(origin='RA13-settled fresh training', candidate='RA14-replay restoration-only bridge',
                          fresh_updates_per_fixture=2000, fresh_ra14_training_updates=0,
                          toy_original_gate=None if old is None else old.get('original_toy_quality_gate'),
                          mnist_original_gate=None, mnist_comparison='All10 metric/LR records match corrected E22',
                          replay=None if new is None else new.get('replay_status'),
                          replay_semantic_exclusion='birth_death.last.eval_seconds only',
                          inherited_final_metrics=None if summary is None else
                          {task:summary['fixtures'][task]['inherited_final_metrics'] for task in ('toy', 'mnist')})


def native_details(e, task, receipt, result):
    native = receipt.get('native') or {}
    plan = receipt.get('original_plan') or {}
    e.check(plan.get('terminal_steps') == TERMINALS and plan.get('observation_steps') == OBSERVATIONS,
            task + ': native34/five-terminal schedule mismatch')
    e.check(all(native.get('event_steps', {}).get(model) == OBSERVATIONS for model in ['live', 'ema']),
            task + ': native event schedule not complete')
    accuracy = native.get('accuracy') or {}
    terminal = accuracy.get('terminal_checks', [])
    e.check([item.get('step') for item in terminal] == TERMINALS,
            task + ': original native five terminal checks absent')
    e.check(all(item.get('metrics', {}).get('n') == 20000 for item in terminal),
            task + ': native terminal metric sample count differs from original20k')
    holdout = accuracy.get('holdout_metrics') or {}
    e.check(holdout.get('n') == 100000, task + ': independent100k holdout absent')
    clouds = [(f'quality_checks/step_{step:06d}.npz', 20000) for step in TERMINALS]
    clouds += [('final_samples.npz', 20000), ('holdout_samples.npz', 100000)]
    for relative, count in clouds:
        shapes = native.get('cloud_shapes', {}).get(relative, {})
        e.check(all(shapes.get(model) == [count, 2] for model in ['live', 'ema', 'target']),
                task + ': original paired-cloud shape receipt incomplete: ' + relative)
    official = native.get('official_status') or {}
    e.check(official.get('accuracy') == receipt.get('quality_status'), task + ': native accuracy verdict mismatch')
    e.check(all(official.get(key) in ('PASS', 'FAIL') for key in ['coverage', 'accuracy']),
            task + ': original native coverage/accuracy verdict missing')
    if official.get('accuracy') == 'PASS':
        e.check(official.get('coverage') == 'PASS' and all(item.get('passed') is True for item in terminal)
                and holdout.get('passed') is True, task + ': native PASS lacks original complete conjunction')
    wanted_metrics = ('precision', 'mass_tv', 'center_rms_sigma', 'abs_cov_trace_bias', 'radial_ks',
                      'accuracy_pass', 'frozen_pass', 'passed', 'n')
    return dict(coverage_status=official.get('coverage'), accuracy_status=official.get('accuracy'),
                terminal_checks=[dict(step=item.get('step'), passed=item.get('passed'),
                                      metrics={key:item.get('metrics', {}).get(key) for key in wanted_metrics})
                                 for item in terminal],
                holdout={key:holdout.get(key) for key in wanted_metrics},
                coverage_final=(result.get('final') or {}),
                paired_clouds='Original five20k terminal clouds + final20k + independent100k; shapes qualified by unchanged collector')


def screen(e, task):
    output = LANE / 'runs' / task
    receipt = e.read(output / 'acceptance-receipt.json')
    result = e.read(output / 'result.json')
    execution = e.read(output / 'execution-receipt.json')
    if not all(item is not None for item in [receipt, result, execution]):
        return None
    e.check(receipt.get('task') == task, task + ': receipt belongs to another task')
    validity = receipt.get('evidence_validity')
    quality = receipt.get('quality_status')
    e.check(validity == 'VALID' and receipt.get('reasons') == [], task + ': original collector evidence is INVALID')
    e.check(quality in ('PASS', 'FAIL') and quality == result.get('status'), task + ': quality verdict mismatch')
    e.check(receipt.get('acceptance_status') == (quality if validity == 'VALID' else 'INVALID'),
            task + ': acceptance conflates quality and validity')
    e.check(receipt.get('result_sha256') == e.inputs[str(output / 'result.json')], task + ': result hash mismatch')
    e.integrity(receipt.get('source_integrity'), task + ' collector')
    for key in ['source_integrity_before', 'source_integrity_after']:
        e.integrity(execution.get(key), task + ' execution ' + key)
    e.check(execution.get('process_exit_code') == 0 and execution.get('resources', {}).get('gpu_uuid') == GPU_UUID,
            task + ': numerical process/device identity mismatch')
    e.check(result.get('header', {}).get('package_sha256') == RUN_SHA, task + ': actual header package differs')
    plan = receipt.get('original_plan') or {}
    e.check(result.get('completed_steps') == plan.get('steps'), task + ': original full budget not completed')
    detail = native_details(e, task, receipt, result) if task in NATIVE else None
    return dict(kind='native' if task in NATIVE else 'portability', queue_kind='screen', task=task,
                evidence_validity=validity, quality_status=quality, receipt=str(output / 'acceptance-receipt.json'),
                final=receipt.get('final'), native=detail,
                selected_backend=(receipt.get('mechanisms') or {}).get('selection', {}).get('actual_backend'))


def moving(e, task):
    output = LANE / 'moving' / task
    receipt = e.read(output / 'COMPLETION.json')
    verdict = e.read(output / 'frames.npz.verdict.json')
    if receipt is None or verdict is None:
        return None
    if receipt.get('status') != 'COMPLETE':
        e.pending.append(task + ': moving receipt not COMPLETE')
        return None
    e.check(receipt.get('task') == task and receipt.get('verdict') == verdict,
            task + ': moving task/verdict receipt mismatch')
    for key in ['source_integrity_before', 'source_integrity_after']:
        e.integrity(receipt.get(key), task + ' moving ' + key)
    protocol = dict(turn_every=500, degrees=30, turns=2, steps=1500, seed=1234, draw=20000)
    e.check(receipt.get('protocol') == protocol, task + ': original moving protocol changed')
    for key in ['scorer_changed', 'schedule_changed', 'thresholds_changed']:
        e.check(receipt.get(key) is False, task + ': ' + key)
    periods = verdict.get('periods', [])
    e.check(verdict.get('turns') == 2 and [row.get('period_end') for row in periods] == [500, 1000, 1500]
            and [row.get('target_deg') for row in periods] == [0, 30, 60],
            task + ': moving two-postturn evidence incomplete')
    # These are the original declared Boolean gate equations on saved JSON scalars.
    postturn = []
    if len(periods) == 3:
        base = periods[0]['hq']
        e.check(verdict.get('pre_turn_hq') == base, task + ': moving baseline inconsistent')
        postturn = [dict(period_end=row['period_end'], target_deg=row['target_deg'], modes=row['modes'],
                         hq=row['hq'], required_hq=.9*base, passed=row['modes'] >= 95 and row['hq'] >= .9*base)
                    for row in periods[1:]]
        passed = sum(row['passed'] for row in postturn)
        e.check(verdict.get('passed_periods') == passed and
                verdict.get('status') == ('PASS' if passed == 2 else 'FAIL'), task + ': moving original gate mismatch')
    e.check(receipt.get('quality_status') == verdict.get('status') and verdict.get('status') in ('PASS', 'FAIL'),
            task + ': moving quality status mismatch')
    e.check(receipt.get('resources', {}).get('gpu_uuid') == GPU_UUID, task + ': moving device identity mismatch')
    return dict(kind='moving', queue_kind='moving', task=task, evidence_validity='VALID',
                quality_status=verdict.get('status'), receipt=str(output / 'COMPLETION.json'),
                baseline=periods[0] if periods else None, postturn=postturn)


def full_suite(e):
    path = LANE / 'FULL-TESTS.json'
    receipt = e.read(path)
    if receipt is None:
        return None
    if receipt.get('status') not in ('PASS', 'FAIL') or 'completed' not in receipt:
        e.pending.append('RA14 full suite has not completed')
        return None
    log = e.raw(LANE / 'full-pytest.log')
    if log is None:
        return None
    e.check(receipt.get('log_sha256') == sha(log), 'Final full-suite log hash mismatch')
    for key in ['source_integrity_before', 'source_integrity_after']:
        e.integrity(receipt.get(key), 'Full suite ' + key)
    try:
        repo_package = Path(receipt['repository']) / 'particlegan'
        declared = {str(Path(path).relative_to(repo_package)): expected
                    for path, expected in receipt.get('sources', {}).items()}
        e.check(declared == e.package_map, 'Full-suite repository source map differs from the actual final package')
    except Exception as error:
        e.defects.append('Full-suite source map unreadable: ' + str(error))
    e.check(receipt.get('status') == ('PASS' if receipt.get('returncode') == 0 else 'FAIL'),
            'Full suite return code/verdict mismatch')
    candidates = [line for line in log.decode(errors='replace').splitlines()
                  if re.search(r'\d+ (?:passed|failed|skipped|errors?)\b', line) and ' in ' in line]
    counts = {}
    if candidates:
        for field in ['passed', 'failed', 'skipped', 'errors', 'xfailed', 'xpassed']:
            pattern = 'errors?' if field == 'errors' else field
            matched = re.search(r'(?:^|[,= ]+)(\d+) ' + pattern + r'\b', candidates[-1])
            counts[field] = int(matched.group(1)) if matched else 0
        matched = re.search(r'(\d+) subtests passed', candidates[-1])
        counts['subtests_passed'] = int(matched.group(1)) if matched else 0
    e.check(bool(counts), 'Actual pytest summary counts unavailable; do not use expected counts')
    e.check(receipt.get('status') != 'PASS' or (counts.get('failed') == 0 and counts.get('errors') == 0),
            'Full suite PASS disagrees with actual pytest summary')
    return dict(status=receipt['status'], evidence_validity='VALID', returncode=receipt['returncode'],
                counts=counts, summary_line=candidates[-1] if candidates else None,
                receipt=str(path), log_sha256=sha(log),
                skip_reasons=[line for line in log.decode(errors='replace').splitlines() if line.startswith('SKIPPED ')])


def append_inventory(e, inventory, complete):
    original = {item['archive_relative_path'] for item in (inventory or {}).get('artifacts', [])}
    files = []
    suffixes = {'.json', '.jsonl', '.log'}
    candidates = set()
    for folder in ['runs', 'moving', 'logs']:
        if (LANE / folder).exists():
            candidates.update(path for path in (LANE / folder).rglob('*') if path.is_file() and path.suffix in suffixes)
    for name in ['SOURCE-FREEZE.json', 'scoreboard-all.json', 'FULL-TESTS.json', 'full-pytest.log']:
        path = LANE / name
        if path.is_file():
            candidates.add(path)
    for path in sorted(candidates):
        relative = str(path.relative_to(ROOT))
        if relative in original:
            continue
        raw = e.raw(path)
        files.append(dict(local_path=str(path), archive_relative_path=relative, bytes=len(raw), sha256=sha(raw),
                          scope='FINALIZED_SMALL_ARTIFACT' if complete else 'SNAPSHOT_REBIND_AFTER_COMPLETE'))
    return dict(status='READY_TO_APPEND' if complete else 'PENDING_FINAL_REBIND',
                original_inventory_sha256=INVENTORY_SHA, files=files, count=len(files),
                bytes=sum(item['bytes'] for item in files),
                excluded='All PT/PTH/checkpoints, NPZ/NPY clouds, images/GIFs and datasets remain local.',
                rule='Append only small files absent by path from the sealed510-file preparation inventory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='New exclusive output directory under release-prep; existing directories are rejected.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == HERE.resolve() or HERE.resolve() not in output.parents:
        parser.error('--output must be a new child directory under local release-prep')
    if output.exists():
        parser.error('retain every output; --output must not already exist')
    e = Evidence()
    inventory, learned = source_and_bridge(e)
    board = e.read(LANE / 'scoreboard-all.json')
    rows = []
    if board:
        listed = board.get('results', [])
        identities = [(item.get('kind'), item.get('task')) for item in listed]
        e.check(len(set(identities)) == len(identities) and set(identities) <= set(PLAN),
                'Queue has duplicate or nonoriginal task records')
        if board.get('status') not in ('PASS', 'FAIL') or len(listed) != 19:
            e.pending.append(f'Original19-gate queue not closed: status={board.get("status")}, records={len(listed)}/19')
        for kind, task in PLAN:
            if (kind, task) not in identities:
                e.pending.append(kind + '/' + task + ' not yet in closed queue results')
                continue
            row = screen(e, task) if kind == 'screen' else moving(e, task)
            if row:
                matching = listed[identities.index((kind, task))]
                e.check(row['quality_status'] == matching.get('status'), kind + '/' + task + ': scoreboard verdict mismatch')
                rows.append(row)
        e.integrity(board.get('integrity'), 'Queue initial')
        if board.get('status') in ('PASS', 'FAIL'):
            e.integrity(board.get('integrity_after'), 'Queue final')
            e.check(board.get('status') == ('PASS' if all(item.get('status') == 'PASS' for item in listed) else 'FAIL'),
                    'Queue aggregate verdict mismatch')
    suite = full_suite(e)
    if len(rows) != 19:
        e.pending.append(f'Corresponding finalized gate receipts incomplete: {len(rows)}/19')
    if len(rows) == 19:
        e.check(Counter(row['kind'] for row in rows) == {'portability': 13, 'moving': 3, 'native': 3},
                'Original13+3+3 scope balance mismatch')
    complete = not e.pending and len(rows) == 19 and suite is not None and board is not None
    complete = complete and board.get('status') in ('PASS', 'FAIL')
    quality = 'PENDING' if not complete else ('PASS' if all(row['quality_status'] == 'PASS' for row in rows)
              and suite and suite['status'] == 'PASS' and learned['toy_original_gate'] == 'PASS' else 'FAIL')
    validity = 'INVALID' if e.defects else ('VALID' if complete else 'PENDING')
    append = append_inventory(e, inventory, complete)
    report = dict(status='COMPLETE' if complete else 'PENDING', evidence_validity=validity,
                  quality_qualification=quality, overall_qualification='INVALID' if e.defects else quality,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  required_gate_count=19, finalized_receipt_count=len(rows),
                  gate_counts={kind:dict(total=sum(row['kind']==kind for row in rows),
                                        passed=sum(row['kind']==kind and row['quality_status']=='PASS' for row in rows),
                                        failed=sum(row['kind']==kind and row['quality_status']=='FAIL' for row in rows))
                               for kind in ['portability', 'moving', 'native']},
                  gates=rows, full_suite=suite, learned=learned,
                  source_identities=dict(run_sha256=RUN_SHA, manifest_sha256=MANIFEST_SHA,
                                         config_sha256=CONFIG_SHA, source_freeze_sha256=SOURCE_FREEZE_SHA),
                  pending=sorted(set(e.pending)), defects=e.defects,
                  validity_scope='Original collectors and execution/source receipts remain the artifact authority; this helper checks finalized JSON/log consistency and source identities, without loading tensors or rescoring.',
                  retained_failures=['RA12 static first false fires and learned regressions',
                                     'RA13 strict native/CPU-map semantic replay FAIL',
                                     'RA14 r1 zero-update bridge normalization failure'],
                  tensor_loads=0, model_calls=0, scoring_calls=0, gpu_operations=0, repository_mutations=0)
    output.mkdir()
    def write(name, value):
        with (output / name).open('x') as handle:
            handle.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')
    write('QUALIFICATION.json' if complete else 'PENDING.json', report)
    write('ARCHIVE-APPEND.json', append)
    write('INPUTS.json', dict(helper_sha256=sha(Path(__file__).read_bytes()), file_sha256=e.inputs))
    if complete:
        table = ['# Final original qualification', '',
                 f'Evidence validity: **{validity}**. Original quality qualification: **{quality}**.', '',
                 '| Scope | Task | Original quality | Evidence | Detail |', '|---|---|---|---|---|']
        for row in rows:
            detail = ('five terminal20k + independent100k' if row['kind']=='native' else
                      'both postturns /1500 updates' if row['kind']=='moving' else 'unchanged original full budget')
            table.append(f'| {row["kind"]} | {row["task"]} | {row["quality_status"]} | {row["evidence_validity"]} | {detail} |')
        table += ['', 'Fresh learned training remains labeled RA13; RA14 inherits it through the explicit helper-only restoration bridge.',
                  'Toy original gate: ' + str(learned['toy_original_gate']) + '. MNIST has no added gate; all10 metric/LR records match corrected E22.',
                  'Actual RA14 full-suite counts: ' + json.dumps(suite['counts'], sort_keys=True) + '.',
                  'Toy and MNIST native versus CPU-map original replay: PASS; only observational birth evaluation duration is excluded.',
                  'Raw checkpoints/datasets/clouds remain local. Prior failures are preserved. No numerical rerun or rescoring was performed.', '']
        write('QUALIFICATION.md', '\n'.join(table))
    hashes = {path.name:sha(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}
    write('FROZEN.json', dict(status='CLOSED_FINAL_QUALIFICATION' if complete else 'CLOSED_PENDING_SNAPSHOT',
                              file_sha256=hashes, evidence_validity=validity, quality_qualification=quality))
    print(json.dumps(dict(status=report['status'], evidence_validity=validity, quality=quality,
                          receipts=len(rows), pending=len(set(e.pending)), defects=len(e.defects), output=str(output)), sort_keys=True))
    return 2 if e.defects else 0


if __name__ == '__main__':
    raise SystemExit(main())
