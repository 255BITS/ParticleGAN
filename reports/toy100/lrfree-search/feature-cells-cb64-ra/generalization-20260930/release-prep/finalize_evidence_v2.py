"""Join the original 19 gates across sealed RA14 lanes using JSON/source/log only.

The root invokes this helper after the corrected moving/native groups and the
original-lane full suite close. Missing prerequisites produce a pending snapshot.
Only new child directories under local release-prep may receive output.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import types

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'
ORIGINAL = ROOT / 'validation-ra14'
CORRECTED = ROOT / 'validation-ra14-r2'
UNLAUNCHED = ROOT / 'validation-ra14-moving-r2'
V1_SHA = '577ea14877b80def725b61bec77cf2ff1067d80d477a0017cf7894b9eae83cd1'
FREEZES = {
    ORIGINAL: 'b998f0760458bf1b39ec8a8ed6ff00af0b3eab2c95e299c3154391949351c0de',
    CORRECTED: '871322377b4f7330087760298dac5b1869f5bec01ae01d93ee8b3c6956760a5f',
    UNLAUNCHED: '1ef7f4e91457f3bac63d6132bae6a61fb38f2f2b64e62cfc8c0eeca4712b76a9',
}
RETAINED = {
    'validation-ra14/scoreboard-all.json': '427071335fe6cd490afcc1aa3714dbdf1f7255384ebccb0f68bdf66221b8fd3f',
    'validation-ra14/scoreboard-native.json': 'd3de827b64701f3615b26c4e9cc8df95650054bc9e96d6e548355ec842d053ce',
    'validation-ra14/logs/moving-grid100.log': '235de678ee8545b8f1864fad38c7d48c5ab7c5eab4b3c342e04484b04f4ec235',
    'validation-ra14/logs/screen-grid100.log': '766f5ae73112816f77a1cc1d522d5ed72b8e8b6c42de22a786721774463cf633',
    'validation-ra14/runs/grid100/result.json': 'a2535b2031ab387ca2678a98aa612def812172c6eb36136230e8ffa2edf0f26b',
    'validation-ra14/runs/grid100/execution-receipt.json': 'a2543843bc29ff6c2e1bec7dfdce85664cb422042e5d3eb9b5748719894bbd10',
    'validation-ra14-r2/ADAPTER-CORRECTIONS.json': '7302f6b12cde2151af82c8ae9c9f26cdfe9ea5d40951ed31c220b20bd0cac545',
    'validation-ra14-moving-r2/OBSERVATION-ADAPTER-CORRECTION.json': '64712dffe02d1ee9eb97588f3a8d0068486d49da868a66ac1ccd636de7de10b6',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


# Execute only the hash-pinned, stdlib-only sealed validator module. This avoids
# importing a candidate, a numerical runner, or a module that can write bytecode.
v1_path = HERE / 'finalize_evidence.py'
v1_raw = v1_path.read_bytes()
if sha(v1_raw) != V1_SHA:
    raise RuntimeError('The sealed v1 validator source changed; retain it byte-for-byte')
v1 = types.ModuleType('sealed_evidence_finalizer_v1')
v1.__file__ = str(v1_path)
exec(compile(v1_raw, str(v1_path), 'exec'), v1.__dict__)


@contextmanager
def route(lane):
    """Select the exact receipt lane and its expected source freeze, sequentially."""
    previous = v1.LANE, v1.SOURCE_FREEZE_SHA
    v1.LANE, v1.SOURCE_FREEZE_SHA = lane, FREEZES[lane]
    try:
        yield
    finally:
        v1.LANE, v1.SOURCE_FREEZE_SHA = previous


class Evidence(v1.Evidence):
    def raw(self, path, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                self.pending.append('Missing ' + str(path))
            return None
        raw = path.read_bytes()
        key, digest = str(path), sha(raw)
        self.check(key not in self.inputs or self.inputs[key] == digest,
                   'Input changed while collecting final evidence: ' + key)
        self.inputs[key] = digest
        return raw

    def integrity(self, record, where):
        super().integrity(record, where)
        if isinstance(record, dict):
            expected_files = 103 if v1.SOURCE_FREEZE_SHA == FREEZES[ORIGINAL] else 104
            self.check(record.get('files') == expected_files, where + ': frozen file-count mismatch')


def frozen_sources(e, lane):
    path = lane / 'SOURCE-FREEZE.json'
    frozen = e.read(path)
    if frozen is None:
        return None
    e.check(e.inputs[str(path)] == FREEZES[lane], str(lane.name) + ': source freeze changed')
    e.check(frozen.get('package_sha256') == v1.RUN_SHA and
            frozen.get('config_sha256') == v1.CONFIG_SHA,
            lane.name + ': shared package/config identity mismatch')
    for flag in ('scorer_changed', 'schedule_changed', 'thresholds_changed'):
        e.check(frozen.get(flag) is False, lane.name + ': ' + flag)
    for name, digest in frozen.get('hashes', {}).items():
        # Every entry in these pinned source freezes is source or an input receipt.
        e.check(Path(name).suffix in ('.py', '.json', '.gz'),
                lane.name + ': unexpected binary source-freeze entry: ' + name)
        if Path(name).suffix not in ('.py', '.json', '.gz'):
            continue
        raw = e.raw(name)
        if raw is not None:
            e.check(sha(raw) == digest, 'Frozen source/input changed: ' + name)
    return frozen


def source_bridges(e):
    freezes = {lane: frozen_sources(e, lane) for lane in FREEZES}
    before, after = freezes[ORIGINAL], freezes[CORRECTED]
    correction = e.read(CORRECTED / 'ADAPTER-CORRECTIONS.json')
    for relative, digest in RETAINED.items():
        raw = e.raw(ROOT / relative)
        if raw is not None:
            e.check(sha(raw) == digest, 'Retained preparation/failure changed: ' + relative)
    if before is None or after is None or correction is None:
        return dict(status='PENDING', source_freezes={str(k): v for k, v in FREEZES.items()})

    def external(frozen, lane):
        return {p: h for p, h in frozen['hashes'].items() if not Path(p).is_relative_to(lane)}

    shared = external(before, ORIGINAL)
    e.check(shared == external(after, CORRECTED),
            'Corrected lane changed package/config/host/scorer/external input hashes')
    if freezes[UNLAUNCHED] is not None:
        e.check(shared == external(freezes[UNLAUNCHED], UNLAUNCHED),
                'Unlaunched preparation lane changed external frozen inputs')
    e.check({k: v for k, v in before.items() if k != 'hashes'} ==
            {k: v for k, v in after.items() if k != 'hashes'},
            'Corrected source freeze changed a declared numerical contract')
    for flag in ('configuration_changed', 'host_changed', 'model_or_optimizer_arithmetic_changed',
                 'package_changed', 'schedule_changed', 'scorer_changed', 'thresholds_changed'):
        e.check(correction.get(flag) is False, 'Adapter bridge: ' + flag)
    e.check(correction.get('original_lane') == str(ORIGINAL) and
            correction.get('corrected_lane') == str(CORRECTED), 'Adapter bridge lane identity mismatch')
    e.check(correction.get('status') == 'OBSERVATION_AND_IMPORT_PATH_CORRECTED_BEFORE_TRAINING',
            'Adapter corrections were not prepared before the corrected run')
    transforms = {
        'run_moving.py': [('surprise=None if trainer.surprise is None else trainer.surprise.diagnostics()',
                           'surprise=None if trainer.policy.surprise is None else trainer.policy.surprise.diagnostics()')],
        'run_screen.py': [('from freeze import CONFIG, PACKAGE, ROOT, verify',
                           'from freeze import CONFIG, HARNESS, PACKAGE, ROOT, verify'),
                          ("            runpy.run_path(str(ROOT / 'screen_current.py'),run_name='__main__')",
                           "            sys.path.insert(0,str(HARNESS))\n"
                           "            runpy.run_path(str(ROOT / 'screen_current.py'),run_name='__main__')")],
        'freeze.py': [("files.update([CONFIG, ROOT / 'SCREEN-ADAPTER.json', STUDY / 'ROOT-REVIEW.json',",
                       "files.update([CONFIG, ROOT / 'SCREEN-ADAPTER.json', ROOT / 'ADAPTER-CORRECTIONS.json', STUDY / 'ROOT-REVIEW.json',")],
    }
    old_local = {str(Path(p).relative_to(ORIGINAL)): h for p, h in before['hashes'].items()
                 if Path(p).is_relative_to(ORIGINAL)}
    new_local = {str(Path(p).relative_to(CORRECTED)): h for p, h in after['hashes'].items()
                 if Path(p).is_relative_to(CORRECTED)}
    e.check(set(new_local) == set(old_local) | {'ADAPTER-CORRECTIONS.json'},
            'Corrected lane introduced an undeclared local source/input')
    e.check(set(correction.get('source_changes', {})) == set(transforms),
            'Adapter bridge declares an unexpected source-change set')
    for name, expected in old_local.items():
        if name not in transforms:
            e.check(new_local.get(name) == expected, 'Unchanged lane authority differs: ' + name)
            continue
        original_raw, corrected_raw = e.raw(ORIGINAL / name), e.raw(CORRECTED / name)
        if original_raw is None or corrected_raw is None:
            continue
        transformed = original_raw.decode()
        for old, new in transforms[name]:
            e.check(transformed.count(old) == 1, 'Adapter bridge replacement is not unique: ' + name)
            transformed = transformed.replace(old, new, 1)
        e.check(transformed.encode() == corrected_raw, 'Adapter-only source equivalence failed: ' + name)
        declared = correction.get('source_changes', {}).get(name, {})
        e.check(declared == dict(before=sha(original_raw), after=sha(corrected_raw)),
                'Adapter bridge source digests mismatch: ' + name)
    return dict(status='VALID' if not e.defects else 'INVALID',
                original_lane=str(ORIGINAL), corrected_lane=str(CORRECTED),
                source_freezes={str(k): v for k, v in FREEZES.items()},
                shared_external_file_count=len(shared),
                shared_external_manifest_sha256=sha(json.dumps(shared, sort_keys=True,
                                                               separators=(',', ':')).encode()),
                corrected_arithmetic='Unchanged; diagnostic owner access and frozen harness import path only',
                changes=correction.get('source_changes'), preparation=str(CORRECTED / 'ADAPTER-CORRECTIONS.json'))


def retained_attempts(e):
    old_all = e.read(ORIGINAL / 'scoreboard-all.json')
    old_native = e.read(ORIGINAL / 'scoreboard-native.json')
    result = e.read(ORIGINAL / 'runs/grid100/result.json')
    execution = e.read(ORIGINAL / 'runs/grid100/execution-receipt.json')
    moving_raw = e.raw(ORIGINAL / 'logs/moving-grid100.log')
    unused = e.read(UNLAUNCHED / 'OBSERVATION-ADAPTER-CORRECTION.json')
    baseline = None
    if old_all:
        e.check(old_all.get('status') == 'ERROR' and old_all.get('group') == 'all' and
                old_all.get('current') == dict(kind='moving', task='grid100') and
                'completed' in old_all, 'Original interrupted ALL attempt was not retained')
    with route(ORIGINAL):
        if old_native:
            e.check(old_native.get('status') == 'ERROR' and old_native.get('results') == [] and
                    old_native.get('current') == dict(kind='screen', task='grid100'),
                    'Original zero-update native failure was not retained')
            e.integrity(old_native.get('integrity'), 'Retained native queue initial')
        if result and execution:
            e.check(result.get('status') == 'ERROR' and result.get('observations') == 0 and
                    result.get('train_seconds') == 0 and 'native100_diagnostics' in result.get('error', ''),
                    'Retained native import failure changed')
            e.check(execution.get('status') == 'ERROR' and execution.get('process_exit_code') == 1,
                    'Retained native execution failure changed')
            for key in ('source_integrity_before', 'source_integrity_after'):
                e.integrity(execution.get(key), 'Retained native ' + key)
    if moving_raw is not None:
        text = moving_raw.decode()
        gates = [json.loads(line[5:]) for line in text.splitlines() if line.startswith('GATE ')]
        e.check(len(gates) == 1 and gates[0] == dict(task='grid100', period_end=500,
                target_deg=0, modes=100, hq=0.9464), 'Retained moving500 baseline changed')
        e.check("AttributeError: 'GANTrainer' object has no attribute 'surprise'" in text,
                'Retained moving diagnostic failure changed')
        baseline = gates[0] if gates else None
    e.check(not any((UNLAUNCHED / name).exists() for name in
                   ('runs', 'moving', 'logs', 'FULL-TESTS.json', 'scoreboard-all.json',
                    'scoreboard-ports.json', 'scoreboard-moving.json', 'scoreboard-native.json')),
            'The retained separate moving preparation lane was launched')
    if unused:
        e.check(unused.get('status') == 'OBSERVATIONAL_ACCESS_CORRECTED_BEFORE_TRAINING',
                'Separate unlaunched preparation record changed')
    return dict(original_all=dict(status=None if old_all is None else old_all.get('status'),
                                  receipt=str(ORIGINAL / 'scoreboard-all.json')),
                original_moving=dict(status='ERROR', completed_updates=500, baseline=baseline,
                                     log=str(ORIGINAL / 'logs/moving-grid100.log')),
                original_native=dict(status='ERROR', completed_updates=0,
                                     receipt=str(ORIGINAL / 'scoreboard-native.json')),
                separate_moving_preparation=dict(status='FROZEN_PREPARATION_NEVER_LAUNCHED',
                                                lane=str(UNLAUNCHED), source_freeze_sha256=FREEZES[UNLAUNCHED]))


def queue_rows(e, lane, name, group, expected, allow_retained_error=False):
    path = lane / name
    board = e.read(path)
    if board is None:
        return [], dict(lane=str(lane), receipt=str(path), status='PENDING')
    listed = board.get('results', [])
    identities = [(item.get('kind'), item.get('task')) for item in listed]
    e.check(len(set(identities)) == len(identities) and set(identities) <= set(expected),
            name + ': duplicate or nonoriginal task records')
    e.check(board.get('group') == group, name + ': group identity mismatch')
    closed = (board.get('status') == 'ERROR' if allow_retained_error else
              board.get('status') in ('PASS', 'FAIL')) and 'completed' in board
    if not closed or set(identities) != set(expected):
        e.pending.append(f'{lane.name}/{name} is not closed for its required scope: '
                         f'status={board.get("status")}, records={len(listed)}/{len(expected)}')
    rows = []
    with route(lane):
        e.integrity(board.get('integrity'), lane.name + '/' + name + ' initial')
        if closed and not allow_retained_error:
            e.integrity(board.get('integrity_after'), lane.name + '/' + name + ' final')
            e.check(board.get('status') == ('PASS' if all(item.get('status') == 'PASS'
                                                         for item in listed) else 'FAIL'),
                    name + ': original group aggregate verdict mismatch')
        for kind, task in expected:
            if (kind, task) not in identities:
                e.pending.append(lane.name + '/' + kind + '/' + task + ' lacks a closed scoreboard result')
                continue
            row = v1.screen(e, task) if kind == 'screen' else v1.moving(e, task)
            if row is None:
                continue
            recorded = listed[identities.index((kind, task))]
            e.check(row['quality_status'] == recorded.get('status'),
                    kind + '/' + task + ': scoreboard/receipt verdict mismatch')
            row.update(source_lane=str(lane), source_freeze_sha256=FREEZES[lane], scoreboard=str(path))
            rows.append(row)
    return rows, dict(lane=str(lane), receipt=str(path), status=board.get('status'),
                     required_count=len(expected), recorded_count=len(listed), closed_for_scope=closed and
                     set(identities) == set(expected), retained_interrupted_all=allow_retained_error)


def historical_suite(e, inventory):
    lane = ROOT / 'validation-ra13-r2'
    receipt = e.read(lane / 'FULL-TESTS.json')
    raw = e.raw(lane / 'full-pytest.log')
    original = {item['archive_relative_path']: item['sha256']
                for item in (inventory or {}).get('artifacts', [])}
    for path in (lane / 'FULL-TESTS.json', lane / 'full-pytest.log'):
        if str(path) in e.inputs:
            e.check(e.inputs[str(path)] == original.get(str(path.relative_to(ROOT))),
                    'Closed historical RA13 suite differs from the original inventory: ' + path.name)
    if receipt is None or raw is None:
        return None
    e.check(receipt.get('status') == 'PASS' and receipt.get('returncode') == 0 and
            receipt.get('log_sha256') == sha(raw), 'Closed RA13 full-suite receipt/log mismatch')
    lines = raw.decode(errors='replace').splitlines()
    summary = [line for line in lines if re.search(r'\d+ passed\b', line) and ' in ' in line]
    return dict(candidate='RA13-settled', status=receipt.get('status'), receipt=str(lane / 'FULL-TESTS.json'),
                log_sha256=sha(raw), summary_line=summary[-1] if summary else None,
                skip_reasons=[line for line in lines if line.startswith('SKIPPED ')],
                inherited_as_ra14_test_counts=False)


def append_inventory(e, inventory, complete):
    original = {item['archive_relative_path']: item for item in (inventory or {}).get('artifacts', [])}
    candidates = set()
    for lane in (ORIGINAL, CORRECTED, UNLAUNCHED):
        if lane.exists():
            candidates.update(p for p in lane.rglob('*') if p.is_file() and
                              p.suffix in ('.json', '.jsonl', '.log', '.py', '.md'))
    for name in ('finalize_evidence_v2.py', 'FINALIZER-V2-PROTOCOL.md', 'FINALIZER-V2-PREPARATION.json'):
        path = HERE / name
        if path.is_file():
            candidates.add(path)
    files = []
    for path in sorted(candidates):
        relative = str(path.relative_to(ROOT))
        raw = e.raw(path)
        if raw is None:
            continue
        if relative in original:
            e.check(sha(raw) == original[relative]['sha256'],
                    'An original archive artifact changed instead of being retained: ' + relative)
            continue
        files.append(dict(local_path=str(path), archive_relative_path=relative,
                          bytes=len(raw), sha256=sha(raw),
                          category='ADAPTER_SOURCE' if path.suffix in ('.py', '.md') else 'SMALL_EVIDENCE',
                          scope='FINALIZED_SMALL_ARTIFACT' if complete else 'SNAPSHOT_REBIND_AFTER_COMPLETE'))
    return dict(status='READY_TO_APPEND' if complete else 'PENDING_FINAL_REBIND',
                original_inventory_sha256=v1.INVENTORY_SHA, files=files, count=len(files),
                bytes=sum(item['bytes'] for item in files),
                excluded='PT/PTH/checkpoints, datasets, NPZ/NPY clouds, images and GIFs remain local.',
                rule='Append small final JSON/JSONL/log evidence and adapter source/protocols absent by path from the sealed 510-file inventory; final output files are archived separately by the root.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == HERE.resolve() or HERE.resolve() not in output.parents:
        parser.error('--output must be a new child directory under local release-prep')
    if output.exists():
        parser.error('retain every output; --output must not already exist')
    e = Evidence()
    e.raw(v1_path)
    preparation = e.read(HERE / 'FINALIZER-V2-PREPARATION.json')
    if preparation:
        for name, expected in preparation.get('file_sha256', {}).items():
            raw = e.raw(HERE / name)
            if raw is not None:
                e.check(sha(raw) == expected, 'Sealed finalizer preparation changed: ' + name)
    with route(ORIGINAL):
        inventory, learned = v1.source_and_bridge(e)
    adapter_bridge = source_bridges(e)
    retained = retained_attempts(e)
    summary = e.read(ROOT / 'mnist/ra14-replay-r2/summary.json')
    if summary:
        e.check(summary.get('fresh_training_updates') == 0 and
                summary.get('fresh_replay_updates_total') == 40 and
                summary.get('fresh_replay_updates_per_fixture_per_branch') == 10,
                'RA14 checkpoint-only continuation scope changed')
        for key in ('inherited_mnist_all10_metric_and_lr_records_equal_corrected_E22',
                    'inherited_toy_all9_postupdate_metric_and_lr_records_equal_original_RA11'):
            e.check(summary.get(key) is True, 'Inherited exact learned comparison missing: ' + key)
        e.check(summary.get('inherited_mnist_quality_gate') is None,
                'An unrequested MNIST numerical gate was introduced')
    learned.update(toy_comparison='All 9 RA13 postupdate metric/LR records match original RA11',
                   mnist_comparison='All 10 RA13 checkpoint metric/LR records match corrected E22',
                   ra14_scope='Checkpoint-only restoration bridge and two 10-update CUDA replay branches per fixture',
                   fresh_replay_updates_total=40, original_ra13_replay_status={'toy': 'FAIL', 'mnist': 'FAIL'})
    rows, boards = [], []
    plans = [(ORIGINAL, 'scoreboard-all.json', 'all', [('screen', task) for task in v1.PORTS], True),
             (CORRECTED, 'scoreboard-moving.json', 'moving', [('moving', task) for task in v1.NATIVE], False),
             (CORRECTED, 'scoreboard-native.json', 'native', [('screen', task) for task in v1.NATIVE], False)]
    for lane, name, group, expected, retained_error in plans:
        found, board = queue_rows(e, lane, name, group, expected, retained_error)
        rows.extend(found)
        boards.append(board)
    with route(ORIGINAL):
        suite = v1.full_suite(e)
    historical = historical_suite(e, inventory)
    if len(rows) != 19:
        e.pending.append(f'Corresponding finalized original gate receipts incomplete: {len(rows)}/19')
    if len(rows) == 19:
        e.check(Counter(row['kind'] for row in rows) == {'portability': 13, 'moving': 3, 'native': 3},
                'Original 13+3+3 scope balance mismatch')
        e.check({(row['queue_kind'], row['task']) for row in rows} == set(v1.PLAN),
                'Final gate set differs from the sealed original19 task set')
    complete = not e.pending and len(rows) == 19 and suite is not None and all(
        board.get('closed_for_scope') is True for board in boards)
    quality = 'PENDING' if not complete else ('PASS' if all(row['quality_status'] == 'PASS' for row in rows)
              and suite['status'] == 'PASS' and learned['toy_original_gate'] == 'PASS' else 'FAIL')
    append = append_inventory(e, inventory, complete)
    validity = 'INVALID' if e.defects else ('VALID' if complete else 'PENDING')
    report = dict(status='COMPLETE' if complete else 'PENDING', evidence_validity=validity,
                  quality_qualification=quality, overall_qualification='INVALID' if e.defects else quality,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), finalizer_version=2,
                  required_gate_count=19, finalized_receipt_count=len(rows),
                  gate_counts={kind: dict(total=sum(row['kind'] == kind for row in rows),
                                         passed=sum(row['kind'] == kind and row['quality_status'] == 'PASS' for row in rows),
                                         failed=sum(row['kind'] == kind and row['quality_status'] == 'FAIL' for row in rows))
                               for kind in ('portability', 'moving', 'native')},
                  gates=rows, scoreboard_routes=boards, full_suite=suite, historical_ra13_full_suite=historical,
                  learned=learned, adapter_bridge=adapter_bridge, retained_attempts=retained,
                  source_identities=dict(run_sha256=v1.RUN_SHA, manifest_sha256=v1.MANIFEST_SHA,
                                         config_sha256=v1.CONFIG_SHA,
                                         source_freezes={str(lane): digest for lane, digest in FREEZES.items()},
                                         sealed_v1_validator_sha256=V1_SHA),
                  pending=sorted(set(e.pending)), defects=e.defects,
                  validity_scope='Unchanged collectors and execution receipts remain the artifact authority. Reused sealed v1 validators check original JSON/log rules and source identities under explicit lane routing; no tensor is loaded and no scorer is run.',
                  retained_failures=['RA12 static first false fires and learned regressions',
                                     'RA13 strict native/CPU-map semantic replay FAIL',
                                     'RA14 replay r1 zero-update bridge normalization failure',
                                     'RA14 original ALL ERROR after13 closed portability receipts and moving500 baseline',
                                     'RA14 original native ERROR with0 updates',
                                     'Separate moving-r2 frozen preparation never launched'],
                  tensor_loads=0, model_calls=0, scoring_calls=0, gpu_operations=0, repository_mutations=0)
    output.mkdir()
    def write(name, value):
        with (output / name).open('x') as handle:
            handle.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')
    write('QUALIFICATION.json' if complete else 'PENDING.json', report)
    write('ARCHIVE-APPEND.json', append)
    write('INPUTS.json', dict(helper_sha256=sha(Path(__file__).read_bytes()),
                            sealed_v1_validator_sha256=V1_SHA, file_sha256=e.inputs))
    if complete:
        table = ['# Final original qualification', '',
                 f'Evidence validity: **{validity}**. Original quality qualification: **{quality}**.', '',
                 '| Scope | Task | Original quality | Evidence | Source lane |', '|---|---|---|---|---|']
        table.extend(f'| {row["kind"]} | {row["task"]} | {row["quality_status"]} | {row["evidence_validity"]} | {Path(row["source_lane"]).name} |'
                     for row in rows)
        table += ['', 'The original19 task set is 13 portability, 3 moving and 3 native gates. Native records retain34 observations, five terminal20k checks and an independent100k holdout. Moving records retain the baseline and both original30-degree postturn checks over1500 updates.',
                  'The original ALL/native runtime errors and moving500 baseline remain archived. The corrected lane changes diagnostic owner access and the frozen harness import path; package, config, host, scorers, seeds, budgets and gates match the original frozen inputs.',
                  'Fresh learned training remains RA13 (2000 updates per fixture). RA14 provides a checkpoint-only restoration bridge and original two10-update CUDA continuation branches per fixture; fresh RA14 training updates:0.',
                  'Toy original gate: ' + str(learned['toy_original_gate']) + '; all9 postupdate metric/LR records match original RA11. MNIST has no added numerical gate; all10 checkpoint metric/LR records match corrected E22.',
                  'Inherited RA13 final metrics: ' + json.dumps(learned['inherited_final_metrics'], sort_keys=True) + '.',
                  'Original RA13 strict replay FAIL is retained; RA14 Toy and MNIST native versus CPU-map replay:PASS. Only observational birth evaluation duration is excluded from semantic state.',
                  'Actual RA14 full-suite counts: ' + json.dumps(suite['counts'], sort_keys=True) + '.',
                  'Actual RA14 pytest summary: ' + str(suite['summary_line']) + '.',
                  'Actual RA14 skip reasons:', '']
        table.extend('- ' + reason for reason in suite['skip_reasons'])
        table += ['', 'Historical RA13 full-suite summary: ' + str(historical['summary_line']) + '. These counts are retained under RA13.',
                  'Raw checkpoints, datasets, clouds and images remain local. This finalizer performed no numerical rerun or rescoring.', '']
        write('QUALIFICATION.md', '\n'.join(table))
    hashes = {path.name: sha(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}
    write('FROZEN.json', dict(status='CLOSED_FINAL_QUALIFICATION' if complete else 'CLOSED_PENDING_SNAPSHOT',
                              file_sha256=hashes, evidence_validity=validity, quality_qualification=quality))
    print(json.dumps(dict(status=report['status'], evidence_validity=validity, quality=quality,
                          receipts=len(rows), pending=len(set(e.pending)), defects=len(e.defects), output=str(output)),
                     sort_keys=True))
    return 2 if e.defects else 0


if __name__ == '__main__':
    raise SystemExit(main())
