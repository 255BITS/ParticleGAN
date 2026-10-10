"""Metadata-only scope and carried accounting for the three original native hosts.

This module owns no scientific loop, model, scorer, sampler, queue, or ledger
mutation. The original19 catalog is reference data; exactly three new rows are
eligible for the maintained execution helper.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


DIRECTORY = 'reports/forge/pr223-native3-continuation-20261004'
SELF = DIRECTORY + '/run_native3.py'
SCHEMA = 'pg_pr223_native3_continuation_v2'
PROTOCOL_SCHEMA = 'pg_pr223_native3_continuation_protocol_v2'
ATTESTATION_SCHEMA = 'pr223_native3_case_attestation_v2'
PREFLIGHT_SCHEMA = 'pg_pr223_native3_copied_source_preflight_v2'
ORDER = ('grid100', 'rotated100', 'staggered100')
IDS = tuple('pr223-native3-continuation-v1-native-' + task for task in ORDER)
CAPS = (1470, 1440, 1380)
PRIOR_CASE_SECONDS = 3165.841891122982
PRIOR_METADATA_SECONDS = 44.37232269323431
PRETRAINING_INVALID_SECONDS = 2.0496059330180287
PREDECESSOR_ANCHOR = DIRECTORY + '/fixtures/first-invalid-debit-anchor.json'
PREDECESSOR_LEDGER = DIRECTORY + '/fixtures/closed-metadata-22.json'
PREDECESSOR_PINS = {
    PREDECESSOR_ANCHOR:('301b8036304c14f0c1a352af280e4c0dd2c8267f675294e62fadf3259fcd5ecb',1948),
    PREDECESSOR_LEDGER:('1cb2f30350c37bedec8e012eb674bda05641093248b9f1658db0bbab8026aaf2',4644),
}
METADATA_CAP = 180
TOTAL_CAP = 10800
CANONICAL_LEDGER = '/ml2/hypergan/.pg-pr223-full-original-retest-20261004.pr223-full19-metadata-cost.json'
ANCHOR_SHA = 'ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369'
ANCHOR_BYTES = 2675
ANCHOR_RELATIVE = DIRECTORY + '/inputs/closed-parent-metadata.json'
ORIGINAL_PROTOCOL_DIGEST = '435cb98301b5759da28fc11ca5f63cd7681807895f00a6f04bd6e59a02016910'
ORIGINAL_PROTOCOL_SHA = '8a5f0f63839e613b051562f030486388e8fcf962b6b6987fe5aaa27ea1f59786'
HISTORY_DIRECTORY = 'reports/forge/pr223-original-full-retest-stopped17-20261004'
HISTORY_PINS = {
    HISTORY_DIRECTORY + '/results.json': '4cca29c719ffe295712775be27167d7465090c40865d3b096aea247e61a3d4f3',
    HISTORY_DIRECTORY + '/FINAL_COST.json': '2335c0da59f1155a329f72390078d1480f253ce1e17a5879a52138d53e3661d3',
    HISTORY_DIRECTORY + '/verification.json': 'a74262e6af196c7686f9275a0723ace989067f65552fb8b554a2c35948c26eb5',
}
PORTABLE_CONTROL_FILES = ('reports/forge/pr223-original-full-retest-20261004/fixtures/native100_score.py.txt',
                          'reports/forge/pr223-original-full-retest-20261004/fixtures/README.md',
                          DIRECTORY + '/fixtures/closed-parent-metadata.json',
                          DIRECTORY + '/fixtures/README.md',*PREDECESSOR_PINS)
PRIOR_SOURCE = {'origin_commit': '2068a661331a45e0e283b0362dd58b7d92263c6f',
                'digest': '00cadbfd06c770e16c35045b42387932b14ee3751954e5320f9e2c32799f7a2f'}
CLAIMS = dict(old_results_are_current_credit=False, qualification_input=False,
              default_adoption=False, current26_qualification=False, speed_ranking=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _equal(actual, expected, label):
    if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0., abs_tol=1e-9):
        raise ValueError(label + ' changed')


def sidecar(output):
    output = Path(output).resolve()
    return output.parent / ('.' + output.name + '.pr223-native3-prepared.json')


def copied_preflight_path(output):
    output = Path(output).resolve()
    return output.parent / ('.' + output.name + '.pr223-native3-copied-preflight.json')


def reference_summary():
    return dict(required=19, source=deepcopy(PRIOR_SOURCE), files_sha256=dict(HISTORY_PINS),
                execution_counts={'PASS': 16, 'INVALID': 1, 'NOT_RUN': 2},
                accepted_original_gate_counts={'PASS': 16, 'FAIL': 0, 'UNAVAILABLE': 3},
                case_charged_seconds=PRIOR_CASE_SECONDS, original_evidence_unchanged=True,
                old_grades_are_current_credit=False)


def predecessor_cost_summary():
    """An explicit additional debit, never an active case or historical grade."""
    return dict(schema='pg_pr223_native3_predecessor_cost_v1',
        source=dict(origin_commit='0fa92c5b599da8ab2c891e544aafc09cf802a70d',
                    digest='b2d0e98000d06a9731a4a40e52e226fedece194c097a1f5d939f501a9360aaa6'),
        files={name:dict(sha256=pin,bytes=size) for name,(pin,size) in PREDECESSOR_PINS.items()},
        execution_counts={'INVALID':1,'NOT_RUN':2},accepted_original_gate_counts={'PASS':0,'FAIL':0,'UNAVAILABLE':3},
        pretraining_invalid_case_charged_seconds=PRETRAINING_INVALID_SECONDS,
        original19_case_charged_seconds=PRIOR_CASE_SECONDS,
        combined_prior_case_charged_seconds=math.fsum((PRIOR_CASE_SECONDS,PRETRAINING_INVALID_SECONDS)),
        canonical_metadata_ledger=CANONICAL_LEDGER,required_closed_prefix_phases=22,
        numerical_credit=False,old_grades_are_current_credit=False,training_started=False)


def predecessor_metadata(root,helper):
    """Read only committed root-sealed metadata; never follow provenance paths."""
    root=Path(root)
    for name,(wanted,size) in PREDECESSOR_PINS.items():
        path=root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=size or sha(path)!=wanted:
            raise ValueError('pretraining-invalid debit metadata changed: '+name)
    anchor=helper.read_json(root/PREDECESSOR_ANCHOR)
    state=helper.read_json(root/PREDECESSOR_LEDGER)
    helper.ledger_module()._validate_snapshot(state)
    if (state['current_phase'] is not None or state['blocked'] or len(state['phases'])!=22
            or anchor['schema']!='pg_pr223_native3_first_invalid_debit_anchor_v1'
            or anchor['source']!=predecessor_cost_summary()['source']
            or anchor['training_started'] is not False or anchor['numerical_credit'] is not False
            or anchor['supervisor']['attempt_status']!='completed'
            or anchor['execution_counts']!={'INVALID':1,'NOT_RUN':2}
            or anchor['closed_metadata_before_sealing']['canonical_path']!=CANONICAL_LEDGER
            or anchor['closed_metadata_before_sealing']['sha256']!=PREDECESSOR_PINS[PREDECESSOR_LEDGER][0]):
        raise ValueError('pretraining-invalid debit source/closed-prefix ownership changed')
    _equal(anchor['case_cost']['charged_seconds'],PRETRAINING_INVALID_SECONDS,'pretraining invalid debit')
    _equal(anchor['old19_case_charged_seconds'],PRIOR_CASE_SECONDS,'original19 prior debit')
    _equal(anchor['closed_metadata_before_sealing']['charged_seconds'],state['charged_seconds'],'closed22 metadata')
    _equal(anchor['case_cost']['reserved_seconds'],0.,'completed pretraining reserve')
    return state


def history(root, helper):
    """Read only the exact committed compact parent cut, never its raw outputs."""
    root = Path(root)
    for relative, wanted in HISTORY_PINS.items():
        if sha(root / relative) != wanted:
            raise ValueError('committed parent history changed: ' + relative)
    raw = helper.read_json(root / HISTORY_DIRECTORY / 'results.json')
    cost = helper.read_json(root / HISTORY_DIRECTORY / 'FINAL_COST.json')
    rows = raw['rows']
    if (len(rows) != 19 or raw['required'] != 19 or raw['source'] != PRIOR_SOURCE
            or [(r['group'], r['task']) for r in rows] != helper.protocol.ORDER
            or [r['execution_status'] for r in rows] != ['PASS'] * 16 + ['INVALID', 'NOT_RUN', 'NOT_RUN']
            or raw['accepted_original_gate_counts'] != {'PASS': 16, 'FAIL': 0, 'UNAVAILABLE': 3}
            or cost['status'] != 'INCOMPLETE' or cost['metadata_phase_count'] != 12
            or cost['original_terminal_cut_results_sha256'] != HISTORY_PINS[HISTORY_DIRECTORY + '/results.json']
            or cost['authoritative_final_metadata_ledger'] != dict(path=CANONICAL_LEDGER, sha256=ANCHOR_SHA, bytes=ANCHOR_BYTES)
            or cost['original_case_verdicts_unchanged'] is not True):
        raise ValueError('parent stopped17 identity/status/history changed')
    _equal(cost['costs']['case_charged_seconds'], PRIOR_CASE_SECONDS, 'parent case charge')
    _equal(cost['costs']['case_paid_wall_seconds'], PRIOR_CASE_SECONDS, 'parent measured charge')
    _equal(math.fsum(r['cost']['charged_seconds'] for r in rows), PRIOR_CASE_SECONDS, 'parent row charges')
    _equal(cost['costs']['metadata']['charged_seconds'], PRIOR_METADATA_SECONDS, 'parent metadata charge')
    if cost['costs']['case_reserved_seconds'] != 0 or cost['costs']['metadata']['reserved_seconds'] != 0:
        raise ValueError('parent reservation changed')
    return raw, cost


def validate_anchor(anchor, helper, *, state=None):
    """Root supplies a SHA-bound CLOSED copy made before any charged new phase."""
    if (not isinstance(anchor, dict) or set(anchor) != {'path', 'sha256', 'bytes'}
            or anchor['sha256'] != ANCHOR_SHA or anchor['bytes'] != ANCHOR_BYTES
            or not Path(anchor['path']).is_absolute()
            or Path(anchor['path']).resolve() == Path(CANONICAL_LEDGER)):
        raise ValueError('immutable root-pinned closed parent ledger copy required')
    if state is None:
        path = Path(anchor['path'])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != ANCHOR_BYTES or sha(path) != ANCHOR_SHA:
            raise ValueError('closed parent ledger bytes changed')
        state = helper.read_json(path)
    helper.ledger_module()._validate_snapshot(state)
    if state['current_phase'] is not None or state['blocked'] or len(state['phases']) != 12:
        raise ValueError('parent metadata copy must be closed with complete phase history')
    if any(p['status'] != 'COMPLETE' for p in state['phases']):
        raise ValueError('interrupted parent phase cannot be reset')
    _equal(state['charged_seconds'], PRIOR_METADATA_SECONDS, 'closed parent metadata')
    _equal(state['reserved_seconds'], 0., 'closed parent reserve')
    return state


def validate_live_history(packet, snapshot, helper):
    """Require the exact old history prefix; never grant a new metadata clock."""
    helper.ledger_module()._validate_snapshot(snapshot)
    original = packet['metadata_history']['closed_state']
    validate_anchor(packet['metadata_history']['closed_anchor'], helper, state=original)
    old=packet['metadata_history']['predecessor_closed_state']
    expected=predecessor_metadata(helper.ROOT,helper)
    if (packet.get('predecessor_cost')!=predecessor_cost_summary() or old!=expected
            or old['phases'][:len(original['phases'])]!=original['phases']):
        raise ValueError('cost-only predecessor or same CLOSED22 ledger prefix changed')
    if (snapshot['phases'][:len(old['phases'])] != old['phases']
            or len(snapshot['phases']) < len(old['phases'])
            or snapshot['charged_seconds'] + 1e-9 < old['charged_seconds']):
        raise ValueError('canonical metadata ledger lost or reset its parent history')
    return snapshot


def _protocol(original, reference):
    rows = []
    for ordinal, (parent, identifier) in enumerate(zip(original['rows'][-3:], IDS), 1):
        row = deepcopy(parent)
        row.update(id=identifier, ordinal=ordinal, parent_retest_id=parent['id'], original19_ordinal=parent['ordinal'])
        rows.append(row)
    return dict(schema=PROTOCOL_SCHEMA, status='DECLARED_NOT_EXECUTED', family='atlas',
                original19_catalog=deepcopy(original), original19_catalog_sha256=digest(original),
                original_required=19, required=3, rows=rows, runtime=deepcopy(original['runtime']),
                parent_reference=reference, claims=deepcopy(CLAIMS),
                budget=dict(prior_case_charged_seconds=PRIOR_CASE_SECONDS,
                    pretraining_invalid_case_charged_seconds=PRETRAINING_INVALID_SECONDS,
                    combined_prior_case_charged_seconds=math.fsum((PRIOR_CASE_SECONDS,PRETRAINING_INVALID_SECONDS)),
                    case_caps_sum_seconds=4290, shared_metadata_cap_seconds=180,
                    shared_metadata_already_charged_seconds=PRIOR_METADATA_SECONDS,
                    shared_metadata_remaining_at_parent_seconds=180-PRIOR_METADATA_SECONDS,
                    aggregate_cap_seconds=10800,
                    maximum_inclusive_campaign_seconds=math.fsum((PRIOR_CASE_SECONDS,PRETRAINING_INVALID_SECONDS,4290,180)),
                    export_grace_seconds=0, retries=0))


def spec(original_protocol_path):
    return dict(id='pr223-native3-continuation-20261004-v2',
        representation_card={'path':original_protocol_path,'sha256':ORIGINAL_PROTOCOL_SHA},
        export_grace_seconds=0,retries=0,total_paid_cap_seconds=10800,case_caps_sum_seconds=4290,
        shared_metadata_and_finalization_seconds=180,prior_case_charged_seconds=PRIOR_CASE_SECONDS,
        pretraining_invalid_case_charged_seconds=PRETRAINING_INVALID_SECONDS,
        resources={'host_memory_mb':2048},diagnostic_independent_tests=True,
        media_contract='original native primary/reference;9existing clocks')


def plan(root, helper, closed_anchor):
    root = Path(root).resolve()
    raw, _ = history(root, helper)
    closed = validate_anchor(closed_anchor, helper)
    predecessor=predecessor_metadata(root,helper)
    if predecessor['phases'][:len(closed['phases'])]!=closed['phases']:
        raise ValueError('pretraining-invalid CLOSED22 prefix is not the original CLOSED12 successor')
    prefix=raw['metadata_phase_history_before_publication']
    if closed['phases'][:len(prefix)] != prefix:
        raise ValueError('closed metadata history is not the parent cut successor')
    base = helper.plan(root)  # Source declarations only; no old outcomes copied.
    if digest(base['protocol']) != ORIGINAL_PROTOCOL_DIGEST:
        raise ValueError('unchanged original19 catalog required')
    card = _protocol(base['protocol'], reference_summary())
    definitions, rows = {}, []
    for declared in card['rows']:
        d = deepcopy(base['case_definitions'][declared['parent_retest_id']])
        d['id'] = declared['id']
        d['fresh_retest'].update(scope='new native3 source; old16 grades are reference only',
                                parent_full19_retest_id=declared['parent_retest_id'])
        definitions[d['id']] = d
        rows.append(dict(id=d['id'], group='native', task=d['task'], status='NOT_RUN',
                         case_sha256=digest(d), timeout_seconds=declared['proposed_inclusive_allowance_seconds'],
                         allowance_seconds=declared['proposed_inclusive_allowance_seconds']))
    closure = helper.source_requirements(root)
    closure['files'].update(HISTORY_PINS)
    closure['files'].update({name:sha(root/name) for name in PORTABLE_CONTROL_FILES})
    if closure['files'][PORTABLE_CONTROL_FILES[0]]!='10cc14edfcd98ab34fd3768aaba2ee835dc2241dc1face2e18998c8f2b687feb':
        raise ValueError('inert portable scorer source fixture changed')
    if closure['files'][DIRECTORY+'/fixtures/closed-parent-metadata.json'] != ANCHOR_SHA:
        raise ValueError('portable parent phase-history fixture changed')
    packet = dict(schema=SCHEMA, required=3, original_required=19, family='atlas',
        spec=spec(helper.protocol.PROTOCOL),
        protocol=card, prior_reference=reference_summary(),
        source={'commit':closure['origin_commit'], 'files_sha256':closure['files']},
        preflight={**base['preflight'], 'cases':3, 'updates':21000, 'original_catalog_cases':19},
        case_definitions=definitions, external_inputs=base['external_inputs'],
        recipe_overrides=deepcopy(base['recipe_overrides']), rows=rows, status='DECLARED',
        spent_seconds=math.fsum((PRIOR_CASE_SECONDS,PRETRAINING_INVALID_SECONDS,predecessor['charged_seconds'])),new_paid_seconds=0.,
        predecessor_cost=predecessor_cost_summary(),
        metadata_history=dict(canonical_path=CANONICAL_LEDGER, closed_anchor=deepcopy(closed_anchor),
                              closed_state=closed,predecessor_closed_state=predecessor,snapshot_relative=ANCHOR_RELATIVE),
        qualification_input=False, default_adoption=False, current_forge_mog_clean_qualification=False,
        speed_ranking=False, old_results_are_current_credit=False)
    return validate_packet(packet, helper)


def validate_packet(packet, helper, *, source=False):
    original = helper.protocol.validate(packet['protocol']['original19_catalog'])
    expected = _protocol(original, reference_summary())
    if digest(original) != ORIGINAL_PROTOCOL_DIGEST or packet['protocol'] != expected:
        raise ValueError('native3 protocol/original catalog/source law changed')
    if (packet.get('schema') != SCHEMA or packet.get('required') != 3 or packet.get('original_required') != 19
            or packet.get('family') != 'atlas' or packet.get('prior_reference') != reference_summary()
            or packet['recipe_overrides'] != {'original_config_sha256':helper.legacy.CONFIG_SHA,
                                             'original_options':helper.legacy.OPTIONS}
            or any(packet.get(k) is not False for k in ('qualification_input', 'default_adoption',
                'current_forge_mog_clean_qualification', 'speed_ranking', 'old_results_are_current_credit'))):
        raise ValueError('native3 scope/old-grade transfer/config changed')
    actual_spec = packet['spec']
    if actual_spec != spec(helper.protocol.PROTOCOL):
        raise ValueError('native3 finite aggregate/carry/resource contract changed')
    history_binding = packet['metadata_history']
    if history_binding.get('canonical_path') != CANONICAL_LEDGER or history_binding.get('snapshot_relative') != ANCHOR_RELATIVE:
        raise ValueError('foreign or reset metadata ledger identity')
    validate_anchor(history_binding['closed_anchor'], helper, state=history_binding['closed_state'])
    if (packet.get('predecessor_cost')!=predecessor_cost_summary()
            or history_binding.get('predecessor_closed_state')!=predecessor_metadata(helper.ROOT,helper)
            or history_binding['predecessor_closed_state']['phases'][:12]!=history_binding['closed_state']['phases']):
        raise ValueError('required explicit pretraining-invalid debit/CLOSED22 prefix omitted or changed')
    if [r.get('id') for r in packet['rows']] != list(IDS) or set(packet['case_definitions']) != set(IDS):
        raise ValueError('only exact ordered native3 execution rows may be present')
    if packet.get('status')=='DECLARED' and any(r.get('status')!='NOT_RUN' or
            any(k in r for k in ('original_gate','full_protocol_complete','media','attempt_token','paid_wall_seconds'))
            for r in packet['rows']):
        raise ValueError('declared native3 rows cannot carry historical grade or attempt credit')
    external_digest = digest(packet['external_inputs']['files'])
    for row, declared, cap in zip(packet['rows'], expected['rows'], CAPS):
        if (row.get('group') != 'native' or row.get('task') != declared['task']
                or row.get('timeout_seconds') != cap or row.get('allowance_seconds') != cap):
            raise ValueError('native host or complete original allowance changed')
        definition = packet['case_definitions'][row['id']]
        if digest(definition) != row.get('case_sha256'):
            raise ValueError('native case fingerprint changed')
        parent = deepcopy(definition)
        retest = parent.pop('fresh_retest'); parent['id'] = retest['parent_original_id']
        old = deepcopy(declared['original_definition']); old['external_inputs_sha256'] = external_digest
        expected_retest = dict(parent_original_id=old['id'], historical_case_sha256=declared['historical_case_sha256'],
            source_derived_parent_sha256=digest(old), goal_media_steps=declared['media_steps'],
            goal_observer_sha256=retest.get('goal_observer_sha256'),
            scope='new native3 source; old16 grades are reference only',
            parent_full19_retest_id=declared['parent_retest_id'])
        if parent != old or retest != expected_retest:
            raise ValueError('original native science/seed/cadence/gate/parent identity changed')
    if source:
        execution = packet['execution_source']; snapshot = Path(execution['snapshot_path']).resolve()
        helper.verify_snapshot(snapshot, execution)
        if (helper.read_json(snapshot/'forge-source.json') != {k:v for k,v in execution.items() if k!='snapshot_path'}
                or packet['source']['execution_digest'] != execution['digest']
                or packet['source']['commit'] != execution['origin_commit']
                or execution['origin_commit'] in {PRIOR_SOURCE['origin_commit'],predecessor_cost_summary()['source']['origin_commit']}
                or execution['digest'] in {PRIOR_SOURCE['digest'],predecessor_cost_summary()['source']['digest']}
                or any(execution['files'].get(p) != h for p,h in packet['source']['files_sha256'].items())):
            raise ValueError('native3 new source manifest/origin changed')
        if any(name not in execution['files'] for name in PORTABLE_CONTROL_FILES):
            raise ValueError('portable copied-source test fixture closure missing')
        for name,(wanted,_) in PREDECESSOR_PINS.items():
            if execution['files'].get(name)!=wanted:
                raise ValueError('copied pretraining-invalid debit source is unbound')
        if predecessor_metadata(snapshot,helper)!=history_binding['predecessor_closed_state']:
            raise ValueError('copied CLOSED22 predecessor prefix changed')
        history(snapshot, helper)
        if helper.protocol.load(snapshot) != original or helper.sha(snapshot/helper.protocol.PROTOCOL) != actual_spec['representation_card']['sha256']:
            raise ValueError('copied original protocol/config changed')
        anchor_path = snapshot/ANCHOR_RELATIVE
        if (helper.sha(anchor_path) != ANCHOR_SHA or anchor_path.stat().st_size != ANCHOR_BYTES
                or helper.read_json(anchor_path) != history_binding['closed_state']
                or execution['files'].get(ANCHOR_RELATIVE) != ANCHOR_SHA):
            raise ValueError('copied closed parent history changed')
        if (packet['metadata_ledger_path'] != CANONICAL_LEDGER
                or packet['snapshot_locations']['package'] != str(snapshot)
                or packet['copied_preflight_receipt_path'] != str(copied_preflight_path(packet['prepared_output']))):
            raise ValueError('prepared source/ledger/output identity changed')
        for definition in packet['case_definitions'].values():
            if definition['fresh_retest']['goal_observer_sha256'] != execution['files'].get(helper.DIRECTORY+'/goal_observer.py'):
                raise ValueError('native3 observer is not bound to actual source')
    return packet


def validate_current_request(request, helper):
    """Current admitted transport is distinct from a pristine DECLARED plan.

    This proves metadata ownership only. The unchanged child lease check still
    requires both real inherited descriptors and their durable source/deadline.
    """
    packet=request['packet'];row=request['row'];worker=request['worker']
    index=next((i for i,r in enumerate(packet['rows']) if r['id']==row.get('id')),None)
    if index is None or packet['rows'][index]!=row or row.get('status')!='RUNNING':
        raise ValueError('native3 request must own exactly its current RUNNING row; no retry')
    keys={'id','group','task','status','case_sha256','timeout_seconds','allowance_seconds','attempt_key','attempt_token'}
    if set(row)!=keys:
        raise ValueError('current native3 row cannot carry prior grade, cost, media or attempt credit')
    for old in packet['rows'][:index]:
        if old.get('status') not in {'PASS','FAIL'} or old.get('full_protocol_complete') is not True or not old.get('media'):
            raise ValueError('current native3 request requires a completed same-study prefix')
    pristine={'id','group','task','status','case_sha256','timeout_seconds','allowance_seconds'}
    for later in packet['rows'][index+1:]:
        if later.get('status')!='NOT_RUN' or set(later)!=pristine:
            raise ValueError('native3 request tail must remain unexecuted and uncredited')
    expected_status='READY' if index<2 else 'INCOMPLETE'
    if (packet.get('status')!=expected_status or packet.get('scientific_status')!=expected_status
            or packet.get('budget_status')!='WITHIN_DECLARED_CAPS' or packet.get('budget_overruns')!=[]
            or packet.get('budget_accounting',{}).get('halt_required') is not False):
        raise ValueError('native3 admission must use saved current state, never DECLARED or halted metadata')
    token=row.get('attempt_token')
    if type(token) is not str or len(token)!=32 or any(c not in '0123456789abcdef' for c in token) or worker.get('token')!=token:
        raise ValueError('native3 current row and worker token differ')
    expected_key=helper.PolicyCoordinator.attempt_key(None,packet,
        {'family':'atlas','recipe_overrides':packet['recipe_overrides']},row)
    if row.get('attempt_key')!=expected_key:
        raise ValueError('native3 current attempt key differs from exact case/source/runtime')
    started=worker.get('started_monotonic');deadline=worker.get('deadline_monotonic')
    if (type(started) not in (int,float) or type(deadline) not in (int,float)
            or not math.isfinite(started) or not math.isfinite(deadline) or started<0
            or deadline-started!=row['allowance_seconds']):
        raise ValueError('native3 current deadline must cover the exact whole allowance')
    fds=worker.get('lease_fds');fd=worker.get('lease_fd')
    if (type(fds) is not list or len(fds)!=2 or any(type(v) is not int or v<0 for v in fds)
            or len(set(fds))!=2 or type(fd) is not int or fd not in fds
            or type(worker.get('physical_gpu')) is not int or worker.get('physical_gpu')!=1 or worker.get('device')!='cuda:0'):
        raise ValueError('native3 current request requires its two descriptor identities and GPU1')
    coordinator=packet['coordinator']
    if (coordinator.get('canonical_output')!=packet['prepared_output']
            or coordinator.get('queue_root')!=packet['queue_root']
            or Path(request['target']).resolve()!=Path(packet['prepared_output'])/'native'/row['task']):
        raise ValueError('native3 current request changed canonical study ownership')
    return request


def require_existing_ledger():
    path = Path(CANONICAL_LEDGER)
    if path.is_symlink() or not path.is_file():
        raise ValueError('same canonical parent metadata ledger must exist; no reset')
    return path


def require_next_reservation(packet, snapshot, next_allowance, helper):
    """Three active rows plus prior case cost and inclusive metadata, each once."""
    validate_live_history(packet, snapshot, helper)
    if [r.get('id') for r in packet['rows']] != list(IDS):
        raise ValueError('native3 accounting requires exact3 rows')
    budget = helper.ledger_module()
    next_allowance = budget._number(next_allowance, 'next full native allowance')
    if next_allowance not in (0, *CAPS):
        raise ValueError('only a full declared native allowance may be reserved')
    paid, reserved, overruns = [], [], []
    for row, cap in zip(packet['rows'], CAPS):
        present = budget._COST_KEYS.intersection(row)
        if not present:
            if row.get('status') not in {'NOT_RUN', 'RUNNING'}:
                raise ValueError('retained native execution lost cost')
            continue
        for key in ('allowance_seconds', 'terminal_status', 'completed_terminal', 'certified',
                    'paid_wall_seconds', 'reserved_seconds', 'unmeasured_interrupt_reserved_seconds',
                    'charged_seconds', 'overrun_seconds'):
            if key not in row:
                raise ValueError('partial native cost mapping')
        if row['allowance_seconds'] != cap or type(row['completed_terminal']) is not bool or type(row['certified']) is not bool:
            raise ValueError('native cap/terminal ownership changed')
        terminal = None if row['terminal_status'] == 'missing' else {
            'attempt_status':row['terminal_status'], 'paid_wall_seconds':row['paid_wall_seconds']}
        if terminal is None and row['paid_wall_seconds']!=0:
            raise ValueError('missing terminal cannot supply measured paid cost')
        expected = budget.case_cost(cap, terminal, certified=row['certified'])
        if any(row[k] != v for k,v in expected.items()):
            raise ValueError('native cost differs from conservative durable terminal accounting')
        paid.append(expected['paid_wall_seconds']);reserved.append(expected['reserved_seconds']);overruns.append(expected['overrun_seconds'])
    current_paid, current_reserved = math.fsum(paid), math.fsum(reserved)
    total = math.fsum((PRIOR_CASE_SECONDS,PRETRAINING_INVALID_SECONDS,current_paid,current_reserved,snapshot['charged_seconds']))
    overrun = math.fsum(overruns) + snapshot['overrun_seconds']
    halt = snapshot['blocked'] or overrun > 0 or total > TOTAL_CAP
    if next_allowance > 0 and (halt or total + next_allowance > TOTAL_CAP):
        raise budget.BudgetExceeded('whole native allowance does not fit the SAME original10800/180 campaign')
    return dict(prior_case_charged_seconds=PRIOR_CASE_SECONDS,
                pretraining_invalid_case_charged_seconds=PRETRAINING_INVALID_SECONDS,
                combined_prior_case_charged_seconds=math.fsum((PRIOR_CASE_SECONDS,PRETRAINING_INVALID_SECONDS)),
                current_case_paid_wall_seconds=current_paid, current_case_reserved_seconds=current_reserved,
                current_case_charged_seconds=current_paid+current_reserved,
                metadata_charged_seconds=snapshot['charged_seconds'], charged_seconds=total,
                cap_seconds=TOTAL_CAP, remaining_seconds=TOTAL_CAP-total, next_allowance_seconds=next_allowance,
                overrun_seconds=overrun, within_cap=total<=TOTAL_CAP, halt_required=halt)


def execution_scope():
    return dict(schema='pg_pr223_native3_execution_scope_v1', active_case_ids=list(IDS),
                original_catalog_required=19, executed_required=3, old_grades_are_current_credit=False)


def validate_grade_metadata(grade):
    """Join recorded original gate labels only; never score retained arrays."""
    gates=grade.get('native_gates')
    if not isinstance(gates,dict) or set(gates)!={'noisy','clean'}:
        raise ValueError('both original native law gate receipts required')
    for values in gates.values():
        if not isinstance(values,dict) or set(values)!={'coverage','accuracy'} or any(v not in {'PASS','FAIL'} for v in values.values()):
            raise ValueError('original native coverage AND accuracy receipts required')
    status='PASS' if all(v=='PASS' for v in gates['noisy'].values()) else 'FAIL'
    if (any(grade.get(k)!=status for k in ('status','original_gate','original_protocol_gate'))
            or grade.get('reported_original_status')!=gates['noisy']['accuracy']
            or grade.get('qualification_input') is not False):
        raise ValueError('native primary gate/diagnostic/original label join changed')
    return grade
