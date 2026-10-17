"""Close current PR155 source through original receipts, finite-horizon proof and actual current gates."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import types
import xml.etree.ElementTree as ET

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'
V3_PATH = HERE / 'finalize_evidence_v3_r3.py'
V3_SHA = '798c0574dadc2075c9c1866b9ff52bcfc364a84ab20343a026893f5a2eccc74a'
if hashlib.sha256(V3_PATH.read_bytes()).hexdigest() != V3_SHA:
    raise RuntimeError('Sealed historical source validator changed')
v3 = types.ModuleType('sealed_historical_source_finalizer')
v3.__file__ = str(V3_PATH)
exec(compile(V3_PATH.read_bytes(), str(V3_PATH), 'exec'), v3.__dict__)
v1, v2 = v3.v1, v3.v2
Evidence, sha, digest = v3.Evidence, v3.sha, v3.digest


def pinned_JSON(e, specification):
    value = e.read(specification['path'])
    if value is not None:
        e.check(e.inputs[specification['path']] == specification['sha256'], 'Pinned JSON changed: ' + specification['path'])
    return value


def history(e, prepared):
    lane = Path(prepared['historical_qualification']['lane'])
    frozen = pinned_JSON(e, prepared['historical_qualification']['closed'])
    if frozen is None:
        return None
    for name, value in frozen['file_sha256'].items():
        e.bind(lane / name, value)
    qualified = e.read(lane / 'QUALIFICATION.json')
    if qualified is None:
        return None
    e.check(qualified.get('evidence_validity') == 'VALID' and qualified.get('quality_qualification') == 'PASS' and
            qualified.get('finalized_receipt_count') == 19, 'Historical RA16 original scope not closed/valid')
    return qualified


def current_source(e, prepared):
    lane = Path(prepared['latest_suite_lane'])
    v3.frozen(e, prepared, lane)
    bridge = pinned_JSON(e, prepared['current_source_bridge'])
    execution = pinned_JSON(e, prepared['execution_bridge'])
    proof = pinned_JSON(e, prepared['noise_floor_proof'])
    if bridge is None or execution is None or proof is None:
        return None
    e.check(bridge.get('status') == 'CPU_PASS_LATEST_BASE_GPU_GATES_PENDING' and
            bridge.get('package_sha256') == prepared['latest_package']['package_sha256'], 'Current source identity/CPU closure mismatch')
    e.check(bridge.get('current_PR155_base') == prepared['qualification_base']['current_PR155_commit'], 'Current PR155 source base mismatch')
    for field in ('original_19_quality_and_2_learned_noise_math_preserved',
                  'lazy_diagnostics_training_outputs_gradients_RNG_math_preserved',
                  'valid_many_site_routing_codes_usage_gradient_math_preserved',
                  'public_DV12_diagnostics_and_checkpoint_key_preserved', 'original_40_replay_ranges_covered'):
        e.check(bridge.get(field) is True, 'Current source bridge lacks: ' + field)
    e.check(bridge.get('fresh_training_required_for_original_noise_floor_fixtures') == [] and
            bridge.get('universal_noise_floor_training_parity_claimed') is False, 'Finite-horizon source scope changed')
    e.check(execution.get('status') == 'PASS_ORIGINAL_FINITE_HORIZON_CURRENT_PR155_SOURCE_EQUIVALENCE_ONLY' and
            execution.get('bridged_quality_task_count') == 19 and execution.get('all_actual_execution_labels_preserved') is True,
            'Current original-execution bridge scope mismatch')
    for path, value in execution.get('read_only_file_sha256', {}).items():
        e.bind(path, value)
    e.check(proof.get('status') == 'PROVEN_UNAFFECTED_FOR_ORIGINAL_FINITE_HORIZONS' and
            proof.get('original_quality_tasks_proven_unaffected') == 19 and
            proof.get('learned_fixture_prefixes_proven_unaffected') == 2 and
            proof.get('minimum_all_step_table_s_bound') == .25 and
            proof.get('tasks_requiring_fresh_training_for_noise_floor_change') == [], 'Original21 finite-horizon proof mismatch')
    records = proof.get('records', [])
    wanted = {'ported/' + task for task in v1.PORTS} | {'moving/' + task for task in v1.NATIVE} | {
        'native/' + task for task in v1.NATIVE} | {'learned/toy', 'learned/mnist'}
    e.check(len(records) == 21 and {row.get('task') for row in records} == wanted, 'Original21 proof record identities mismatch')
    for row in records:
        e.check(row.get('unaffected_by_noise_floor_change') is True and
                row.get('fresh_training_required_for_noise_floor_change') is False and
                row.get('all_step_table_s_lower_bound', 0) >= .25 and
                row.get('witness_role') == 'table', 'Original all-step proof record fails: ' + str(row.get('task')))
        e.bind(row['path'], row['checkpoint_sha256'])
    e.check(proof.get('original_learned_replay', {}).get('within_proved_training_prefix') is True and
            proof.get('original_learned_replay', {}).get('total_updates') == 40, 'Original40 replay range not covered')
    return dict(source_bridge=prepared['current_source_bridge'], execution_bridge=prepared['execution_bridge'],
                noise_floor_proof=prepared['noise_floor_proof'], current_PR155_commit=bridge['current_PR155_base'],
                CPU_contracts_passed=bridge['CPU_contracts_passed'], original_quality_tasks=19,
                original_learned_prefixes=2, minimum_all_step_table_s_bound=.25,
                finite_original_horizons_only=True, universal_noise_floor_training_parity_claimed=False)


def original_gates(e, prepared, historical):
    rows = []
    if historical is None:
        return rows
    for saved in historical.get('gates', []):
        lane = Path(saved['source_lane'])
        with v3.route(e, prepared, lane):
            row = v1.moving(e, saved['task']) if saved['kind'] == 'moving' else v1.screen(e, saved['task'])
        if row is None:
            continue
        for field, value in row.items():
            e.check(saved.get(field) == value, 'Original closed receipt changed: ' + saved['kind'] + '/' + saved['task'] + '/' + field)
        kept = dict(saved)
        kept.update(qualified_current_source_candidate=prepared['latest_candidate'],
                    current_source_math_bridge=prepared['current_source_bridge']['sha256'],
                    fresh_RA17_execution=False)
        rows.append(kept)
    e.check(len(rows) == 19 and {(row['queue_kind'], row['task']) for row in rows} == set(v1.PLAN), 'Original19 task set changed')
    e.check(Counter(row['kind'] for row in rows) == {'portability': 13, 'moving': 3, 'native': 3}, 'Original13+3+3 balance changed')
    return rows


def actual_suite(e, prepared):
    package = prepared['latest_package']
    path = Path(package['path']) / 'particlegan'
    mapping, h = {}, hashlib.sha256()
    for source in sorted(path.rglob('*.py')):
        raw, relative = e.raw(source), str(source.relative_to(path))
        mapping[relative] = sha(raw)
        h.update(relative.encode() + b'\0' + raw + b'\0')
    e.check(mapping == package['source_sha256'] and h.hexdigest() == package['package_sha256'], 'Current package source identity changed')
    e.package_map = mapping
    lane = Path(prepared['latest_suite_lane'])
    with v3.route(e, prepared, lane):
        suite = v1.full_suite(e)
    if suite is None:
        return None
    receipt = e.read(lane / 'FULL-TESTS.json')
    junit = e.raw(lane / 'full-pytest-junit.xml')
    if junit is None:
        return None
    e.check(receipt.get('junit_sha256') == sha(junit), 'Actual latest JUnit hash mismatch')
    root = ET.fromstring(junit)
    cases = list(root.iter('testcase'))
    actual = []
    for classname, name in prepared['required_CUDA_nodes']:
        matched = [node for node in cases if node.get('classname') == classname and node.get('name') == name]
        passed = len(matched) == 1 and not any(child.tag in ('skipped', 'failure', 'error') for child in matched[0])
        actual.append(dict(classname=classname, name=name, records=len(matched), status='PASS' if passed else 'FAIL'))
    e.check(receipt.get('required_actual_CUDA_test_coverage') == actual, 'Actual CUDA node capture/receipt mismatch')
    e.check(all(row['status'] == 'PASS' for row in actual), 'Required current accelerator/portability CUDA nodes did not execute and pass')
    for source, value in {**receipt.get('sources', {}), **receipt.get('tests', {})}.items():
        e.bind(source, value)
    suite.update(actual_required_CUDA_nodes=actual, junit_sha256=sha(junit), current_source=True)
    return suite


def actual_CI_smoke(e, prepared):
    receipt = pinned_JSON(e, prepared['CI_smoke']['receipt'])
    report = pinned_JSON(e, prepared['CI_smoke']['report'])
    if receipt is None or report is None:
        return None
    e.check(receipt.get('status') == 'PASS' and receipt.get('returncode') == 0 and
            receipt.get('package_sha256') == prepared['latest_package']['package_sha256'] and
            receipt.get('frozen_package_on_pythonpath') is True, 'Exact CI CPU smoke source/process mismatch')
    e.check(receipt.get('report_sha256') == e.inputs[prepared['CI_smoke']['report']['path']], 'CI CPU smoke report hash mismatch')
    for key, wanted in dict(device='cpu', sites=3, tokens=2, particles=16, steps=2,
                            warmup_steps=1, probe_interval=1000, structural_evaluations=0).items():
        e.check(report.get(key) == wanted, 'Exact upstream CI CPU smoke settings mismatch: ' + key)
    dictionaries = report.get('last_dv12_applications', [])
    e.check(len(dictionaries) == 2 and all(isinstance(row, dict) and row and
            all(type(value) is float and math.isfinite(value) for value in row.values()) for row in dictionaries),
            'CI CPU smoke ordinary finite DV12 dictionaries missing')
    timing = report.get('milliseconds_per_update')
    e.check(isinstance(timing, (int, float)) and math.isfinite(timing) and timing > 0, 'CI CPU smoke timing invalid')
    e.check(set(report.get('host_scalar_extractions', {})) == {'two_generator_forwards', 'complete_update'},
            'CI CPU smoke scalar-read diagnostics missing')
    for relative, value in receipt.get('upstream_source_sha256', {}).items():
        e.bind(Path(prepared['repository']) / relative, value)
    return dict(status='PASS', device='cpu', exact_upstream_CI_arguments=True, receipt=prepared['CI_smoke']['receipt'],
                report=prepared['CI_smoke']['report'], structural_evaluations=0, probe_interval=1000,
                finite_DV12_dictionaries=2, quality_gate_or_timing_threshold_added=False)


def append_inventory(e, inventory, complete):
    base = v3.append_inventory(e, inventory, complete)
    original = {row['archive_relative_path']: row for row in inventory['artifacts']}
    known = {row['archive_relative_path'] for row in base['files']}
    paths = set()
    for relative in ('validation-ra17', 'mnist/ra17-replay', 'portability/ra17-current-pr155',
                     'pkg-RA17-current-pr155', 'diagnostics/upstream-noise-floor-applicability'):
        paths.update(path for path in (ROOT / relative).rglob('*') if path.is_file() and
                     path.suffix in ('.py', '.md', '.json', '.jsonl', '.log', '.xml', '.patch', '.toml', '.cfg', '.ini'))
    paths.update(path for path in HERE.rglob('*') if path.is_file() and path.suffix in ('.toml', '.cfg', '.ini', '.xml'))
    paths.add(ROOT / 'configs/RA17-current-pr155.json')
    for path in sorted(paths):
        relative = str(path.relative_to(ROOT))
        if relative in original:
            e.bind(path, original[relative]['sha256'])
            continue
        if relative in known:
            continue
        raw = e.raw(path)
        base['files'].append(dict(local_path=str(path), archive_relative_path=relative, bytes=len(raw), sha256=sha(raw),
            category='SOURCE_OR_PROTOCOL' if path.suffix in ('.py', '.md', '.patch', '.toml', '.cfg', '.ini') else 'SMALL_EVIDENCE',
            scope='FINALIZED_SMALL_ARTIFACT' if complete else 'SNAPSHOT_REBIND_AFTER_COMPLETE'))
    base.update(count=len(base['files']), bytes=sum(row['bytes'] for row in base['files']),
        excluded='Weights/checkpoints, datasets, clouds and unrelated images are excluded. The user-authorized shift visualization is a separate pinned artifact.')
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if HERE.resolve() not in output.parents or output.exists():
        parser.error('--output must be a new child directory under release-prep')
    e = Evidence()
    prepared = e.read(HERE / 'FINALIZER-V4-PREPARATION.json')
    if prepared is None:
        parser.error('Seal current-source preparation before invoking')
    for name, value in prepared['helper_file_sha256'].items():
        e.bind(HERE / name, value)
    e.bind(V3_PATH, V3_SHA)
    historical = history(e, prepared)
    bridge = current_source(e, prepared)
    gates = original_gates(e, prepared, historical)
    with v3.route(e, prepared, v2.ORIGINAL):
        inventory, original_learned = v1.source_and_bridge(e)
    suite = actual_suite(e, prepared)
    replay = v3.latest_replay(e, prepared)
    smoke = actual_CI_smoke(e, prepared)
    complete = not e.pending and len(gates) == 19 and all(value is not None for value in (bridge, suite, replay, smoke))
    quality = 'PENDING' if not complete else ('PASS' if all(row['quality_status'] == 'PASS' for row in gates) and
              suite['status'] == replay['status'] == smoke['status'] == 'PASS' and original_learned['toy_original_gate'] == 'PASS' else 'FAIL')
    append = append_inventory(e, inventory, complete)
    if complete and e.pending:
        complete, quality = False, 'PENDING'
        append['status'] = 'PENDING_FINAL_REBIND'
        for row in append['files']:
            row['scope'] = 'SNAPSHOT_REBIND_AFTER_COMPLETE'
    validity = 'INVALID' if e.defects else ('VALID' if complete else 'PENDING')
    learned = dict(original_learned, current_candidate=prepared['latest_candidate'], latest_replay=replay,
        original_fresh_training_source='RA13-settled', fresh_current_learned_training_updates=0,
        current_source_scope='Original finite-prefix source equivalence and actual current native/CPU-map CUDA40 continuation')
    report = dict(status='COMPLETE' if complete else 'PENDING', finalizer_version=4,
        candidate=prepared['latest_candidate'], qualification_base=prepared['qualification_base'],
        evidence_validity=validity, quality_qualification=quality, overall_qualification='INVALID' if e.defects else quality,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), required_gate_count=19, finalized_receipt_count=len(gates),
        gates=gates, gate_counts={kind: dict(total=sum(row['kind'] == kind for row in gates),
            passed=sum(row['kind'] == kind and row['quality_status'] == 'PASS' for row in gates),
            failed=sum(row['kind'] == kind and row['quality_status'] == 'FAIL' for row in gates))
            for kind in ('portability', 'moving', 'native')},
        current_source_bridge=bridge, full_suite=suite, learned=learned, exact_upstream_CI_CPU_smoke=smoke,
        historical_RA16_qualification=prepared['historical_qualification'],
        actual_quality_execution_labels='15 RA14 executions plus4 fresh RA15 affected gates; none relabelled freshRA17',
        current_source_training_equivalence_scope='Original19 gates and2 learned prefixes only; not a universal noise-floor parity claim',
        retained_failures=None if historical is None else historical.get('retained_failures'),
        pending=sorted(set(e.pending)), defects=e.defects,
        tensor_loads=0, model_calls=0, scorer_calls=0, GPU_operations=0, repository_mutations=0)
    output.mkdir()
    def write(name, value):
        with (output / name).open('x') as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')
    write('QUALIFICATION.json' if complete else 'PENDING.json', report)
    write('ARCHIVE-APPEND.json', append)
    write('INPUTS.json', dict(helper_sha256=digest(Path(__file__)), file_sha256=e.inputs))
    if complete:
        lines = ['# Current PR155 original qualification', '',
            f'Evidence: **{validity}**. Original quality qualification: **{quality}**.', '',
            'Current PR155 base: ' + prepared['qualification_base']['current_PR155_commit'] + '.',
            'Frozen current package: ' + prepared['latest_package']['package_sha256'] + '.', '',
            '| Scope | Task | Quality | Actual executed source |', '|---|---|---|---|']
        lines.extend(f'| {row["kind"]} | {row["task"]} | {row["quality_status"]} | {row["execution_candidate"]} |' for row in gates)
        lines += ['', report['actual_quality_execution_labels'] + '.',
            'The current source bridge preserves original finite-horizon math through all21 prefixes: lifetime stationary decisions imply table s≥.25 at every step. Lazy diagnostics and valid routing preserve forward/gradient/RNG/schema law. No universal noise-floor training parity is claimed.',
            'Moving quality remains actual scorer periods, both original turns over1500 updates. Native coverage retains34 observations, five20k terminal checks and100k independent holdouts.',
            'Fresh learned training remains RA13,2000 updates per fixture. Toy original gate: ' + str(learned['toy_original_gate']) + '; all9 metric/LR records match RA11. MNIST all10 records match corrected E22; no numerical MNIST gate was added.',
            'Inherited final learned metrics: ' + json.dumps(learned['inherited_final_metrics'], sort_keys=True) + '.',
            'Actual current learned replay:40 CUDA updates, strict native/CPU-map loss/state/sample parity and original native control agreement.',
            'Actual current full suite: ' + str(suite['summary_line']) + '.',
            'Required current CUDA nodes: ' + json.dumps(suite['actual_required_CUDA_nodes'], sort_keys=True) + '.',
            'Exact upstream CI CPU CLI:PASS, sites3/tokens2/particles16/steps2/warmup1. No timing quality threshold was added.', '',
            'Actual skip reasons:', '']
        lines.extend('- ' + reason for reason in suite['skip_reasons'])
        lines += ['', 'Historical failures, old-base closure, unlaunched lanes and earlier metadata failures remain retained. Weights, datasets and clouds stay local. The user-authorized shift visualization is tracked separately. This finalizer performed no numerical work.', '']
        write('QUALIFICATION.md', '\n'.join(lines))
    files = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    write('FROZEN.json', dict(status='CLOSED_FINAL_QUALIFICATION' if complete else 'CLOSED_PENDING_SNAPSHOT',
        file_sha256=files, evidence_validity=validity, quality_qualification=quality))
    print(json.dumps(dict(status=report['status'], evidence_validity=validity, quality=quality,
        receipts=len(gates), defects=len(e.defects), pending=len(set(e.pending)), output=str(output)), sort_keys=True))
    return 2 if e.defects else 0


if __name__ == '__main__':
    raise SystemExit(main())
