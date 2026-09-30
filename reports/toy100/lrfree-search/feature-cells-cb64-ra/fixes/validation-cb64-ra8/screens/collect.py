"""Collect exact original verdicts, independent validity, mechanisms, and GPU costs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from lane import CONFIG_SHA, ENV, GPU0_UUID, HARNESS, NATIVE, PACKAGE_SHA, PORTABILITY, ROOT, SCREEN_SHA, TASKS, now, read, sha, task_plan, verify_frozen, write

def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def validate_native(out, task, result, reasons):
    import numpy as np
    fixture = read(HARNESS / 'tasks/native100_fixture.json')
    actual = read(out / 'native-fixture.json') if (out / 'native-fixture.json').exists() else {}
    matches = {role: {name: actual.get('initial', {}).get(role, {}).get(name) == expected for name, expected in params.items()} for role, params in fixture['expected_parameters'].items()}
    if not all((value for params in matches.values() for value in params.values())):
        reasons.append('canonical native initial G/prior parameter mismatch or receipt absent')
    if actual.get('prior_range') != fixture['prior_range']:
        reasons.append('canonical native prior range mismatch or receipt absent')
    if result.get('native_fixture') != actual:
        reasons.append('native fixture result and sidecar disagree')
    plan = task_plan(task)
    d = out / 'native-noisy'
    summary = read(d / 'summary.json') if (d / 'summary.json').exists() else {}
    config = summary.get('config', {})
    for key in ('steps', 'seed', 'num_particles', 'z_dim', 'batch_size'):
        if config.get(key) != plan[key]:
            reasons.append(f'native config {key}: {config.get(key)} != {plan[key]}')
    for key, expected in dict(device='cuda:0', eval_samples=20000, threads=1).items():
        if config.get(key) != expected:
            reasons.append(f'native config {key}: {config.get(key)} != {expected}')
    if summary.get('eval_steps') != plan['observation_steps']:
        reasons.append('native official summary does not contain the original 34-observation schedule')
    if summary.get('accuracy_check_steps') != plan['terminal_steps']:
        reasons.append('native terminal schedule differs from 6000,6250,6500,6750,7000')
    if summary.get('accuracy', {}).get('holdout_samples') != 100000:
        reasons.append('native official summary does not declare the independent 100k holdout')
    events = jsonl(d / 'events.jsonl')
    event_steps = {model: [row['step'] for row in events if row.get('event') == 'eval' and row.get('model') == model] for model in ('live', 'ema')}
    for model, steps in event_steps.items():
        if steps != plan['observation_steps']:
            reasons.append(f'native {model} event schedule differs from original')
    shapes = {}
    for rel, count in [(f'quality_checks/step_{step:06d}.npz', 20000) for step in plan['terminal_steps']] + [('final_samples.npz', 20000), ('holdout_samples.npz', 100000)]:
        path = d / rel
        try:
            with np.load(path, allow_pickle=False) as cloud:
                shapes[rel] = {name: list(cloud[name].shape) for name in ('live', 'ema', 'target')}
                if any((shape != [count, 2] for shape in shapes[rel].values())):
                    reasons.append(f'native noisy {rel} shapes are {shapes[rel]}')
        except Exception as error:
            reasons.append(f'native noisy {rel} unreadable: {error!r}')
    verdict = read(d / 'verdict.json') if (d / 'verdict.json').exists() else {}
    statuses = {kind: verdict.get(kind, {}).get('status') for kind in ('coverage', 'accuracy')}
    if any((value not in ('PASS', 'FAIL') for value in statuses.values())):
        reasons.append(f'official native verdict evidence is invalid or absent: {statuses}')
    if verdict.get('sources') != fixture['host_source_sha256']:
        reasons.append('official native scorer source receipt differs from frozen source map')
    if result.get('status') != statuses['accuracy']:
        reasons.append('original result status differs from official noisy live accuracy gate status')
    if statuses['accuracy'] == 'PASS' and statuses['coverage'] != 'PASS':
        reasons.append('official accuracy PASS lacks required coverage PASS')
    return dict(initial_parameter_match=matches, prior_range_match=actual.get('prior_range') == fixture['prior_range'], fixture=actual, official_status=statuses, event_steps=event_steps, cloud_shapes=shapes, coverage=verdict.get('coverage'), accuracy=verdict.get('accuracy'))

def mechanisms(out, rows):
    record = dict(final_diagnostics=rows[-1].get('diag', {}) if rows else {})
    state_path = out / 'final-state.pt'
    if state_path.exists():
        for key, value in ENV.items():
            os.environ[key] = value
        import torch
        torch.set_num_threads(1)
        state = torch.load(state_path, map_location='cpu', weights_only=False)['trainer']
        bd = state.get('birth_death')
        if bd:
            record['birth_death'] = {key: bd.get(key) for key in ('backend', 'settings', 'counters', 'last', 'snapshot_serial')}
        ev = state.get('row_evidence')
        if ev:
            neff = ev['W'].square() / ev['S'].clamp_min(1e-30)
            record['row_evidence'] = dict(counters=ev.get('counters'), fraction=ev.get('fraction'), n_eff_min=float(neff.min()), n_eff_max=float(neff.max()), n_eff_mean=float(neff.mean()))
    else:
        record['final_state_missing'] = True
    return record

def collect(task, require_attempt=False):
    out = ROOT / 'runs' / task
    path = out / 'result.json'
    execution_path = out / 'execution-receipt.json'
    if not path.exists() and (not execution_path.exists()) and (not require_attempt):
        return dict(task=task, primary_status='PENDING', acceptance_status='PENDING', canonical_fixture_validity='UNVERIFIED')
    result = read(path) if path.exists() else dict(status='ERROR', task=task, error='missing original result.json')
    execution = read(execution_path) if execution_path.exists() else {}
    rows = jsonl(out / 'metrics.jsonl')
    plan = task_plan(task)
    reasons = []
    try:
        integrity = verify_frozen()
    except Exception as error:
        integrity = dict(status='INVALID', error=repr(error))
        reasons.append(str(error))
    primary = result.get('status')
    if primary not in ('PASS', 'FAIL', 'ERROR'):
        reasons.append(f'original status is not PASS/FAIL/ERROR: {primary!r}')
    if execution.get('source_integrity_before', {}).get('status') != 'VALID' or execution.get('source_integrity_after', {}).get('status') != 'VALID':
        reasons.append('execution source integrity receipts are missing or invalid')
    if execution.get('process_exit_code') != (1 if primary == 'ERROR' else 0):
        reasons.append('process exit code is missing or inconsistent with original verdict')
    header = result.get('header', {})
    if header.get('package_sha256') != PACKAGE_SHA:
        reasons.append('original header candidate digest is missing or differs')
    if header.get('device') != 'cuda:0' or header.get('cuda_visible_devices') != '0':
        reasons.append('original header is not physical GPU0/cuda:0')
    if execution.get('resources', {}).get('gpu_uuid') != GPU0_UUID:
        reasons.append('authorized physical GPU0 UUID was not confirmed by the resource guard')
    options = header.get('options', {})
    expected_options = dict(eval_output_noise=True, strict_streams=True, save_final_state=True, diagnostics=True, evaluation_generate='indexed', serial_backward_argument=True, initialization='batch_feature_zero', image_prior_perturb=False, ring_frozen_control=False)
    if options != expected_options:
        reasons.append(f'original resolved options differ: {options}')
    if result.get('stream_deviations') != 0:
        reasons.append(f"original stream deviations: {result.get('stream_deviations')}")
    warnings = result.get('warnings', [])
    fixture_warnings = [warning for warning in warnings if 'construction RNG' in warning or 'initial receipt unavailable' in warning]
    if primary in ('PASS', 'FAIL'):
        if result.get('completed_steps') != plan['steps']:
            reasons.append(f"completed steps {result.get('completed_steps')} != {plan['steps']}")
        if [row.get('step') for row in rows] != plan['observation_steps']:
            reasons.append('recorded observations do not match the original full-budget schedule')
        if result.get('observations') != len(plan['observation_steps']):
            reasons.append('result observation count differs from original schedule')
        for key in ('num_particles', 'z_dim', 'batch_size'):
            if result.get('recipe', {}).get(key) != plan[key]:
                reasons.append(f'final recipe resource {key} differs from original task')
        if not (out / 'final-state.pt').exists():
            reasons.append('requested final state was not saved')
    native_evidence = None
    if task in NATIVE and primary in ('PASS', 'FAIL'):
        native_evidence = validate_native(out, task, result, reasons)
    validity = 'INVALID' if reasons else 'UNVERIFIED' if primary == 'ERROR' else 'VALID'
    acceptance = primary if primary in ('PASS', 'FAIL') and validity == 'VALID' else 'ERROR'
    record = dict(task=task, collected_at=now(), primary_status=primary if primary in ('PASS', 'FAIL', 'ERROR') else 'ERROR', acceptance_status=acceptance, canonical_fixture_validity=validity, validity_reasons=reasons, legacy_fixture_comparison='MISMATCH/WARNING' if fixture_warnings else 'MATCH' if primary in ('PASS', 'FAIL') else 'UNVERIFIED', legacy_fixture_warnings=fixture_warnings, canonical_gpu_acceptance=acceptance, verdict_scope='original frozen CUDA, live noisy primary', expected=plan, result_sha256=sha(path) if path.exists() else None, execution_receipt_sha256=sha(execution_path) if execution_path.exists() else None, source_integrity=integrity, completed_steps=result.get('completed_steps'), observations=result.get('observations'), first_arrival=result.get('first_arrival'), final_streak=result.get('final_streak'), final=result.get('final'), ema_final=result.get('ema_final'), clean_status=result.get('clean_status'), clean_final=result.get('clean_final'), native=result.get('native'), segments=result.get('segments'), thresholds=result.get('thresholds'), pass_rule=result.get('pass_rule'), warnings=warnings, error=result.get('error'), native_evidence=native_evidence, original_screen_sha256=SCREEN_SHA, candidate_package_sha256=PACKAGE_SHA, config_sha256=CONFIG_SHA, timing=dict(original_seconds=result.get('seconds'), original_context_seconds=result.get('train_seconds'), wrapper_wall_seconds=execution.get('wall_seconds'), meaning='screen/context times include evaluation and diagnostics; not training-only throughput'), gpu_memory=dict(peak_allocated_mib=execution.get('peak_allocated_gpu_mib'), peak_reserved_mib=execution.get('peak_reserved_gpu_mib'), original_peak_reserved_mib=result.get('max_gpu_mib'), resources=execution.get('resources')))
    final = result.get('final') or {}
    holdout = (result.get('native') or {}).get('holdout') or {}
    margins = {}
    for key, operation, limit in result.get('thresholds') or []:
        value = final.get(key)
        if isinstance(value, (int, float)):
            margins[key] = dict(value=value, operation=operation, limit=limit, passing_margin=value - limit if operation == '>=' else limit - value)
        if key.startswith('acc_') and isinstance(holdout.get(key[4:]), (int, float)):
            value = holdout[key[4:]]
            margins['holdout_' + key[4:]] = dict(value=value, operation=operation, limit=limit, passing_margin=value - limit if operation == '>=' else limit - value)
    record['threshold_margins'] = margins
    if out.exists():
        try:
            record['mechanisms'] = mechanisms(out, rows)
        except Exception as error:
            record['mechanism_collection_error'] = repr(error)
        write(out / 'acceptance-receipt.json', record)
    return record

def report(records):
    counts = {name: dict(Counter((row['acceptance_status'] for row in records if row['task'] in tasks))) for name, tasks in (('portability', PORTABILITY), ('native', NATIVE))}
    primary_counts = {name: dict(Counter((row['primary_status'] for row in records if row['task'] in tasks))) for name, tasks in (('portability', PORTABILITY), ('native', NATIVE))}
    complete = all((row['acceptance_status'] != 'PENDING' for row in records))
    all_pass = complete and all((row['acceptance_status'] == 'PASS' for row in records))
    suite_status = 'PASS' if all_pass else 'ERROR' if complete and any((row['acceptance_status'] == 'ERROR' for row in records)) else 'FAIL' if complete else 'PENDING'
    summary = dict(collected_at=now(), status=suite_status, acceptance_counts=counts, primary_counts=primary_counts, tasks=records, scope='13 original CUDA portability gates and 3 native 7000-update gates; old CPU evidence is separate')
    write(ROOT / 'leaderboard.json', summary)
    lines = ['# Original frozen CUDA screens', '', f"State: **{summary['status']}**. Acceptance counts: portability {counts['portability']}; native {counts['native']}.", '', 'Primary verdicts are copied from original result.json. Validity is recorded separately; source, stream, canonical fixture, runtime or evidence errors cannot become accepted quality results.', '', '| Task | Original verdict | Fixture/evidence | Accepted verdict | Steps | Peak reserved MiB |', '|---|---|---|---|---:|---:|']
    for row in records:
        lines.append(f"| {row['task']} | {row['primary_status']} | {row['canonical_fixture_validity']} | {row['acceptance_status']} | {row.get('completed_steps', '—')} | {row.get('gpu_memory', {}).get('peak_reserved_mib', '—')} |")
    lines += ['', 'The original CUDA screen, its original hosts and scorers, and the frozen candidate are unchanged. Each native run uses seed 1234, N=20000, z=2, batch 2048, all 34 observations, five terminal 20k clouds and an independent 100k holdout. Live noisy scoring decides the verdict; clean and EMA metrics are diagnostics.', '', 'Old CPU results remain separate evidence. Archived E22 GPU results may be cited as noncontemporary controls, with their source/task/options identities; no new E22 native jobs or seed sweeps are scheduled.', '', 'Screen and context elapsed times include evaluations and diagnostics. GPU memory comes from the guarded execution receipt, with fraction .2 on physical GPU0. These screen timings are not training-only throughput.', '', '## Native metrics', '', '| Task | Coverage | Accuracy | Terminal checks | Holdout precision | Center RMS/σ | Absolute trace bias | Radial KS |', '|---|---|---|---|---:|---:|---:|---:|']
    for row in records:
        if row['task'] not in NATIVE:
            continue
        n = row.get('native') or {}
        h = n.get('holdout') or {}
        lines.append(f"| {row['task']} | {n.get('coverage_status', '—')} | {n.get('accuracy_status', '—')} | {n.get('terminal_accuracy', '—')} | {h.get('precision', '—')} | {h.get('center_rms_sigma', '—')} | {h.get('abs_cov_trace_bias', '—')} | {h.get('radial_ks', '—')} |")
    lines += ['', '## Mechanism activity', '', '| Task | Ordinary evaluations | Discoveries | Ordinary moves | Isolation evaluations | Isolation moves |', '|---|---:|---:|---:|---:|---:|']
    for row in records:
        c = row.get('mechanisms', {}).get('birth_death', {}).get('counters') or {}
        if c:
            lines.append(f"| {row['task']} | {c.get('cell_evals')} | {c.get('cell_discoveries')} | {c.get('ordinary_moves')} | {c.get('iso_evals')} | {c.get('iso_moves')} |")
    failures = [row for row in records if row.get('validity_reasons') or row.get('error')]
    if failures:
        lines += ['', '## Runtime and validity details', '']
        for row in failures:
            lines.append(f"- {row['task']}: {'; '.join(row.get('validity_reasons', []))}; runtime {row.get('error')}")
    warning_rows = [row for row in records if row.get('legacy_fixture_warnings')]
    if warning_rows:
        lines += ['', '## Original warning-only fixture comparisons', '', 'These historical construction RNG/reference comparisons have the original warning-only semantics. Mandatory native tensor/range/source and strict stream checks remain active.']
        for row in warning_rows:
            lines.append(f"- {row['task']}: {'; '.join(row['legacy_fixture_warnings'])}")
    if complete:
        activity = [row.get('mechanisms', {}).get('birth_death', {}).get('counters') or {} for row in records]
        moved = sum((int(c.get('ordinary_moves', 0)) for c in activity))
        lines += ['', f'Ordinary move total: {moved}. ' + ('The ordinary feature-cell reaction path acts on the recorded frozen tasks.' if moved else 'These frozen tasks do not establish an acting ordinary reaction path.'), '', 'Recommendation: ' + ('the candidate satisfies these frozen CUDA gates; assess the paired learned-model comparisons before replacement.' if all_pass else 'retain the passing reference until the reported failures are addressed and independently retested; the candidate does not satisfy this CUDA screen suite.')]
    else:
        lines += ['', 'Recommendation is pending completion of all authorized original CUDA jobs.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary

def manifest():
    files = {str(path.relative_to(ROOT)): dict(sha256=sha(path), bytes=path.stat().st_size) for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name not in ('artifact_manifest.json', '.screen-lane.lock') and ('__pycache__' not in path.parts)}
    write(ROOT / 'artifact_manifest.json', dict(created_at=now(), files=files))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--manifest', action='store_true')
    args = parser.parse_args()
    if args.task:
        record = collect(args.task, require_attempt=True)
        print(json.dumps({key: record.get(key) for key in ('task', 'primary_status', 'canonical_fixture_validity', 'acceptance_status', 'validity_reasons', 'gpu_memory')}), flush=True)
    if args.all or args.report:
        records = [collect(task) for task in TASKS]
        summary = report(records)
        print(json.dumps(dict(event='screen_summary', status=summary['status'], counts=summary['acceptance_counts'])), flush=True)
    if args.manifest:
        manifest()
if __name__ == '__main__':
    main()
