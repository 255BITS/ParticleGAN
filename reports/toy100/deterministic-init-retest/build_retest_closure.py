#!/usr/bin/env python3
"""Freeze existing retest coverage and report progress; no launches or learner imports."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
QUEUE = HERE / 'research-screen-queue'
OUT = HERE / 'retest-closure'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def scope():
    queue = read(QUEUE / 'queue.json')
    definitions = {r['id']: r for r in queue['rows']}
    planned = []
    for directory in (QUEUE, QUEUE / 'simple-probe-preparation'):
        for row in read(directory / 'prepared-index.json')['rows']:
            bundle = Path(row['directory'])
            proof = Path(row['required_review'])
            value = read(proof)
            assert value['status'] == 'PASS' and value['cuda_initialized'] is False and value['learner_steps'] == 0
            assert value['manifest_sha256'] == sha(bundle / 'manifest.json') == row['manifest_sha256']
            assert all(value['checks'].values())
            planned.append(dict(queue_row=row['queue_row'], case_id=bundle.name,
                source_definition_digest=definitions[row['queue_row']]['source_definition_digest'],
                preparation_manifest=pin(bundle / 'manifest.json'), cpu_proof=pin(proof)))
    ka2 = HERE / 'research-mode-hold-preparation'
    proof = HERE / 'research-mode-hold-review/cpu-constructor-proof.json'
    value = read(proof)
    assert value['status'] == 'PASS' and value['cuda_initialized'] is False and value['learner_steps'] == 0
    assert value['manifest_sha256'] == sha(ka2 / 'manifest.json') and all(value['checks'].values())
    planned.append(dict(queue_row='priority:KA2', case_id='research-ka2-new-init',
        source_definition_digest=definitions['priority:KA2']['source_definition_digest'],
        preparation_manifest=pin(ka2 / 'manifest.json'), cpu_proof=pin(proof)))
    selected = {r['queue_row'] for r in planned}
    duplicates, excluded = [], []
    for row in queue['rows']:
        if row['id'] in selected:
            continue
        if row['status'] == 'COMPATIBLE_SOURCE_GROUP_REQUIRES_CANDIDATE_CONSTRUCTOR_REVIEW':
            matches = [r for r in planned if definitions[r['queue_row']]['active_learner_digest'] == row['active_learner_digest']]
            assert len(matches) == 1
            duplicate = definitions[matches[0]['queue_row']]
            assert row['configuration'] == duplicate['configuration']
            assert row['probe_sha256'] == duplicate['probe_sha256']
            assert row['binding_group'] == duplicate['binding_group']
            assert row['active_local_sources'] == duplicate['active_local_sources']
            duplicates.append(dict(queue_row=row['id'], covered_by=matches[0]['queue_row'],
                active_learner_digest=row['active_learner_digest'],
                reason='Exact active local source bytes, configuration, probe and host binding match an already reviewed execution; no duplicate run required.'))
            continue
        status = row['status']
        reasons = {
            'PENDING_ADAPTER_OR_BINDING_REVIEW': ('NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER',
                'The retained original probe has a different construction/hook interface. Its exact initializer binding and fresh CPU constructor contract were not completed before bounded closure; substituting the shared wrapper would be unverified.'),
            'PENDING_EXACT_SOURCE_BINDING': ('NOT_RETESTED_MISSING_EXACT_SOURCE_BINDING',
                'No unique immutable source/configuration/host binding was established for this historical row. A similarly named candidate cannot substitute for it.'),
            'EXACT_PACKAGE_DEFINITION_REQUIRES_DIFFERENT_ADAPTER': ('NOT_RETESTED_OLDER_PACKAGE_INTERFACE',
                'Original package, configuration and extra options are pinned. Its older host/package interface needs a separate initializer-preserving adapter and independent CPU proof; neither was completed in this bounded pass.'),
            'OUT_OF_SCOPE_UNTESTED_DRAFT': ('NOT_RETESTED_UNTESTED_DRAFT',
                'PD2 was a retained untested draft, not an executed leaderboard candidate. Running it would create a new mechanism experiment.')}
        code, reason = reasons[status]
        item = dict(queue_row=row['id'], candidate=row['candidate'], status=code, reason=reason,
            original_source_status=status, historical_scores_preserved=True,
            source_definition_digest=row.get('source_definition_digest'),
            probe_sha256=row.get('probe_sha256'), binding_group=row.get('binding_group'),
            source_directory=row.get('source_directory'), runtime_binding=row.get('runtime_binding'),
            constructor_binding_concerns=row.get('constructor_binding_concerns', []))
        if row.get('candidate_archive'):
            archive = row['candidate_archive']
            assert sha(Path(archive['path'])) == archive['sha256']
            item['retained_candidate_archive'] = archive
        if row['id'] == 'priority:historical-base/constraints_simple_regularization':
            item['reason'] = 'Original config and worker are retained (original-config.json and worker.py); the differently shaped reference worker lacks a reviewed new-initializer adapter. This is not a missing-config claim.'
        if '/game_dynamics/extragradient' in row['id'] or '/k3p_extragradient/' in row['id']:
            item['reason'] += ' This mechanism also replaces host functions dynamically; full transformed-source closure and initializer ordering must be bound before execution.'
        excluded.append(item)
    assert len(planned) == 97 and len(duplicates) == 4 and len(excluded) == 89
    assert len(definitions) == len(planned) + len(duplicates) + len(excluded) == 190
    return dict(schema=1, status='FROZEN_EXISTING_CANDIDATE_SCOPE_NO_EXPANSION',
        scope='Finish only the already reviewed fixed retest cases and approved shortlist followups, update PR/body, then stop owned search. No new mechanisms, reconstructions or GPU cases are created here.',
        sources=dict(research_queue=pin(QUEUE / 'queue.json'), grouped85=pin(QUEUE / 'prepared-index.json'),
            simple11=pin(QUEUE / 'simple-probe-preparation/prepared-index.json'),
            external_configuration_recovery=pin(QUEUE / 'external-binding-recovery/index.json'),
            older_package_archive_review=pin(QUEUE / 'older-package-archive-review.json')),
        counts=dict(coverage_rows=254, public_controls=3, api_historical_configurations=46,
            research_definition_rows=190, research_alias_rows=15, reviewed_research_execution_cases=97,
            exact_duplicate_definition_rows=4, not_retested_definition_rows=89,
            not_retested_reasons=dict(Counter(r['status'] for r in excluded))),
        reviewed_research_cases=planned, exact_duplicates=duplicates, not_retested=excluded,
        aliases=queue['aliases'],
        notes=['Counts are historical definition/alias rows, not claims of that many independent algorithms.',
            'Existing research quality does not establish public GANTrainer qualification; actual schedules and source eligibility remain separate.',
            'Twelve older external configurations were recovered exactly; this does not supply missing adapter/constructor proof.',
            '29 legacy PR records and the old RG5 floor variant remain unbound; current similarly named in-repository PR107/140/143 packages are distinct.',
            'API completeness includes45 ordinary source ports and one explicit RP1 worker-owned CUDA-eager diagnostic, alongside3 public controls; the diagnostic is not a silently modified public baseline.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    frozen = scope()
    scope_path = OUT / 'coverage-scope.json'
    if scope_path.exists():
        assert read(scope_path) == frozen, 'Frozen coverage changed; no automatic expansion allowed'
    else:
        write(scope_path, frozen)
    cases = {r['queue_row']: dict(r, status='NOT_YET_ARCHIVED') for r in frozen['reviewed_research_cases']}
    ledger_path = HERE / 'research-results.json'
    for score in read(ledger_path)['results']:
        archive_path = HERE / score['archive_manifest']
        archive = read(archive_path)
        assert all(score[k] == archive[k] for k in ('id', 'candidate', 'status', 'summary', 'audit_sha256'))
        plan = read(archive_path.parent / 'source-plan.json')
        row_id = plan.get('source_queue_row') or 'priority:KA2'
        assert row_id in cases
        audit_path = Path(score['audit']); assert sha(audit_path) == score['audit_sha256']
        audit = read(audit_path)
        assert audit['status'] == 'PASS' and audit['quality_status'] == score['status']
        cases[row_id].update(status='TERMINAL_ARCHIVED_AND_AUDITED', quality=score['status'],
            summary=score['summary'], archive_manifest=pin(archive_path), independent_audit=pin(audit_path))
    records = read(Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/batch.json'))
    by_case = {r['case_id']: r for r in cases.values()}
    for record in records:
        for case_id in record.get('candidates', []):
            if case_id in by_case:
                row = by_case[case_id]
                row['launch'] = {k:record[k] for k in ('lane','pid','directory','probe_spec_sha256') if k in record}
                if row['status'] != 'TERMINAL_ARCHIVED_AND_AUDITED':
                    row['status'] = 'LAUNCHED_AWAITING_ARCHIVED_AUDIT'
    completed = [r for r in cases.values() if r['status'] == 'TERMINAL_ARCHIVED_AND_AUDITED']
    stop_path = OUT / 'user-stop/stop-receipt.json'
    stopped = read(stop_path) if stop_path.exists() else None
    if stopped:
        assert stopped['status'] == 'STOPPED_USER_REQUEST' and not stopped['owned_processes_remaining']
        remaining = {r['case_id'] for r in cases.values() if r['status'] != 'TERMINAL_ARCHIVED_AND_AUDITED'}
        assert remaining == set(stopped['not_run_case_ids'])
        for row in cases.values():
            if row['case_id'] in remaining:
                row.update(status='NOT_RUN_USER_STOP', reason='Reviewed exact-source case left unrun after the user requested immediate wrap-up; not a quality failure or missing-source claim.')
    api_ledger = read(HERE / 'screen-results.json')['results']
    full_api = [r for r in api_ledger if r['status'] in ('PASS','FAIL')]
    assert len(full_api) == 49 and len({r['candidate'] for r in full_api}) == 49
    progress = dict(schema=1, status='STOPPED_USER_REQUEST_PARTIAL_COVERAGE' if stopped else ('COMPLETE_FIXED_RESEARCH_SCOPE' if len(completed)==97 else 'IN_PROGRESS_FIXED_SCOPE'),
        recorded_utc=datetime.now(timezone.utc).isoformat(), frozen_scope=pin(scope_path),
        research_score_ledger=pin(ledger_path), api_score_ledger=pin(HERE / 'screen-results.json'),
        research_counts=dict(planned=97, archived_and_independently_audited=len(completed),
            awaiting_archive_or_execution=97-len(completed),
            quality=dict(Counter(r['quality'] for r in completed)), status=dict(Counter(r['status'] for r in cases.values()))),
        api_counts=dict(completed_quality_windows=49, quality=dict(Counter(r['status'] for r in full_api)),
            retained_logging_errors=sum(r['status']=='ERROR' for r in api_ledger)),
        cases=list(cases.values()), remaining_case_ids=[r['case_id'] for r in cases.values() if r['status']!='TERMINAL_ARCHIVED_AND_AUDITED'])
    if stopped:
        progress['stop_receipt'] = pin(stop_path)
        progress['full_leaderboard_retest_complete'] = False
        progress['research_counts']['not_run_user_stop'] = len(stopped['not_run_case_ids'])
        progress['research_counts']['awaiting_archive_or_execution'] = 0
    write(OUT / 'coverage-progress.json', progress)
    groups = defaultdict(list)
    for row in frozen['not_retested']:
        groups[(row['status'], row['probe_sha256'], row['binding_group'])].append(row)
    text = ['# Bounded retest coverage', '',
        (f"Search stopped at the user's request. {len(completed)}/97 reviewed research cases are archived and independently audited; {97-len(completed)} are NOT_RUN_USER_STOP. All owned workers have exited and launch STOP markers are installed. Coverage is partial, not a completed full-leaderboard retest." if stopped else f"Scope is frozen at97 reviewed research cases (85 grouped + KA2 +11 simple). At {progress['recorded_utc']}, {len(completed)}/97 are archived and independently audited; {97-len(completed)} still await execution or archive."), '',
        'All49 API/control quality windows are complete (46 historical configurations, including the explicitly scoped RP1 eager diagnostic, plus3 public controls):4 PASS and45 FAIL. Four earlier logging ERROR records remain preserved. A passing tiny screen does not override a later breadth failure.', '',
        'The190 research definition rows comprise97 reviewed execution cases,4 exact duplicate definitions, and89 NOT_RETESTED rows. Another15 rows are explicit priority aliases. These mappings plus46 API configurations and3 public controls account for all254 coverage rows.', '',
        '| NOT_RETESTED reason | Definition rows |', '|---|---:|']
    for reason, count in frozen['counts']['not_retested_reasons'].items():
        text.append(f'| {reason} | {count} |')
    text += ['', 'These89 gaps are additional to the22 reviewed cases stopped by user request. Fifty rows have retained source but no completed reviewed initialization adapter; this is unfinished work, not missing or impossible source. Eight older package definitions likewise retain sources but need a different reviewed binding. Thirty rows lack a unique exact binding, and one is an untested draft. No unavailable adapter is counted as a quality failure or as completed retesting.', '',
        '| Source/binding group | Rows | Historical labels |', '|---|---:|---|']
    for (reason, probe, binding), rows in groups.items():
        text.append(f"| {reason}; probe {(probe or 'unbound')[:12]} / binding {(binding or 'unbound')[:12]} | {len(rows)} | "+', '.join(r['candidate'] for r in rows)+' |')
    text += ['', 'The [frozen scope](coverage-scope.json) lists every exact case, duplicate mapping, alias and untested reason. [Progress](coverage-progress.json) binds current archive/audit hashes. The frozen scope cannot be expanded by rerunning this report.', '',
        'Refresh bookkeeping only with `python reports/toy100/deterministic-init-retest/build_retest_closure.py`. The user-stop receipt freezes partial coverage and exact unrun IDs. `--require-complete` intentionally fails because the full97 were not run. This command never runs training, changes a score, launches a worker or stops a process.', '']
    (OUT / 'README.md').write_text('\n'.join(text))
    print(json.dumps(dict(research=progress['research_counts'],api=progress['api_counts'],scope_sha256=sha(scope_path)),indent=2))
    if args.require_complete and len(completed)!=97:
        raise SystemExit('Fixed research scope is not fully archived yet; keep IN_PROGRESS, not COMPLETE.')


if __name__ == '__main__':
    main()
