#!/usr/bin/env python3
"""Refresh source/port/execution bookkeeping only; never imports or runs a learner."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import io
import gzip
import json
import os
import zipfile

HERE=Path(__file__).resolve().parent
INVENTORY=HERE.parent/'continuous-api-search/fixed-init-retest-inventory'
DEFAULT_BATCH=Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/batch.json')
INITIALIZER='c720645ecae6b648e9fc6034e9d6b48ccff06ed3'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p): return dict(path=str(p),sha256=sha(p))
def atomic(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def cpu_receipt(p,declaration):
    if not p.exists():return dict(status='PENDING')
    v=read(p)
    ok=v.get('status')=='PASS' and v.get('cpu_initialization',{}).get('status')=='PASS' and v.get('declaration_sha256')==sha(declaration) and v.get('cuda_initialized') is False
    return dict(status='PASS' if ok else 'RECEIPT_REQUIRES_REVIEW',receipt=pin(p),quality_test=False)
def process_alive(pid):
    try:
        raw=Path('/proc')/str(pid)/'stat'
        return raw.read_text().split(') ',1)[1][0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def attach_archived_results(rows):
    """Link retained score receipts; a runtime audit PASS is not a quality PASS."""
    ledger=HERE/'screen-results.json'
    if not ledger.exists():return None
    declared=read(ledger)
    if declared.get('initializer_commit')!=INITIALIZER:raise ValueError('wrong ledger initializer')
    by_candidate={r['candidate']:r for r in rows}
    public_path=HERE/'public3-runtime-audit.json'
    public=read(public_path) if public_path.exists() else {}
    for score in declared['results']:
        row=by_candidate.get(score['candidate'])
        if row is None:continue
        archive=HERE/score['archive_manifest'];retained=read(archive)
        if any(retained.get(k)!=score.get(k) for k in ('candidate','status','summary','complete','artifacts')):
            raise ValueError('ledger/archive mismatch: '+score['candidate'])
        for rel,item in retained['artifacts'].items():
            if sha(HERE/rel)!=item['sha256']:raise ValueError('archive content changed: '+rel)
        declaration=archive.parent/'declaration.json';result=read(archive.parent/'result.json')
        if sha(declaration)!=row['declaration']['sha256'] or read(declaration).get('initializer_commit')!=INITIALIZER:
            raise ValueError('archive/current declaration mismatch: '+score['candidate'])
        row.update(execution_status='ARCHIVED_PENDING_INDEPENDENT_AUDIT',quality_status='PENDING_INDEPENDENT_RESULT_AUDIT',archive=pin(archive),reported_status=score['status'],reported_summary=score['summary'],complete_quality_window_reported=score['complete'])
        audit_path=HERE/(row['id']+'-runtime-audit.json')
        audit=read(audit_path) if audit_path.exists() else None
        if audit is not None:
            good=(audit.get('status')=='PASS' and audit.get('candidate')==score['candidate'] and audit.get('result')==score['status'] and audit.get('summary')==score['summary'] and audit.get('declaration_sha256')==sha(declaration) and audit.get('source_zip_sha256')==sha(archive.parent/'source.zip') and audit.get('sampling_rows_verified')==1200 and audit.get('observations_verified')==24 and audit.get('initial_tensors_equal_cpu_preflight') is True)
        else:
            audit=next((v for v in public.get('results',[]) if v.get('candidate')==score['candidate']),None)
            good=(public.get('status')=='PASS' and audit is not None and audit.get('status')=='AUDIT_PASS_SCORE_'+score['status'] and audit.get('result')==result and audit.get('source_zip_sha256')==sha(archive.parent/'source.zip') and audit.get('frozen_sampling_batches_verified')==1200 and audit.get('observations_verified')==24 and audit.get('initial_models_equal_cpu_preflight') is True)
            audit_path=public_path
        if good and score['complete'] and result.get('status')==score['status']:
            row.update(execution_status='TERMINAL_ARCHIVED_AND_AUDITED',quality_status='AUDITED_'+score['status'],independent_runtime_audit=pin(audit_path))
        elif audit is not None:
            row['audit_receipt_status']='PRESENT_REQUIRES_REVIEW';row['independent_runtime_audit']=pin(audit_path)
    return pin(ledger)

def attach_research_progress(rows):
    """Keep research-host results separate from public/API score authority."""
    folder=HERE/'research-screen-queue';queue_path=folder/'queue.json'
    if not queue_path.exists():return None
    queue=read(queue_path);by_id={r['id']:r for r in rows}
    for definition in queue['rows']:
        row=by_id.get(definition['id'])
        if row is None:continue
        row['research_binding_status']=definition['status']
        row['port_status']=definition['status']
        row['research_binding_group']=definition.get('binding_group')
        row['research_source_definition_digest']=definition.get('source_definition_digest')
    for alias in queue['aliases']:
        row=by_id.get(alias['id'])
        if row is not None:
            row['linked_queue_rows']=alias['linked_rows'];row['port_status']='FOLLOW_EXACT_LINKED_DEFINITION'
    prepared_rows=[]
    for preparation_folder in (folder,folder/'simple-probe-preparation'):
        index_path=preparation_folder/'prepared-index.json'
        if index_path.exists():prepared_rows.extend((preparation_folder,r) for r in read(index_path)['rows'])
    if prepared_rows:
        for preparation_folder,prepared in prepared_rows:
            row=by_id.get(prepared['queue_row'])
            if row is None:continue
            root=preparation_folder/prepared['directory_relative'] if prepared.get('directory_relative') else Path(prepared['directory'])
            proof_path=preparation_folder/prepared['required_review_relative'] if prepared.get('required_review_relative') else Path(prepared['required_review'])
            assert sha(root/'manifest.json')==prepared['manifest_sha256']
            row['port_status']='SOURCE_PREPARED_REQUIRES_CPU_PROOF';row['research_preparation']=prepared
            if proof_path.exists():
                proof=read(proof_path)
                bound=(proof.get('status')=='PASS' and proof.get('cuda_initialized') is False and proof.get('learner_steps')==0 and proof.get('manifest_sha256')==prepared['manifest_sha256'] and proof.get('source_plan_sha256')==prepared['source_plan_sha256'] and proof.get('bridge_sha256')==prepared['bridge_sha256'] and proof.get('runner_sha256')==prepared['worker_sha256'] and all(proof.get('checks',{}).get(k) is True for k in ('all_initial_tensors_match_public_host','repeat_without_rng_reset','constructor_rng_cursor_preserved','initializer_rng_neutral','historical_prior_registration_preserved','all_bindings_restore_on_exception','batch_distance_scope_explicit')))
                row['cpu_preflight']=dict(status='PASS' if bound else 'RECEIPT_REQUIRES_REVIEW',receipt=pin(proof_path),quality_test=False)
                if bound:row['port_status']='SOURCE_AND_CPU_REVIEWED_RESEARCH_HOST'
    ledger=HERE/'research-results.json'
    if not ledger.exists():return dict(source_queue=pin(queue_path))
    for score in read(ledger)['results']:
        archive=HERE/score['archive_manifest'];retained=read(archive)
        assert score['scope']=='RESEARCH_HOST' and all(retained.get(k)==score.get(k) for k in ('id','candidate','status','summary','artifacts','audit_sha256'))
        for rel,item in retained['artifacts'].items():assert sha(HERE/rel)==item['sha256']
        plan=read(archive.parent/'source-plan.json');assert plan['initializer_commit']==INITIALIZER
        row_id=plan.get('source_queue_row') or ('priority:KA2' if score['candidate']=='RESEARCH-KA2-new-init' else None)
        if row_id not in by_id:raise ValueError('unmapped research archive: '+score['candidate'])
        row=by_id[row_id];audit_path=Path(score['audit']);assert sha(audit_path)==score['audit_sha256']
        audit=read(audit_path)
        own_initial_proof=(audit.get('initial_cuda_material_matches_cpu_proof') is True or audit.get('initial_cuda_material_matches_own_cpu_proof') is True)
        assert audit['status']=='PASS' and audit['quality_status']==score['status'] and audit['observations']==score['summary']['observations'] and audit['passing_observations']==score['summary']['passing'] and audit['passing_suffix']==score['summary']['final_suffix'] and own_initial_proof
        row.update(port_status='SOURCE_AND_CPU_REVIEWED_RESEARCH_HOST',execution_status='TERMINAL_ARCHIVED_AND_AUDITED',quality_status='AUDITED_'+score['status'],quality_scope='RESEARCH_HOST_NOT_PUBLIC_API',archive=pin(archive),independent_runtime_audit=pin(audit_path),reported_summary=score['summary'],reported_status=score['status'],continuous_eligibility=score['continuous_eligibility'],seconds=score['seconds'])
    return dict(source_queue=pin(queue_path),score_ledger=pin(ledger))

def attach_followups(rows):
    """A later task failure limits qualification without erasing a screen pass."""
    ledger=HERE/'followup-results.json'
    if not ledger.exists():return None
    by_candidate={r['candidate']:r for r in rows}
    archives=[(p,read(p)) for p in (HERE/'followup-evidence').glob('*/archive-manifest.json')]
    for audit in read(ledger)['results']:
        row=by_candidate.get(audit['candidate'])
        if row is None:raise ValueError('unmapped followup candidate')
        if 'archive_manifest' in audit:
            path=HERE/audit['archive_manifest'];archive=read(path)
            assert all(archive[k]==audit[k] for k in ('candidate','task','quality_status','audit_sha256'))
            audit_path=Path(audit['audit']);assert sha(audit_path)==audit['audit_sha256']
            original=read(audit_path)
            assert original=={k:v for k,v in audit.items() if k not in ('archive_manifest','audit','audit_sha256')}
            for rel,item in archive['artifacts'].items():
                retained=HERE/item['path'];assert sha(retained)==item['sha256']
                raw=gzip.decompress(retained.read_bytes()) if retained.suffix=='.gz' else retained.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==item['original_sha256']
        else:
            matches=[(p,v) for p,v in archives if v.get('audit')==audit]
            if len(matches)!=1:raise ValueError('followup audit/archive binding is not unique')
            path,archive=matches[0]
            for rel,item in archive['retained'].items():assert sha(path.parent/rel)==item['sha256']
        assert audit['status']=='PASS'
        row.setdefault('audited_followups',[]).append(dict(task=audit['task'],quality_status='AUDITED_'+audit['quality_status'],observations=audit['observations'],passing_observations=audit['passing_observations'],passing_suffix=audit['passing_suffix'],first_arrival=audit.get('first_arrival'),final=audit['final'],failed_bounds=audit.get('final_failed_bounds',[]),archive=pin(path),limitations=audit.get('limits',[])))
        if audit['quality_status']=='FAIL':row['qualification_status']='REJECTED_BY_OWN_COMPLETED_FOLLOWUP; INITIAL_SCREEN_SCORE_PRESERVED'
    return pin(ledger)

def attach_current_eligibility(rows):
    """Current source findings and pruned work never overwrite historical scores."""
    by_id={r['id']:r for r in rows};receipts=[]
    for path in sorted((HERE/'research-eligibility-audits').glob('*-configuration.json')):
        audit=read(path);row=by_id[audit['candidate']]
        for item in audit['evidence']:
            if 'path' in item:assert sha(Path(item['path']))==item['sha256']
            else:
                with zipfile.ZipFile(item['archive']) as archive:
                    members=item['member'].split('!/')
                    raw=archive.read(members[0])
                    for member in members[1:]:
                        with zipfile.ZipFile(io.BytesIO(raw)) as nested:raw=nested.read(member)
                    assert hashlib.sha256(raw).hexdigest()==item['sha256']
        authority=audit['quality_authority']
        score=next(v for v in read(Path(authority['path']))['results'] if v['id']==authority['row_id'])
        assert sha(HERE/score['archive_manifest'])==authority['archive_manifest_sha256']
        assert score['status']==audit['new_initialization_quality']['status']
        row['current_configuration_eligibility']=audit['current_configuration_eligibility']
        row['current_source_eligibility_audit']=pin(path)
        row['historical_eligibility_label_preserved']=audit['historical_eligibility_label_unchanged']
        receipts.append(pin(path))
    path=HERE/'precision-three-single-shift-preparation/qualification-status.json'
    if path.exists():
        status=read(path)
        for candidate,case in status['cases'].items():
            row=by_id[candidate]
            row['further_ring_preparation_status']=case
            row['further_ring_status_receipt']=pin(path)
            if case['ring']=='NOT_RUN':
                assert any(v['task']=='img_bars4' and v['quality_status']=='AUDITED_FAIL' for v in row.get('audited_followups',[])), candidate
        receipts.append(pin(path))
    return receipts

def attach_bounded_closure(rows):
    path=HERE/'retest-closure/coverage-scope.json'
    if not path.exists():return None
    scope=read(path);by_id={r['id']:r for r in rows}
    assert scope['status']=='FROZEN_EXISTING_CANDIDATE_SCOPE_NO_EXPANSION'
    for item in scope['not_retested']:
        row=by_id[item['queue_row']]
        assert row['quality_status']=='NOT_RUN', 'Untested closure cannot erase a quality result'
        row['execution_status']='NOT_RETESTED_AT_BOUNDED_CLOSURE'
        row['closure_status']=item['status'];row['not_retested_reason']=item['reason']
    for item in scope['exact_duplicates']:
        row=by_id[item['queue_row']]
        row['execution_status']='COVERED_BY_EXACT_DEFINITION_LINK'
        row['linked_queue_rows']=[item['covered_by']]
        row['duplicate_source_proof']=item
    for alias in scope['aliases']:
        row=by_id[alias['id']]
        row['execution_status']='ALIAS_FOLLOWS_LINKED_ROWS';row['linked_queue_rows']=alias['linked_rows']
    stop_path=HERE/'retest-closure/user-stop/stop-receipt.json'
    if stop_path.exists():
        stop=read(stop_path)
        assert stop['status']=='STOPPED_USER_REQUEST' and not stop['owned_processes_remaining']
        planned={r['case_id']:r['queue_row'] for r in scope['reviewed_research_cases']}
        for case_id in stop['not_run_case_ids']:
            row=by_id[planned[case_id]]
            assert row['quality_status']=='NOT_RUN'
            row.update(execution_status='NOT_RUN_USER_STOP',not_retested_reason='Exact-source reviewed case left unrun after the user requested wrap-up.',user_stop_receipt=pin(stop_path))
    return pin(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,action='append',default=[])
    args=parser.parse_args()
    inv=read(INVENTORY/'inventory.json');api=read(INVENTORY/'api-screening-candidates.json')
    rows=[];references=[]
    for name in ('public-k3p','public-ka2','public-ka2-constant'):
        p=HERE/(name+'-declaration.json');de=read(p)
        rows.append(dict(id=name,candidate=de['candidate'],category='PUBLIC_CONTROL',lane='reference',source_status='PINNED_PUBLIC_PACKAGE',port_status='PORTED',declaration=pin(p),package_sha256=de['package_sha256'],recipe_overrides=de['recipe_overrides'],cpu_preflight=cpu_receipt(HERE/(name+'-cpu-preflight.json'),p),execution_status='PENDING',quality_status='NOT_RUN',continuous_eligibility='SCHEDULED_REFERENCE; NEW_INIT_QUALITY_DOES_NOT_ESTABLISH_RUN_FOREVER',prior_scores_inherited=False))
    for a in api['candidate_rows']:
        if a['lane']=='reference':references.append(dict(candidate=a['candidate'],meaning='Historical released identity underlying explicit public reference protocols; not an additional duplicate mechanism.',authority=a['source_authority']));continue
        id=a['candidate'].lower();row=dict(id=id,candidate=a['candidate']+'-new-init',category='API_EXPERIMENTAL',lane=a['lane'],source_status='IMMUTABLE_SOURCE_READY',historical_source=a['source_authority'],historical_declaration=a['configuration_authority'],historical_package_digest=a['package_digest'],historical_records=a['historical_record_ids'],declared_historical_recipe=a['declared_recipe'],port_status='PENDING',cpu_preflight={'status':'PENDING'},execution_status='PENDING',quality_status='NOT_RUN',continuous_eligibility='OWN_NEW_PACKAGE_AND_FULL_QUALIFICATION_REQUIRED',prior_scores_inherited=False)
        # Only the canonical candidate directory is authoritative. Rebuild/debug/conflict directories do not count as ports.
        dest=HERE/'port-source'/id;manifest=dest/'port-manifest.json';decl=dest/'candidate-declaration.json';archive=dest/'package.zip'
        if manifest.exists():
            m=read(manifest);row['port_manifest']=pin(manifest)
            if m.get('status')=='SOURCE_PORTED_NOT_QUALITY_QUALIFIED' and not m.get('unresolved_conflicts') and decl.exists() and archive.exists():
                de=read(decl);ok=sha(archive)==m.get('package_zip_sha256') and de.get('initializer_commit')==INITIALIZER and m.get('source_zip_sha256')==a['source_authority']['sha256']
                with zipfile.ZipFile(archive) as z:package={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('.py')}
                ok=ok and package==de['package_sha256']
                row['port_status']='PORTED' if ok else 'PORT_RECEIPT_MISMATCH';row['declaration']=pin(decl);row['new_package']=pin(archive)
                row['cpu_preflight']=cpu_receipt(HERE/(id+'-cpu-preflight.json'),decl)
                independent=HERE/(id+'-independent-init-audit.json')
                if independent.exists():
                    audit=read(independent)
                    ready=(audit.get('status')=='PASS_INITIALIZATION_ONLY' and audit.get('declaration_sha256')==sha(decl) and audit.get('port_manifest_sha256')==sha(manifest) and audit.get('cpu_receipt_sha256')==row['cpu_preflight'].get('receipt',{}).get('sha256') and audit.get('package_zip_sha256')==sha(archive))
                    row['independent_initialization_audit']=dict(status='PASS_INITIALIZATION_ONLY' if ready else 'RECEIPT_REQUIRES_REVIEW',receipt=pin(independent))
            else:row['port_status']='PORT_BLOCKED_OR_INCOMPLETE';row['unresolved_conflicts']=m.get('unresolved_conflicts');row['integration_blocker']=m.get('integration_blocker')
        rows.append(row)
    eligibility={}
    for e in inv['research_priority_entries']:
        if 'historical_continuous_eligibility' in e:
            for id in e.get('research_entries',[]):eligibility[id]=e['historical_continuous_eligibility']
    for e in inv['research_candidate_entries']:
        rows.append(dict(id='research:'+e['id'],candidate=e['candidate'],category='RESEARCH_COHORT',lane=e['lane'],source_status=e['source_status'],source_bindings=e['source_bindings'],historical_evidence=e['evidence_input'],historical_row_indices=e['evidence_row_indices'],historical_status_counts=e['historical_status_counts'],port_status='PENDING_RESEARCH_HOST_BINDING',cpu_preflight={'status':'PENDING'},execution_status='PENDING',quality_status='NOT_RUN',continuous_eligibility=eligibility.get(e['id'],{'status':'UNVERIFIED'}),prior_scores_inherited=False))
    # Priority/old alias rows are a coverage map. Their linked scored cohorts stay explicit; do not count every alias as another mechanism.
    for i,e in enumerate(inv['research_priority_entries']):
        links=['research:'+x for x in e.get('research_entries',[])];status='ALIAS_LINKED_TO_RESEARCH_ROWS' if links else ('SOURCE_CONFIGURATION_AVAILABLE' if e.get('source_binding') or e.get('definition_source') or e.get('sources') or e.get('source_receipt') else 'MISSING_EXACT_SOURCE_BINDING')
        rows.append(dict(id='priority:'+e.get('id',e['candidate']),candidate=e['candidate'],category='RESEARCH_PRIORITY_MAPPING',lane='research',source_status=status,inventory_index=i,linked_queue_rows=links,port_status='FOLLOW_LINKED_ROWS' if links else 'PENDING_RESEARCH_HOST_BINDING',cpu_preflight={'status':'PENDING'},execution_status='PENDING',quality_status='NOT_RUN',continuous_eligibility=e.get('historical_continuous_eligibility',{'status':'UNVERIFIED'}),prior_scores_inherited=False))
    for e in inv['late_research_recovery']:
        rows.append(dict(id='late-research:'+e['candidate'],candidate=e['candidate'],category='RESEARCH_PARTIAL_OR_RECOVERED',lane='research',source_status='RETAINED_SOURCE_AND_RESULT_RECEIPTS',source_receipts=e['source_and_result_receipts'],historical_completion=e['classification'],port_status='OUT_OF_SCOPE_UNTESTED_DRAFT' if e['classification']=='UNTESTED_DRAFT' else 'PENDING_RESEARCH_HOST_BINDING',cpu_preflight={'status':'PENDING'},execution_status='NOT_QUEUED_DRAFT' if e['classification']=='UNTESTED_DRAFT' else 'PENDING',quality_status='NOT_RUN',continuous_eligibility='UNVERIFIED',prior_scores_inherited=False))
    for e in inv['legacy_pr_summary_entries']:
        rows.append(dict(id=e['id'],candidate=e['candidate'],category='LEGACY_PR_SOURCE_RECOVERY',lane='research',source_status='MISSING_PINNED_SOURCE_BINDING',summary=e['historical_summary'],port_status='BLOCKED_SOURCE_RECOVERY',cpu_preflight={'status':'PENDING'},execution_status='PENDING',quality_status='NOT_RUN',continuous_eligibility='UNVERIFIED',prior_scores_inherited=False))
    by_candidate={x['candidate']:x for x in rows if x['category'] in ['PUBLIC_CONTROL','API_EXPERIMENTAL']};by_id={x['id']:x for x in rows};batchpins=[];unmatched=[]
    research_cases={'research-ka2-new-init':'priority:KA2'}
    for prep in (HERE/'research-screen-queue',HERE/'research-screen-queue/simple-probe-preparation'):
        if (prep/'prepared-index.json').exists():
            for entry in read(prep/'prepared-index.json')['rows']:research_cases[Path(entry['directory']).name]=entry['queue_row']
    batches=args.batch or [DEFAULT_BATCH]
    for batch in batches:
        if not batch.exists():continue
        records=read(batch);batchpins.append(pin(batch))
        for record in records:
            directory=Path(record['directory']);resultpaths=list(directory.glob('*/repo/reports/fixed-init-mode-hold/result.json'))+list(directory.glob('*/repo/reports/fixed-init-mode-hold/*/result.json'))
            for assigned in record.get('candidates',[record['lane']]):
                target=by_id.get(assigned) or by_id.get(research_cases.get(assigned))
                if target is not None:
                    target['launch_record']={k:record[k] for k in ['lane','directory','pid','base'] if k in record};target['driver_alive_observed']=process_alive(record['pid']);target['execution_status']='BATCH_ACTIVE_AWAITING_RESULT' if target['driver_alive_observed'] else 'DRIVER_EXITED_NO_TERMINAL_RESULT'
            if record['lane'].startswith('research-'):
                for resultpath in directory.glob('*/repo/reports/reviewed-probe-output/*/*/result.json'):
                    plan_path=resultpath.parent/'source-plan.json';receipt_path=resultpath.parent/'initialization-receipt.json'
                    if not plan_path.exists() or not receipt_path.exists():continue
                    plan=read(plan_path);receipt=read(receipt_path)
                    row=by_id.get(plan.get('source_queue_row') or ('priority:KA2' if plan.get('candidate')=='RESEARCH-KA2-new-init' else None))
                    if row is None:unmatched.append(pin(resultpath));continue
                    assert plan['initializer_commit']==INITIALIZER and receipt['source_plan_sha256']==sha(plan_path)
                    result=read(resultpath)
                    row.update(execution_status='TERMINAL_RECORDED_PENDING_AUDIT',result=pin(resultpath),reported_status=result.get('status'),seconds=result.get('seconds'),quality_status='PENDING_INDEPENDENT_RESULT_AUDIT',quality_scope='RESEARCH_HOST_NOT_PUBLIC_API')
            for resultpath in resultpaths:
                result=read(resultpath);row=by_candidate.get(result.get('candidate'));decl=resultpath.parent/'declaration.json'
                if row is None:unmatched.append(pin(resultpath));continue
                if not decl.exists() or read(decl).get('initializer_commit')!=INITIALIZER or ('declaration' in row and sha(decl)!=row['declaration']['sha256']):
                    row['execution_status']='TERMINAL_IDENTITY_MISMATCH';row['result']=pin(resultpath);continue
                row['execution_status']='TERMINAL_RECORDED_PENDING_AUDIT';row['result']=pin(resultpath);row['reported_status']=result.get('status');row['reported_metrics']=result.get('metrics');row['seconds']=result.get('seconds');row['quality_status']='PENDING_INDEPENDENT_RESULT_AUDIT'
                # Explicitly distinguish a 0-update preflight or runtime error from a complete quality run.
                m=result.get('metrics',{});row['complete_quality_window_reported']=m.get('updates')==1200 and m.get('verdict',{}).get('convergence',{}).get('complete') is True
    score_ledger=attach_archived_results(rows)
    research_progress=attach_research_progress(rows)
    followup_ledger=attach_followups(rows)
    current_eligibility=attach_current_eligibility(rows)
    bounded_closure=attach_bounded_closure(rows)
    data=dict(schema=1,recorded_utc=datetime.now(timezone.utc).isoformat(),initializer_commit=INITIALIZER,scope='Authoritative coverage/progress queue, not a winner leaderboard. Source, port, CPU checks, execution, quality and eligibility are independent. Historical scores unchanged.',inventory=pin(INVENTORY/'inventory.json'),api_authority=pin(INVENTORY/'api-screening-candidates.json'),refresh_script=pin(Path(__file__)),batch_receipts=batchpins,counts=dict(rows=len(rows),by_category=dict(Counter(x['category'] for x in rows)),api_experimental_port_status=dict(Counter(x['port_status'] for x in rows if x['category']=='API_EXPERIMENTAL')),execution_status=dict(Counter(x['execution_status'] for x in rows))),historical_reference_mappings=references,conditional_diagnostics=[dict(id='public-ka2-decay-historical-config',status='DECLARED_HISTORICAL_CONFIGURATION_NOT_LAUNCHED',source='continuous-api-search/fixed-init-retest-inventory/api-authority.json#mandatory_controls',condition='Retain exact historical settings as a separate diagnostic; no need to repeat decay before new-init constant result justifies it.')],rows=rows,unmatched_terminal_results=unmatched,policy=['All46 existing API configurations and public3 controls are mapped; source-ported does not mean runtime/quality PASS.','Research priority aliases may point to scored cohort rows; raw row count is not a count of independent algorithms or required duplicate runs.','Never drop prior quality failures because they failed under old initialization. No historical pass transfers.','Preserve each declared learner policy; scheduled historical quality can improve without becoming run-forever eligible.','Do not load old tensor weights. New initializer and caller/data/latent/noise cursor contracts are separate.','Do not execute draft proposals under the retest instruction. Recover missing source identity while ready candidates proceed.'])
    data['score_ledger']=score_ledger
    data['research_progress']=research_progress
    data['followup_ledger']=followup_ledger
    data['current_eligibility_receipts']=current_eligibility
    data['bounded_closure']=bounded_closure
    data['counts']['research_port_status']=dict(Counter(x['port_status'] for x in rows if x['category'] not in ('PUBLIC_CONTROL','API_EXPERIMENTAL')))
    data['counts']['quality_status']=dict(Counter(x['quality_status'] for x in rows))
    data['counts']['api_experimental_cpu_status']=dict(Counter(x['cpu_preflight']['status'] for x in rows if x['category']=='API_EXPERIMENTAL'))
    data['counts']['api_independent_initialization_status']=dict(Counter(x.get('independent_initialization_audit',{}).get('status','PENDING') for x in rows if x['category']=='API_EXPERIMENTAL'))
    assert sum(x['category']=='API_EXPERIMENTAL' for x in rows)==46 and sum(x['category']=='PUBLIC_CONTROL' for x in rows)==3
    assert len({x['id'] for x in rows})==len(rows)
    atomic(HERE/'retest-queue.json',data)
    text=['# Fixed-initialization retest queue','',f"Updated {data['recorded_utc']}. This is the progress/coverage authority; historical scores remain in their original ledgers. No entry is promoted by being ported or CPU-checked.",'','## Public controls and all46 API configurations','', '| Candidate | Source | Initialization port | CPU construction | Execution | Quality audit |','|---|---|---|---|---|---|']
    for r in rows:
        if r['category'] in ['PUBLIC_CONTROL','API_EXPERIMENTAL']:text.append(f"| {r['candidate']} | {r['source_status']} | {r['port_status']} | {r['cpu_preflight']['status']} | {r['execution_status']} | {r['quality_status']} |")
    followups=[(r,v) for r in rows for v in r.get('audited_followups',[])]
    if followups:
        text+=['','## Audited followups','','The original screen score is preserved. A failure on another frozen task prevents qualification.','','| Candidate | Task | Quality | Passing observations | Final passing suffix | Failed final bounds |','|---|---|---|---|---|---|']
        for row,result in followups:
            bounds='; '.join(f"{v['metric']}={v['value']:.7g} (requires {v['op']} {v['threshold']})" for v in result['failed_bounds'])
            text.append(f"| {row['candidate']} | {result['task']} | {result['quality_status']} | {result['passing_observations']}/{result['observations']} | {result['passing_suffix']} | {bounds} |")
    text+=['','## Research coverage and source recovery','','These entries map old scored rows and aliases; overlapping rows are not requests for duplicate identical runs. Research-host results remain separate from public API evidence. Missing historical binding stays visible while ready ports proceed. Source-disqualified scheduled configurations retain that eligibility status even if new-init quality improves.','','| Candidate | Historical identity | Source | Port / execution | Quality |','|---|---|---|---|---|']
    for r in rows:
        if r['category'] not in ['PUBLIC_CONTROL','API_EXPERIMENTAL']:text.append(f"| {r['candidate']} | {r['id']} | {r['source_status']} | {r['port_status']} / {r['execution_status']} | {r['quality_status']} |")
    eligibility_rows=[r for r in rows if r.get('current_configuration_eligibility')]
    if eligibility_rows:
        text+=['','## Current source eligibility','','Quality scores and historical labels remain unchanged. These findings apply to the exact retested configuration.','','| Candidate | Earned quality | Current source eligibility |','|---|---|---|']
        for r in eligibility_rows:text.append(f"| {r['candidate']} | {r['quality_status']} | {r['current_configuration_eligibility']} |")
    pruned=[r for r in rows if r.get('further_ring_preparation_status',{}).get('ring')=='NOT_RUN']
    if pruned:
        text+=['','Further ring, vector and long-run qualifications are **NOT_RUN** for '+', '.join(r['candidate'] for r in pruned)+': each has an independently audited failure on its own new-initializer bars4 task. Prepared sources and CPU receipts remain available; they do not authorize additional execution.']
    if bounded_closure:text+=['','Search is STOPPED at the user’s request. The [partial coverage closure](retest-closure/README.md) records75 audited research cases and22 NOT_RUN_USER_STOP cases from97 reviewed preparations, plus89 additional untested definition rows,4 exact duplicates and15 aliases. No unrun result is inferred as a failure or pass. Retained source without a reviewed adapter is unfinished work, not missing or impossible source. No new launches are authorized.']
    text+=['','Full exact source/config links, package hashes, prior evidence pointers, CPU receipt hashes, launch PIDs and terminal-result pointers are in [retest-queue.json](retest-queue.json). The old decayed KA2 arm is retained as a historical conditional diagnostic, separately from default KA2 and the mandatory constant arm; it is not an additional authorized launch after bounded closure.','','Refresh bookkeeping without training: `python reports/toy100/deterministic-init-retest/refresh_retest_queue.py`. Additional already-authorized batch receipts can be passed with repeated `--batch PATH`. The script reads only the specified batch process/artifact receipts, standard-library source ZIPs and existing CPU reports.']
    (HERE/'retest-queue.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(data['counts'],indent=2))

if __name__=='__main__':main()
