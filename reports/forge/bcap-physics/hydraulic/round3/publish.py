"""Publish compact source-bound metrics and existing-observation GIFs only."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import render
from experiments.forge.execution_policy import group_blockers
from reports.forge.regenerate_technique_inventory import _evaluator_summary

CAMPAIGN = 'hydraulic-local-shape-round3-v1'
REPORT = ROOT/'reports/forge/bcap-physics/hydraulic/round3'
CONTROL = 'bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36'
CANDIDATE = 'hydraulic-local-shape-v3'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,required=True)
    parser.add_argument('--media',action='store_true')
    args=parser.parse_args()
    state=read_json(args.queue_root/'queue/state.json')
    entries={k:e for k,e in state['submissions'].items() if e['request']['campaign_id']==CAMPAIGN}
    if len(entries)!=2:
        raise ValueError('exactly the candidate and matched winner requests are required')
    requests=[e['request'] for e in entries.values()]
    if len({r['source']['digest'] for r in requests})!=1 or len({stable_hash(r['runtime']) for r in requests})!=1:
        raise ValueError('matched scientific source/runtime mismatch')
    frozen_source=requests[0]['source']
    if any(not (ROOT/name).is_file() or file_hash(ROOT/name)!=digest
           for name,digest in frozen_source['files'].items()):
        raise ValueError('publication checkout differs from the actual executed scientific files')
    by_candidate={}
    receipts=[]
    media=[]
    complete=True
    spec=importlib.util.spec_from_file_location('hydraulic_saved_native_renderer',ROOT/'reports/forge/gaussian-smoke-inventory/export_media.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    for request_id, entry in entries.items():
        request=entry['request'];name=request['candidate']['id'];label='candidate' if name==CANDIDATE else 'control'
        rows=[]
        executed={row['task_id']:row for definition in request['jobs']
                  for row in (state['jobs'][definition['compatibility_key']].get('result') or {}).get('task_results',[])}
        for declaration in request['jobs']:
            job=state['jobs'][declaration['compatibility_key']]
            blockers=group_blockers(request,declaration)
            if blockers:
                for task_id in declaration['task_ids']:
                    rows.append(dict(task_id=task_id,gate_status='BLOCKED',raw_status='NOT_RUN',
                                     reason='; '.join(blockers),paid_seconds=0.,attempted=False))
                continue
            # A diagnostic request can finish with a dependency-disabled global
            # job still pending. Report the actual failed prerequisite; do not
            # mutate queue state or invent an attempt/evidence receipt.
            members=set(declaration['task_ids'])
            dependencies={d['task'] if isinstance(d,dict) else d for member in members
                          for d in request['tasks'][member].get('dependencies',[])}-members
            failed=[d for d in sorted(dependencies) if d in executed and executed[d]['gate_status']!='PASS']
            if not job.get('result') and failed:
                for task_id in declaration['task_ids']:
                    rows.append(dict(task_id=task_id,gate_status='BLOCKED',raw_status='NOT_RUN',
                        reason='prerequisites are unsatisfied: '+', '.join(d+': '+executed[d]['gate_status'] for d in failed),
                        queue_job_status=job['status'],paid_seconds=0.,attempted=False))
                continue
            if job['status'] not in {'terminal','blocked'}:
                complete=False
            result=job.get('result')
            if not result:
                for task_id in declaration['task_ids']:
                    rows.append(dict(task_id=task_id,gate_status='BLOCKED' if job['status']=='blocked' else 'INCOMPLETE',raw_status='NOT_RUN',status=job['status'],attempted=False,paid_seconds=0.,blockers=job.get('blockers'),reason=job.get('reason')))
                continue
            identity=result['attempt_id']
            certificate_dir=ROOT/'reports/forge/attempts'/identity
            for row in result['task_results']:
                task_id=row['task_id'];evidence=row.get('evidence',{})
                compact={k:row.get(k) for k in ('task_id','gate_status','raw_status','reason','reasons','metrics','evaluator_result','cost','initializer','initialization','extensions','prior')}
                compact['evaluator_result']=_evaluator_summary(row.get('evaluator_result',{}))
                compact.update(attempt_id=identity,compatibility_key=row['compatibility_key'],recipe=row.get('recipe'),
                               hydraulic=evidence.get('hydraulic'),guards=evidence.get('guards'),
                               host=evidence.get('host'),data_sha256=evidence.get('data_sha256'))
                end_hashes=evidence.get('provenance_checkpoint',{}).get('named_stream_state_sha256',{})
                compact['stream_proof']={'/'.join(binding[k] for k in ('family','component','purpose')):
                    dict(seed=binding['seed'],initial=binding['initial_state_sha256'],final=end_hashes.get(key))
                    for key,binding in row.get('rng',{}).get('bindings',{}).items()}
                rows.append(compact)
                receipt=dict(attempt_id=identity,task_id=task_id,candidate_id=name,
                    candidate_revision=result['candidate_revision'],source_digest=request['source']['digest'],
                    runtime_sha256=stable_hash(request['runtime']),protocol_sha256=stable_hash(request['protocol']),
                    compatibility_key=row['compatibility_key'],result_sha256=stable_hash(result),
                    artifact_root=evidence.get('artifact_root'),provenance_checkpoint=evidence.get('provenance_checkpoint'))
                if certificate_dir.exists():
                    cert=read_json(certificate_dir/'evidence.json')
                    envelope=read_json(certificate_dir/'request.json')
                    certified_request=envelope.get('request',envelope)
                    assert cert['result_hash']==stable_hash(result)
                    assert cert['source']==certified_request['source']==request['source']
                    assert cert['runtime']==request['runtime']
                    assert read_json(Path(cert['local_artifact_root'])/'result.json')==result
                    receipt['certificate_files']={p.name:file_hash(p) for p in certificate_dir.glob('*.json')}
                    receipt['local_artifact_root']=cert['local_artifact_root']
                    local=Path(cert['local_artifact_root'])
                else:
                    local=None
                receipts.append(receipt)
                if args.media and evidence and row['gate_status'] in {'PASS','FAIL'}:
                    if local is None:
                        raise ValueError('certified local observations are required for GIFs')
                    destination=REPORT/'media'/label/(task_id+'.gif')
                    if request['tasks'][task_id]['adapter']=='native100':
                        media_receipt=renderer.render_native(request['tasks'][task_id],row,destination)
                    else:
                        media_receipt=render(request['tasks'][task_id],row,local,destination)
                    media.append(dict(candidate_id=name,attempt_id=identity,gif=destination.relative_to(REPORT).as_posix(),**media_receipt))
        by_candidate[label]=dict(candidate_id=name,candidate_revision=request['candidate_revision'],request_id=request_id,
            status=entry['status'],source_origin_commit=request['source']['origin_commit'],
            outcomes=dict(Counter(r.get('gate_status',r.get('status')) for r in rows)),tasks=rows)
    parity=[]
    candidate_rows={r['task_id']:r for r in by_candidate['candidate']['tasks']}
    control_rows={r['task_id']:r for r in by_candidate['control']['tasks']}
    for task_id in candidate_rows.keys() & control_rows.keys():
        a,b=candidate_rows[task_id],control_rows[task_id]
        if 'stream_proof' not in a or 'stream_proof' not in b:
            continue
        checks={name:a.get(name)==b.get(name) for name in ('initialization','recipe','prior','host','data_sha256')}
        # Only comparing the initial host description; host contains no outcome.
        checks['task_declaration']=requests[0]['tasks'][task_id]==requests[1]['tasks'][task_id]
        stream_names=set(a['stream_proof']) & set(b['stream_proof'])
        training_streams={name:a['stream_proof'][name]==b['stream_proof'][name]
                          for name in stream_names if not name.startswith('eval/')}
        checks['consumed_training_streams']=all(training_streams.values())
        parity.append(dict(task_id=task_id,checks=checks,training_streams=training_streams))
        if not all(checks.values()):
            raise ValueError('matched condition/stream mismatch: '+task_id)
    output=dict(schema_version=1,study_id='hydraulic-local-shape-candidate-round3-v1',campaign_id=CAMPAIGN,
        qualification_input=False,scope='bounded_local_finite_shape_diagnostic',complete=complete,
        source_digest=requests[0]['source']['digest'],runtime=requests[0]['runtime'],
        compute_profiles=requests[0]['compute_profiles'],protocol=requests[0]['protocol'],
        campaign=state['campaigns'][CAMPAIGN],comparison=by_candidate,matched_parity=parity)
    atomic_json(REPORT/'results.json',output)
    atomic_json(REPORT/'provenance.json',dict(schema_version=1,receipts=receipts,
        trained_source=dict(digest=frozen_source['digest'],origin_commit=frozen_source['origin_commit'],
                            verified_file_count=len(frozen_source['files']),publication_scientific_files_unchanged=True),
        publication_updates_added=0,publication_sampling_draws_added=0,software_checks='Focused software checks recorded in README; no scientific qualification'))
    if args.media:
        atomic_json(REPORT/'media/index.json',dict(schema_version=1,items=media,optimizer_updates_added=0,sampling_draws_added=0,qualification_input=False))
    print({'complete':complete,'outcomes':{k:v['outcomes'] for k,v in by_candidate.items()},
           'paid_worker_seconds':output['campaign']['spent_seconds'],'reserved_seconds':output['campaign']['reserved_seconds']},flush=True)

if __name__=='__main__':main()
