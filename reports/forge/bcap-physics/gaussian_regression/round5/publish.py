"""Publish certified, saved observations; adds no updates or random draws."""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.tier1_media import render, _scored_outputs
from reports.forge.regenerate_technique_inventory import project_receipt

QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/queue')
OUT = Path(__file__).resolve().parent
REQUESTS = read_json(Path('/tmp/bcap-physics-round5-20261009/gaussian_regression/progress.json'))['requests']
DELTAS = {'constraint_geometry_mode'}


def render_saved(task, row, local, gif):
    """Existing host metadata may contain lists; display scalar metrics only."""
    from unittest.mock import patch
    from benchmarks.toy_audit import api_run
    original=api_run.render_gif
    def scalar_display(case, records, *args, **kwargs):
        records=[{**r, 'metrics': {k:v for k,v in r['metrics'].items() if type(v) in (int,float)}} for r in records]
        return original(case,records,*args,**kwargs)
    with patch.object(api_run,'render_gif',scalar_display):
        receipt=render(task,row,local,gif)
    receipt['display_metrics']='Saved scalar metrics; structured scorer metadata remains in original evidence.'
    atomic_json(gif.with_suffix('.json'),receipt)
    return receipt


def pending():
    state = read_json(QUEUE / 'queue/state.json')
    return [rid for rid in REQUESTS.values() if state['submissions'][rid]['status'] in {'queued', 'running', 'paused'}]


def main():
    torch.set_num_threads(1)
    state = read_json(QUEUE / 'queue/state.json')
    if pending():
        raise ValueError('all three declared recipes must finish before final publication')
    rows, receipts, media, proof = [], [], [], []
    for role, rid in REQUESTS.items():
        request = state['submissions'][rid]['request']
        for job in state['jobs'].values():
            if rid not in job['subscribers']:
                continue
            if not job.get('result'):
                rows.append(dict(role=role, task_id=job['definition']['task_id'], gate_status='BLOCKED',
                                 reason=request['tasks'][job['definition']['task_id']].get('preflight_blockers') or job.get('reason', 'own prerequisite did not pass'), metrics={}))
                continue
            aid = job['result']['attempt_id']
            durable = ROOT / 'reports/forge/attempts' / aid
            assert all((durable / (name + '.json')).exists() for name in ('request', 'result', 'evidence')), aid
            compact = project_receipt(ROOT, aid)
            receipts.append(compact)
            envelope, result, certificate = [read_json(durable / (n + '.json')) for n in ('request', 'result', 'evidence')]
            assert certificate['result_hash'] == stable_hash(result)
            assert certificate['source'] == request['source']
            local = Path(certificate['local_artifact_root'])
            assert read_json(local / 'result.json') == result
            for row in result['task_results']:
                task = request['tasks'][row['task_id']]
                evidence = row['evidence']
                item = dict(role=role, task_id=row['task_id'], attempt_id=aid, gate_status=row['gate_status'],
                            metrics=row.get('metrics', {}), evaluator_summary=next(x['evaluator_summary'] for x in compact['task_results'] if x['task_id']==row['task_id']),
                            paid_seconds=next(x['seconds'] for x in state['charges'] if x['attempt_id']==aid),
                            cost=row.get('cost',{}), sampling={k:evidence.get(k) for k in ('sampling_law','eval_output_noise','scoring_weights')})
                descriptor=evidence.get('provenance_checkpoint')
                if descriptor:
                    root = Path(descriptor['artifact_root'])
                    verify_artifacts(root, descriptor['artifact_manifest'])
                    path = root / descriptor['path'];assert file_hash(path)==descriptor['sha256']
                    saved=torch.load(path, map_location='cpu', weights_only=False)
                    if row['execution_path']=='public_components':
                        item['constraint_geometry']=[opt['constraint_geometry']['stats'] for opt in saved['optimizers']['generator'] if 'constraint_geometry' in opt]
                    else:
                        # GANTrainer state is wrapped by the scalar protocol.
                        def find(value):
                            if isinstance(value,dict):
                                if 'constraint_geometry' in value: return [value['constraint_geometry']['stats']]
                                return [out for sub in value.values() for out in find(sub)]
                            if isinstance(value,(list,tuple)):return [out for sub in value for out in find(sub)]
                            return []
                        item['constraint_geometry']=find(saved)
                def progress_stats(value):
                    if isinstance(value, dict):
                        return [{**value[key]['stats'], 'mode':key} for key in ('strict_progress','direction_blend') if key in value] + [x for sub in value.values() for x in progress_stats(sub)]
                    if isinstance(value, (list, tuple)):
                        return [x for sub in value for x in progress_stats(sub)]
                    return []
                item['strict_progress'] = progress_stats(saved) if descriptor else []
                samples, inputs = _scored_outputs(task, evidence, local)
                if samples and 'samples' in samples[0]:
                    scalar=row['task_id'].startswith('gaussian')
                    if scalar:
                        from benchmarks.toy_audit.gaussian1d_quality import score_samples
                    else:
                        from benchmarks.transfer_suite.vector_tasks import score_samples
                    spec={**task['execution']['host_definition'],'thresholds':task['evaluation']['thresholds']}
                    for record,observed in zip(samples,evidence['observations']):
                        if scalar and row['task_id']=='gaussian1d_stability' and record['step']>4000:
                            spec['means']=[[3.]]
                        rescored=score_samples(record['samples'],spec,record['step'])
                        assert rescored=={k:v for k,v in observed.items() if k!='step'},(role,row['task_id'],record['step'])
                    item['metric_sets_reproduced']=len(samples)
                if samples and 'views' in samples[0]:
                    predictions=torch.stack([x['views'][0]['samples'].float() for x in samples])
                    item['scored_prediction_motion']=dict(intervals=len(predictions)-1,
                        mean_rms=float((predictions[1:]-predictions[:-1]).square().flatten(1).mean(1).sqrt().mean()),
                        maximum_rms=float((predictions[1:]-predictions[:-1]).square().flatten(1).mean(1).sqrt().max()),
                        note='Movement between scored states; not per-step accepted norm or projection fraction.')
                    if row['task_id'] in ('trajectory','residual_student'):
                        from benchmarks.locked_shared.trajectory import identity_mse
                        from benchmarks.locked_shared.hosts.residual_student import landing_stats
                        target=samples[-1]['views'][0]['target'].reshape(12,16)
                        prediction=samples[-1]['views'][0]['samples'].reshape(12,16)
                        recomputed=identity_mse(prediction,target)
                        assert abs(recomputed-item['metrics']['identity_mse'])<1e-7
                        distances=(prediction[:,None,:]-target[None,:,:]).square().mean(2)
                        item['conditional_endpoint_diagnostics']=dict(
                            per_row_identity_mse=[float(x) for x in (prediction-target).square().mean(1)],
                            nearest_target_row=[int(x) for x in distances.argmin(1)],
                            nearest_target_is_own=int((distances.argmin(1)==torch.arange(12)).sum()),
                            note='Saved full conditional panel; diagnostic allocation, not replacement gate.')
                        mse_curve=[point['identity_mse'] for point in evidence['observations']]
                        item['identity_mse_increasing_intervals']=sum(b>a for a,b in zip(mse_curve,mse_curve[1:]))
                        permutation=target.roll(1,0)
                        controls=dict(oracle=dict(identity_mse=identity_mse(target,target)),
                                      wrong_identity=dict(identity_mse=identity_mse(permutation,target)))
                        if row['task_id']=='residual_student':
                            controls['oracle'].update(landing_stats(target,target));controls['wrong_identity'].update(landing_stats(permutation,target))
                        def passed(values):
                            return all(values[name]<=bound if op=='<=' else values[name]>=bound if op=='>=' else values[name]==bound
                                       for name,op,bound in task['evaluation']['thresholds'])
                        controls.update(oracle_pass=passed(controls['oracle']),wrong_identity_pass=passed(controls['wrong_identity']),
                                        information_access='target-informed scorer controls only; not learned baseline or alternate initializer')
                        assert controls['oracle_pass'] and not controls['wrong_identity_pass']
                        assert controls['oracle']['identity_mse']==0 and controls['wrong_identity']['identity_mse']>.02
                        item['scorer_controls']=controls
                gif=OUT/'media'/f'{role}-{row["task_id"]}.gif'
                if gif.exists() and gif.with_suffix(".json").exists():
                    artifact=read_json(gif.with_suffix(".json"))
                    assert artifact["gif_sha256"]==file_hash(gif)
                    assert artifact["observations_sha256"]==stable_hash(evidence["observations"])
                else:
                    artifact=render_saved(task,row,local,gif)
                media.append({**artifact,'role':role,'task_id':row['task_id'],'gif':str(gif.relative_to(OUT))})
                from audit import data_proof
                proof.append(dict(role=role,task_id=row['task_id'],attempt_id=aid,
                                  source_digest=request['source']['digest'],runtime=request['runtime'],
                                  recipe=row.get('recipe',row.get('applied',{}).get('recipe')),
                                  initialization=row.get('initialization',row.get('applied',{}).get('initialization')),
                                  data_sha256=data_proof(task,row,saved),
                                  named_stream_state_sha256=(descriptor or {}).get('named_stream_state_sha256'),
                                  original_certificates={n:file_hash(durable/(n+'.json')) for n in ('request','result','evidence')},
                                  retained_inputs=inputs,provenance_checkpoint=descriptor))
                rows.append(item)
    matched=[]
    for task in sorted({x['task_id'] for x in proof}):
        group=[x for x in proof if x['task_id']==task]
        first=group[0]
        for row in group[1:]:
            assert first['source_digest']==row['source_digest'],task
            assert first['runtime']==row['runtime'],task
            assert first['initialization']==row['initialization'],task
            assert first['data_sha256']==row['data_sha256'],task
            assert first['named_stream_state_sha256']==row['named_stream_state_sha256'],task
            recipes=[{k:v for k,v in x['recipe'].items() if k not in DELTAS} for x in [first,row]]
            assert recipes[0]==recipes[1],task
        matched.append(dict(task_id=task,arms=[x['role'] for x in group],source_runtime_equal=True,
                            initialization_equal=True,data_batch_digest_equal=True,consumed_named_streams_equal=True,
                            base_effective_recipe_equal=True))
    campaign=state['campaigns']['gaussian_regression-round5-v1']
    assert campaign['reserved_seconds']==0
    summary=dict(schema_version=1,qualification_input=False,scope='research_diagnostic',
                 arms={role:dict(candidate_id=state['submissions'][rid]['request']['candidate']['id'],
                                revision=state['submissions'][rid]['request']['candidate_revision']) for role,rid in REQUESTS.items()},
                 requests=REQUESTS,source_digest=first['source_digest'],
                 source_commit=state['submissions'][REQUESTS['local']]['request']['source']['origin_commit'],
                 reservation_ceiling=21180,planned_full_reservations=18360,
                 executed_full_reservations=sum(j['definition']['budget_seconds']*len(j['attempts']) for j in state['jobs'].values()),
                 campaign_accounting=campaign,paid_seconds=campaign['spent_seconds'],
                 scientific_attempts=sum(len(j['attempts']) for j in state['jobs'].values()),
                 scientific_retries=sum(max(0,len(j['attempts'])-1) for j in state['jobs'].values()),
                 outcomes={role:{status:sum(r['role']==role and r['gate_status']==status for r in rows) for status in ('PASS','FAIL','BLOCKED','INCOMPLETE','INVALID')} for role in REQUESTS},
                 task_results=rows,optimizer_updates_added_by_publication=0,sampling_draws_added_by_publication=0)
    atomic_json(OUT/'results.json',summary)
    atomic_json(OUT/'receipts.json',receipts)
    atomic_json(OUT/'provenance.json',dict(schema_version=1,qualification_input=False,matched_conditions=matched,proofs=proof))
    atomic_json(OUT/'media/index.json',dict(schema_version=1,qualification_input=False,media=media,optimizer_updates_added=0,sampling_draws_added=0))
    print(json.dumps(dict(outcomes=summary['outcomes'],media=len(media),paid_seconds=summary['campaign_accounting']['spent_seconds']),default=str))


if __name__=='__main__':main()
