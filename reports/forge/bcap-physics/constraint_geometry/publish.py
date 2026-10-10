"""Publish certified, saved observations; adds no updates or random draws."""
from pathlib import Path
import json
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.tier1_media import render, _scored_outputs
from reports.forge.regenerate_technique_inventory import project_receipt

QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue')
OUT = Path(__file__).resolve().parent
REQUESTS = {'candidate': '7d22bb9602d65151123f8494', 'control': 'f30bde03964b05286609b536'}


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
        raise ValueError('both declared recipes must finish before final publication')
    rows, receipts, media, proof = [], [], [], []
    for role, rid in REQUESTS.items():
        request = state['submissions'][rid]['request']
        for job in state['jobs'].values():
            if rid not in job['subscribers']:
                continue
            if not job.get('result'):
                rows.append(dict(role=role, task_id=job['definition']['task_id'], gate_status='BLOCKED',
                                 reason=job.get('reason', 'own prerequisite did not pass'), metrics={}))
                continue
            aid = job['result']['attempt_id']
            durable = ROOT / 'reports/forge/attempts' / aid
            central = Path('/home/martyn/dev/ParticleGAN/reports/forge/attempts') / aid
            durable.mkdir(parents=True, exist_ok=True)
            for name in ('request', 'result', 'evidence'):
                if not (durable / (name + '.json')).exists():
                    shutil.copyfile(central / (name + '.json'), durable / (name + '.json'))
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
                            paid_seconds=result.get('raw',{}).get('seconds', row.get('cost',{}).get('runner_seconds')),
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
                samples, inputs = _scored_outputs(task, evidence, local)
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
                proof.append(dict(role=role,task_id=row['task_id'],attempt_id=aid,
                                  source_digest=request['source']['digest'],runtime=request['runtime'],
                                  recipe=row.get('recipe',row.get('applied',{}).get('recipe')),
                                  initialization=row.get('initialization',row.get('applied',{}).get('initialization')),
                                  data_sha256=evidence.get('data_sha256'),
                                  named_stream_state_sha256=(descriptor or {}).get('named_stream_state_sha256'),
                                  original_certificates={n:file_hash(durable/(n+'.json')) for n in ('request','result','evidence')},
                                  retained_inputs=inputs,provenance_checkpoint=descriptor))
                rows.append(item)
    for task in sorted({x['task_id'] for x in proof}):
        pair=[x for x in proof if x['task_id']==task]
        if len(pair)==2:
            assert pair[0]['source_digest']==pair[1]['source_digest']
            assert pair[0]['runtime']==pair[1]['runtime']
            if pair[0]['initialization'] and pair[1]['initialization']:
                assert pair[0]['initialization']==pair[1]['initialization'], task
            if pair[0]['recipe'] and pair[1]['recipe']:
                recipes=[dict(item['recipe']) for item in pair]
                for recipe in recipes:recipe.pop('constraint_geometry_mode',None)
                assert recipes[0]==recipes[1], task
            if pair[0]['data_sha256'] and pair[1]['data_sha256']:
                assert pair[0]['data_sha256']==pair[1]['data_sha256'], task
    summary=dict(schema_version=1,qualification_input=False,scope='research_diagnostic',
                 candidate='constraint_geometry-nonascent-v1',control='constraint_geometry-control-v1',
                 requests=REQUESTS,source_digest=state['submissions'][REQUESTS['candidate']]['request']['source']['digest'],
                 candidate_revision=state['submissions'][REQUESTS['candidate']]['request']['candidate_revision'],
                 control_revision=state['submissions'][REQUESTS['control']]['request']['candidate_revision'],
                 reservation_ceiling=12840,campaign_accounting=state['campaigns']['constraint_geometry-round-v1'],
                 outcomes={role:{status:sum(r['role']==role and r['gate_status']==status for r in rows) for status in ('PASS','FAIL','BLOCKED','INCOMPLETE','INVALID')} for role in REQUESTS},
                 task_results=rows,optimizer_updates_added_by_publication=0,sampling_draws_added_by_publication=0)
    atomic_json(OUT/'results.json',summary)
    atomic_json(OUT/'receipts.json',receipts)
    atomic_json(OUT/'provenance.json',dict(schema_version=1,qualification_input=False,proofs=proof))
    atomic_json(OUT/'media/index.json',dict(schema_version=1,qualification_input=False,media=media,optimizer_updates_added=0,sampling_draws_added=0))
    print(json.dumps(dict(outcomes=summary['outcomes'],media=len(media),paid_seconds=summary['campaign_accounting']['spent_seconds']),default=str))


if __name__=='__main__':main()
