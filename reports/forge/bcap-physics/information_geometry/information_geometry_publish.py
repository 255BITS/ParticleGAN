"""Publish certified saved outputs for the one bounded spectral-half comparison.

No model construction, training, sampling, or scientific regrading occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import _scored_outputs, render
import importlib.util

_spec = importlib.util.spec_from_file_location("information_geometry_image_renderer", ROOT / "reports/forge/bcap-convolution/publish.py")
_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_images)
render_image = _images.render_image
from reports.forge.regenerate_technique_inventory import project_receipt

CAMPAIGN = 'information_geometry_campaign_v1'
CANDIDATE = 'information_geometry_spectral_half_v1'
CONTROL = 'bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36'


def image_diagnostics(task, row, local):
    """Posthoc matched nearest-template region errors; original gate unchanged."""
    import torch
    records, _ = _scored_outputs(task, row['evidence'], local)
    sample, targets = records[-1]['samples'].double(), records[-1]['targets'].double()
    error = sample[:, None] - targets[None, :]
    assignment = error.square().flatten(2).mean(2).argmin(1)
    target = targets[assignment]
    difference = sample - target
    background = target.abs() <= .001
    signal = ~background
    def rms(mask):
        return float(difference[mask].square().mean().sqrt()) if bool(mask.any()) else None
    return {'scope':'posthoc exact saved endpoint outputs; nearest-template assignment, zero training/sampling, no gate change',
            'background_mask':'absolute target pixel <= .001',
            'background_rmse':rms(background), 'signal_rmse':rms(signal),
            'background_signed_bias':float(difference[background].mean()) if bool(background.any()) else None,
            'signal_signed_bias':float(difference[signal].mean()) if bool(signal.any()) else None,
            'assigned_template_counts':torch.bincount(assignment,minlength=len(targets)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root', type=Path, required=True)
    parser.add_argument('--certificate-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    state = read_json(args.queue_root / 'queue/state.json')
    entries = [v for v in state['submissions'].values() if v['request'].get('campaign_id') == CAMPAIGN]
    assert len(entries) == 2 and {v['request']['candidate']['id'] for v in entries} == {CANDIDATE, CONTROL}
    assert all(v['status'] not in {'queued','running','paused'} for v in entries)
    assert state['campaigns'][CAMPAIGN]['reserved_seconds'] == 0
    requests = {v['request']['candidate']['id']:v['request'] for v in entries}
    assert requests[CANDIDATE]['source']['digest'] == requests[CONTROL]['source']['digest']
    assert requests[CANDIDATE]['runtime'] == requests[CONTROL]['runtime']
    assert all(v['protocol']['seed'] == 0 for v in requests.values())
    rows, receipts, media, audits = [], [], [], []
    parity = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in sorted((args.queue_root / CAMPAIGN).iterdir()):
        if not directory.is_dir() or not (directory / 'result.json').exists():
            continue
        envelope = read_json(directory / 'request.json')
        request, candidate = envelope['request'], envelope['request']['candidate']['id']
        assert candidate in requests and request['source']['digest'] == requests[candidate]['source']['digest']
        receipt = project_receipt(args.certificate_root, directory.name)
        assert receipt['certificate_validated'] and receipt['attempt_status'] == 'completed'
        receipts.append(receipt)
        result, raw = read_json(directory / 'result.json'), read_json(directory / 'raw-result.json')
        applied = raw.get('applied', raw)
        recipe = {k:v['value'] for k,v in applied.get('field_ownership',{}).get('recipe_fields',{}).items() if v.get('status') == 'effective'}
        recipe.update(applied['recipe'])
        assert recipe['optimizer_family'] == ('information_geometry_spectral_half' if candidate == CANDIDATE else 'dualnorm')
        for key, expected in {'loss':'non_saturating','lr':.012,'d_lr_mult':1.5,'prior_lr_mult':2.5,
                              'optimizer_smoothing':.001,'optimizer_momentum':0.,'optimizer_convolution':'per_offset',
                              'reg_arm':'b_cap','reg_coeff':1.,'reg_kappa':1.,'reg_every':1,
                              'ema_decay':0.,'input_noise_std':0.,'output_noise_std':0.}.items():
            assert recipe[key] == expected, (candidate,key,recipe[key])
        for row in result['task_results']:
            task_id = row['task_id'];task = request['tasks'][task_id]
            compact = {'candidate_id':candidate,'task_id':task_id,'attempt_id':directory.name,
                       'status':row['gate_status'],'metrics':row.get('metrics',{}),'cost':row['cost'],
                       'evaluator_result':row.get('evaluator_result',{}),
                       'reasons':row.get('reasons',[]), 'full_gate':task['evaluation'],
                       'execution':task['execution'], 'resolved_recipe':recipe}
            # Do not duplicate source file dictionaries in the compact task record.
            compact['full_gate']={k:v for k,v in compact['full_gate'].items() if k!='sources'}
            rows.append(compact)
            proof=row['evidence']['provenance_checkpoint']
            assert file_hash(Path(proof['artifact_root'])/proof['path']) == proof['sha256']
            import torch
            checkpoint=torch.load(Path(proof['artifact_root'])/proof['path'],map_location='cpu',weights_only=False)
            def optimizer_metadata(value):
                found=[]
                if isinstance(value,dict):
                    if 'dualnorm' in value:
                        meta=value['dualnorm']
                        found.append({k:meta[k] for k in ['schema','family','momentum','smoothing','convolution'] if k in meta})
                        assert meta['family']==recipe['optimizer_family'] and meta['momentum']==0 and meta['smoothing']==.001
                    else:
                        for item in value.values(): found.extend(optimizer_metadata(item))
                elif isinstance(value,(list,tuple)):
                    for item in value: found.extend(optimizer_metadata(item))
                return found
            metadata=optimizer_metadata(checkpoint.get('trainer',checkpoint)['optimizers'])
            assert metadata
            stream_hashes=proof['named_stream_state_sha256']
            datum={'initialization':applied.get('initialization'),
                   'data_digest':row['evidence'].get('data_sha256'),
                   'data_streams':{k:v for k,v in stream_hashes.items() if '"data"' in k},
                   'rng_binding':applied.get('rng',raw.get('rng')),
                   'checkpoint':{k:proof[k] for k in ['path','sha256','state_sha256','completed_steps','named_stream_keys']},
                   'all_stream_state_sha256':stream_hashes,
                   'checkpoint_optimizer_metadata':metadata,
                   'resolved_recipe':recipe, 'task_execution_hash':stable_hash(task['execution']),
                   'task_evaluation_hash':stable_hash(task['evaluation'])}
            parity.setdefault(task_id,{})[candidate]=datum
            output=args.output/'media'/('candidate' if candidate==CANDIDATE else 'control')/(task_id+'.gif')
            output.parent.mkdir(parents=True,exist_ok=True)
            if output.exists() and output.with_suffix('.json').exists():
                rendered=read_json(output.with_suffix('.json'))
                assert rendered['gif_sha256']==file_hash(output) and rendered['observations_sha256']==stable_hash(row['evidence']['observations'])
                _scored_outputs(task,row['evidence'],directory)
            elif task['adapter']=='transfer_image':
                rendered=render_image(task,row,directory,output)
            else:
                rendered=render(task,row,directory,output)
            if task['adapter']=='transfer_image':
                compact['image_diagnostics']=image_diagnostics(task,row,directory)
            media.append({'candidate_id':candidate,'attempt_id':directory.name,'gif':str(output.relative_to(args.output)),**rendered})
            print({'event':'saved_media_exported','candidate':candidate,'task':task_id,'status':row['gate_status']},flush=True)
    for task_id, pair in sorted(parity.items()):
        if set(pair)!={CANDIDATE,CONTROL}:
            continue
        a,b=pair[CANDIDATE],pair[CONTROL]
        # Own Gaussian continuation states are different trained weights; initial
        # public tensors and named data consumption remain bound by smoke evidence.
        is_continuation=task_id=='gaussian1d_stability'
        assert a['initialization']==b['initialization'] or is_continuation
        assert a['data_streams']==b['data_streams']
        assert a['data_digest']==b['data_digest']
        assert a['rng_binding']==b['rng_binding']
        recipe_delta={k:(b['resolved_recipe'][k],v) for k,v in a['resolved_recipe'].items() if b['resolved_recipe'].get(k)!=v}
        assert set(recipe_delta)=={'optimizer_family'}, (task_id,recipe_delta)
        assert a['task_execution_hash']==b['task_execution_hash'] and a['task_evaluation_hash']==b['task_evaluation_hash']
        audits.append({'task_id':task_id,'initialization_equal':a['initialization']==b['initialization'],
                       'own_checkpoint_continuation':is_continuation,'data_end_streams_equal':True,
                       'data_digest_equal':True,'data_digest_present':a['data_digest'] is not None,
                       'initial_named_stream_bindings_equal':True,'consumed_recipe_delta':recipe_delta,'task_contracts_equal':True,'proofs':pair})
    measured={(r['candidate_id'],r['task_id']) for r in rows}
    for candidate,request in requests.items():
        for task in request['tasks']:
            if (candidate,task) not in measured:
                keys=[j['compatibility_key'] for j in request['jobs'] if task in j.get('task_ids',[j['task_id']])]
                jobs=[state['jobs'][k] for k in keys]
                rows.append({'candidate_id':candidate,'task_id':task,'status':'BLOCKED',
                             'reason':'Own required dependency did not pass; no training or relaxed fixture',
                             'queue_job_status':[j['status'] for j in jobs]})
    campaign=state['campaigns'][CAMPAIGN]
    summary={'schema_version':1,'scope':'research_diagnostic','qualification_input':False,
             'candidate_id':CANDIDATE,'control_id':CONTROL,'campaign_id':CAMPAIGN,
             'seed':0,'source_digest':requests[CANDIDATE]['source']['digest'],
             'source_origin_commit':requests[CANDIDATE]['source']['origin_commit'],
             'runtime':requests[CANDIDATE]['runtime'], 'study_ids':[e['request']['study']['id'] for e in entries],
             'request_ids':[e['request']['request_id'] for e in entries],
             'total_full_reservation_seconds':12840,'campaign_ceiling_seconds':14400,
             'paid_worker_seconds':campaign['spent_seconds'],'remaining_reserved_seconds':campaign['reserved_seconds'],
             'status_counts':{c:dict(Counter(r['status'] for r in rows if r['candidate_id']==c)) for c in requests},
             'tasks':rows,'scientific_retries':0,'publication_optimizer_updates_added':0,'publication_sampling_draws_added':0}
    atomic_json(args.output/'results.json',summary)
    atomic_json(args.output/'provenance.json',{'schema_version':1,'qualification_input':False,'attempts':receipts,
                'matched_conditions':audits,'queue_root':str(args.queue_root),'certificate_root':str(args.certificate_root),
                'publication_source_sha256':file_hash(Path(__file__))})
    atomic_json(args.output/'media/index.json',{'schema_version':1,'qualification_input':False,'items':media})
    print({'event':'publication_complete','paid_worker_seconds':summary['paid_worker_seconds'],'counts':summary['status_counts']},flush=True)


if __name__=='__main__':
    main()
