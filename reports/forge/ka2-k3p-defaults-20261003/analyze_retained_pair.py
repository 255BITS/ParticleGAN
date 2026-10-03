"""Draw-free diagnostics of two immutable failed image attempts.

Reads existing sample arrays and checkpoint dictionaries. No public fixture,
model, sampler, restore, scorer or optimizer is called. Original recorded
scientific verdicts are copied, never replaced by the descriptive partitions.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
RAW = Path('/ml2/hypergan/forge-ka2-k3p-defaults-20261003')
CASE_ID = 'image-develop-img_intensity2-source-transpose12'
COMMIT = '26ff278c3796d775969391adc0bde52e3af11149'
EXPECTED = {
    'ka2': {'study': '58721647a201b569c1ac0a992880d410ca5375836986807afbc547397e89aabf',
            'receipt': 'b60afb0fd92a55d3cd97dc5bdeaeb53af5004061653cc1722cf7a96b77f7aeb7'},
    'k3p': {'study': '385a48da6efb236c9b2a388e5eb8fbebf43c0add3d1a34270c36ad71be749461',
            'receipt': '2cb069fdd6cf3560385984f3a9d35d651e0d0df3e1aca28d864ea0582679c480'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def finite_summary(value):
    total = nonfinite = tensors = 0
    def walk(v):
        nonlocal total, nonfinite, tensors
        if isinstance(v, torch.Tensor):
            tensors += 1
            if v.is_floating_point():
                total += v.numel()
                nonfinite += int((~torch.isfinite(v)).sum())
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v: walk(x)
    walk(value)
    return dict(tensors=tensors, floating_elements=total, nonfinite_floating_elements=nonfinite)


def differences(left, right):
    changes, maxdiff, tensor_count, scalar_equal = [], 0.0, 0, 0
    def walk(a, b, path):
        nonlocal maxdiff, tensor_count, scalar_equal
        if type(a) is not type(b):
            changes.append(dict(path=path, kind='type')); return
        if isinstance(a, torch.Tensor):
            tensor_count += 1
            if a.dtype != b.dtype or a.shape != b.shape:
                changes.append(dict(path=path, kind='shape_or_dtype')); return
            if not torch.equal(a, b):
                d = float((a.to(torch.float64)-b.to(torch.float64)).abs().max())
                maxdiff = max(maxdiff, d)
                changes.append(dict(path=path, kind='tensor', max_abs_difference=d,
                                    unequal_elements=int((a != b).sum())))
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b), key=str):
                if key not in a or key not in b:
                    changes.append(dict(path=path+'/'+str(key), kind='missing_key'))
                else: walk(a[key], b[key], path+'/'+str(key))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b): changes.append(dict(path=path, kind='length'))
            for i, (x, y) in enumerate(zip(a,b)): walk(x,y,path+'/'+str(i))
        elif a != b:
            changes.append(dict(path=path, kind='scalar', left=a, right=b))
        else: scalar_equal += 1
    walk(left,right,'')
    return dict(exact_equal=not changes, tensor_leaves=tensor_count,
                differing_leaves=len(changes), max_abs_tensor_difference=maxdiff,
                changes=changes, equal_scalar_leaves=scalar_equal)


def image_attribution(samples, target):
    # Descriptive arithmetic only. Keep the source's already-recorded gates.
    # Float64 distances avoid an additional float32 threshold ambiguity; no
    # rejected sample is close enough to the cutoff here for this to matter.
    unique, counts = np.unique(samples.reshape(len(samples), -1), axis=0, return_counts=True)
    unique = unique.reshape(-1,1,8,8)
    distance = np.sqrt(np.mean((unique[:,None].astype(np.float64)-target[None])**2,
                               axis=(2,3,4)))
    mode = distance.argmin(1)
    nearest = distance.min(1)
    accepted = nearest <= .06
    patch = unique[:,0,2:6,2:6].mean((1,2))
    background = np.concatenate((unique[:,0,:2].reshape(len(unique),-1),
                                 unique[:,0,6:].reshape(len(unique),-1),
                                 unique[:,0,2:6,:2].reshape(len(unique),-1),
                                 unique[:,0,2:6,6:].reshape(len(unique),-1)), axis=1).mean(1)
    return dict(distinct_outputs=len(unique),
                retained_draw_nearest_counts=np.bincount(mode, weights=counts, minlength=2).astype(int).tolist(),
                retained_draw_accepted_counts=np.bincount(mode[accepted],weights=counts[accepted],minlength=2).astype(int).tolist(),
                unique_output_nearest_counts=np.bincount(mode,minlength=2).tolist(),
                unique_output_accepted_counts=np.bincount(mode[accepted],minlength=2).tolist(),
                distinct_output_multiplicity_range=[int(counts.min()),int(counts.max())],
                full32_population_observed=len(unique)==32,
                rejected_unique_outputs=[dict(nearest_mode=int(mode[i]), draws=int(counts[i]),
                                               nearest_rmse=float(nearest[i]), central_patch_mean=float(patch[i]),
                                               background_mean=float(background[i]))
                                         for i in np.flatnonzero(~accepted)])


inputs, loaded, rows = {}, {}, []
for family in ('ka2','k3p'):
    study_path = RAW/family/'study.json'
    assert sha(study_path) == EXPECTED[family]['study']
    study = json.loads(study_path.read_text())
    trial = next(t for t in study['trials'] if t['family']==family)
    row = next(r for r in trial['cases'] if r['id']==CASE_ID)
    receipt_path=Path(row['receipt_path'])
    assert sha(receipt_path) == EXPECTED[family]['receipt'] == row['receipt_sha256']
    receipt=json.loads(receipt_path.read_text())
    assert receipt['source']['commit']==COMMIT and receipt['source_unchanged']
    assert receipt['status']=='COMPLETE' and receipt['completed_updates']==600
    assert receipt['default_protocol_complete'] and receipt['verdict']=='FAIL'
    assert [r['step'] for r in receipt['observations']]==list(range(0,601,25))
    files={'study':artifact(study_path),'receipt':artifact(receipt_path)}
    for name, pin in receipt['artifacts'].items():
        a=artifact(receipt_path.parent/name)
        assert a['sha256']==pin['sha256'] and a['bytes']==pin['bytes']
        files[name]=a
    inputs.update({x['path']:x['sha256'] for x in files.values()})
    with np.load(receipt_path.parent/'observations.npz',allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in z.files}
    state=torch.load(receipt_path.parent/'final-state.pt',map_location='cpu',weights_only=True)
    api=state['api_state']; opts=api['optimizers']
    with Image.open(receipt_path.parent/'goal.gif') as gif:
        media=dict(frames=gif.n_frames,width=gif.width,height=gif.height)
    assert media['frames']==9 and api['completed_steps']==600
    trace=[]
    for observation in receipt['observations']:
        step=observation['step']; x=arrays[f'step{step}_view0_samples']; t=arrays[f'step{step}_view0_target']
        assert x.shape==(1024,1,8,8) and x.dtype==np.float32 and np.isfinite(x).all()
        assert hashlib.sha256(t.astype('<f4').tobytes()).hexdigest()=='05489b1025607126a07709640796daa66e6d0a841697fc6160dff46636879512'
        trace.append(dict(step=step, recorded_passed=observation['passed'],
                          recorded_failed_bounds=observation['failed_bounds'],
                          recorded_metrics=observation['metrics'], attribution=image_attribution(x,t)))
    recipe=receipt['recipe']
    source_records=opts[1]['regularizer']['record']
    latent=opts[0]['regularizer']['latent']['state']
    rows.append(dict(family=family,candidate_id=trial['id'], status=trial['status'],
                     case_id=CASE_ID, original_gate=row['original_gate'], study_gate=row['study_gate'],
                     acquisition_hold=row['acquisition_hold'], completed_updates=600, remaining_unknown_cases=[x['id'] for x in trial['cases'] if x['status']=='UNKNOWN'],
                     artifacts=files, source=receipt['source'], runtime=receipt['runtime'], recipe=recipe,
                     requested_recipe_overrides=receipt['requested_recipe_overrides'], protocol=receipt['protocol'],
                     original_failed_bounds=receipt['failed_bounds'], original_metric_passed=receipt['metric_passed'],
                     original_sustained_metric_passed=receipt['sustained_metric_passed'],
                     numeric_pass_count_including0=sum(bool(x['passed']) for x in receipt['observations']),
                     capture_count=len(trace), final_metrics=receipt['observations'][-1]['metrics'],
                     best_recorded_finite_template_tv=min(x['metrics']['finite_template_tv'] for x in receipt['observations'] if x['step']>0),
                     supervised_paid_seconds=row['paid_wall_seconds'], acquisition_elapsed_seconds=receipt['elapsed_seconds'],
                     final_policy_metadata=api['policy'], absent_continuous_owners=[k for k in ('controller','backend_selection','output_noise','lr_settle','birth_death','row_evidence','surprise','reopen_guard') if k not in api],
                     role_rates=dict(initial=api['initial_lrs'], final_generator=opts[0]['param_groups'][0]['lr'], final_prior=opts[0]['param_groups'][1]['lr'], final_critic=opts[1]['param_groups'][0]['lr']),
                     final_critic_record=source_records, critic_spike_clips=opts[1]['regularizer']['guard']['clipped_tensors'],
                     latent_row_damping=dict(**latent, cumulative_observation_rate=latent['observed']/latent['total'],
                                            direct_response=opts[0]['regularizer']['direct'],
                                            applied_step_count_not_recorded=True),
                     owner_finite_summary=finite_summary({'models':api['models'],'optimizers':opts}),
                     image_ema_serving_differences={name:differences(api['models'][name],api['models']['ema_'+name]) for name in ('G','prior')},
                     gif=media, trace=trace))
    loaded[family]=(receipt,arrays,state)

rl,al,sl=loaded['ka2']; rr,ar,sr=loaded['k3p']
array_comparison=[]
for observation in rl['observations']:
    step=observation['step']; key=f'step{step}_view0_samples'; x,y=al[key],ar[key]
    array_comparison.append(dict(step=step,samples_bitwise_equal=bool(np.array_equal(x.view(np.uint32),y.view(np.uint32))),
                                 target_bitwise_equal=bool(np.array_equal(al[f'step{step}_view0_target'].view(np.uint32),ar[f'step{step}_view0_target'].view(np.uint32))),
                                 max_pixel_difference=float(np.abs(x.astype(np.float64)-y).max())))
comparisons={key:differences(sl['api_state'][key],sr['api_state'][key])
             for key in ('models','optimizers','initial_lrs','completed_steps','policy','streams','cpu_rng','cuda_rng')}
comparisons['data_generator']=differences(sl['data_generator'],sr['data_generator'])
recipe_diffs={key:dict(ka2=rl['recipe'].get(key,'OMITTED_DEFAULT'),k3p=rr['recipe'].get(key,'OMITTED_DEFAULT'))
              for key in sorted(set(rl['recipe'])|set(rr['recipe'])) if rl['recipe'].get(key)!=rr['recipe'].get(key)}
for path,pin in inputs.items(): assert sha(path)==pin
assert not torch.cuda.is_initialized()
paid=sum(r['supervised_paid_seconds'] for r in rows)
packet=dict(schema='ka2_k3p_retained_failure_diagnosis_v1', source_commit=COMMIT,
            requested_shared_tuple={'lr':.006375,'prior_lr_mult':1.,'d_lr_mult':1.},
            denominators=dict(families=2,definitions_per_family=8,full_scientific_runs=2,
                              original_pass=0,original_fail=2,study_pass=0,study_fail=2,
                              remaining_unknown=14,zero_update_capacity_supported=16),
            cost=dict(new_supervised_pair_seconds=paid,prior_campaign_seconds=113.99425188452005,
                      cumulative_campaign_seconds=113.99425188452005+paid,
                      original_campaign_cap_seconds=15360.,new_science_by_this_analysis_seconds=0),
            operations=dict(model_constructions=0,restores=0,draws=0,scorer_calls=0,
                            optimizer_updates=0,cuda_initialized=False,
                            attribution='descriptive arithmetic on immutable retained image arrays; recorded gates unchanged'),
            inputs_unchanged=True, records=rows,
            cross_family=dict(recipe_differences=recipe_diffs,arrays=array_comparison,
                              owner_comparisons=comparisons),
            missing_artifacts=['Intermediate model/prior/optimizer snapshots between0 and600',
                               'Per-update losses, gradients and penalty-phase events (public runner discards step returns)',
                               'A retained row-ID-to-output map; unique32 output coverage determines population counts but not which latent row owns each image',
                               'A paired EMA-generated image law; checkpointed EMA parameters are not scored primary outputs'],
            limits=['Saved equal image arrays through475 do not certify every unobserved update or complete optimizer-state equality at those earlier times.',
                    'Late common schedules and divergent critic anchor metadata explain a mechanism difference, not the reason for a particular bright output overshoot.',
                    '19/13 population allocation alone is below the.10 nearest-TV limit; the off-template bright-output deficit is necessary to explain the recorded finite-template rejection.',
                    'Both a larger prior rate and any family-mechanism contrast are unexecuted hypotheses, not fixes or winners.',
                    'Zero-clock16SUPPORTED capacity is not reachability/terminal noisy-native/full-family qualification.'])
(OUT/'pair-intensity-analysis.json').write_text(json.dumps(packet,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'records':len(rows),'first_retained_sample_difference':next(x['step'] for x in array_comparison if not x['samples_bitwise_equal']),
                  'paid':paid,'cumulative':packet['cost']['cumulative_campaign_seconds'],
                  'all_inputs_unchanged':True,'cuda_initialized':False}))
