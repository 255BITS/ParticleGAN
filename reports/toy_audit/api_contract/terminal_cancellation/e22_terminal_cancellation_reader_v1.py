"""Independent saved-CPU reduction, CPU60; no producer/native/model/API imports.

Read five endpoint tensors, four same-displacement precision arrays and plain
saved native tensor states. No model construction, policy restore or forward.
The opaque inputs.pt containing a Geometry object is SHA-bound, not unpickled.
"""
import time
STARTED=time.monotonic()
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

TASK='routed_terminal_cancellation_v1'
ARMS=('ordinary_BF16','ordinary_FP32','particle_BF16','particle_FP32')
STEPS=(0,64,128,256,384,512)
IDS=[s for s in range(6) for _ in range(8)]
LIMIT=60
CHECKS={}


def budget():
    if time.monotonic()-STARTED>LIMIT:raise TimeoutError('savedCPU60 startup-through-write limit')


def check(name,value):
    CHECKS[name]=bool(value)
    if not value:raise AssertionError(name)
    budget()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    # Storage identity format declared in the frozen producer, no producer import.
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x):h.update(str(key).encode());add(x[key])
        elif isinstance(x,(tuple,list)):
            for item in x:add(item)
        else:h.update(json.dumps(x,sort_keys=True).encode())
    add(value);return h.hexdigest()


def values(tensor,name,shape=None):
    check(name+'_CPU_finite_float',isinstance(tensor,torch.Tensor) and tensor.device.type=='cpu'
          and tensor.is_floating_point() and bool(torch.isfinite(tensor).all()))
    if shape is not None:check(name+'_shape',tuple(tensor.shape)==shape)
    return tensor.double().reshape(-1).tolist()


def mean_squared(vals):return math.fsum(v*v for v in vals)/len(vals)


def metric(tensor,name):
    vals=values(tensor,name,(48,16,16));width=16*16
    return {'rmse':math.sqrt(mean_squared(vals)),
            'by_source':{str(s):math.sqrt(mean_squared(vals[s*8*width:(s+1)*8*width])) for s in range(6)},
            'source_counts':{str(s):8 for s in range(6)}}


def near(name,a,b):
    check(name,math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-12+2e-10*max(abs(a),abs(b)))


def same_metric(name,a,b):
    near(name+'_aggregate',a['rmse'],b['rmse'])
    check(name+'_counts',a['source_counts']==b['source_counts'])
    for s in range(6):near(name+'_source'+str(s),a['by_source'][str(s)],b['by_source'][str(s)])


def secant(before,after,name):
    a=values(before,name+'_before');b=values(after,name+'_after')
    check(name+'_same_shape',tuple(before.shape)==tuple(after.shape))
    dy=[v-u for u,v in zip(a,b)];m0=mean_squared(a);m1=mean_squared(b)
    linear=2*math.fsum(u*d for u,d in zip(a,dy))/len(a);q=mean_squared(dy)
    out={'MSE0':m0,'MSE1':m1,'delta_MSE':m1-m0,'output_linear':linear,'Q':q}
    check(name+'_physical_identity',abs(out['delta_MSE']-linear-q)<=1e-14+1e-12*max(abs(v) for v in out.values()))
    return out


def dot(named_gradient,named_delta,name):
    check(name+'_all_optimizer_coordinates',set(named_gradient)==set(named_delta)
          and 'table' in named_gradient and 'log_output_sigma' in named_gradient)
    total=[]
    for key in named_gradient:
        g,d=named_gradient[key],named_delta[key]
        check(name+'_'+key+'_F64_shape',g.dtype==d.dtype==torch.float64 and g.shape==d.shape)
        a=values(g,name+'_'+key+'_gradient');b=values(d,name+'_'+key+'_delta')
        total.extend(x*y for x,y in zip(a,b))
    return math.fsum(total)


def reduce_precision(raw,observed):
    keys={'BF16_before','BF16_after','FP32_before','FP32_after','BF16_gradient','FP32_gradient','delta'}
    check('precision_exact_raw_keys',set(raw)==keys)
    reduced={m:secant(raw[m+'_before'],raw[m+'_after'],m) for m in ('BF16','FP32')}
    slopes={m:dot(raw[m+'_gradient'],raw['delta'],m) for m in ('BF16','FP32')}
    errors={m:abs(reduced[m]['delta_MSE']-slopes[m]) for m in reduced}
    defined=reduced['BF16']['Q']>0 and errors['BF16']>0
    ratios={'Q':reduced['FP32']['Q']/reduced['BF16']['Q'],'discrepancy':errors['FP32']/errors['BF16']} if defined else None
    status='NA' if not defined else 'PASS' if all(r<=.75 for r in ratios.values()) else 'FAIL'
    helpful=slopes['BF16']<0<reduced['BF16']['delta_MSE']
    check('precision_status_independently_same',status==observed['precision_status'])
    check('helpful_slope_finite_harm_independently_same',helpful==observed['helpful_parameter_slope_and_finite_harm'])
    for m in reduced:
        for key,value in reduced[m].items():near('precision_'+m+'_'+key,value,observed[m][key])
        near('slope_'+m,slopes[m],observed['parameter_slopes'][m])
        near('discrepancy_'+m,errors[m],observed['absolute_discrepancies'][m])
    if defined:
        for key,value in ratios.items():near('ratio_'+key,value,observed['ratios'][key])
    else:check('undefined_ratios_preserved',observed['ratios'] is None)
    by_source={}
    for s in range(6):
        by_source[str(s)]={}
        for m in reduced:
            result=secant(raw[m+'_before'][s*8:(s+1)*8],raw[m+'_after'][s*8:(s+1)*8],m+'_source'+str(s))
            by_source[str(s)][m]=result
            for key,value in result.items():near('source'+str(s)+'_'+m+'_'+key,value,observed['by_source'][str(s)][m][key])
    return {'precision_status':status,'BF16':reduced['BF16'],'FP32':reduced['FP32'],'parameter_slopes':slopes,
            'absolute_discrepancies':errors,'ratios':ratios,'helpful_parameter_slope_and_finite_harm':helpful,'by_source':by_source,
            'scope':'Independent saved-output/VJP local same-delta reduction, not a sustained-convergence or unique-cause finding.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','card','external','manifest','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--prep',type=Path)
    parser.add_argument('--launcher',type=Path,default=Path('/tmp/bounded-particle-api-campaign-launcher-v1.py'))
    args=parser.parse_args();error=None;result=None;writable=False;code=2
    rng=torch.get_rng_state().clone();threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        check('exclusive_report_and_companion',not args.out.exists() and not Path(str(args.out)+'.completion.json').exists())
        args.out.parent.mkdir(parents=True,exist_ok=True);writable=True
        rp,cp=args.run/'report.json',args.run/'completion.json'
        report=json.loads(rp.read_text());completion=json.loads(cp.read_text());card=json.loads(args.card.read_text())
        external=json.loads(args.external.read_text());manifest=json.loads(args.manifest.read_text())
        pinned={str(p):sha(p) for p in (Path(__file__),args.card,rp,cp,args.external,args.manifest,args.launcher)}
        if args.prep is not None:pinned[str(args.prep)]=sha(args.prep)
        check('complete_valid_scientific_verdict',report['complete'] and completion['complete'] and external['complete']
              and report['task']==completion['task']==card['task']==TASK
              and report['scientific_status']==completion['scientific_status']==('PASS' if report['gate']['pass'] else 'FAIL'))
        expected_exit=0 if report['gate']['pass'] else 1
        check('canonical_completed_negative_is_complete',completion['exit_code']==external['exit_code']==expected_exit
              and external['child_scientific_status']==report['scientific_status'] and not external['timed_out'])
        check('producer_external_budget_and_companion',completion['seconds']<=300 and external['seconds']<=300
              and completion['report_sha256']==sha(rp) and external['child_completion_sha256']==sha(cp))
        check('external_manifest_exact',external['manifest_sha256']==sha(args.manifest)==sha(external['manifest'])
              and external['launcher_sha256']==sha(args.launcher)
              and all(external['checks_before'].values()) and all(external['checks_after'].values()))
        check('frozen_manifest_current',all(sha(p)==expected for p,expected in manifest['files'].items()))
        root=Path(manifest['cwd'])
        check('protocol_sources_exact',report['protocol_sha256']==sha(args.card) and report['source_identity']==card['sources']
              and all(sha(root/path)==value for path,value in card['sources'].items()))
        check('fixed_task_law',card['arms']==list(ARMS) and card['steps']==512 and card['sites']==['input','edit']
              and card['edit_amplitude']==1/64 and card['bank_N']==32 and card['bank_Z']==4 and card['batch_size']==4)
        package=Path(report['imported_native_observed']['imported_package']);h=hashlib.sha256()
        for p in sorted(package.rglob('*.py')):h.update(str(Path('particlegan')/p.relative_to(package)).encode());h.update(p.read_bytes())
        check('actually_imported_native_current_no_allowlist',h.hexdigest()==report['imported_native_observed']['python_sha256'])
        if args.prep is not None:
            prep=json.loads(args.prep.read_text())
            check('prep_source_card_identity',prep['source_identity']==card['sources'] and prep['protocol_sha256']==sha(args.card))
        files={'endpoint-residuals.pt':'endpoint_residual_sha256','observed-media.pt':'observed_media_sha256',
               'precision-witness.pt':'precision_raw_sha256','inputs.pt':'input_sha256'}
        for path,key in files.items():check(path+'_SHA',sha(args.run/path)==report[key]);pinned[str(args.run/path)]=sha(args.run/path)
        endpoints=torch.load(args.run/'endpoint-residuals.pt',map_location='cpu',weights_only=True)
        check('five_fixed_terminal_arrays',set(endpoints)==set(ARMS)|{'zero_code'})
        accuracy={a:metric(y,a) for a,y in endpoints.items()}
        for a,value in accuracy.items():same_metric(a+'_reported_accuracy',value,report['accuracy'][a])
        media=torch.load(args.run/'observed-media.pt',map_location='cpu',weights_only=True)
        check('media_fixed_task_frames_and_indices',media['task']==TASK and media['steps']==list(STEPS)
              and media['indices']==[0,8,16,24,32,40,1,9] and media['source_ids']==[0,1,2,3,4,5,0,1]
              and set(media['actual_residuals'])==set(ARMS) and media['captures_immutable'])
        check('fixed_true_zero_target',media['target_residual'].shape==(8,16,16) and not bool(media['target_residual'].count_nonzero()))
        for a in ARMS:
            check(a+'_all_media_steps',set(media['actual_residuals'][a])=={str(s) for s in STEPS})
            for step,value in media['actual_residuals'][a].items():values(value,a+'_media_'+step,(8,16,16))
            check(a+'_media_endpoint_exact',torch.equal(media['actual_residuals'][a]['512'],endpoints[a][media['indices']]))
        for m in ('BF16','FP32'):check(m+'_initial_ordinary_particle_exact',torch.equal(media['actual_residuals']['ordinary_'+m]['0'],media['actual_residuals']['particle_'+m]['0']))
        live={};streams={};events={};independent_norms={}
        for a in ARMS:
            path=args.run/(a+'.jsonl');check(a+'_trace_SHA',sha(path)==report['trace_sha256'][a]);pinned[str(path)]=sha(path)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            check(a+'_exact512_steps',[r['step'] for r in rows]==list(range(1,513)))
            check(a+'_finite_native_losses',all(math.isfinite(r[k]) for r in rows for k in ('loss_g','loss_d_game','penalty')))
            check(a+'_B4_FIT_indices',all(len(r['batch_indices'])==4 and all(isinstance(i,int) and 0<=i<48 for i in r['batch_indices']) for r in rows))
            live[a]={'bank':sum(bool(r['bank_live']) for r in rows[1:]),'query':sum(bool(r['query_live']) for r in rows[1:])}
            check(a+'_live_recount',live[a]==report['live'][a])
            fields=('step','batch_indices','paired_bases','data_rng','paired_rng','penalty_globals')
            streams[a]=digest([{k:r[k] for k in fields} for r in rows])
            check(a+'_external_stream_digest',streams[a]==report['matched_stream_sha256'][a])
            proposals=[r['move'] for r in rows if r['move']]
            events[a]={'proposal_events':len(proposals),'accepted_proposals':sum(bool(e.get('accepted',False)) for e in proposals),
                       'accepted_row_moves':sum(int(e.get('moves',0)) for e in proposals)}
            state_path=args.run/(a+'-512.pt');pinned[str(state_path)]=sha(state_path)
            saved=torch.load(state_path,map_location='cpu',weights_only=True);native=saved['native']
            check(a+'_saved_boundary_recipe_data',saved['arm']==a and saved['data_digest']==report['data_digest']
                  and native['completed_steps']==512 and native['recipe']==report['recipes'][a])
            gen=native['models']['generator']
            check(a+'_frozen_student_teacher_equal',all(torch.equal(gen['backbone.'+student],gen['teacher_backbone.'+teacher])
                  for student,teacher in (('carrier.weight','carrier.weight'),('carrier.bias','carrier.bias'),
                    ('input.base.weight','input.weight'),('input.base.bias','input.bias'),
                    ('edit.base.weight','edit.weight'),('edit.base.bias','edit.bias'),('final.matrix.weight','final.matrix.weight'))))
            for role,mapping in native['models'].items():
                for key,tensor in mapping.items():
                    if tensor.is_floating_point():values(tensor,a+'_'+role+'_'+key)
            values(native['table'],a+'_external_table',(32,4))
            values(native['output_noise'],a+'_external_sigma')
            if a.startswith('particle'):
                independent_norms[a]={}
                for site in ('input','edit'):
                    c=gen['backbone.'+site+'.bridge.weight'][:,4:]
                    up=gen['backbone.'+site+'.particle_up.weight']
                    independent_norms[a][site]={'C':math.sqrt(math.fsum(v*v for v in values(c,a+'_'+site+'_C'))),
                                               'particle_up':math.sqrt(math.fsum(v*v for v in values(up,a+'_'+site+'_codeUp')))}
                    for key,value in independent_norms[a][site].items():
                        recorded=report['norms'][a][site][key]
                        check(a+'_'+site+'_'+key+'_F32_norm_agreement',math.isfinite(recorded) and abs(value-recorded)<=1e-10+2e-7*value)
        check('fourarms_matched_external_draws',len(set(streams.values()))==1)
        check('within_architecture_recipes_exact',report['recipes']['ordinary_BF16']==report['recipes']['ordinary_FP32']
              and report['recipes']['particle_BF16']==report['recipes']['particle_FP32'])
        ordinary,candidate,zero=(accuracy[k] for k in ('ordinary_FP32','particle_FP32','zero_code'))
        norms=independent_norms['particle_FP32']
        gate={'aggregate':candidate['rmse']<=ordinary['rmse']*.999,
              'no_source_harm':all(candidate['by_source'][str(s)]<=ordinary['by_source'][str(s)]+1e-6 for s in range(6)),
              'code_aggregate':zero['rmse']>=candidate['rmse']*1.001,
              'code_each_source':all(zero['by_source'][str(s)]>candidate['by_source'][str(s)] for s in range(6)),
              'bank_live':live['particle_FP32']['bank']>=.9*511,'query_live':live['particle_FP32']['query']>=.9*511,
              'heads_live':set(norms)=={'input','edit'} and all(math.isfinite(v[k]) and v[k]>0 for v in norms.values() for k in ('C','particle_up'))}
        check('fixed_science_bounds_reduced',gate==report['gate']['checks'] and all(gate.values())==report['gate']['pass']
              and [k for k,v in gate.items() if not v]==report['gate']['failed_bounds'])
        check('initial_source_proofs',all(report['initial_prerequisite'][k] for k in ('same_precision_initial_outputs_exact','common_named_DownUp_exact','EMA_disjoint_and_precision_preserved','teacher_independent_BF16','captures_native_caller_RNG_modes_gradients_immutable')))
        raw=torch.load(args.run/'precision-witness.pt',map_location='cpu',weights_only=True)
        for m in ('BF16','FP32'):
            for suffix in ('before','after'):values(raw[m+'_'+suffix],m+'_'+suffix,(48,16,16))
        check('precision_BF16_after_is_actual_endpoint',torch.equal(raw['BF16_after'],endpoints['particle_BF16']))
        precision=reduce_precision(raw,report['precision_witness'])
        check('zero_unused_sigma_accuracy_gradients',all(not bool(raw[m+'_gradient']['log_output_sigma'].count_nonzero()) for m in ('BF16','FP32')))
        check('no_model_native_API_imports','particlegan' not in __import__('sys').modules and not any(k.startswith('examples.') for k in __import__('sys').modules))
        check('CPU_only_no_CUDA',not torch.cuda.is_initialized())
        check('all_inputs_current_unchanged',all(sha(p)==value for p,value in pinned.items()))
        result={'complete':True,'qualification':'PASS','task':TASK,'scientific_status_unchanged':report['scientific_status'],
                'checks':CHECKS,'accuracy':accuracy,'science_checks':gate,'live':live,'norms':independent_norms,'population_events':events,'precision_witness':precision,
                'source_identity':card['sources'],'protocol_sha256':sha(args.card),'artifact_sha256':pinned,
                'scope':'Saved-tensor numerical and byte qualification, including final head norms and paired frozen state values. Initial EMA/owner/RNG/noisy-forward/VJP origins remain source-bound producer proofs; no model/public-API replay. Opaque inputs.pt not unpickled.',
                'native_updates':0,'model_forwards':0,'API_calls':0};code=0
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        torch.set_num_threads(threads)
        if not torch.equal(rng,torch.get_rng_state()):error={'type':'AssertionError','message':'CPU callerRNG changed'}
        if time.monotonic()-STARTED>LIMIT:error={'type':'TimeoutError','message':'CPU cleanup60s exceeded'}
        if error is not None:result={'complete':False,'qualification':'FAIL','task':TASK,'checks':CHECKS,'error':error};code=2
        if writable:
            result.update(seconds=time.monotonic()-STARTED,limit_seconds=LIMIT)
            with args.out.open('x') as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
            companion={'complete':result['complete'],'qualification':result['qualification'],'report_sha256':sha(args.out),
                       'seconds':time.monotonic()-STARTED,'limit_seconds':LIMIT,'native_updates':0,'model_forwards':0,'API_calls':0}
            dest=Path(str(args.out)+'.completion.json')
            with dest.open('x') as handle:handle.write(json.dumps(companion,indent=2,allow_nan=False)+'\n')
            if time.monotonic()-STARTED>LIMIT:
                companion.update(complete=False,qualification='FAIL',error='post-write60s overrun');dest.write_text(json.dumps(companion,indent=2)+'\n');code=2
        print(json.dumps({'complete':bool(result and result['complete']),'qualification':result['qualification'] if result else 'FAIL',
                          'checks':len(CHECKS),'error':error,'seconds':time.monotonic()-STARTED},allow_nan=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
