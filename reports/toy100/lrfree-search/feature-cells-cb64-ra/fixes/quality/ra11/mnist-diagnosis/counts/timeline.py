#!/usr/bin/env python3
"""Closed JSON/log-only factual timeline. Does not import candidate modules."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
AREA = ROOT / 'quality/ra11/mnist-diagnosis/counts'
PREV = ROOT.parent / 'scaling-portability-20260929/validation'
CASES = {
    'RA11': ROOT / 'validation-cb64-ra11/learned/training/mnist/CB64-RA11',
    'RA4': ROOT / 'validation-ra4/learned/training/mnist/CB64-RA4',
    'E22': PREV / 'runs/mnist/E22',
}
LOGS = {
    'RA11': ROOT / 'validation-cb64-ra11/logs/learned-mnist-CB64-RA11.log',
    'RA4': ROOT / 'validation-ra4/logs/learned-mnist-CB64-RA4.log',
}
STEPS = [0,100,250,500,750,1000,1250,1500,1750,2000]
COUNTERS = ['evals','discoveries','moves','ordinary_moves','iso_moves',
    'novel_birth_attempts','novel_birth_moves','mean_evals','mean_witness_fires',
    'mean_moves','mean_preview_rows','iso_flagged','iso_skipped']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def records(path):
    out=[]
    for line in Path(path).read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(row,dict): out.append(row)
    return out

def seal():
    files=[Path(__file__),AREA/'PLAN.md',PREV/'models_metrics.py',PREV/'run.py',
        PREV/'data-receipt.json',ROOT/'quality/ra11/READY.json',
        ROOT/'validation-cb64-ra11/source-freeze.json',
        ROOT/'validation-ra4/source-freeze.json']
    for case in CASES.values():
        files += [case/name for name in ['metrics.jsonl','result.json','config.json','evaluator.json']]
    files += list(LOGS.values())
    files += sorted((AREA/'preparation-attempt1').glob('*'))
    for pkg in ['pkg-CB64-RA11','pkg-CB64-RA4']:
        files += sorted((ROOT/pkg/'particlegan').glob('*.py'))
    for lane in ['validation-cb64-ra11','validation-ra4']:
        files += [ROOT/lane/'learned'/name for name in ['common.py','run_training.py','INPUTS.json','PROTOCOL.md']]
    assert all(f.is_file() for f in files)
    for name,case in CASES.items():
        result=read(case/'result.json')
        assert result['problem']=='mnist' and result['steps']==2000
        assert result['final']['step']==2000
        if name!='E22': assert result['status']=='COMPLETE'
    out={'status':'FROZEN_BEFORE_DERIVED_JSON_TIMELINE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'scope':'JSON_AND_RECORDED_LOG_SCALARS_ONLY','protected_sha256':{str(f):sha(f) for f in files},
         'PT_objects_loaded':0,'Torch_imported':False,'forward_or_draw_or_training':0}
    target=AREA/'SOURCE-INPUTS-FROZEN.json'
    assert not target.exists()
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'sha256':sha(target),'files':len(out['protected_sha256'])}))

def one(row,cfg):
    m=row['metrics']; d=row['diagnostics']; bd=d.get('birth_death',{}); last=bd.get('last',{})
    mean=last.get('mean_transport',{})
    lease=bd.get('paired_average',{})
    settle=d.get('lr_settle',d.get('stationarity',{}))
    g1=settle.get('g1',{})
    scalar={k:m.get(k) for k in ['class_mass_tv','confident_class_coverage',
        'confident_fraction','mean_classifier_confidence','pixel_clipping_fraction']}
    scalar['native_flat_embedding']={k:m[k] for k in ['embedding_frechet','embedding_precision','embedding_recall'] if k in m} or None
    scalar['raw_embedding']=m.get('raw_embedding')
    scalar['active_embedding']=m.get('active_embedding')
    out={'step':row['step'],'metrics':scalar,'output_sigma':d.get('output_sigma'),
         'lrs':d.get('lr'),'controller_closed':d.get('controller',{}).get('closed'),
         'counters':{k:bd.get('counters',{}).get(k) for k in COUNTERS},
         'eligible_now':bd.get('eligible_now'),'last_eligible':last.get('eligible'),
         'last_iso_flagged':last.get('iso_flagged'),'last_iso_moves':last.get('iso_moves'),
         'last_moves':last.get('moves'),'actual_cells':last.get('cells'),'chart_rank':last.get('metric_rank'),
         'learned_groups':last.get('mass_topology',{}).get('groups'),
         'requested_cells':bd.get('settings',{}).get('cells'),
         'count_multiplicity':last.get('count_multiplicity'),'mean':mean or None,
         'lease':lease or None,'lease_age_real_rows':bd.get('paired_average_age_real_rows'),
         'table_settler':{k:g1.get(k) for k in ['s','b','last','counts','population_active','last_population']},
         'novel_attempted_cells_last':last.get('novel_birth_attempts'),
         'ordinary_inaccessible_birth_quota_last':last.get('ordinary_inaccessible_birth_quota')}
    if mean.get('output_dim'):
        out['selected_axes_fraction']=len(mean.get('selected_axes',[]))/mean['output_dim']
    else: out['selected_axes_fraction']=None
    return out

def run():
    freeze=read(AREA/'SOURCE-INPUTS-FROZEN.json')
    for path,digest in freeze['protected_sha256'].items(): assert sha(path)==digest,path
    timeline={}; configs={}; evaluators={}; progress={}
    for name,case in CASES.items():
        cfg=read(case/'config.json'); configs[name]=cfg; evaluators[name]=read(case/'evaluator.json')
        rows=records(case/'metrics.jsonl'); assert [r['step'] for r in rows]==STEPS
        timeline[name]=[one(row,cfg) for row in rows]
        if name in LOGS:
            log=records(LOGS[name])
            checkpoints=[r for r in log if r.get('event')=='training_checkpoint']
            assert [r['step'] for r in checkpoints]==STEPS
            progress[name]=[{k:r.get(k) for k in ['event','step','loss_d','loss_g','output_sigma']}
                for r in log if r.get('event')=='training_progress']
    identities={k:[configs[n].get(k) for n in CASES] for k in
        ['seed','steps','initial_generator_sha256','initial_critic_sha256','initial_prior_sha256']}
    assert all(v[0]==v[1]==v[2] for v in identities.values())
    evaluator_fields=['active_dimensions','active_mask_sha256','active_mean_sha256','active_std_sha256',
        'raw_reference_sha256','active_reference_sha256','evaluator_model_sha256']
    exact_evaluator={k:evaluators['RA11'].get(k)==evaluators['RA4'].get(k) for k in evaluator_fields}
    assert all(exact_evaluator.values())
    common_steps={n:[r['step'] for r in timeline[n]] for n in CASES}
    axis_union=sorted({a for row in timeline['RA11'] for a in (row['mean'] or {}).get('selected_axes',[])})
    out={'status':'PASS_JSON_FACTUAL_DIAGNOSIS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_input_freeze_sha256':sha(AREA/'SOURCE-INPUTS-FROZEN.json'),'identities':identities,
        'RA11_RA4_evaluator_fields_exact':exact_evaluator,'milestones':common_steps,'timeline':timeline,
        'progress':progress,'recipes':{n:configs[n]['recipe'] for n in CASES},
        'sampled_RA11_axis_union':axis_union,'sampled_RA11_axis_union_count':len(axis_union),
        'sampled_RA11_axis_union_fraction':len(axis_union)/784,
        'PT_objects_loaded':0,'Torch_imported':False,'model_forwards':0,'new_draws':0,'new_training_updates':0,
        'limits':['Native E22 flat embedding normalization differs from later active/raw metrics; FD values are not interchangeable.',
          'Recorded milestone axes/reasons do not describe every intervening reaction; cumulative counters bound accepted actions.',
          'Matching initialization/fixture receipts do not isolate causal changes among package/rate/serving differences.',
          'No output-coordinate variance coverage or raw/EMA/G/D geometry is measured from JSON.']}
    target=AREA/'result.json'; assert not target.exists()
    target.write_text(json.dumps(out,indent=2,allow_nan=True)+'\n')
    for n in CASES:
        print(n)
        for r in timeline[n]:
            m=r['metrics']; b=r['counters']; mean=r['mean'] or {}; lease=r['lease'] or {}
            print(json.dumps({'step':r['step'],'activeFD':(m['active_embedding'] or {}).get('embedding_frechet'),
                'nativeFD':(m['native_flat_embedding'] or {}).get('embedding_frechet'),'rawFD':(m['raw_embedding'] or {}).get('embedding_frechet'),
                'coverage':m['confident_class_coverage'],'confident':m['confident_fraction'],'TV':m['class_mass_tv'],
                'clip':m['pixel_clipping_fraction'],'sigma':r['output_sigma'],'moves':b.get('moves'),
                'mean_fires':b.get('mean_witness_fires'),'mean_moves':b.get('mean_moves'),'mean_status':mean.get('status'),
                'mean_reason':mean.get('reason'),'last_eligible':r['last_eligible'],'flags':r['last_iso_flagged'],
                'groups':r['learned_groups'],'lease_eligible':lease.get('eligible'),'lease_coherent':lease.get('coherent_rows'),
                'lease_ema_eligible':lease.get('ema_eligible_rows')}))
    print(json.dumps({'status':out['status'],'result_sha256':sha(target),'sampled_axis_union_count':len(axis_union)}))
    for path,digest in freeze['protected_sha256'].items(): assert sha(path)==digest,path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','run']);a=p.parse_args()
    seal() if a.mode=='seal' else run()
