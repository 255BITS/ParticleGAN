import json, sys, datetime
from pathlib import Path
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-transfer-vectors')
from benchmarks.transfer_suite import vector_tasks as v
out=Path('/tmp/pr36-transfer-vectors-20260922')
manifest=json.loads((out/'frozen_manifest.json').read_text())
assert v.fingerprint()==manifest['protocol']
plan=[]
for family in ('unequal_mass','unequal_width'):
    base=next(x for x in v.TASKS if x['family']==family)
    plan.append({'control':'fixed_constant','spec':base,'policy':v.fixed_policy('constant')})
    plan.append({'control':'fixed_cosine_r1r2_0p1','spec':dict(base,reg_arm='a_r1r2',reg_coeff=.1),'policy':v.fixed_policy('cosine')})
(out/'additional_reference_plan.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'The initial cosine control failed component spread in both families. These four permitted simple controls test whether changing schedule or penalty helps; thresholds, data and architecture remain frozen.','attempts':plan},indent=2,allow_nan=False)+'\n')
rows=[]
for attempt in plan:
    spec,control,policy=attempt['spec'],attempt['control'],attempt['policy']
    print(f'START {spec["name"]} {control}',flush=True)
    result=v.run_episode(spec,policy,fixed=True)
    record={'name':spec['name'],'control':control,'spec':v.resolve(spec),'policy':policy,'protocol':manifest['protocol'],'result':result}
    (out/(spec['name']+'__'+control+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    row={'name':spec['name'],'control':control,'status':result['status'],'seconds':result['seconds'],'convergence':result['convergence'],'metrics':{k:result['live'].get(k) for k,_,_ in spec['thresholds']},'error':result.get('error')}
    rows.append(row)
    (out/'additional_summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)
print('COMPLETE extra references',flush=True)
