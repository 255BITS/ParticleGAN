import json, sys, time, datetime
from pathlib import Path
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-transfer-vectors')
from benchmarks.transfer_suite import vector_tasks as v
out=Path('/tmp/pr36-transfer-vectors-20260922')
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tasks':v.TASKS, 'reserved':v.RESERVED, 'protocol':v.fingerprint(), 'plan':{'initial':['fixed_cosine on all 8 development tasks'], 'optional':'At most four simple fixed reference rates or R1+R2 settings on failed ranking tasks. No threshold/spec changes, no controller fitting, no reserved evaluation.'}}
(out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
rows=[]
for spec in v.TASKS:
    name=spec['name']
    print(f'START {name} fixed_cosine steps={spec["steps"]}',flush=True)
    policy=v.fixed_policy('cosine')
    result=v.run_episode(spec,policy,fixed=True)
    record={'name':name,'control':'fixed_cosine','spec':v.resolve(spec),'policy':policy,'protocol':manifest['protocol'],'result':result}
    (out/(name+'__fixed_cosine.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    row={'name':name,'tier':spec['tier'],'status':result['status'],'seconds':result['seconds'],'convergence':result['convergence'],'metrics':{k:result['live'].get(k) for k,_,_ in spec['thresholds']},'error':result.get('error')}
    rows.append(row)
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)
print('COMPLETE',flush=True)
