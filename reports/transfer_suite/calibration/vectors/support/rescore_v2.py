import datetime, hashlib, json, math, shutil, sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-transfer-vectors')
import torch
from benchmarks.transfer_suite import vector_tasks as v
base=Path('/tmp/pr36-transfer-vectors-20260922')
out=base/'v2_rescored'
out.mkdir(exist_ok=False)
root=Path('/ml2/hypergan/ParticleGAN-transfer-vectors')
protocol=v.fingerprint()
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':v.TASKS,'reserved':v.RESERVED,'protocol':protocol,'correction':'Before controller fitting, extend the existing minimum component eigenvalue >= .15 bound to every identifiable mixture. Existing v1 metric values are re-scored; no retraining, tier changes or reserved evaluation. Original v1 files are unchanged.','v1_manifest_sha256':hashlib.sha256((base/'frozen_manifest.json').read_bytes()).hexdigest()}
(out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
for name,digest in protocol['source_sha256'].items():
    src=root/name
    assert hashlib.sha256(src.read_bytes()).hexdigest()==digest
    dest=out/'source_snapshot'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
rows=[]
for original in sorted(base.glob('vector_*__*.json')):
    record=json.loads(original.read_text())
    spec=deepcopy(record['spec'])
    dev=next(x for x in v.TASKS if x['name']==spec['name'])
    spec['thresholds']=deepcopy(dev['thresholds'])
    result=deepcopy(record['result'])
    expected={math.ceil(i*spec['steps']/24) for i in range(1,25)}
    result['convergence']=v.sustained(result['observations'],spec['thresholds'],expected_steps=expected)
    if 'error' not in result:
        result['status']='PASS' if v.passes(result['live'],spec['thresholds']) else 'FAIL'
    rescored={'name':record['name'],'control':record['control'],'spec':spec,'policy':record['policy'],'execution_protocol':record['protocol'],'scoring_protocol':protocol,'execution_spec':record['spec'],'rescore_provenance':{'original_file':str(original),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'retrained':False,'timing':'Copied from original training; not a fresh timing measurement.'},'result':result}
    (out/original.name).write_text(json.dumps(rescored,indent=2,allow_nan=False)+'\n')
    row={'name':spec['name'],'control':record['control'],'tier':spec['tier'],'final_pass':v.passes(result['live'],spec['thresholds']),'sustained':result['convergence'].get('stable_from_step') is not None,'suffix':result['convergence']['passing_suffix'],'confirmed_step':result['convergence']['confirmed_step'],'seconds':result['seconds'],'metrics':{k:result['live'].get(k) for k,_,_ in spec['thresholds']},'failed_bounds':[[k,op,bound,result['live'].get(k)] for k,op,bound in spec['thresholds'] if not v.passes(result['live'],[[k,op,bound]])]}
    rows.append(row)
(out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
torch.set_num_threads(1)
oracles=[]
for spec in v.TASKS:
    real=v.sample_target(spec,v.EVAL_SAMPLES,torch.Generator().manual_seed(993),spec['steps'])
    metric=v.score_samples(real,spec,spec['steps'])
    oracles.append({'name':spec['name'],'metrics':metric,'pass':v.passes(metric,spec['thresholds'])})
assert all(x['pass'] for x in oracles)
(out/'target_oracle.json').write_text(json.dumps({'description':'Independent target-sampler sanity controls, using the exact same evaluation streams as unit tests; these are not GAN training or seed-search experiments. No reserved samples produced.','results':oracles},indent=2,allow_nan=False)+'\n')
lines=['# Vector fixed-reference calibration (protocol v2)','','Re-scored from retained v1 training; no new training, no reserved evaluation. All rows have 24/24 observations. Minimum eigenvalue >=0.15 applies to every separated mixture. Timing is original CPU1 elapsed time, including evaluation.','','| Task | Control | Tier | Final | Sustained | Tail / 24 | Seconds | Failed final bounds |','|---|---|---|---:|---:|---:|---:|---|']
for row in rows:
    failed=', '.join(f'{k}={value:.4g} ({op}{bound:g})' for k,op,bound,value in row['failed_bounds']) or 'none'
    lines.append(f'| {row["name"]} | {row["control"]} | {row["tier"]} | {"PASS" if row["final_pass"] else "FAIL"} | {"PASS" if row["sustained"] else "FAIL"} | {row["suffix"]} | {row["seconds"]:.2f} | {failed} |')
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
