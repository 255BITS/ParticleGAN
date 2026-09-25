from pathlib import Path
from collections import Counter
import gzip,hashlib,json,shutil,sys
PUB=Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup')
BATCH=Path('/ml2/hypergan/gan-attempts/formulations-20260925T002058Z')
RUN=BATCH/'critic_scale/20260925T002058Z-2110671'
SRC=RUN/'repo/reports/toy100/critic-scale-attempt'
C=SRC/'candidates/dimension_rms_hybrid'
OUT=PUB/'reports/toy100/dimension-rms-base';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
declaration=json.loads((C/'declaration.json').read_text())
for name,h in declaration['file_hashes'].items():
 assert sha(C/name)==h
 shutil.copy2(C/name,OUT/name)
shutil.copy2(C/'declaration.json',OUT/'original-declaration.json')
manifest=json.loads((SRC/'prepared/prepared-sources.json').read_text())
for name,h in manifest['cuda'].items():assert sha(SRC/'prepared/repos/cuda'/name)==h,name
shutil.copy2(SRC/'prepared/prepared-sources.json',OUT/'prepared-sources.json')
sys.path.insert(0,str(SRC/'prepared/repos/cuda'))
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from benchmarks.transfer_suite.public_default_verification import load_declaration,declared_spec
from benchmarks.transfer_suite.protocol import test_verdict
config=json.loads((OUT/'config.json').read_text())|{'device':'cuda:0'}
recipe,_,_=declared_recipe(config);jobs,profile=load_declaration();rows=[]
for p in sorted(C.glob('*/result.json')):
 r=json.loads(p.read_text());task=r['task']
 expected,_,_=declared_spec(next(j for j in jobs if j['spec']['name']==task),profile,recipe)
 assert r['spec']==expected and r['config']==config
 assert test_verdict(expected,r['result'])==r['verdict']
 assert r['status']==r['verdict']['status']
 assert r['worker_sha256']==declaration['file_hashes']['probe.py']
 fixture=PUB/'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures'/task/'initial-values.pt'
 if not fixture.exists():
  fixture=SRC/'initialization-fixtures'/task/'initial-values.pt'
  dest=OUT/'initialization-fixtures'/task;dest.mkdir(parents=True)
  shutil.copy2(fixture,dest/'initial-values.pt')
  shutil.copy2(fixture.with_name('result.json'),dest/'result.json')
 assert sha(fixture)==r['initialization_fixture_sha256']
 fp=fixture.with_name('result.json.gz')
 init=json.loads(gzip.decompress(fp.read_bytes())) if fp.exists() else json.loads(fixture.with_name('result.json').read_text())
 assert init['proof']['adam_calls']==0 and init['proof']['initial_optimizers']==r['proof']['initial_optimizers']
 assert r['proof']['adam_calls']==2*expected['steps']
 assert all(o['calls']==expected['steps'] and o['device']=='cuda:0' and o['state_devices']==['cuda:0'] and o['parameter_dtypes']==['torch.float32'] for o in r['proof']['optimizers'].values())
 assert r['regularizer_receipt']['calls']==expected['steps'] and r['regularizer_receipt']['extra_critic_forwards']==0
 reference=PUB/'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init'/(task+'.json.gz')
 rng_compared=reference.exists()
 if rng_compared:
  ref=json.loads(gzip.decompress(reference.read_bytes()));assert r['randomness']==ref['randomness']
 else:ref=json.loads((PUB/'reports/toy100/gpu-known-winner-control/runs/simpler22_reference'/task/'result.json').read_text())
 assert r['spec']['thresholds']==ref['spec']['thresholds'] and r['result']['actions']==ref['result']['actions']
 dest=OUT/'results'/(task+'.json.gz');dest.parent.mkdir(exist_ok=True)
 dest.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
 shutil.copy2(p.parent/'audit.json',dest.with_suffix('').with_suffix('.audit.json'))
 rows.append(dict(gate=task,status=r['status'],metrics=r['result']['live'],passing_suffix=r['verdict']['convergence']['passing_suffix'],steps=expected['steps'],adam_calls=r['proof']['adam_calls'],rng_control_compared=rng_compared,artifact=str(dest.relative_to(OUT))))
assert len(rows)==7 and Counter(r['status'] for r in rows)=={'PASS':6,'FAIL':1}
for name in ('check_mechanism.py','dimension_rms_hybrid-mechanism-check.json','check_cuda_adam_state.py','cuda-adam-state-check.json','final-audit.json','commands.jsonl'):
 shutil.copy2(SRC/name,OUT/name)
shutil.copy2(RUN/'result.md',OUT/'original-attempt-report.md')
shutil.copy2(RUN/'tests.jsonl',OUT/'original-tests.jsonl')
(OUT/'audit.json').write_text(json.dumps(dict(status='PASS',candidate='dimension_rms_hybrid',source_files=1614,executed_gates=7,passes=6,failures=1,checks=['exact candidate/config hashes','archived source hashes','canonical declared specs and frozen sustained verdicts','pre-update CPU fixtures','CUDA FP32 model/gradient/Adam states','native RNG for six retained controls','frozen schedule/actions','no added model evaluations'],rows=rows),indent=2)+'\n')
# A compact deduplicated review of the full round keeps failed alternatives visible.
round_rows=[]
for lane in json.loads((BATCH/'batch.json').read_text()):
 run=sorted(Path(lane['directory']).glob('20*'))[-1]
 unique={}
 for line in (run/'tests.jsonl').read_text().splitlines():
  x=json.loads(line)
  if x['candidate']=='regression' or x['status']=='SKIPPED':continue
  artifact=Path(x.get('artifact',''))
  if not artifact.is_file() or artifact.name!='result.json':continue
  unique[str(artifact)]=x
 for artifact,x in unique.items():
  r=json.loads(Path(artifact).read_text())
  assert r['verdict']==test_verdict(r['spec'],r['result'])
  round_rows.append(dict(lane=lane['lane'],candidate=x['candidate'],gate=x['gate'],status=r['status'],metrics=r['result']['live'],passing_suffix=r['verdict']['convergence']['passing_suffix'],original_artifact=artifact))
assert len(round_rows)==24
(OUT/'round-summary.json').write_text(json.dumps(dict(proposals=9,executed_training_gates=24,counts=dict(Counter(r['status'] for r in round_rows)),audit_error_corrections=2,rows=round_rows),indent=2)+'\n')
print(json.dumps(dict(published=str(OUT),selected_counts=dict(Counter(r['status'] for r in rows)),round_counts=dict(Counter(r['status'] for r in round_rows)))))
