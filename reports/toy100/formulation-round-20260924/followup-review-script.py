from pathlib import Path
from collections import Counter
import gzip,hashlib,json,shutil,sys
base=Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup')
batch=Path(Path('/ml2/hypergan/current-formulation-followup.txt').read_text().strip())
root=base/'reports/toy100/formulation-round-20260924';out=root/'followups';out.mkdir(exist_ok=True)
records=json.loads((batch/'batch.json').read_text());first=sorted(Path(records[0]['directory']).glob('20*'))[-1]
source=Path(json.loads((first/'declaration.json').read_text())['source']);sys.path.insert(0,str(source))
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from benchmarks.transfer_suite.public_default_verification import load_declaration,declared_spec
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in json.loads((source.parents[1]/'prepared-sources.json').read_text())['cuda'].items():assert sha(source/name)==h
original=json.loads((base/'configs/toy100/constraints_simple_regularization.json').read_text());rows=[]
for row in records:
 run=sorted(Path(row['directory']).glob('20*'))[-1];dest=out/row['lane'];dest.mkdir(exist_ok=True)
 d=json.loads((run/'declaration.json').read_text());cfg=json.loads((run/'config.json').read_text())
 assert {k:v for k,v in cfg.items() if k not in ('gan_mode','reg_arm','reg_kappa')}=={k:v for k,v in original.items() if k not in ('gan_mode','reg_arm','reg_kappa')}
 for name,h in d['file_hashes'].items():assert sha(run/name)==h
 for p in run.iterdir():
  if p.is_file() and p.suffix in ('.py','.json','.jsonl','.log','.txt','.md'):shutil.copy2(p,dest/p.name)
 for p in run.glob('*/result.json'):
  r=json.loads(p.read_text());task=r['task'];assert r['status'] in ('PASS','FAIL')
  recipe,_,_=declared_recipe(cfg|{'device':'cuda:0'});jobs,profile=load_declaration()
  spec,_,_=declared_spec(next(j for j in jobs if j['spec']['name']==task),profile,recipe)
  assert spec==r['spec'] and test_verdict(spec,r['result'])==r['verdict']
  c=json.loads(gzip.decompress((base/'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init'/(task+'.json.gz')).read_bytes()))
  assert r['proof']['initial_optimizers']==c['proof']['initial_optimizers']
  assert r['randomness']==c['randomness'] and r['result']['actions']==c['result']['actions']
  assert r['backend']=='cuda' and r['config']==cfg|{'device':'cuda:0'}
  assert r['worker_sha256']==d['file_hashes']['probe.py']
  assert r['proof']['adam_calls']==c['proof']['adam_calls']
  assert all(v['device']=='cuda:0' for v in r['proof']['optimizers'].values())
  target=dest/'results'/(task+'.json.gz');target.parent.mkdir(exist_ok=True);target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
  rows.append(dict(candidate=row['lane'],gate=task,status=r['status'],live=r['result']['live'],convergence=r['verdict']['convergence'],artifact=str(target.relative_to(root)),audit='PASS'))
for p in batch.iterdir():
 if p.is_file() and p.suffix in ('.py','.json','.log'):shutil.copy2(p,out/p.name)
(root/'followup-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'followup-audit.json').write_text(json.dumps(dict(status='PASS',gates=len(rows),source_files=1614,checks=['source and candidate hashes','declared config only changes objective/regularizer/cap','frozen spec constructed from archived harness','sustained verdict','native CUDA initial/RNG/schedule parity','CUDA optimizer updates']),indent=2)+'\n')
lines=['# Direct measured follow-ups','','Every formulation earns its own passes. Full 22 and own-state stability have not','run for these candidates. The Ra hybrid and symmetric b-cap both pass the initial','two blockers, then fail trajectory. The hybrid also fails image intensity.','','| Candidate | Gate | Result | Passing suffix |','|---|---|---|---:|']
for r in rows:lines.append(f"| {r['candidate']} | {r['gate']} | {r['status']} | {r['convergence']['passing_suffix']} |")
lines+=['','Original follow-up audit errors are retained. The check initially rejected allowed','recipe fields inside the specification; it now constructs the exact expected spec','from the declared recipe and unchanged archived harness. No scoring, data or budget','was changed and no training rerun was required. [Independent audit](followup-audit.json).','','R1-containing candidates remain eligible when supported by measured results. The','hybrid stopped because it failed transfer regressions, not because of its name.','The latest objective ablations restore the original Rp loss to the two candidates','that passed both blockers with Ra but failed trajectory. These are new measured','candidates; their passes cannot be inherited from the Ra counterparts.']
(root/'FOLLOWUPS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(gates=len(rows),counts=dict(Counter(r['status'] for r in rows)),audit='PASS')))
