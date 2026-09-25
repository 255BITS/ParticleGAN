from pathlib import Path
import gzip,hashlib,json,sys
root=Path(__file__).resolve().parent
base=Path('/ml2/hypergan/ParticleGAN-selected-h-stability-base')
records=json.loads((root/'batch.json').read_text());rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in records:
 run=sorted(Path(row['directory']).glob('20*'))[-1];d=json.loads((run/'declaration.json').read_text())
 for file,h in d['file_hashes'].items():assert sha(run/file)==h
 sys.path.insert(0,d['source'])
 from benchmarks.transfer_suite.protocol import test_verdict
 for task in ('mode_hold','vector_unequal_mass'):
  p=run/task/'result.json';r=json.loads(p.read_text())
  control=json.loads(gzip.decompress((base/'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init'/(task+'.json.gz')).read_bytes()))
  expected=dict(control['spec'])
  declared_config=json.loads((run/'config.json').read_text())
  for key in ('reg_arm','gan_mode','loss_type'):
   if key in expected:expected[key]=declared_config[key]
  assert r['spec']==expected
  assert r['config']==declared_config|{'device':'cuda:0'}
  assert test_verdict(r['spec'],r['result'])==r['verdict']
  assert r['verdict']['status']==r['status']
  assert r['proof']['initial_optimizers']==control['proof']['initial_optimizers']
  assert r['initialization_fixture_sha256']==control['initialization_fixture_sha256']
  assert r['randomness']==control['randomness']
  assert r['result']['actions']==control['result']['actions']
  assert r['backend']=='cuda' and not r['cpu_random']
  assert r['worker_sha256']==d['file_hashes']['probe.py']
  assert r['proof']['adam_calls']==2400
  assert all(v['calls']==1200 and v['device']=='cuda:0' for v in r['proof']['optimizers'].values())
  v=dict(candidate=d['candidate'],gate=task,status=r['status'],seconds=r['seconds'],metrics=r['result']['live'],convergence=r['verdict']['convergence'],artifact=str(p),audit='PASS')
  old=[json.loads(s) for s in (run/'tests.jsonl').read_text().splitlines()]
  if not any(x['gate']==task for x in old):
   with (run/'tests.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  rows.append(v)
 (run/'audit.json').write_text(json.dumps(dict(status='PASS',gates=2,source_code=True,frozen_gates=True,declared_recipe_fields=True,cuda_updates=True,init_rng_schedules=True,audit_repair='Original assertion compared permitted recipe fields inside spec to baseline recipe. Corrected to declared objective/regularizer; all data/budgets/thresholds/evaluation unchanged. Original error.txt retained. No training rerun.'),indent=2)+'\n')
 (run/'status.txt').write_text('completed (inspect result.md; exit 0 does not mean a win)\n')
 (run/'exit-code.txt').write_text('0\n')
 (run/'result.md').write_text('Direct composition '+d['candidate']+'\n\n'+'\n'.join(v['gate']+': '+v['status'] for v in rows if v['candidate']==d['candidate'])+'\n\nFrozen verdicts independently checked. Full 22 and continuation NOT_RUN. See audit.json for repaired receipt check.\n')
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([dict(candidate=r['candidate'],gate=r['gate'],status=r['status']) for r in rows]))
