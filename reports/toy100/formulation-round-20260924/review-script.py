"""Independently check and retain the completed 18-proposal round without retraining."""
from collections import Counter
from pathlib import Path
import gzip, hashlib, json, shutil, sys
BASE=Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup')
BATCH=Path('/ml2/hypergan/gan-attempts/formulations-20260924T233426Z')
OUT=BASE/'reports/toy100/formulation-round-20260924'
MAP=dict(critic_formulation='critic-formulation-attempt',particle_geometry='particle-geometry',game_dynamics='game-dynamics')
records=json.loads((BATCH/'batch.json').read_text())
first=sorted(Path(records[0]['directory']).glob('20*'))[-1]/'repo/reports/toy100'/MAP[records[0]['lane']]
sys.path.insert(0,str(first/'prepared/repos/cuda'))
from benchmarks.transfer_suite.protocol import test_verdict
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True)
allrows=[];lane_counts={};audits=[]
for lane in records:
 name=lane['lane'];run=sorted(Path(lane['directory']).glob('20*'))[-1]
 src=run/'repo/reports/toy100'/MAP[name];dst=OUT/name;dst.mkdir(exist_ok=True)
 manifest=json.loads((src/'prepared/prepared-sources.json').read_text())
 for file,h in manifest['cuda'].items():assert sha(src/'prepared/repos/cuda'/file)==h,(name,file)
 shutil.copy2(src/'prepared/prepared-sources.json',dst/'prepared-sources.json')
 for p in src.iterdir():
  if p.is_file() and p.suffix in ('.py','.json','.jsonl','.md','.sh','.log'):
   shutil.copy2(p,dst/p.name)
 shutil.copy2(run/'result.md',dst/'attempt-report.md');shutil.copy2(run/'tests.jsonl',dst/'tests.jsonl')
 shutil.copytree(src/'candidates',dst/'candidates',ignore=shutil.ignore_patterns('runs','__pycache__','*.pyc','*.pt'),dirs_exist_ok=True)
 ledger=[json.loads(line) for line in (run/'tests.jsonl').read_text().splitlines()]
 primary=[r for r in ledger if r['candidate']!='regression' and r['status'] in ('PASS','FAIL','ERROR')]
 assert len(primary)==12
 for row in primary:
  artifact=Path(row['artifact']);r=json.loads(artifact.read_text());task=row['gate'];cand=row['candidate']
  control=json.loads(gzip.decompress((BASE/'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init'/(task+'.json.gz')).read_bytes()))
  expected_spec=dict(control['spec'])
  if name=='critic_formulation':
   for key in ('reg_arm','gan_mode','loss_type'):
    if key in expected_spec:expected_spec[key]=r['config'][key]
  assert r['spec']==expected_spec,(cand,'frozen spec')
  assert test_verdict(r['spec'],r['result'])==r['verdict'],cand
  assert r['status']==row['status']==r['verdict']['status']
  assert r['proof']['initial_optimizers']==control['proof']['initial_optimizers']
  assert r['initialization_fixture_sha256']==control['initialization_fixture_sha256']
  assert r['backend']=='cuda' and not r['cpu_random']
  assert all(v['device']=='cuda:0' for v in r['proof']['optimizers'].values())
  code=src/'candidates'/cand;decl=json.loads((code/'declaration.json').read_text())
  if name=='critic_formulation':
   expected={'probe.py':decl['probe_sha256'],'config.json':decl['config_sha256']}
   allowed=json.loads((BASE/'configs/toy100/constraints_simple_regularization.json').read_text())|decl['changed_fields']|{'device':'cuda:0'}
   assert r['config']==allowed
  elif name=='particle_geometry':expected=decl['code_sha256']
  else:expected=decl['file_hashes']
  for file,h in expected.items():assert sha(code/file)==h,(cand,file)
  assert r['worker_sha256']==expected['probe.py']
  extra=(name=='game_dynamics' and cand=='extragradient')
  if extra:
   correction=r['proof']['game_correction']
   assert r['proof']['adam_calls']==4800 and correction['accepted_optimizer_updates']==2400
   assert correction['rng_replays']==1200 and not correction['pending']
  else:
   assert r['proof']['adam_calls']==2400
   assert r['randomness']==control['randomness'],(cand,'RNG')
  if name!='critic_formulation':assert r['config']==control['config']
  if extra and task=='mode_hold':
   actual=Counter(json.dumps(a,sort_keys=True) for a in r['result']['actions'])
   expected_actions=Counter(json.dumps(a,sort_keys=True) for a in control['result']['actions'])
   assert actual==Counter({k:2*v for k,v in expected_actions.items()}),(cand,'predictor/corrector schedules')
  else:assert r['result']['actions']==control['result']['actions'],(cand,'schedules')
  target=dst/'results'/cand/(task+'.json.gz');target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(gzip.compress(artifact.read_bytes(),mtime=0))
  allrows.append(dict(lane=name,candidate=cand,gate=task,status=r['status'],live=r['result']['live'],convergence=r['verdict']['convergence'],shortfall=r['verdict']['shortfall'],seconds=r['seconds'],adam_calls=r['proof']['adam_calls'],equal_compute=not extra,artifact=str(target.relative_to(OUT))))
 lane_counts[name]=dict(Counter(r['status'] for r in primary))
 audits.append(dict(lane=name,gates=12,prepared_files=len(manifest['cuda']),status='PASS',checks=['frozen specification and complete sustained verdict','candidate source hashes','original initialization','CUDA optimizer device and updates','native RNG except declared extragradient replay','original schedules','declared configuration']))
assert len(allrows)==36 and len(set((r['lane'],r['candidate']) for r in allrows))==18
summary=dict(candidates=18,training_gates=36,counts=dict(Counter(r['status'] for r in allrows)),per_lane=lane_counts,qualified_candidates=[],full22='NOT_RUN for every candidate',own_state_stability='NOT_RUN for every candidate',audit='PASS',rows=allrows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'independent-audit.json').write_text(json.dumps(dict(status='PASS',training_gates=36,checks=audits),indent=2)+'\n')
lines=['# Formulation round: 18 proposals, no joint winner','','**36 GPU training gates: 7 PASS, 29 FAIL, 0 ERROR.** No candidate passes both','ring and unequal mass, so none advances to the full 22 or post-convergence tests.','The full native GPU reference remains 16/22.','','| Candidate | Ring modes / HQ | Ring verdict | Unequal eigen ratio | Unequal verdict |','|---|---:|---|---:|---|']
for lane in MAP:
 names=list(dict.fromkeys(r['candidate'] for r in allrows if r['lane']==lane))
 for cand in names:
  pair={r['gate']:r for r in allrows if r['lane']==lane and r['candidate']==cand};ring=pair['mode_hold'];rare=pair['vector_unequal_mass']
  lines.append(f"| {cand} | {ring['live']['modes']} / {ring['live']['hq']:.4f} | {ring['status']} ({ring['convergence']['passing_suffix']}/5 suffix) | {rare['live']['component_min_eigen_ratio']:.5f} | {rare['status']} ({rare['convergence']['passing_suffix']}/5 suffix) |")
lines+=['','A suffix over five is sufficient only when all frozen metrics pass. Displayed','final values do not override sustained verdicts. In particular optimistic Adam','reaches eight ring modes but has only three final passing checks.','','Ra logistic + R1+R2 (`c05_ra_r1r2`) passes ring but over-spreads the rare component','(covariance error 1.34110 versus the .85 gate). R1 only fixes unequal mass but loses','ring coverage. Five particle updates pass unequal mass and fail ring. These are','partial results from different formulations, not interchangeable passes.','','All training uses audited CPU initial parameters with CUDA models, gradients and','optimizer states, fixed seeds and frozen 1200-step budgets. Original schedules and','auxiliary host losses are retained. ExtraAdam performs two gradient blocks per','outer step: its result is equal-step, not equal-compute. All other proposals add no','model forward/backward evaluations, though tensor-operation costs differ.','','The three agent audits reported 12 passing mechanism/audit gates (not 12 toy','passes). An independent review additionally regraded all 36 runs and checked their','frozen specifications, exact candidate sources, initial parameters, schedules,','CUDA updates and random receipts. [Audit](independent-audit.json) · [Results](summary.json).','','Candidate code, prelaunch declarations, raw results compressed as JSON, original','command receipts and lane reports are retained here. Prepared baseline sources','are reconstructed using the existing checksum-verifying source preparer, avoiding','another copy of the archive. Failed final-state checkpoints remain in the original','local attempt directories; no continuation or qualification depends on them.','','Next measured follow-ups: Ra + real-point R1 / fake-point cap; and Ra + R1+R2','combined separately with exposure-mean and shared-coordinate particle updates.','Each composition must earn its own two blocker passes. No formulation is promoted.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
