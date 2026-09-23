"""Audit frozen D-only attempts and write a readable width/rare handoff."""
from dataclasses import asdict
import gzip,hashlib,json,math,tarfile
from pathlib import Path
from benchmarks.transfer_suite.compare_defaults import plan,effective_spec,candidate,ema_verdict
from benchmarks.transfer_suite.shared_variants import architecture_spec,architecture_identity
from benchmarks.transfer_suite.shared_discriminator_search import recipe
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.locked_shared.baseline import score_metrics
ROOT=Path('/tmp/pr38-shared-width-search')
def read(p):return json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
originals={j['spec']['name']:json.loads(json.dumps(j)) for j in plan()}
records=[];payloads={};seen=set();source_checks=0
for phase in ['screen','refinement','last_refinement','cross']:
 folder=ROOT/phase
 if not folder.exists():continue
 protocol=read(folder/'protocol.json')
 with tarfile.open(folder/'source.tar.gz') as archive:
  for path,want in protocol['source_sha256'].items():assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==want;source_checks+=1
 phase_rows=read(folder/'index.json')['records']
 assert len(phase_rows)==len(protocol['declaration']['architectures'])*len(protocol['declaration']['tasks'])
 assert {(r['architecture'],r['spec']['name']) for r in phase_rows}=={(a,t) for a in protocol['declaration']['architectures'] for t in protocol['declaration']['tasks']}
 for row in phase_rows:
  raw=gzip.decompress((folder/row['artifact']).read_bytes());assert hashlib.sha256(raw).hexdigest()==row['uncompressed_sha256']
  payload=json.loads(raw);spec=payload['spec'];result=payload['result'];name=spec['name'];card=payload['discriminator_variant'];original=originals[name]
  assert payload['recipe']==json.loads(json.dumps(recipe().to_dict()))
  assert payload['original_spec']==original['spec']
  assert payload['reference_sha256']==original['reference_sha256']
  assert spec==effective_spec(architecture_spec(original['spec'],card),recipe())
  assert payload['candidate']==json.loads(json.dumps(asdict(candidate(recipe()))))
  assert payload['source_sha256']==protocol['source_sha256']
  identity=(name,json.dumps(architecture_identity(spec),sort_keys=True));assert identity not in seen;seen.add(identity)
  assert row['verdict']==payload['verdict']==test_verdict(spec,result)
  assert row['ema_verdict']==payload['ema_verdict']==ema_verdict(spec,result)
  assert not result.get('error') and row['verdict']['convergence']['complete']
  assert len(result['observations'])==24 and [o['step'] for o in result['observations']]==[math.ceil(i*spec['steps']/24) for i in range(1,25)]
  assert {g['role'] for g in payload['applied']}=={'g','d','prior'}
  for group in payload['applied']:
   assert group['lr']=={'g':.00425,'d':.00425,'prior':.0085}[group['role']] and group['betas']==[0.,.99]
  indexed=row|dict(phase=phase,artifact=f"{phase}/{row['artifact']}");records.append(indexed);payloads[(row['architecture'],name)]=payload
assert sum(r['phase']=='screen' for r in records)==16
assert sum(r['phase']=='refinement' for r in records)==12
assert sum(r['phase']=='last_refinement' for r in records)==6
width=[r for r in records if r['spec']['name']=='vector_unequal_width'];winners=[r['architecture'] for r in width if r['verdict']['passed']]
assert {r['architecture'] for r in records if r['spec']['name']=='vector_unequal_mass'}==set(winners)
width.sort(key=lambda r:(not r['verdict']['passed'],r['verdict']['shortfall'],-r['verdict']['convergence']['passing_suffix'],r['architecture']))
write(ROOT/'index.json',dict(records=records))
curves={}
for row in records:
 payload=payloads[(row['architecture'],row['spec']['name'])]
 curves.setdefault(row['architecture'],{})[row['spec']['name']]=dict(artifact=row['artifact'],thresholds=row['spec']['thresholds'],observations=[o|{'all_live_bounds_pass':all(m['status']=='PASS' for m in score_metrics(o,row['spec']['thresholds']))} for o in payload['result']['observations']])
write(ROOT/'curves.json',curves)
write(ROOT/'audit.json',dict(episodes=len(records),complete24=len(records),errors=0,source_instances_checked=source_checks,all_gzip_and_source_hashes_verified=True,all_verdicts_and_EMA_recomputed=True,all_recipes_exact_shared_c6=True,all_actual_optimizer_roles_LRs_betas_verified=True,only_discriminator_changes=True,all_targets_generators_resources_budgets_gates_preserved=True,no_normalized_architecture_duplicates=True,width_winners=winners))
lines=['# Unequal-width discriminator stability search','',
'Every trial uses exact `shared_c6`: Rp logistic, cap6/κ1.25, spread .05, no particle L2, Adam(0,.99), G/D LR .00425 and particle LR .0085, delayed cosine60%/floor5%. Only the discriminator changes; original G, data, initialization rules,256 particles,batch128,1200 updates and all behavioral bounds remain fixed. Seed0, CPU, one Torch thread. EMA is separate.',
'', 'Sixteen initial raw-input smooth architectures were declared before results and checked against prior normalized discriminator identities. After that negative screen, a separately frozen12-card head/skip refinement was authorized and recorded in scope-v2; the original plan and numerical sources were preserved. A final explicitly authorized six-card scope-v3 tests Softplus sharpness and main-branch gains for the SiLU skip critic. Thus34 unique width architectures are measured; no failed attempt is omitted. Every failed curve is retained. A live PASS requires all24 measurements and at least five final passing checks. Final-only success is a failure. Every width winner is checked on rare mass with exactly the same architecture and recipe; other data tests are unrun for these cards.',
'', '**Sustained width winners:** '+(', '.join(winners) if winners else 'none')+'. '+str(len(records))+' episodes are retained: '+str(len(width))+' width trials and '+str(len(records)-len(width))+' rare cross-checks.',
'', '| Discriminator | D params | Width live | Final streak | Width EMA | Rare live | Final width failing bounds |', '| --- | ---: | --- | ---: | --- | --- | --- |']
for row in width:
 name=row['architecture'];v=row['verdict'];params=next(g['parameters'] for g in row['applied'] if g['role']=='d');rare=next((r for r in records if r['architecture']==name and r['spec']['name']=='vector_unequal_mass'),None)
 bad=', '.join(f"{m['metric']}={m['value']:.6g}" for m in v['metrics'] if m['status']!='PASS') or ('none' if v['passed'] else 'none; stability fails')
 rarecell='unrun' if rare is None else f"[{rare['verdict']['status']}]({rare['artifact']})"
 lines.append(f"| {name} | {params} | [{v['status']}]({row['artifact']}) | {v['convergence']['passing_suffix']}/24 | {row['ema_verdict']['status']} | {rarecell} | {bad} |")
lines+=['', '## Last five measurements of the strongest width results', '', 'All24 live and EMA observations for every attempt are in [curves.json](curves.json). The columns below are steps1000,1050,1100,1150,1200.', '', '| D | Metric / bound | 1000 | 1050 | 1100 | 1150 | 1200 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for row in width[:max(3,len(winners))]:
 data=curves[row['architecture']]['vector_unequal_width'];late=data['observations'][-5:]
 for key,op,bound in data['thresholds']:
  lines.append(f"| {row['architecture']} | {key} {op} {bound} | "+' | '.join(f"{p[key]:.6g}" for p in late)+' |')
 lines.append(f"| {row['architecture']} | All live bounds | "+' | '.join('PASS' if p['all_live_bounds_pass'] else 'FAIL' for p in late)+' |')
lines+=['', 'These are inspected development cases. Architecture support within a fixed recipe does not establish one universal discriminator or a production default. All raw-input cards reuse the exact `shared_critic_v1` implementation; source changes only add a separate catalog and a wrapper around the canonical architecture runner. Static checks exercised pointwise scoring and active-cap double backprop; the1000× head scaling belongs only to static derivative checks and is never used in training.',
'', '[Predeclared study/full cards](study_plan.json) · [Actual recipes, optimizer receipts and all records](index.json) · [Audit](audit.json) · [Static derivative/dedup checks](static_checks.json) · [Screen plan](screen_plan.json) · [Screen log](screen.log) · [Screen exact source archive](screen/source.tar.gz) · [Separate refinement scope](refinement_scope.json) · [Refinement full cards/plan](refinement_plan.json) · [Refinement checks](refinement_checks.json) · [Refinement log](refinement.log) · [Refinement source](refinement/source.tar.gz) · [Final scope-v3](last_refinement_scope.json) · [Final plan](last_refinement_plan.json) · [Final checks](last_refinement_checks.json) · [Final log](last_refinement.log) · [Final exact source](last_refinement/source.tar.gz) · [Rare cross plan](cross_plan.json) · [Rare cross log](cross.log) · [Rare cross source](cross/source.tar.gz).', '', 'Reproduce the initial catalog with `python -m benchmarks.transfer_suite.shared_width_search --plan screen_plan.json --output /tmp/new-width-screen`; use `benchmarks.transfer_suite.shared_width_refinement` and `refinement_plan.json` for the second stage, and `benchmarks.transfer_suite.shared_width_last_refinement` with `last_refinement_plan.json` for the third. Run from the integrated checkout; use new output directories.', '', 'No generated aggregate or production files were edited. Parent integration should append these indexes to the existing `shared_c6` row after independent replay of any witness.', '']
(ROOT/'README.md').write_text('\n'.join(lines))
print(json.dumps(dict(episodes=len(records),width_winners=winners,rare_winners=[r['architecture'] for r in records if r['spec']['name']=='vector_unequal_mass' and r['verdict']['passed']],seconds=sum(r['seconds'] for r in records)),indent=2))
