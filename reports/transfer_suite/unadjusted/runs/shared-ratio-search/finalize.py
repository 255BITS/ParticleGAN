"""Audit every episode and produce a standalone report for the bounded ratio study."""
from collections import defaultdict
from dataclasses import asdict
import gzip,hashlib,json,math,tarfile
from pathlib import Path
from particlegan import Recipe
from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite.compare_defaults import candidate,effective_spec,ema_verdict,plan
from benchmarks.transfer_suite.protocol import test_verdict
ROOT=Path('/tmp/pr38-shared-ratio-search')
read=lambda p:json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
study=read(ROOT/'study_plan.json');selection=read(ROOT/'selection.json');declared={j['spec']['name']:json.loads(json.dumps(j)) for j in plan()}
records=[];details={};sources=0
for phase,count in [('screen',42),('completion',24)]:
 folder=ROOT/phase;protocol=read(folder/'protocol.json');rows=read(folder/'index.json')['records'];assert len(rows)==count
 suite.verify_source(protocol)
 with tarfile.open(folder/'source.tar.gz') as archive:
  for name,want in protocol['source_sha256'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==want;sources+=1
 for row in rows:
  raw=gzip.decompress((folder/row['artifact']).read_bytes());assert hashlib.sha256(raw).hexdigest()==row['uncompressed_sha256']
  payload=json.loads(raw);spec=payload['spec'];result=payload['result'];recipe=Recipe(**payload['recipe']);name=recipe.name;task=spec['name'];key=(name,task)
  assert key not in details
  original=declared[task]['spec'];assert payload['original_spec']==original
  assert spec==effective_spec(original,recipe)
  assert payload['architecture']==declared[task]['architecture']
  assert payload['reference_sha256']==declared[task]['reference_sha256']
  assert payload['source_sha256']==protocol['source_sha256']
  assert payload['candidate']==json.loads(json.dumps(asdict(candidate(recipe))))
  assert not result.get('error') and len(result['observations'])==24
  assert [p['step'] for p in result['observations']]==[math.ceil(i*spec['steps']/24) for i in range(1,25)]
  assert row['verdict']==payload['verdict']==test_verdict(spec,result)
  assert row['ema_verdict']==payload['ema_verdict']==ema_verdict(spec,result)
  assert row['verdict']['convergence']['complete']
  assert payload['applied']
  for group in payload['applied']:
   role=group['role'];assert group['lr']==recipe.lr*{'g':1.,'d':recipe.d_lr_mult,'prior':recipe.prior_lr_mult}[role]
   assert group['betas']==list((recipe.prior_betas or recipe.betas) if role=='prior' else recipe.betas)
  records.append(row|dict(phase=phase,artifact=f"{phase}/{row['artifact']}"));details[key]=payload
assert len(records)==66
rows=[]
for card in study['screen']['candidates']:
 name=card['name'];trials=[r for r in records if r['recipe']['name']==name];full=name in selection['selected'];assert len(trials)==(19 if full else 7)
 recipes=[r['recipe'] for r in trials];assert all(p==recipes[0] for p in recipes)
 expected=next(r for r in study['resolved_recipes'] if r['name']==name);assert recipes[0]==expected
 counts={kind:dict(passed=sum(r['verdict']['passed'] for r in trials if r['spec']['runner']==kind),attempted=sum(r['spec']['runner']==kind for r in trials),total=total) for kind,total in [('legacy',9),('vector',6),('image',4)]}
 rows.append(dict(name=name,recipe=recipes[0],passed=sum(r['verdict']['passed'] for r in trials),attempted=len(trials),counts=counts,shortfall=sum(r['verdict']['shortfall'] for r in trials),overall='PASS' if full and all(r['verdict']['passed'] for r in trials) else 'FAIL' if full else 'INCOMPLETE',seconds=sum(r['seconds'] for r in trials),failures=[r['spec']['name'] for r in trials if not r['verdict']['passed']],records={r['spec']['name']:r for r in trials}))
rows.sort(key=lambda r:(r['attempted']!=19,-r['passed'],r['shortfall'],r['name']))
write(ROOT/'leaderboard.json',dict(rows=rows))
write(ROOT/'audit.json',dict(episodes=66,complete24=66,errors=0,source_instances_checked=sources,unchanged_recipes_per_candidate=True,all_actual_absolute_optimizer_rates_and_betas_checked=True,all_original_setups_architectures_gates_resources_and_budgets_preserved=True,all_verdicts_recomputed=True,all_episode_and_source_hashes_verified=True))
lines=['# Shared learning-rate ratio search','',
'One fixed recipe per candidate across every test. This bounded study changes only global G:D:particle learning-rate ratios; all six candidates use Rp logistic, b_cap3/κ1.25, spread .05, Adam(0,.99), no particle L2, and the same delayed cosine (60% hold,5% floor). The canonical19-case reference architecture profile, targets, resources, initialization, budgets and metric bounds are unchanged. Seed0, CPU, one Torch thread.',
'', 'Six cards were frozen before42 screening episodes. Two selected cards complete the other12 cases each, without reruns or altered settings:66 episodes total. A complete PASS requires19/19 live cases, every metric sustained for at least five final observations in a complete24-point curve. EMA is separate. Partial candidates cannot compete as complete defaults.',
'', '| Candidate | G / D / particle LR | Required | Data | Images | Live pass / attempted | Overall |', '| --- | --- | ---: | ---: | ---: | ---: | --- |']
for row in rows:
 p=row['recipe'];c=row['counts'];cells=[f"{c[k]['passed']}/{c[k]['total']} ({c[k]['attempted']} tried)" for k in ['legacy','vector','image']]
 lines.append(f"| {row['name']} | {p['lr']:.6g} / {p['lr']*p['d_lr_mult']:.6g} / {p['lr']*p['prior_lr_mult']:.6g} | "+' | '.join(cells)+f" | {row['passed']}/{row['attempted']} | {row['overall']} |")
lines+=['', '**Frozen finalist rule:** '+study['selection'], '', '| Candidate | Failed measured tests |', '| --- | --- |']
for row in rows:lines.append(f"| {row['name']} | "+', '.join(row['failures'])+' |')
lines+=['', '## Complete candidates, every test', '', '| Test | '+' | '.join(selection['selected'])+' |','| --- | '+' | '.join('---' for _ in selection['selected'])+' |']
for name in declared:
 cells=[]
 for candidate_name in selection['selected']:
  row=next(r for r in rows if r['name']==candidate_name);r=row['records'][name];v=r['verdict']
  cells.append(f"[{v['status']} ({v['convergence']['passing_suffix']}/24 final streak)]({r['artifact']})")
 lines.append(f"| {name} | "+' | '.join(cells)+' |')
lines+=['', 'These are inspected development cases. A better score in this search is not proof of a production or out-of-distribution default. No candidate mixes per-task optimizer/formulation settings and no production default was changed.',
'', '[Predeclared study and resolved recipes](study_plan.json) · [Screen plan](screen_plan.json) · [Frozen-rule selection](selection.json) · [Completion plan](completion_plan.json) · [Full summary/metrics](leaderboard.json) · [Audit](audit.json) · [Screen raw episodes/source](screen/index.json) · [Completion raw episodes/source](completion/index.json) · [Screen log](screen.log) · [Completion log](completion.log).',
'', 'Exact commands, run from the isolated checkout at78c872236b70f6e5527169db7143559536e052a1:', '', '```bash', study['command'], 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_default_search --plan /tmp/pr38-shared-ratio-search/completion_plan.json --output /tmp/pr38-shared-ratio-search/completion > /tmp/pr38-shared-ratio-search/completion.log 2>&1', '```', '', 'Use new output directories when reproducing; the runner refuses to overwrite archived runs.', '']
(ROOT/'README.md').write_text('\n'.join(lines))
print(json.dumps(dict(episodes=66,rows=[dict(name=r['name'],passed=r['passed'],attempted=r['attempted'],failures=r['failures']) for r in rows]),indent=2))
