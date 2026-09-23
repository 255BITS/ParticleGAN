"""Validate immutable numerical evidence, then write readable handoff indexes."""
from collections import Counter
import gzip,hashlib,json,math,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-d')
from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite.protocol import test_verdict
ROOT=Path('/tmp/pr36-valid-softplus-refinement')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
assert (ROOT/'run.log').read_text().rstrip().endswith('COMPLETE 24')
plan=read(ROOT/'plan.json');protocol=read(ROOT/'protocol.json');records=read(ROOT/'index.json')['records'];selection=read(ROOT/'selection.json')
assert len(records)==24 and len({(r['candidate']['name'],r['spec']['name']) for r in records})==24
assert Counter(r['phase'] for r in records)=={'screen':18,'full_six_completion':6}
suite.verify_source(protocol)
for relative,expected in protocol['experiment_source_sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected
archive=tarfile.open(ROOT/'source.tar.gz')
for name,expected in protocol['source_sha256'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected
specs={s['name']:s for s in read(ROOT/'task_specs.json')};checked=[]
for record in records:
 raw=gzip.decompress((ROOT/record['artifact']).read_bytes());assert hashlib.sha256(raw).hexdigest()==record['uncompressed_sha256']
 payload=json.loads(raw);result=payload['result'];spec=payload['spec'];original=payload['original_spec'];card=payload['candidate']
 assert original==specs[spec['name']]
 expected=original|card['overrides']|{'research_discriminator':card['architecture']}
 assert spec==expected
 assert payload['source_sha256']==protocol['source_sha256']
 assert payload['experiment_source_sha256']==protocol['experiment_source_sha256']
 assert test_verdict(spec,result)==record['verdict']==payload['verdict']
 assert not result.get('error') and record['verdict']['convergence']['complete']
 assert len(result['observations'])==24
 assert [o['step'] for o in result['observations']]==[math.ceil(i*spec['steps']/24) for i in range(1,25)]
 assert record['live']==result['live'] and record['ema']==result['ema']
 checked.append(dict(artifact=record['artifact'],status=record['verdict']['status'],suffix=record['verdict']['convergence']['passing_suffix']))
checks={r['candidate']:r for r in read(ROOT/'architecture_checks.json')}
rows=[]
for card in plan['candidates']:
 trials=[r for r in records if r['candidate']['name']==card['name']]
 rows.append(dict(candidate=card['name'],architecture=card['architecture'],overrides=card['overrides'],parameters=checks[card['name']]['discriminator_parameters'],passed=sum(r['verdict']['passed'] for r in trials),attempted=len(trials),seconds=sum(r['seconds'] for r in trials),cases={r['spec']['name']:dict(status=r['verdict']['status'],suffix=r['verdict']['convergence']['passing_suffix'],confirmed_step=r['verdict']['convergence']['confirmed_step'],artifact=r['artifact'],final_live=r['live'],final_ema=r['ema']) for r in trials}))
write(ROOT/'leaderboard.json',dict(rows=rows,selected=selection['selected'],selection=plan['selection']))
audit=dict(episodes_checked=len(records),complete24=len(records),errors=0,all_episode_uncompressed_hashes_verified=True,all_exact_source_hashes_verified=True,all_verdicts_recomputed=True,all_original_targets_thresholds_G_optimizer_recipe_and_budgets_preserved=True,only_discriminator_activation_width_depth_changed=True,worktree_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/ml2/hypergan/ParticleGAN-pr36-valid-d',text=True).strip(),source_files=len(protocol['source_sha256']),checks=checked)
write(ROOT/'audit.json',audit)
short=lambda name:name.replace('vector_','')
lines=['# Smooth discriminator refinement, original recipe','',
'An adaptive six-card follow-up to the previously observed Softplus β5 / D64×2 near-miss. The six candidates and selection rule were frozen before these episodes. All 24 new attempts, including failures, are retained. These are inspected development tasks; this is architecture search, not independent validation.',
'', 'Every episode uses the original shared recipe: Rp logistic, b_cap coefficient 3 / κ1.25, prior regularization .05, no particle L2, Adam (0,.99), G LR .001, D LR .0015, prior LR .01, cosine, 256 particles, batch128, 1:1 updates, unchanged G. D alone changes: axis Fourier2, Softplus β, width and depth. Budgets remain 1200 updates except spiral1600. Seed0 and one CPU thread; no seed sweeps.',
'', 'PASS requires unchanged live bounds and five final passing observations from all24 scheduled checkpoints. `FAIL(n)` displays the final passing suffix length. EMA is recorded separately and never determines success. A missing case is unrun, not a failure or pass.',
'', '| Discriminator | D params | Rare mass | Unequal width | Overlap | Two broad | Anisotropic | Spiral | Sustained / attempted | Seconds |',
'| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |']
order=plan['screen_tasks']+plan['validation_tasks']
for row in rows:
 cells=[]
 for task in order:
  case=row['cases'].get(task)
  cells.append('unrun' if case is None else f"[{case['status']}({case['suffix']})]({case['artifact']})")
 lines.append(f"| {row['candidate']} | {row['parameters']} | "+' | '.join(cells)+f" | {row['passed']}/{row['attempted']} | {row['seconds']:.2f} |")
rare=[r['candidate'] for r in rows if r['cases']['vector_unequal_mass']['status']=='PASS']
lines+=['',('Rare-mass sustained witnesses: '+', '.join(rare)+'.') if rare else 'No new rare-mass sustained witness was found in this bounded refinement.',
'', 'Different discriminator architectures may support different cases under the same recipe. This table does not claim a single discriminator solves all data tests. Timings are observed serial CPU wall times including metrics, and are not a controlled speed comparison.',
'', '**Finalist rule:** '+plan['selection'], '', '**Selected:** '+', '.join(selection['selected'])+'.',
'', '| Candidate | Task | Final live failing bounds |', '| --- | --- | --- |']
for r in records:
 failures=[m for m in r['verdict'].get('metrics',[]) if m['status']!='PASS']
 text=', '.join(f"{m['metric']} {m['value']:.6g} (bound {m['threshold']})" for m in failures) if failures else ('none; sustained PASS' if r['verdict']['passed'] else 'none; fewer than five final passing observations')
 lines.append(f"| {r['candidate']['name']} | {short(r['spec']['name'])} | {text} |")
lines+=['', '[Frozen plan](plan.json) · [Exact task specs](task_specs.json) · [Resolved episodes, metrics, EMA and hashes](index.json) · [Leader cells](leaderboard.json) · [Finalist selection](selection.json) · [Audit](audit.json) · [Static copy/gradient parity checks](architecture_checks.json) · [Runtime and source hashes](protocol.json) · [Exact numerical source](source.tar.gz) · [Driver](run.py) · [Tailable execution log](run.log).',
'', 'Source implementation was copied byte-for-byte from the smooth-D study (SHA256 `'+plan['source_origin_sha256']+'`). All six constructed models matched that source exactly in initial states, outputs, input gradients, cap penalty, and parameter gradients under the same seed0. This is static copy parity, not a rerun of the previous training episode.', '']
(ROOT/'README.md').write_text('\n'.join(lines))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'}
write(ROOT/'archive_manifest.json',dict(files=manifest,immutable_handoff=True))
print(json.dumps(dict(episodes=len(records),rare_witnesses=rare,selected=selection['selected'],source_files=len(protocol['source_sha256']))))
