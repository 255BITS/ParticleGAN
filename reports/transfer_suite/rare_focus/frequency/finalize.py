"""Read-only numerical audit and report assembly after the declared experiments."""
from collections import Counter
import gzip,hashlib,json,math,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-d')
from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.locked_shared.baseline import score_metrics
ROOT=Path('/tmp/pr36-rare-frequency-round4')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
plan=read(ROOT/'plan.json');protocol=read(ROOT/'protocol.json');records=read(ROOT/'index.json')['records'];selection=read(ROOT/'selection.json')
expected=8+5*len(selection['selected'])
assert (ROOT/'run.log').read_text().rstrip().endswith(f'COMPLETE {expected}')
assert len(records)==expected and len({(r['candidate']['name'],r['spec']['name']) for r in records})==expected
assert sum(r['phase']=='screen' for r in records)==8
assert sum(r['phase']=='full_six_completion' for r in records)==expected-8
assert len({c['name'] for c in plan['candidates']})==8
suite.verify_source(protocol)
for relative,want in protocol['experiment_source_sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==want
with tarfile.open(ROOT/'source.tar.gz') as archive:
 for name,want in protocol['source_sha256'].items():assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==want
specs={s['name']:s for s in read(ROOT/'task_specs.json')};checked=[];curves={}
for record in records:
 raw=gzip.decompress((ROOT/record['artifact']).read_bytes());assert hashlib.sha256(raw).hexdigest()==record['uncompressed_sha256']
 payload=json.loads(raw);result=payload['result'];spec=payload['spec'];original=payload['original_spec'];card=payload['candidate']
 assert original==specs[spec['name']]
 assert spec==original|card['overrides']|{'research_discriminator':card['architecture']}
 assert payload['source_sha256']==protocol['source_sha256']
 assert payload['experiment_source_sha256']==protocol['experiment_source_sha256']
 assert test_verdict(spec,result)==record['verdict']==payload['verdict']
 assert not result.get('error') and record['verdict']['convergence']['complete']
 assert len(result['observations'])==24
 assert [o['step'] for o in result['observations']]==[math.ceil(i*spec['steps']/24) for i in range(1,25)]
 assert record['live']==result['live'] and record['ema']==result['ema']
 checked.append(dict(artifact=record['artifact'],status=record['verdict']['status'],suffix=record['verdict']['convergence']['passing_suffix']))
 if spec['name']=='vector_unequal_mass':
  curves[card['name']]=dict(artifact=record['artifact'],thresholds=spec['thresholds'],verdict=record['verdict'],observations=[o|{'all_live_bounds_pass':all(m['status']=='PASS' for m in score_metrics(o,spec['thresholds']))} for o in result['observations']])
checks={r['candidate']:r for r in read(ROOT/'architecture_checks.json') if 'candidate' in r}
rows=[]
for card in plan['candidates']:
 trials=[r for r in records if r['candidate']['name']==card['name']]
 rare=next(r for r in trials if r['spec']['name']=='vector_unequal_mass')
 assert len(trials)==(6 if rare['verdict']['passed'] else 1)
 rows.append(dict(candidate=card['name'],architecture=card['architecture'],overrides=card['overrides'],parameters=checks[card['name']]['discriminator_parameters'],passed=sum(r['verdict']['passed'] for r in trials),attempted=len(trials),seconds=sum(r['seconds'] for r in trials),rare_verdict=rare['verdict'],cases={r['spec']['name']:dict(status=r['verdict']['status'],suffix=r['verdict']['convergence']['passing_suffix'],confirmed_step=r['verdict']['convergence']['confirmed_step'],artifact=r['artifact'],final_live=r['live'],final_ema=r['ema']) for r in trials}))
write(ROOT/'leaderboard.json',dict(rows=rows,selected=selection['selected'],selection=plan['selection']))
write(ROOT/'rare_curves.json',curves)
audit=dict(episodes_checked=len(records),rare_cards=8,complete24=len(records),errors=0,all_episode_uncompressed_hashes_verified=True,all_exact_source_hashes_verified=True,all_verdicts_recomputed=True,all_original_targets_thresholds_G_optimizer_recipe_and_budgets_preserved=True,only_discriminator_frequency_basis_beta_changed=True,worktree_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/ml2/hypergan/ParticleGAN-pr36-valid-d',text=True).strip(),source_files=len(protocol['source_sha256']),checks=checked)
write(ROOT/'audit.json',audit)
lines=['# Rare-mode frequency refinement, final round 4','',
'Eight frequency-basis cards were frozen before this final adaptive architecture round. Frozen-critic forensics identified dominant contraction from the lowest pi harmonic. This follow-up tests lower generic frequencies and raw-only controls. It is a hypothesis about trainable feature bases, not a demonstrated causal remedy. Every failed architecture and all24 live/EMA observations are retained. All sustained rare-case winners receive the other five valid data tests without changes. These are inspected development cases, not held-out confirmation.',
'', 'Original shared recipe: Rp logistic, b_cap coefficient3 / κ1.25, prior regularization .05, no particle L2, Adam (0,.99), G LR .001, D LR .0015, prior LR .01, cosine, 256 particles, batch128, 1:1 updates, unchanged G. Only discriminator Fourier frequencies change around Softplus β5/6; width96 and depth2 stay fixed. Two axis bands use frequencies [pi,2pi] multiplied by .125/.25/.5; raw-only controls use zero Fourier bands. Frequency choices are generic constants, independent of data, labels, target geometry, or component identities. Feature amplitudes are unchanged. Every rare run uses1200 updates; any cross-check retains1200 except spiral1600. Seed0, one CPU thread; no seed sweeps.',
'', 'PASS means unchanged live metric bounds, all24 expected observations, and at least five final passing observations. EMA never determines success. Missing cross-checks are unrun.',
'', 'Names encode Softplus beta (`b5` or `b6`) and the global frequency multiplier; raw-only cards omit Fourier features. Prior architecture inventories and sibling confirmation excluded exact raw-only D96×2 Softplus5/6 duplicates before execution.',
'', '| Discriminator | D params | Rare sustained | Final passing suffix | Passing observations /24 | Final mean normalized shortfall | Seconds |',
'| --- | ---: | --- | ---: | ---: | ---: | ---: |']
ordered=sorted(rows,key=lambda r:(not r['rare_verdict']['passed'],r['rare_verdict']['shortfall'],-r['rare_verdict']['convergence']['passing_suffix'],r['candidate']))
for row in ordered:
 v=row['rare_verdict'];c=row['cases']['vector_unequal_mass']
 lines.append(f"| {row['candidate']} | {row['parameters']} | [{v['status']}]({c['artifact']}) | {v['convergence']['passing_suffix']} | {v['convergence']['passing_observations']}/24 | {v['shortfall']:.6g} | {row['seconds']:.2f} |")
lines+=['', 'Rare sustained winners: '+(', '.join(selection['selected']) if selection['selected'] else '**none**')+'.',
'', '## Complete late curves for the three closest rare results',
'', 'The five columns are steps1000,1050,1100,1150,1200. A good final value alone cannot pass. [All24 observations for every architecture](rare_curves.json) include live and EMA values, measurement times, and recomputed per-check success.',
'', '| Discriminator | Metric / unchanged bound | 1000 | 1050 | 1100 | 1150 | 1200 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for row in ordered[:3]:
 data=curves[row['candidate']];tail=data['observations'][-5:]
 for metric,op,bound in data['thresholds']:
  lines.append(f"| {row['candidate']} | {metric} {op} {bound} | "+' | '.join(f"{o[metric]:.6g}" for o in tail)+' |')
 lines.append(f"| {row['candidate']} | ALL live bounds | "+' | '.join('PASS' if o['all_live_bounds_pass'] else 'FAIL' for o in tail)+' |')
if selection['selected']:
 lines+=['','## Same-architecture full-six checks','','| Discriminator | Rare mass | Unequal width | Overlap | Two broad | Anisotropic | Spiral | Sustained /6 |','| --- | --- | --- | --- | --- | --- | --- | ---: |']
 for row in rows:
  if row['candidate'] not in selection['selected']:continue
  names=['vector_unequal_mass','vector_unequal_width','vector_overlap','vector_two_broad','vector_anisotropic','vector_spiral']
  cells=[f"[{row['cases'][n]['status']} ({row['cases'][n]['suffix']})]({row['cases'][n]['artifact']})" for n in names]
  lines.append(f"| {row['candidate']} | "+' | '.join(cells)+f" | {row['passed']}/6 |")
lines+=['', 'All numerical sources, episode bytes, original targets/gates and complete observation schedules were audited. The research `FrequencyScaledCritic` subclasses the exact existing smooth critic and changes only its fixed Fourier frequency buffer. Neutral multiplier1 exactly matches baseline states, outputs, input/parameter gradients, and cap penalties for Fourier2 and raw/Fourier0. All eight cards pass finite input-Hessian and active cap double-backprop checks. Only those static checks multiply output weights1000× to activate the cap; training uses normal initialization and the unchanged cap. Cap differentiation remains in original data coordinates. Architecture support across cases must be labeled separately from a single shared discriminator.',
'', '[Frozen plan](plan.json) · [Exact task specs](task_specs.json) · [Resolved records, metrics, EMA and hashes](index.json) · [Audit](audit.json) · [Static checks](architecture_checks.json) · [Runtime/source hashes](protocol.json) · [Exact source](source.tar.gz) · [Driver](run.py) · [Execution log](run.log).',
'', 'Timings are observed CPU episode wall times including measurements and are not a controlled speed comparison. No required regression, image, stress or diagnostic cases were used to select these architectures.', '']
(ROOT/'README.md').write_text('\n'.join(lines))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='archive_manifest.json'}
write(ROOT/'archive_manifest.json',dict(files=manifest,immutable_handoff=True))
print(json.dumps(dict(episodes=len(records),rare_witnesses=selection['selected'],source_files=len(protocol['source_sha256']),seconds=sum(r['seconds'] for r in records))))
