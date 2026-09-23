import gzip
import hashlib
import json
from pathlib import Path
import shutil
from benchmarks.transfer_suite.protocol import test_verdict

ROOT=Path.cwd()
OUT=Path('/tmp/pr36-valid-support-report')
OUT.mkdir(exist_ok=True)
(OUT/'reused').mkdir(exist_ok=True)
HARD=['vector_unequal_mass','vector_unequal_width','vector_overlap']
ALL=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
numerical=['benchmarks/transfer_suite/vector_tasks.py','benchmarks/smart_descent/controller.py',
           'benchmarks/smart_descent/evaluate.py','benchmarks/learned_lr_evaluation.py',
           'benchmarks/locked_shared/observation.py','lib/toy_models.py','lib/toy_metrics.py']
numerical += [str(p.relative_to(ROOT)) for p in (ROOT/'particlegan').glob('*.py')]

def reuse(path,alias):
 raw=path.read_bytes(); payload=json.loads(gzip.decompress(raw))
 for name in numerical:
  assert payload['source_sha256'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),name
 spec=payload.get('spec',payload.get('task')); result=payload['result']
 original=payload.get('original_spec',spec)
 assert all(spec[k]==v for k,v in {'reg_arm':'b_cap','reg_coeff':3.,'reg_kappa':1.25,'prior_reg':.05,'lr':.001,'d_lr_mult':1.5,'prior_lr_mult':10.,'betas':[0.,.99],'hidden':64,'layers':2,'fourier':2,'z_dim':4}.items())
 filename=alias+'__'+spec['name']+'.json.gz'; (OUT/'reused'/filename).write_bytes(raw)
 return dict(candidate=dict(name=alias,overrides=dict(particles=spec['particles'],batch=spec['batch'])),spec=spec,original_spec=original,
             verdict=test_verdict(spec,result),seconds=result['seconds'],live=result['live'],ema=result.get('ema',{}),
             source_path=str(path.relative_to(ROOT)),original_artifact_sha256=hashlib.sha256(raw).hexdigest(),
             artifact='reused/'+filename,source='reused exact numerical source and settings')

records=[]; base=[]
for task in HARD:
 records.append(reuse(ROOT/f'reports/transfer_suite/solvability/vectors/screen/episodes/particles1024__{task}.json.gz','p1024_b128'))
for task in ALL:
 base.append(reuse(ROOT/f'reports/transfer_suite/study/episodes/cosine__{task}.json.gz','baseline_p256_b128'))
index=Path('/tmp/pr36-valid-support-screen/index.json')
if index.exists():
 for record in json.loads(index.read_text())['records']:
  record['source']='new screen';record['artifact']='/tmp/pr36-valid-support-screen/'+record['artifact'];records.append(record)
validation=Path('/tmp/pr36-valid-support-validation/index.json')
if validation.exists():
 for record in json.loads(validation.read_text())['records']:
  record['source']='new validation';record['artifact']='/tmp/pr36-valid-support-validation/'+record['artifact'];records.append(record)
groups={}
for record in records:
 name=record['candidate']['name'];groups.setdefault(name,{})[record['spec']['name']]=record

def key(item):
 name,rs=item
 hard=[rs[t] for t in HARD if t in rs]
 complete=len(hard)==3
 return (not complete,-sum(r['verdict']['passed'] for r in hard),sum(r['verdict']['shortfall'] for r in hard)/len(hard),
         hard[0]['spec']['particles'],hard[0]['spec']['batch'])
ranked=sorted(groups.items(),key=key)
lines=['# Particle support and batch resources — fixed formulation','',
       'All cards preserve Rp logistic, b_cap coefficient 3 / kappa 1.25, prior regularization 0.05, networks, all LRs/Adam settings, targets and thresholds. '
       'Only particle count and batch change. Each Gaussian task has 1200 updates; spiral validation keeps its original 1600. '
       'Seed 0; 24 evaluations and at least five final passing observations using live weights. EMA is separate. '
       'More particles increase support parameters; larger batches process more samples at the same update budget.', '',
       '| Shared particles / batch | Hard sustained /3 | Rare 2% mode | Unequal width | Overlap | Final shortfall | Seconds (3 hard) |',
       '| --- | ---: | --- | --- | --- | ---: | ---: |']
def cell(r):
 if r is None:return '—'
 verdict=r['verdict'];suffix=verdict.get('convergence',{}).get('passing_suffix',0)
 return f"{verdict['status']} (tail {suffix}/24)"
for name,rs in [('baseline_p256_b128',{r['spec']['name']:r for r in base}),*ranked]:
 r=next(iter(rs.values()));hard=[rs[t] for t in HARD if t in rs]
 shortfall=sum(x['verdict']['shortfall'] for x in hard)/len(hard)
 lines.append(f"| {r['spec']['particles']} / {r['spec']['batch']} | {sum(x['verdict']['passed'] for x in hard)}/3 | "+
              ' | '.join(cell(rs.get(t)) for t in HARD)+f" | {shortfall:.4f} | {sum(x['seconds'] for x in hard):.1f} |")
lines+=['','A final snapshot can pass while its final suffix is too short; that remains FAIL. '
        'Selection first uses hard-task sustained count, then average final normalized shortfall, then lower resources. '
        'All twelve cards were declared before training. The 1024/128 card and baseline are reused byte-for-byte from prior runs after checking every numerical source hash.', '',
        '| Shared particles / batch | Unequal-mass mean covariance ≤.85 | Unequal-mass min-eigen ≥.15 | Unequal-width mean covariance ≤.85 | Width min-eigen ≥.15 | Overlap mean error ≤.15 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
for name,rs in ranked:
 if len(rs)<3:continue
 r=next(iter(rs.values()));m=rs[HARD[0]]['live'];w=rs[HARD[1]]['live'];o=rs[HARD[2]]['live']
 lines.append(f"| {r['spec']['particles']} / {r['spec']['batch']} | {m.get('component_covariance_error',float('nan')):.4f} | {m.get('component_min_eigen_ratio',float('nan')):.4f} | {w.get('component_covariance_error',float('nan')):.4f} | {w.get('component_min_eigen_ratio',float('nan')):.4f} | {o.get('mean_error',float('nan')):.4f} |")
lines+=['','## All six valid-data tasks','', '| Shared card | Sustained /6 | '+' | '.join(ALL)+' |', '| --- | ---: | '+' | '.join('---' for _ in ALL)+' |']
for name,rs in [('baseline_p256_b128',{r['spec']['name']:r for r in base}),*ranked]:
 if len(rs)!=6:continue
 lines.append('| '+name+f" | {sum(x['verdict']['passed'] for x in rs.values())}/6 | "+' | '.join(cell(rs.get(t)) for t in ALL)+' |')
lines+=['','Every failure remains in the raw archives. These inspected development cases do not establish a universal default or fresh transfer. '
        'No images, artificial dynamics stressors, seeds, formulation knobs or longer-budget trials enter this sweep.', '']
(OUT/'README.md').write_text('\n'.join(lines))
(OUT/'index.json').write_text(json.dumps(dict(records=records,baseline=base,ranked_candidates=[n for n,_ in ranked]),indent=2))
print('Updated',len(records),'resource records;',len(base),'reused baseline records')
print('Best complete cards',[(n,sum(r['verdict']['passed'] for r in rs.values())) for n,rs in ranked if len(rs)>=3][:3])
