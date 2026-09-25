"""Audit completed search evidence using only the standard library; no training."""
from pathlib import Path
import gzip,hashlib,io,json,math,tarfile
ROOT=Path('/tmp/pr36-valid-recipe')
sha=lambda b:hashlib.sha256(b).hexdigest()
plan=json.loads((ROOT/'screen_plan.json').read_text())
assert len(plan['candidates'])==18
cards={c['name']:c for c in plan['candidates']}
rows=json.loads((ROOT/'index.json').read_text())['records']
seen=set();protocol=None
fixed={'gan_mode':'rp','loss_type':'logistic','reg_arm':'b_cap','reg_coeff':3.,'reg_kappa':1.25,'prior_reg':.05,'particles':256}
for phase in sorted({r['phase'] for r in rows}):
 p=json.loads((ROOT/phase/'protocol.json').read_text())
 if protocol is None:protocol=p
 assert protocol==p
 with tarfile.open(ROOT/phase/'source.tar.gz','r:gz') as tf:
  members=tf.getmembers();assert {x.name for x in members}==set(p['source_sha256'])
  for member in members:assert sha(tf.extractfile(member).read())==p['source_sha256'][member.name]

def passes(metrics,bounds):
 return all(isinstance(metrics.get(k),(int,float)) and not isinstance(metrics[k],bool) and math.isfinite(metrics[k]) and (metrics[k]>=b if op=='>=' else metrics[k]<=b) for k,op,b in bounds)
for r in rows:
 raw=gzip.decompress((ROOT/r['artifact']).read_bytes());assert sha(raw)==r['uncompressed_sha256']
 e=json.loads(raw);name=e['candidate']['name'];task=e['spec']['name'];key=(name,task)
 assert key not in seen;seen.add(key)
 assert e['candidate']==cards[name]
 spec=e['spec'];orig=e['original_spec'];result=e['result'];v=e['verdict']
 assert not result.get('error')
 assert spec['runner']=='vector' and spec['tier']=='ranking' and spec['split']=='development'
 assert spec==orig|e['candidate']['overrides']
 assert spec['thresholds']==orig['thresholds'] and spec['steps']==orig['steps']
 assert all(spec[k]==value for k,value in fixed.items())
 assert spec['hidden']==64 and spec['layers']==2 and spec['fourier']==2 and spec['batch']==128 and spec['z_dim']==4
 assert e['source_sha256']==protocol['source_sha256']
 assert len(result['observations'])==24
 assert [p['step'] for p in result['observations']]==[math.ceil(i*spec['steps']/24) for i in range(1,25)]
 suffix=0
 for point in reversed(result['observations']):
  if not passes(point,spec['thresholds']):break
  suffix+=1
 assert suffix==v['convergence']['passing_suffix']
 final=passes(result['live'],spec['thresholds'])
 assert (final and suffix>=5)==v['passed']
 assert result['update_counts']=={role:len(range(0,spec['steps'],spec[f'{role}_every'])) for role in ('d','g')}
 assert r['verdict']==v and r['live']==result['live'] and r['ema']==result['ema']
ver={'verified':True,'episodes':len(rows),'live_observations':24*len(rows),'distinct_cards':len({r['candidate']['name'] for r in rows}),
     'source_files':len(protocol['source_sha256']),'recorded_wall_seconds':sum(r['seconds'] for r in rows),
     'formulation_unchanged':fixed,'original_architecture_and_budgets':True,'complete_card_count':sum(sum(r['candidate']['name']==name for r in rows)==6 for name in cards)}
(ROOT/'validation.json').write_text(json.dumps(ver,indent=2,sort_keys=True)+'\n')
print(json.dumps(ver,indent=2))
