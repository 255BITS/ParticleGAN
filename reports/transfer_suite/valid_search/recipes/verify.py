"""Portable byte and numerical-verdict audit; never trains a model."""
from pathlib import Path
import gzip,hashlib,json,math,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):return json.loads(gzip.decompress((ROOT/path).read_bytes()))
inv=json.loads((ROOT/'inventory.json').read_text());roundtrips=0
for file in inv['files']:
 data=(ROOT/file['path']).read_bytes()
 assert sha(data)==file['sha256'] and len(data)==file['bytes'],file['path']
 if file['path'].endswith('.gz'):
  raw=gzip.decompress(data);roundtrips+=1
  assert sha(raw)==file['uncompressed_sha256'] and len(raw)==file['uncompressed_bytes'],file['path']
protocol=read('screen/protocol.json.gz')
assert read('cross/protocol.json.gz')==read('cross_secondary/protocol.json.gz')==protocol
with tarfile.open(ROOT/'source.tar.gz','r:gz') as archive:
 members=archive.getmembers();assert {m.name for m in members}==set(protocol['source_sha256'])
 for m in members:assert m.isfile() and sha(archive.extractfile(m).read())==protocol['source_sha256'][m.name]
plan=read('screen_plan.json.gz');cards={c['name']:c for c in plan['candidates']}
assert len(cards)==18
rows=read('index.json.gz')['records'];assert len(rows)==63
seen=set();full={};wall=0.;ema_passes=0
fixed={'gan_mode':'rp','loss_type':'logistic','reg_arm':'b_cap','reg_coeff':3.,'reg_kappa':1.25,'prior_reg':.05,'particles':256}
def passes(m,bounds):
 return all(isinstance(m.get(k),(int,float)) and not isinstance(m[k],bool) and math.isfinite(m[k]) and (m[k]>=b if op=='>=' else m[k]<=b) for k,op,b in bounds)
for row in rows:
 raw=gzip.decompress((ROOT/row['artifact']).read_bytes());assert sha(raw)==row['uncompressed_sha256']
 e=json.loads(raw);s=e['spec'];o=e['original_spec'];r=e['result'];v=e['verdict'];name=e['candidate']['name'];key=(name,s['name'])
 assert key not in seen;seen.add(key)
 assert e['candidate']==cards[name] and s==o|e['candidate']['overrides']
 assert s['thresholds']==o['thresholds'] and s['steps']==o['steps']
 assert s['runner']=='vector' and s['tier']=='ranking' and s['split']=='development'
 assert all(s[k]==value for k,value in fixed.items())
 assert s['hidden']==64 and s['layers']==2 and s['fourier']==2 and s['batch']==128 and s['z_dim']==4
 assert not r.get('error') and len(r['observations'])==24
 assert [p['step'] for p in r['observations']]==[math.ceil(i*s['steps']/24) for i in range(1,25)]
 assert r['update_counts']=={role:len(range(0,s['steps'],s[f'{role}_every'])) for role in ('d','g')}
 assert e['source_sha256']==protocol['source_sha256']
 suffix=0
 for point in reversed(r['observations']):
  if not passes(point,s['thresholds']):break
  suffix+=1
 assert suffix==v['convergence']['passing_suffix']
 assert v['passed']==(passes(r['live'],s['thresholds']) and suffix>=5)
 assert v==row['verdict'] and r['live']==row['live'] and r['ema']==row['ema']
 ema_passes+=passes(r['ema'],s['thresholds']);wall+=r['seconds']
 full.setdefault(name,[]).append(v['passed'])
counts={name:sum(values) for name,values in full.items() if len(values)==6}
assert counts=={'b999_lr075_d2_p30':4,'b995_lr075_d2_p20':4,'b999_lr075_d3_p5':3}
result={'verified':True,'episodes':63,'live_observations':1512,'source_files':len(protocol['source_sha256']),
        'gzip_roundtrips':roundtrips,'fully_checked_cards':counts,'ema_final_passes_reported_separately':ema_passes,
        'recorded_wall_seconds':wall,'original_formulation_architecture_budgets_and_thresholds':True}
(ROOT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
