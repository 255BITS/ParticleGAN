"""Portable audit of the two declared failed rare-mass followups."""
from pathlib import Path
import gzip,hashlib,json,math,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(name):return json.loads(gzip.decompress((ROOT/name).read_bytes()))
inv=json.loads((ROOT/'inventory.json').read_text());roundtrips=0
for f in inv['files']:
 data=(ROOT/f['path']).read_bytes();assert sha(data)==f['sha256'] and len(data)==f['bytes'],f['path']
 if f['path'].endswith('.gz'):
  raw=gzip.decompress(data);roundtrips+=1;assert sha(raw)==f['uncompressed_sha256'] and len(raw)==f['uncompressed_bytes']
p=read('rare64/protocol.json.gz');assert p==read('rare96/protocol.json.gz')
for name,h in p['experiment_source_sha256'].items():assert sha((ROOT/'scripts'/name).read_bytes())==h
with tarfile.open(ROOT/'source.tar.gz','r:gz') as tf:
 assert {m.name for m in tf.getmembers()}==set(p['source_sha256'])
 for m in tf.getmembers():assert m.isfile() and sha(tf.extractfile(m).read())==p['source_sha256'][m.name]
index=read('index.json.gz');rows=index['records'];declaration=read('declaration.json.gz');cards={c['name']:c for c in declaration['cards']}
assert len(rows)==2 and index['cross_evaluations']==0

def passes(m,b):return all(isinstance(m.get(k),(int,float)) and not isinstance(m[k],bool) and math.isfinite(m[k]) and (m[k]>=v if op=='>=' else m[k]<=v) for k,op,v in b)
widths=[];ema_passes=[]
for row in rows:
 raw=gzip.decompress((ROOT/row['artifact']).read_bytes());assert sha(raw)==row['uncompressed_sha256']
 e=json.loads(raw);c=e['candidate'];o=e['original_spec'];s=e['spec'];r=e['result'];v=e['verdict']
 assert c==cards[c['name']] and s==o|c['overrides']|{'research_discriminator':c['architecture']}
 assert c['overrides']=={'particles':512,'d_hidden':c['architecture']['hidden'],'d_layers':2}
 assert s['name']=='vector_unequal_mass' and s['steps']==1200 and s['thresholds']==o['thresholds']
 assert e['resource_profile']==declaration['resource_profile']
 for k,x in {'hidden':64,'layers':2,'fourier':2,'z_dim':4,'particles':512,'batch':128,'lr':.001,'d_lr_mult':1.5,'prior_lr_mult':10.,'prior_reg':.05,'betas':[0.,.99],'reg_arm':'b_cap','reg_coeff':3.,'reg_kappa':1.25,'d_every':1,'g_every':1}.items():assert s[k]==x
 assert s.get('gan_mode','rp')=='rp' and s.get('loss_type','logistic')=='logistic'
 assert c['architecture']['features']=='axis' and c['architecture']['activation']=='softplus' and c['architecture']['beta']==5.
 assert not r.get('error') and len(r['observations'])==24
 assert [q['step'] for q in r['observations']]==list(range(50,1201,50))
 assert r['update_counts']=={'d':1200,'g':1200}
 assert e['source_sha256']==p['source_sha256'] and e['experiment_source_sha256']==p['experiment_source_sha256']
 suffix=0
 for point in reversed(r['observations']):
  if not passes(point,s['thresholds']):break
  suffix+=1
 assert suffix==v['convergence']['passing_suffix']==0
 assert v['passed']==(passes(r['live'],s['thresholds']) and suffix>=5)==False
 assert row['verdict']==v and row['live']==r['live'] and row['ema']==r['ema']
 widths.append(s['d_hidden']);ema_passes.append(passes(r['ema'],s['thresholds']))
assert sorted(widths)==[64,96]
result={'verified':True,'episodes':2,'live_observations':48,'full_six_followups':0,'sustained_live_passes':0,'ema_final_passes_separate':sum(ema_passes),'source_files':len(p['source_sha256']),'gzip_roundtrips':roundtrips,'recorded_wall_seconds':sum(r['seconds'] for r in rows),'unchanged_recipe_and_budget':True,'resource_profile_particles':512}
(ROOT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
