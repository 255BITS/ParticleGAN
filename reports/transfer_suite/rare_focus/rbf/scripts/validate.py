"""Validate research-only architectural changes and retained live verdicts."""
from pathlib import Path
import gzip,hashlib,json,math,tarfile
ROOT=Path('/tmp/pr36-valid-local-d')
sha=lambda b:hashlib.sha256(b).hexdigest()
rows=json.loads((ROOT/'index.json').read_text())['records'];seen=set();protocol=None;by={};ema_final=0
for phase in sorted({r['phase'] for r in rows}):
 p=json.loads((ROOT/phase/'protocol.json').read_text())
 if protocol is None:protocol=p
 assert p==protocol
 for name,h in p['experiment_source_sha256'].items():assert sha((ROOT/name).read_bytes())==h
 with tarfile.open(ROOT/phase/'source.tar.gz','r:gz') as tf:
  assert {m.name for m in tf.getmembers()}==set(p['source_sha256'])
  for m in tf.getmembers():assert sha(tf.extractfile(m).read())==p['source_sha256'][m.name]
def passes(metrics,bounds):
 return all(isinstance(metrics.get(k),(int,float)) and not isinstance(metrics[k],bool) and math.isfinite(metrics[k]) and (metrics[k]>=b if op=='>=' else metrics[k]<=b) for k,op,b in bounds)
for row in rows:
 raw=gzip.decompress((ROOT/row['artifact']).read_bytes());assert sha(raw)==row['uncompressed_sha256']
 e=json.loads(raw);s=e['spec'];o=e['original_spec'];r=e['result'];v=e['verdict'];c=e['candidate'];key=(c['name'],s['name'])
 assert key not in seen;seen.add(key)
 assert s==o|{'research_discriminator':c['architecture']}
 assert s['runner']=='vector' and s['tier']=='ranking' and s['split']=='development'
 for k,x in {'hidden':64,'layers':2,'fourier':2,'z_dim':4,'particles':256,'batch':128,'lr':.001,'d_lr_mult':1.5,'prior_lr_mult':10.,'prior_reg':.05,'betas':[0.,.99],'reg_arm':'b_cap','reg_coeff':3.,'reg_kappa':1.25,'d_every':1,'g_every':1}.items():assert s[k]==x
 assert s.get('gan_mode','rp')=='rp' and s.get('loss_type','logistic')=='logistic'
 assert not r.get('error') and len(r['observations'])==24
 assert [p['step'] for p in r['observations']]==[math.ceil(i*s['steps']/24) for i in range(1,25)]
 assert r['update_counts']=={'g':s['steps'],'d':s['steps']}
 assert e['source_sha256']==protocol['source_sha256'] and e['experiment_source_sha256']==protocol['experiment_source_sha256']
 suffix=0
 for point in reversed(r['observations']):
  if not passes(point,s['thresholds']):break
  suffix+=1
 assert suffix==v['convergence']['passing_suffix'] and v['passed']==(passes(r['live'],s['thresholds']) and suffix>=5)
 assert v==row['verdict'] and row['live']==r['live'] and row['ema']==r['ema']
 by.setdefault(c['name'],[]).append(v['passed']);ema_final+=passes(r['ema'],s['thresholds'])
assert len(rows)==8 and all(not r['verdict']['passed'] for r in rows)
summary={'verified':True,'episodes':len(rows),'live_observations':24*len(rows),'architecture_cards':len(by),'source_files':len(protocol['source_sha256']),
 'recorded_wall_seconds':sum(r['seconds'] for r in rows),'complete_profiles':{n:sum(v) for n,v in by.items() if len(v)==6},
 'ema_final_passes_separate':ema_final,'only_research_discriminator_changed':True,'source_commit_base':'981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45'}
(ROOT/'validation.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
