from pathlib import Path
import gzip,hashlib,json
ROOT=Path('/tmp/pr36-valid-smooth512');OUT=Path('/tmp/pr36-valid-smooth512-handoff')
rows=[]
for phase in ('rare64','rare96'):
 for r in json.loads((ROOT/phase/'index.json').read_text())['records']:
  r['phase']=phase;r['artifact']=phase+'/'+r['artifact'];rows.append(r)
assert len(rows)==2 and all(not r['verdict']['passed'] for r in rows)
index={'records':rows,'episodes':2,'cross_evaluations':0,'stopped_reason':'Both predeclared rare-mass checks failed sustained live PASS; no cross-evaluation allowed by this bounded plan.','resource_profile':'512 particles, batch128, original G and original recipe; D64x2 or96x2, axis Fourier2 Softplus beta5'}
(ROOT/'index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
lines=['# Targeted 512-particle smooth-critic check','','**Both declared rare-mass checks fail.** Training stopped after these two episodes; no full-six followup was performed. These results are separate from every 256-particle experiment and do not establish an all-six supported 512-particle profile.','',
 '| Critic | Sustained live | Suffix | HQ | Mass TV | Covariance error | Minimum eigenvalue ratio | Min mass ratio | Wall seconds |',
 '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
 m=r['live'];lines.append(f"| Softplus5 D{r['spec']['d_hidden']}×2 / Fourier2 | {r['verdict']['status']} | {r['verdict']['convergence']['passing_suffix']}/24 | {m['hq']:.6f} | {m['mass_tv']:.6f} | {m['component_covariance_error']:.6f} | {m['component_min_eigen_ratio']:.6f} | {m['min_mass_ratio']:.6f} | {r['seconds']:.3f} |")
lines+=['','The 64-wide critic fails covariance error (6.54936 > .85) while its other final bounds pass. The 96-wide critic fails minimum eigenvalue ratio (.011894 < .15) while its other final bounds pass. The latter assigns only .659% of generated mass to the 2% target component; that clears the declared rare-occupancy floor but still has poor within-component geometry. A higher particle count does not automatically fix this trained rare-mode collapse.','',
 'Every episode uses 512 particles, batch128, original G64×2/z4, Adam(0,.99), G/D/prior LRs .001/.0015/.01, Rp logistic, b_cap coefficient3/kappa1.25, prior regularization .05, no particle L2, cosine and 1:1 updates. The only changes from the original vector task are the declared support size and smooth-D architecture. There are 1,200 outer steps, 24 fixed live observations and an unchanged requirement for at least five final passing observations. EMA is retained separately. Seed0 only.','',
 f"Exactly **2 GAN episodes**, **48 live observations**, **{sum(r['seconds'] for r in rows):.6f} seconds** summed recorded wall time on a shared CPU host. All failures and full curves remain retained.",'',
 'Read [index.json](index.json) for original/effective specs, candidate architecture, verdicts, live/EMA summaries and artifact paths. Each episode includes complete curves, actions and actual D/G update counts. The [declaration](declaration.json) was saved before training and requires stopping if neither rare check passes. No other resource profile is pooled into these results.']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
OUT.mkdir(exist_ok=True);items=[];sha=lambda b:hashlib.sha256(b).hexdigest()
def add(name,data,raw=None):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);entry={'path':name,'sha256':sha(data),'bytes':len(data)}
 if raw is not None:entry.update(uncompressed_sha256=sha(raw),uncompressed_bytes=len(raw))
 items.append(entry)
for p in sorted(ROOT.glob('*.json')):
 raw=p.read_bytes();add(p.name+'.gz',gzip.compress(raw,mtime=0),raw)
for phase in ('rare64','rare96'):
 for p in sorted((ROOT/phase).glob('*.json')):
  raw=p.read_bytes();add(f'{phase}/{p.name}.gz',gzip.compress(raw,mtime=0),raw)
 for p in sorted((ROOT/phase/'episodes').glob('*.gz')):add(f'{phase}/episodes/{p.name}',p.read_bytes(),gzip.decompress(p.read_bytes()))
 add(f'{phase}/run.log',(ROOT/f'{phase}.log').read_bytes())
source=(ROOT/'rare64/source.tar.gz').read_bytes();assert gzip.decompress(source)==gzip.decompress((ROOT/'rare96/source.tar.gz').read_bytes());add('source.tar.gz',source,gzip.decompress(source))
for p in sorted(ROOT.glob('*.py')):add('scripts/'+p.name,p.read_bytes())
text=(ROOT/'README.md').read_text().replace('(index.json)','(index.json.gz)').replace('(declaration.json)','(declaration.json.gz)')
text+='''
## Reproduction and archive audit

Source commit `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45`; the [source archive](source.tar.gz) contains all 58 exact numerical dependency files. [D64 protocol](rare64/protocol.json.gz) and [D96 protocol](rare96/protocol.json.gz) have identical source hashes. [Driver](scripts/run.py) is exact and separately hashed in each protocol. Every JSON preserves original uncompressed bytes, with both compressed and uncompressed hashes in the [inventory](inventory.json).

Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; Python `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. Commands: `python -u run.py --width 64 --phase rare`, followed by `python -u run.py --width 96 --phase rare`. The driver retains original absolute paths; reproduce in those paths or adapt a separate copy, never overwrite retained evidence. It replaces only the discriminator constructor for each serial episode and restores it afterward.

Run `python verify.py` to check source hashes, gzip byte roundtrips, both full episode invariants and independent numerical verdicts without training. [Verification](verification.json) records the result. No new implementation commit was needed for this experiment.
'''
add('README.md',text.encode());(OUT/'inventory.json').write_text(json.dumps({'episodes':2,'live_observations':48,'files':items},indent=2,sort_keys=True)+'\n');print(OUT)
