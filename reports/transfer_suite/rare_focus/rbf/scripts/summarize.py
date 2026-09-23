from pathlib import Path
from collections import defaultdict
import json
ROOT=Path('/tmp/pr36-valid-local-d');rows=[]
for phase in ('screen','cross'):
 f=ROOT/phase/'index.json'
 if f.exists():
  for r in json.loads(f.read_text())['records']:r['phase']=phase;r['artifact']=phase+'/'+r['artifact'];rows.append(r)
by=defaultdict(list)
for r in rows:by[r['candidate']['name']].append(r)
ordered=sorted(by.items(),key=lambda item:(-sum(r['verdict']['passed'] for r in item[1]),sum(r['verdict']['shortfall'] for r in item[1])/len(item[1]),item[0]))
lines=['# Pointwise local-feature discriminator study','','Architecture-only rare-mode screen under the original recipe. No target-derived centers or features, extra objectives, normalization, optimizer changes or extra training. Gaussian centers use one fixed local seed0 standard-normal initialization; declared octave widths may remain fixed or learn through the existing D loss.','',
 '| Card | D parameters | Rare sustained | Final suffix | HQ | Mass TV | Covariance error | Min eigen ratio | Min mass ratio | Seconds |',
 '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for name,group in ordered:
 r=next(q for q in group if q['spec']['name']=='vector_unequal_mass');m=r['live'];v=r['verdict']
 fmt=lambda k:f"{m[k]:.6f}" if isinstance(m.get(k),(int,float)) else 'ERROR'
 lines.append(f"| {name} | {r['candidate']['parameter_count']} | {v['status']} | {v.get('convergence',{}).get('passing_suffix',0)}/24 | {fmt('hq')} | {fmt('mass_tv')} | {fmt('component_covariance_error')} | {fmt('component_min_eigen_ratio')} | {fmt('min_mass_ratio')} | {r['seconds']:.3f} |")
lines+=['','Original D has4,929 parameters; architectural capacity changes above are explicit. G stays64×2/z4. All cards use256 particles,batch128,1,200 rare-task steps,Adam(0,.99),G/D/prior LRs .001/.0015/.01,Rp logistic,b_cap3/kappa1.25,prior regularization .05,no particle L2,cosine and1:1 updates. The original five/six behavioral bounds and final-five-of24 stability rule are unchanged. EMA is separate.','']
if any(r['phase']=='cross' for r in rows):
 tasks=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
 lines+=['| Card | '+' | '.join(tasks)+' |','| --- | '+' | '.join(['---']*6)+' |']
 for name,group in ordered:
  if len(group)==1:continue
  bt={r['spec']['name']:r for r in group};lines.append('| '+name+' | '+' | '.join(bt[t]['verdict']['status'] if t in bt else 'Not run' for t in tasks)+' |')
lines += ['',f"Actual GAN episodes: **{len(rows)}**; live observations: **{24*len(rows)}**; summed recorded wall time: **{sum(r['seconds'] for r in rows):.3f}s**.",'','[Combined index](index.json) retains exact cards, original/effective specs, full-episode artifact paths and hashes, all failures, live/EMA curves, actions and runtime. [Architecture checks](architecture_checks.json) verify pointwise behavior, reproducible initialization and active cap backward for every card.']
(ROOT/'MATRIX.md').write_text('\n'.join(lines)+'\n');(ROOT/'index.json').write_text(json.dumps({'records':rows,'episodes':len(rows),'source_commit_base':'981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45','research_module':'benchmarks/transfer_suite/local_critic_research.py'},indent=2,sort_keys=True)+'\n')
for name,group in ordered:
 print(name,[(r['spec']['name'],r['verdict']['status'],r['verdict'].get('convergence',{}).get('passing_suffix',0),[(m['metric'],round(m['value'],5) if isinstance(m['value'],(int,float)) else m['value']) for m in r['verdict'].get('metrics',[]) if m['status']=='FAIL']) for r in group])
