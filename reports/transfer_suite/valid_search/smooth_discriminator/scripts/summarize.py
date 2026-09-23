from pathlib import Path
from collections import defaultdict
import json
ROOT=Path('/tmp/pr36-valid-smooth-d')
TASKS=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
rows=[]
for phase in ('screen','cross'):
 path=ROOT/phase/'index.json'
 if path.exists():
  for row in json.loads(path.read_text())['records']:
   row['phase']=phase;row['artifact']=phase+'/'+row['artifact'];rows.append(row)
by=defaultdict(list)
for row in rows:by[row['candidate']['name']].append(row)
ranked=sorted(by.items(),key=lambda kv:(-sum(r['verdict']['passed'] for r in kv[1]),-len(kv[1]),sum(r['verdict']['shortfall'] for r in kv[1])/len(kv[1]),kv[0]))
lines=['# Smooth discriminator architecture screen','','Research architecture variants only. All cards retain the original vector recipe: Rp logistic, b_cap3/kappa1.25, prior regularization .05, Adam(0,.99), G LR .001/D LR .0015/prior LR .01, cosine and 1:1 updates. Original G,256 particles,batch128 and budgets are unchanged (1,200; spiral1,600). Seed0 only. Live PASS needs every metric passing for the final five of24 observations. EMA is separate.','',
       '| Research D | Broad | Rare mass | Unequal width | Anisotropic | Overlap | Spiral | Sustained / attempted | Mean final shortfall | Seconds |',
       '| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |']
for name,group in ranked:
 bt={r['spec']['name']:r for r in group};cells=[]
 for task in TASKS:
  r=bt.get(task)
  if r is None:cells.append('Not run');continue
  v=r['verdict'];f=bool(v.get('metrics')) and all(c['status']=='PASS' for c in v['metrics'])
  cells.append(('PASS' if v['passed'] else 'Final only' if f else v['status'])+f" ({v.get('convergence',{}).get('passing_suffix',0)}/24)")
 lines.append('| '+name+' | '+' | '.join(cells)+f" | {sum(r['verdict']['passed'] for r in group)}/{len(group)} | {sum(r['verdict']['shortfall'] for r in group)/len(group):.4f} | {sum(r['seconds'] for r in group):.2f} |")
lines+=['','## Architecture declarations','','All critics retain width64 and two hidden layers. Axis features exactly retain the original 10-dimensional input (raw2 + sin/cos at pi and2pi on each axis). Random-oriented4 features also yield10 dimensions. Random-oriented8 yields18 dimensions and modestly increases D parameters. Every orientation uses one fixed local seed0, no target data and no seed selection.','',
       '| D | Features | Activation | Radial frequencies / pi | Parameters |','| --- | --- | --- | --- | ---: |']
checks={c['name']:c for c in json.loads((ROOT/'architecture_checks.json').read_text())}
for c in json.loads((ROOT/'screen/plan.json').read_text())['candidates']:
 a=c['architecture'];act=a['activation']+(f" beta{a['beta']}" if 'beta'in a else '')
 lines.append(f"| {c['name']} | {a['features']} | {act} | {a.get('radial_bands','axis1,2')} | {checks[c['name']]['parameters']} |")
lines+=['',f"Actual GAN episodes: **{len(rows)}**; fixed live observations: **{len(rows)*24}**; summed recorded wall time: **{sum(r['seconds'] for r in rows):.2f}s** on a shared CPU host.",'','Full cards/original and effective specs/verdicts and links to complete live+EMA curves/actions/update counts: [index.json](index.json). All failures retained. Architecture derivative/reproducibility checks are static tests, not GAN training episodes.']
(ROOT/'MATRIX.md').write_text('\n'.join(lines)+'\n')
(ROOT/'index.json').write_text(json.dumps({'records':rows,'episodes':len(rows),'source_commit':'a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff','research_source':'benchmarks/transfer_suite/smooth_critic_research.py'},indent=2,sort_keys=True)+'\n')
for name,g in ranked:print(name,sum(r['verdict']['passed'] for r in g),'/',len(g),'shortfall',round(sum(r['verdict']['shortfall'] for r in g)/len(g),5),[(r['spec']['name'],r['verdict']['status'],r['verdict'].get('convergence',{}).get('passing_suffix',0)) for r in g])
