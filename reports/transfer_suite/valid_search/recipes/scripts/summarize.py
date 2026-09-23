"""Readable live leaderboard; all executed episode records stay in phase directories."""
from pathlib import Path
from collections import defaultdict
import argparse,gzip,json
OUT=Path('/tmp/pr36-valid-recipe')
TASKS=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
PHASES=['screen','cross','cross_secondary']
rows=[]
for phase in PHASES:
 path=OUT/phase/'index.json'
 if path.exists():
  for row in json.loads(path.read_text())['records']:
   row['phase']=phase;row['artifact']=phase+'/'+row['artifact'];rows.append(row)
by=defaultdict(list)
for row in rows:by[row['candidate']['name']].append(row)
ranked=sorted(by.items(),key=lambda item:(-sum(r['verdict']['passed'] for r in item[1]),-len(item[1]),sum(r['verdict']['shortfall'] for r in item[1])/len(item[1]),sum(r['verdict']['confirmation_fraction'] for r in item[1])/len(item[1]),item[0]))
lines=['# Shared vector optimizer recipes','','Rp logistic, b_cap coefficient 3 / kappa 1.25, prior regularization .05, no particle L2. Every card uses the original generator/critic architecture, 256 particles, batch 128, cosine schedule and original outer budgets (1,200; spiral 1,600). All runs use seed 0, 24 fixed live observations and a final passing suffix of at least five. EMA is separate.','',
       'The 18 new coordinated optimizer cards were declared before training; no task-specific recipe overrides. The first three hard data cases screen candidates. Up to three advance to all six valid data toys. Untested cells do not imply success. These are inspected development cases; this matrix does not include required or image validation.','',
       '| Card | Broad | Rare mass | Unequal width | Anisotropic | Overlap | Spiral | Sustained / attempted | Mean final shortfall | Wall seconds |',
       '| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |']
for name,group in ranked:
 bytask={r['spec']['name']:r for r in group};cells=[]
 for task in TASKS:
  r=bytask.get(task)
  if not r:cells.append('Not run');continue
  v=r['verdict'];suffix=v.get('convergence',{}).get('passing_suffix',0)
  final=bool(v.get('metrics')) and all(c['status']=='PASS' for c in v['metrics'])
  cells.append(('PASS' if v['passed'] else 'Final only' if final else v['status'])+f' ({suffix}/24)')
 lines.append(f"| {name} | "+' | '.join(cells)+f" | {sum(r['verdict']['passed'] for r in group)}/{len(group)} | {sum(r['verdict']['shortfall'] for r in group)/len(group):.4f} | {sum(r['seconds'] for r in group):.2f} |")
lines+=['','## Exact shared cards','', '| Card | G LR | D / G LR | Prior / G LR | Adam betas | D every | G every |','| --- | ---: | ---: | ---: | --- | ---: | ---: |']
for card in json.loads((OUT/'screen_plan.json').read_text())['candidates']:
 o=card['overrides'];lines.append(f"| {card['name']} | {o['lr']} | {o['d_lr_mult']} | {o['prior_lr_mult']} | {o['betas']} | {o['d_every']} | {o['g_every']} |")
lines+=['','D/G cadence is measured on the same declared outer-step clock. Fewer role updates are explicit, not hidden extra compute. Actual counts and EMA metrics are retained in each full episode.','',f"Actual episodes: **{len(rows)}**. Fixed live observations: **{24*len(rows)}**. Recorded wall time: **{sum(r['seconds'] for r in rows):.2f} seconds** on a shared CPU host.",'','Full per-episode artifacts and source hashes: [index.json](index.json). Original phase logs, plans and exact source archives are retained.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'index.json').write_text(json.dumps({'records':rows,'episodes':len(rows),'source_commit':'a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff'},indent=2,sort_keys=True)+'\n')
for name,group in ranked:
 print(name, 'passed',sum(r['verdict']['passed'] for r in group),'/',len(group),'shortfall',round(sum(r['verdict']['shortfall'] for r in group)/len(group),4),[(r['spec']['name'],r['verdict']['status'],r['verdict'].get('convergence',{}).get('passing_suffix',0)) for r in group])
