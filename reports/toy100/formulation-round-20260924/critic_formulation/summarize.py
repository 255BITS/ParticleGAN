"""Rebuild concise metric leaderboard from retained frozen verdicts."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
names=['mode_hold','vector_unequal_mass','trajectory','img_intensity2','img_bars4','img_blobs4','two_pole','unipolar','mid_scale_identity','cover_leftover','residual_student','img_stripes2','vector_overlap','vector_unequal_width','vector_two_broad','vector_anisotropic','vector_spiral','ae_gan_hold','unused_token_hold','grid100','rotated100','staggered100']
rows=[]
for candidate in sorted((root/'candidates').iterdir()):
 if not (candidate/'declaration.json').exists():continue
 results={p.parent.name:json.loads(p.read_text()) for p in (candidate/'runs').glob('*/result.json')} if (candidate/'runs').exists() else {}
 if not results:continue
 r=dict(candidate=candidate.name,gates={name:results[name]['status'] if name in results else 'NOT_RUN' for name in names},full22='NOT_RUN',own_state_stability='NOT_RUN',executed=len(results),passes=sum(d['status']=='PASS' for d in results.values()),primary_passes=sum(results.get(t,{}).get('status')=='PASS' for t in names[:2]),mean_final_shortfall=sum(results.get(t,{}).get('verdict',{}).get('shortfall',2.) for t in names[:2])/2)
 for task,key in zip(names[:2],('ring','mass')):
  d=results.get(task,{});c=d.get('verdict',{}).get('convergence',{});r[key]=dict(d.get('result',{}).get('live',{}),passing_suffix=c.get('passing_suffix'),passing_observations=c.get('passing_observations'))
 rows.append(r)
rows.sort(key=lambda r:(-r['primary_passes'],r['mean_final_shortfall']))
(root/'leaderboard.json').write_text(json.dumps(rows,indent=2)+'\n')
def fmt(value):return 'NOT_RUN' if value is None else (f'{value:.5g}' if isinstance(value,float) else str(value))
lines=['# Critic formulation search','','Ranking: sustained primary passes, then lower mean frozen final normalized shortfall. Partial scores are diagnostic; no candidate is qualified without its own 22 gates and continuation.','','| Rank | Candidate | Sustained primary | Ring modes / HQ | Ring suffix | Unequal eigen ratio | Unequal covariance error | Unequal suffix | Full 22 | Own-state stability |','|---:|---|---:|---|---:|---:|---:|---:|---|---|']
for i,r in enumerate(rows,1):
 g,m=r['ring'],r['mass'];lines.append(f"| {i} | {r['candidate']} | {r['primary_passes']}/2 | {fmt(g.get('modes'))} / {fmt(g.get('hq'))} | {fmt(g.get('passing_suffix'))}/5 | {fmt(m.get('component_min_eigen_ratio'))} | {fmt(m.get('component_covariance_error'))} | {fmt(m.get('passing_suffix'))}/5 | {r['full22']} | {r['own_state_stability']} |")
lines+=['','All acquisition verdicts use the frozen sustained gate, including at least five passing observations at the end. Every primary screen uses its full 1,200 D + 1,200 G/prior updates. No seed experiments.','', 'Gate matrix:','', '| Gate | '+' | '.join(r['candidate'] for r in rows)+' |','|---|'+'---|'*len(rows)]
for task in names:lines.append('| '+task+' | '+' | '.join(r['gates'][task] for r in rows)+' |')
(root/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([dict(candidate=r['candidate'],passes=r['passes'],executed=r['executed'],mean_final_shortfall=r['mean_final_shortfall']) for r in rows]))
