"""Offline evidence summary; never imported by training."""
import collections,json
from pathlib import Path
root=Path(__file__).resolve().parent;repo=root.parents[2];attempt=repo.parent
rows=[]
for c in sorted((root/'candidates').iterdir()):
    if not (root/'runs'/c.name).exists():continue
    gates={}
    for name in ['mode_hold','vector_unequal_mass','trajectory','img_intensity2','img_bars4','img_blobs4']:
        path=root/'runs'/c.name/name/'result.json'
        if path.exists():
            d=json.loads(path.read_text());v=d.get('verdict',{});live=d.get('result',{}).get('live',{})
            gates[name]=dict(status=d['status'],seconds=d['seconds'],metrics={k:x for k,x in live.items() if not isinstance(x,(list,dict))},suffix=v.get('convergence',{}).get('passing_suffix'),shortfall=v.get('shortfall'),adam_calls=d['proof']['adam_calls'],optimizer_calls=[x['calls'] for x in d['proof']['optimizers'].values()],correction=d['proof']['game_correction'],artifact=str(path))
        else:gates[name]=dict(status='NOT_RUN')
    first=[gates[x] for x in ['mode_hold','vector_unequal_mass']]
    rows.append(dict(candidate=c.name,blocker_passes=sum(x['status']=='PASS' for x in first),blocker_completed=sum(x['status']!='NOT_RUN' for x in first),mean_final_shortfall=sum(x.get('shortfall',2.) or 0 for x in first)/2,gates=gates,full22='NOT_RUN',own_state_continuation='NOT_RUN'))
rows.sort(key=lambda r:(-r['blocker_passes'],r['mean_final_shortfall']))
(root/'leaderboard.json').write_text(json.dumps(rows,indent=2)+'\n')
ledger=[json.loads(line) for line in (attempt/'tests.jsonl').read_text().splitlines()] if (attempt/'tests.jsonl').exists() else []
bench=[r for r in ledger if r['candidate']!='regression' and r['status']!='SKIPPED'];reg=[r for r in ledger if r['candidate']=='regression']
lines=['# Game dynamics formulation search','',f'Measured candidates: {len(rows)}. No candidate has cleared both frozen blockers; no release or stability qualification.', '',
'Ranking uses sustained blocker passes, then mean frozen final-metric shortfall (offline only). A favorable final point never overrides the five-observation passing suffix.', '',
'| Rank | Candidate | Ring: sustained / modes / HQ / suffix | Unequal: sustained / mass ratio / min eigen ratio / suffix | Four follow-ups | Full 22 | Own-state stability |',
'|---:|---|---|---|---|---|---|']
for i,row in enumerate(rows,1):
    ring=row['gates']['mode_hold'];mass=row['gates']['vector_unequal_mass'];r=ring.get('metrics',{});m=mass.get('metrics',{})
    def fmt(x):return f'{x:.5g}' if isinstance(x,(int,float)) else str(x)
    lines.append(f"| {i} | {row['candidate']} | {ring['status']} / {r.get('modes','—')} / {fmt(r.get('hq','—'))} / {ring.get('suffix','—')} | {mass['status']} / {fmt(m.get('min_mass_ratio','—'))} / {fmt(m.get('component_min_eigen_ratio','—'))} / {mass.get('suffix','—')} | NOT_RUN | NOT_RUN | NOT_RUN |")
lines+=['',f"Executed GAN gates: {len(bench)}; "+', '.join(f'{k}={v}' for k,v in sorted(collections.Counter(r['status'] for r in bench).items()))+'.',f"Mechanism/regression checks: {len(reg)}; "+', '.join(f'{k}={v}' for k,v in sorted(collections.Counter(r['status'] for r in reg).items()))+'.', '',
'All blocker budgets are 1,200 outer steps. The one-evaluation mechanisms perform 1,200 D and 1,200 G Adam calls. ExtraAdam performs 1,200 predictor and 1,200 corrector calls per player; predictor moments are discarded, leaving 1,200 accepted updates per player. It costs two model/gradient blocks per outer step. These are equal-step comparisons, and ExtraAdam is **not** an equal-compute result. Wall times and full update receipts are in each result.', '',
'## Declarations and evidence','',f'- Executed code and prelaunch declarations: `{root}/candidates/<candidate>/`.',f'- Raw metrics, complete frozen curves and CUDA/randomness proofs: `{root}/runs/<candidate>/<gate>/result.json`.',f'- Prepared baseline (one prepare invocation): `{root}/prepared/repos/cuda/`; hash manifest `{root}/prepared/prepared-sources.json`.',f'- Retained initialization fixtures reused: `{repo}/reports/toy100/cpu-recipe-gpu-port/initialization-fixtures/`.',f'- Exact commands, environments, declaration hashes: `{root}/commands.jsonl`.',f'- Machine-readable leaderboard: `{root}/leaderboard.json`.',f'- Append-only gate ledger: `{attempt}/tests.jsonl`.',f'- Small CUDA mechanism checks: `{root}/check_mechanisms.py`, receipts `{root}/mechanism-checks.json`.', '',
'## Replay','',f'From `{repo}`:', '', '```bash', '/tmp/pr38-default-env/bin/python -u reports/toy100/game-dynamics/run_batch.py CANDIDATE mode_hold vector_unequal_mass', 'tail -F reports/toy100/game-dynamics/batch.log', 'tail -F reports/toy100/game-dynamics/runs/CANDIDATE/mode_hold.log', '```','',
'Existing run destinations are intentionally never overwritten. For exact standalone replay, take the command from commands.jsonl and change only `--output` to a fresh directory inside this checkout. Sources are already prepared; do not prepare again. The driver enforces one foreground worker, configured physical GPU, deterministic FP32 and one CPU thread; the probe disables TF32.', '',
'No architecture, data, seeds, thresholds, evaluation, host auxiliary terms, original learning-rate decay or noise schedule was changed. Model parameters, gradients, Adam moments and correction tensors use CUDA; scalar Adam counters retain the original control behavior. The new mechanisms receive no evaluation metrics, mode labels, centers or target statistics.']
lines += ['', f"Planned gate/stage entries explicitly SKIPPED (NOT_RUN): {sum(r['status']=='SKIPPED' for r in ledger)}. These are not executed tests.", '', (root/'conclusions.md').read_text()]
(attempt/'result.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'candidates':len(rows),'GAN_gate_totals':dict(collections.Counter(r['status'] for r in bench)),'regression_totals':dict(collections.Counter(r['status'] for r in reg))}))
