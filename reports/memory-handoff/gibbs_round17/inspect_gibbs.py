"""Compact completed diagnostic tables, including matched architecture contrasts."""
import json
from pathlib import Path

report = Path(__file__).resolve().parent
rows = json.loads((report/'gibbs.json').read_text())['results']
lines = ['# Same-event diagnostics', '', '1024 fresh histories, prefix32; all errors below are evaluation-only MSE.', '',
         '|Model|Point error|Zero M error|Shuffled M error|Zero h output change|Shuffled h output change|K real>fake|K pair>wrong history|',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    p = row['prefixes']['32']; h = p['latent']; k = p['joint_critic']
    vals = [row['name'], f"{p['point_mse']:.5f}", *[f"{p['memory_interventions'][n]['point_mse']:.5f}" for n in ('zero','shuffle')]]
    vals += [f"{h['interventions'][n]['output_change_mse']:.6f}" if h else '—' for n in ('zero','shuffle')]
    vals += [f"{k[n]['positive_rank_fraction']:.1%}" if k else '—' for n in ('real_vs_generated','real_vs_other_episode_pair')]
    lines.append('|'+ '|'.join(vals)+'|')
lines += ['', '## Inference refinement, context held fixed', '', '|Model|1 decode error|2 decodes|3 decodes|7 decodes|Producer vs reencoded h error|', '|---|---:|---:|---:|---:|---:|']
for row in rows:
    p=row['prefixes']['32']
    if p['refinement']:
        vals=[row['name']]+[f"{p['refinement'][str(n)]['point_mse']:.5f}" for n in (1,2,3,7)]
        vals.append(f"{p['latent']['producer_vs_inferred_generated_mse']:.6f}")
        lines.append('|'+ '|'.join(vals)+'|')
lines += ['', 'Latent distances are representation-specific. Interventions measure local sensitivity and error, not autonomous survival. Extra inference iterations can be out of training support. A good joint rank or small latent discrepancy is not proof of equilibrium or circle completion.']
(report/'gibbs_comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
