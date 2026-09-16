"""Apply the predeclared round13 gates, only after all five scouts finish."""
import json
from pathlib import Path

queue = Path('runs/memory_path/write_match_round13')
report = Path('reports/memory-handoff/write_match_round13')
assert (queue/'SEALED').exists()
assert not list((queue/'running').glob('*.json'))
assert not list((queue/'pending').glob('*.json'))
assert not list((queue/'failed').glob('*.json'))
assert len(list((queue/'done').glob('*.json'))) == 5
rows = json.loads((report/'results.json').read_text())
done_names = sorted(json.loads(p.read_text())['name'] for p in (queue/'done').glob('*.json'))
assert sorted(r['name'] for r in rows if not r.get('comparison_only')) == done_names
base = next(r for r in rows if r['name'] == 'match_shuffle25')
comparisons = []
for row in rows:
    if row.get('comparison_only'):
        continue
    stop = row['metrics']['generated_long']['late_stopped_fraction']
    stop_ok = stop <= base['metrics']['generated_long']['late_stopped_fraction']+.01
    point_checks = {}
    for n in (8, 32):
        key = f'prefix{n}'
        q, bq = row['orbit_progress'][key], base['orbit_progress'][key]
        f, bf = row['metrics'][key]['fidelity_long'], base['metrics'][key]['fidelity_long']
        changes = dict(quality_gain=q['quality']/bq['quality']-1,
            late_quality_gain=q['quality_last256']/bq['quality_last256']-1,
            radial_change=f['relative_radial_rmse']/bf['relative_radial_rmse']-1,
            direction_change=f['reference_direction_consistency']-bf['reference_direction_consistency'],
            warm_pass_gain=f['reference_orbit_fraction']-bf['reference_orbit_fraction'])
        changes['secondary_pass'] = (changes['quality_gain'] >= .2 and changes['late_quality_gain'] >= 0
            and changes['radial_change'] <= .05 and changes['direction_change'] >= -.02)
        point_checks[key] = changes
    primary = all(c['warm_pass_gain'] > 0 for c in point_checks.values())
    secondary = all(c['secondary_pass'] for c in point_checks.values())
    comparisons.append(dict(name=row['name'], qualifies=stop_ok and (primary or secondary),
        primary_route=primary, secondary_route=secondary, stopping_ok=stop_ok,
        comparisons=point_checks))
qualified = {c['name'] for c in comparisons if c['qualifies']}
ordered = sorted((r for r in rows if r['name'] in qualified), key=lambda r: (
    -min(r['metrics'][f'prefix{n}']['fidelity_long']['reference_orbit_fraction'] for n in (8,32)),
    -min(r['orbit_progress'][f'prefix{n}']['quality'] for n in (8,32))))
result = dict(all_five_completed=True, baseline=base['name'], criteria='plan.md, fixed before training',
    comparisons=comparisons, selected=[r['name'] for r in ordered[:2]])
(report/'extension_decision.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
