"""Completed-only comparisons for separate G state and D clock scouts."""
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
rows = json.loads((REPORT/'results.json').read_text())
names = json.loads((REPORT/'scouts.json').read_text())
by_name = {r['name']: r for r in rows}
assert all(n in by_name for n in names)

def quality(r):
    return min(r['orbit_progress'][f'prefix{n}']['quality'] for n in (8,32))

lines = ['# Matched state and clock comparisons', '',
    '|G update|Q without D clock|Q with D clock|Clock change|',
    '|---|---:|---:|---:|']
for mode in ('embedded','intent','hybrid'):
    a,b = by_name[mode+'8'],by_name[mode+'8_dclock']
    lines.append(f'|{mode}|{quality(a):.6f}|{quality(b):.6f}|{quality(b)/quality(a)-1:+.1%}|')
a,b = by_name['match_shuffle25'],by_name['baseline_dclock']
lines.append(f'|Shared D memory baseline|{quality(a):.6f}|{quality(b):.6f}|{quality(b)/quality(a)-1:+.1%}|')
lines += ['', 'Q is minimum warm1024 quality over prefixes8/32, not a probability.', '',
    '|Internal update vs embedded observation control|Q change|', '|---|---:|']
for suffix in ('','_dclock'):
    control = quality(by_name['embedded8'+suffix])
    for mode in ('intent','hybrid'):
        name=mode+'8'+suffix
        lines.append(f'|{name}|{quality(by_name[name])/control-1:+.1%}|')

for filename,kind,title in [('g_information.json','Mg','G memory'),('information.json','M','D memory')]:
    result=json.loads((REPORT/filename).read_text())
    lines += ['', f'## {title}: held-out nonlinear process probes', '',
              '|Model|Clean radius/speed R2|After8|After32|After128|Real-write128|',
              '|---|---:|---:|---:|---:|---:|']
    for row in result['results']:
        entries={(r['training_domain'],r['depth']):r['test'] for r in row['rows']
                 if r['probe']=='mlp' and r['features']==kind}
        values=[]
        for key in [('generated',0),('generated',8),('generated',32),('generated',128),('real',128)]:
            e=entries[key];values.append(f"{e['radius_r2']:.3f}/{e['signed_speed_r2']:.3f}")
        lines.append('|'+row['name']+'|'+'|'.join(values)+'|')
    lines += ['', 'Independent held-out histories; particles use the learned table. Regression belongs only to diagnostics. Finite probes do not establish information-theoretic erasure.']
    if kind == 'Mg':
        lines += ['', '### Real-trained probe transfer into generated G state', '',
            '|Model|After1 radius/speed R2|After8|After32|After128|Generated128 with particle|',
            '|---|---:|---:|---:|---:|---:|']
        for row in result['results']:
            entries={r['depth']:r['transfer_generated_test'] for r in row['rows']
                     if r['probe']=='mlp' and r['features']=='Mg' and r['training_domain']=='real'}
            values=[f"{entries[n]['radius_r2']:.3f}/{entries[n]['signed_speed_r2']:.3f}" for n in (1,8,32,128)]
            e=next(r['test'] for r in row['rows'] if r['probe']=='mlp' and r['features']=='Mg+z'
                   and r['depth']==128 and r['training_domain']=='generated')
            values.append(f"{e['radius_r2']:.3f}/{e['signed_speed_r2']:.3f}")
            lines.append('|'+row['name']+'|'+'|'.join(values)+'|')

lines += ['', '## Persistent G-state read interventions (prefix32)', '',
    '|Model|Normal first32 error|Zero Mg|Shuffle Mg|', '|---|---:|---:|---:|']
for name in names:
    r=by_name[name]['metrics']['prefix32']
    if 'g_zero' not in r:
        continue
    for control in ('zero','shuffle'):
        assert r[control] == r['fidelity_256'], (name, control, 'D leaked into G runtime')
    vals=[r[k]['position_error_first32'] for k in ('fidelity_256','g_zero','g_shuffle')]
    lines.append('|'+name+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|')
lines += ['', 'D zero/shuffle metrics are exactly equal to normal256 for all separated scouts. G interventions replace the read state at every generated step; sensitivity is not proof of useful retention.']

result=json.loads((REPORT/'process.json').read_text())
lines += ['', '## Behavioral response to changed real prefixes', '',
    '|Model|Radius response median (ideal1)|Speed response median (ideal1)|Both directions correct|',
    '|---|---:|---:|---:|']
for r in result['results']:
    e=r['response']
    lines.append(f"|{r['name']}|{e['radius_response_median_ideal1']:.4g}|{e['speed_response_median_ideal1']:.4g}|{e['direction_correct_in_both_fraction']:.1%}|")
lines += ['', 'Responses measured late in1024-step evaluation. Direction here is late mean sign, weaker than full-orbit success.']
(REPORT/'state_comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
