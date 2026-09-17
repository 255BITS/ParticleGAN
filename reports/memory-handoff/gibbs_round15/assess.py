"""Render completed-round comparison and apply unchanged extension gates."""
import argparse
import json
from pathlib import Path


def assess(report):
    rows = json.loads((report/'results.json').read_text())
    names = json.loads((report/'scouts.json').read_text())
    base = next(r for r in rows if r['name'] == 'match_shuffle25')
    comparisons = []
    for row in rows:
        if row['name'] not in names:
            continue
        stop_ok = row['metrics']['generated_long']['late_stopped_fraction'] <= base['metrics']['generated_long']['late_stopped_fraction']+.01
        checks = {}
        for n in (8, 32):
            key = f'prefix{n}'
            q, bq = row['orbit_progress'][key], base['orbit_progress'][key]
            f, bf = row['metrics'][key]['fidelity_long'], base['metrics'][key]['fidelity_long']
            c = dict(quality_gain=q['quality']/bq['quality']-1,
                late_quality_gain=q['quality_last256']/bq['quality_last256']-1,
                radial_change=f['relative_radial_rmse']/bf['relative_radial_rmse']-1,
                direction_change=f['reference_direction_consistency']-bf['reference_direction_consistency'],
                warm_pass_gain=f['reference_orbit_fraction']-bf['reference_orbit_fraction'])
            c['secondary_pass'] = c['quality_gain'] >= .2 and c['late_quality_gain'] >= 0 and c['radial_change'] <= .05 and c['direction_change'] >= -.02
            checks[key] = c
        primary = all(c['warm_pass_gain'] > 0 for c in checks.values())
        secondary = all(c['secondary_pass'] for c in checks.values())
        comparisons.append(dict(name=row['name'], qualifies=stop_ok and (primary or secondary),
            primary_route=primary, secondary_route=secondary, stopping_ok=stop_ok, comparisons=checks))
    qualifies = {c['name'] for c in comparisons if c['qualifies']}
    ordered = sorted(rows, key=lambda r: (
        -min(r['metrics'][f'prefix{n}']['fidelity_long']['reference_orbit_fraction'] for n in (8,32)),
        -min(r['orbit_progress'][f'prefix{n}']['quality'] for n in (8,32))))
    decision = dict(baseline=base['name'], comparisons=comparisons,
                    selected=[r['name'] for r in ordered if r['name'] in qualifies][:2])
    (report/'extension_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    lines = ['# Completed scout comparison', '', '|Model|Min warm Q|Q32|Late Q32|Radial32|Warm passes8/32|Cold late stopped|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in ordered:
        q = r['orbit_progress']['prefix32']
        f = r['metrics']['prefix32']['fidelity_long']
        passes = '/'.join(str(round(r['metrics'][f'prefix{n}']['fidelity_long']['reference_orbit_fraction']*128)) for n in (8,32))
        lines.append(f"|{r['name']}|{min(r['orbit_progress'][f'prefix{n}']['quality'] for n in (8,32)):.6f}|{q['quality']:.6f}|{q['quality_last256']:.6f}|{f['relative_radial_rmse']:.3f}|{passes}|{r['metrics']['generated_long']['late_stopped_fraction']:.1%}|")
    lines += ['', 'Extension qualifiers: '+(', '.join(decision['selected']) or 'none')+'.', '',
              'Full warm passes are out of128 at1024 steps. Q is a continuous diagnostic, not a success probability.']
    information = report/'information.json'
    if information.exists():
        baselines = json.loads(Path('reports/memory-handoff/information_round14/diagnosis.json').read_text())
        probes = baselines['results']+json.loads(information.read_text())['results']
        earlier = Path('reports/memory-handoff/gibbs_round15/information.json')
        if report.name == 'gibbs_round16' and earlier.exists():
            comparison_names = {r['name'] for r in rows}
            probes += [r for r in json.loads(earlier.read_text())['results'] if r['name'] in comparison_names]
        lines += ['', '## Held-out history information', '', '|Model|Radius/speed R2 clean|After8 writes|After32 writes|After128 writes|', '|---|---:|---:|---:|---:|']
        for r in probes:
            entries = {x['depth']: x['test'] for x in r['rows'] if x['training_domain']=='generated' and x['probe']=='mlp' and x['features']=='M'}
            vals = [f"{entries[n]['radius_r2']:.3f}/{entries[n]['signed_speed_r2']:.3f}" for n in (0,8,32,128)]
            lines.append('|'+r['name']+'|'+'|'.join(vals)+'|')
        lines += ['', 'Probe regression is evaluation-only. Histories are held out; particles come from the learned table. Finite-probe failure is not proof of information erasure.']
    transition = report/'transition.json'
    if transition.exists():
        probes = json.loads(transition.read_text())['results']
        lines += ['', '## One-write read response (prefix32)', '', '|Model|Next-read MSE real/generated|Normalized M gap|K real>fake|Correct-anchor margin > shuffled|K real>wrong-history successor|', '|---|---:|---:|---:|---:|---:|']
        for r in probes:
            p = r['prefixes']['32']; k = p['transition_critic']
            rank = f"{k['real_vs_generated']['positive_rank_fraction']:.1%}" if k else '—'
            anchor = f"{k['paired_margin_correct_vs_shuffled_anchor']['positive_rank_fraction']:.1%}" if k else '—'
            wrong = f"{k['real_vs_other_episode_successor']['positive_rank_fraction']:.1%}" if k and 'real_vs_other_episode_successor' in k else '—'
            lines.append(f"|{r['name']}|{p['next_read_mse']['real']:.5f}/{p['next_read_mse']['generated']:.5f}|{p['normalized_successor_memory_mse']:.5f}|{rank}|{anchor}|{wrong}|")
        lines += ['', 'MSE is evaluation-only. K margins are not calibrated across separately trained models; hybrid/anchor interventions are descriptive.']
    (report/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    assess(parser.parse_args().report)
