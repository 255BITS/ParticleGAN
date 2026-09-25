"""Verify replay evidence against pinned PR checkouts; no training runs."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--pr140', type=Path, required=True)
p.add_argument('--pr143', type=Path, required=True)
p.add_argument('--audit', type=Path, default=Path(__file__).parent)
a = p.parse_args()

def read(path):
    return json.loads(path.read_text())

summary = {'status': 'MATCH', 'prs': {}, 'prefix_controls': {}}
for number, repo, name in [(140, a.pr140, 'delayed-arm-g-lr'),
                           (143, a.pr143, 'delayed-arm-width-hold')]:
    submitted = read(repo / f'reports/toy100/continuous-evidence/{name}/stay-avx2/stay.json')
    replay = read(a.audit / f'pr{number}-intel-dispatch-stay/stay.json')
    assert submitted['records'] == replay['records'], f'PR{number} update trace differs'
    assert submitted['diagnostic'] == replay['diagnostic'], f'PR{number} quality trace differs'
    assert len(replay['records']) == 2400 and len(replay['diagnostic']) == 240
    late = [r for r in replay['diagnostic'] if r['step'] > 1200]
    misses = [r for r in late if r['modes'] != 8 or r['hq'] < .9]
    receipt = read(a.audit / f'pr{number}-intel-dispatch-stay/declaration.json')
    for name, sha in receipt['source'].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == sha, name
    summary['prs'][str(number)] = dict(update_records_equal=2400, quality_checks_equal=240,
        passing_stay_checks=len(late)-len(misses), stay_checks=len(late),
        min_modes=min(r['modes'] for r in late), min_hq=min(r['hq'] for r in late),
        failing_steps=[r['step'] for r in misses], final=late[-1],
        step2190=next(r for r in late if r['step']==2190), source_hashes_match=len(receipt['source']))

expected = a.pr143 / 'reports/toy100/continuous-evidence/delayed-arm-width-hold'
assert read(expected/'warm-avx2/forks/summary.json') == read(a.audit/'pr143-intel-dispatch-warm/forks/summary.json')
summary['warm_summary_and_state_hashes_equal'] = True

def strip_timing(value):
    if isinstance(value, dict):
        return {k:strip_timing(v) for k,v in value.items() if k not in ('seconds','elapsed','stable_from_seconds','confirmed_seconds')}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value
for task in ('trajectory','mode_hold'):
    x = read(expected/f'cold-avx2/{task}.json')
    y = read(a.audit/f'pr143-intel-dispatch-cold/{task}.json')
    assert strip_timing(x) == strip_timing(y), f'cold {task} differs beyond wall-clock timing'
summary['cold_results_equal_excluding_timing'] = ['trajectory','mode_hold']

base = read(a.audit/'prefixes/default.json')
intel = read(a.audit/'prefixes/intel-dispatch.json')
assert base['initial'] == intel['initial']
summary['initial_model_prior_rng_tensors_equal'] = len(base['initial'])
for path in sorted((a.audit/'prefixes').glob('*.json')):
    row = read(path)['records'][0]
    summary['prefix_controls'][path.stem] = dict(sharpness=row['critic_sharpness'],g_factor=row['g']['factor'])
for name in ('deterministic','interop1','mkl-verbose'):
    assert read(a.audit/f'prefixes/{name}.json')['records'] == base['records'], name
assert base['records'] != intel['records']
print(json.dumps(summary,indent=2))
