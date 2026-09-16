"""Audit completed artifacts and render compact diagnostic comparisons."""
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[3]
report = Path(__file__).resolve().parent
queue = root/'runs/memory_path/information_round14'
baseline = root/'runs/memory_path/principles_round12/runs/match_shuffle25'
jobs = [json.loads(p.read_text()) for p in (queue/'done').glob('*.json')]
assert len(jobs) == len(json.loads((report/'scouts.json').read_text()))
for state in ('running', 'pending', 'failed'):
    assert not list((queue/state).glob('*.json'))
summaries = [json.loads((Path(job['out'])/'summary.json').read_text()) for job in jobs]
provenance = [json.loads((Path(job['out'])/'provenance.json').read_text()) for job in jobs]
assert all(p['sources'] == provenance[0]['sources'] for p in provenance)
panel_keys = ('continuation_reference', 'observed_prefix8', 'observed_prefix32')
with np.load(baseline/'trajectories.npz') as arrays:
    panel = {k: arrays[k].copy() for k in panel_keys}
for job in jobs:
    with np.load(Path(job['out'])/'trajectories.npz') as arrays:
        assert all(np.array_equal(arrays[k], panel[k]) for k in panel)
execution = dict(done=len(jobs), failed=0, devices=sorted({j['device'] for j in jobs}),
    queue_wall_seconds=max(j['finished'] for j in jobs)-min(j['started'] for j in jobs),
    training_gpu_seconds=sum(s['train_seconds'] for s in summaries),
    identical_training_sources=True, sources=provenance[0]['sources'],
    reference_and_observed_prefixes_bitwise_equal_baseline=True,
    runs=[dict(name=s['name'], train_seconds=s['train_seconds'],
               seconds_per_update=s['seconds_per_update']) for s in summaries])
(report/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')

process = json.loads((report/'process.json').read_text())['results']
signal = json.loads((report/'signal.json').read_text())['results']
lines = ['# Completed-model diagnostics, round14', '',
    'All models evaluated on CPU; original and counterfactual histories recomputed on the same device.',
    'Response windows contain32 points starting after the indicated number of generated writes.',
    'Ideal radius/speed response is1. Nonzero sensitivity alone does not establish correct retention.', '']
for title, metric in (
    ('Median normalized radius response', 'radius_response_median_ideal1'),
    ('Median normalized signed-speed response', 'speed_response_median_ideal1'),
    ('Correct mean direction in both original and flipped history', 'direction_correct_in_both_fraction'),
):
    times = ('0', '1', '8', '32', '128', '512')
    lines += [f'## {title}', '', '| Model | '+ ' | '.join(times)+' | Last256 |',
              '|---|'+'---:|'*7]
    for row in process:
        values = [row['response_over_time'][n][metric] for n in times]+[row['response'][metric]]
        lines.append('| '+row['name']+' | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines.append('')
lines += ['## Prefix32 point-head nearest-history ranking', '',
    '| Model | Clean | After .25 write | After full write | Clean local MSE | Full-write local MSE |',
    '|---|---:|---:|---:|---:|---:|']
for row in signal:
    clean = row['ranking']['prefix32']
    mild = row['ranking_after_write']['0.25']['prefix32']
    full = row['ranking_after_write']['1.0']['prefix32']
    values = [r['nearest']['correct_rank_fraction'] for r in (clean, mild, full)]
    values += [clean['generated_next_point_mse'], full['generated_next_point_mse']]
    lines.append('| '+row['name']+' | '+' | '.join(f'{v:.5f}' for v in values)+' |')
future = json.loads((report/'future_signal.json').read_text())['results']
lines += ['', '## Prefix32 nearest-history future ranking', '',
    '| Model | Horizon conditioned | h0 clean | h4 clean | h12 clean | h4 full write | h12 full write |',
    '|---|---|---:|---:|---:|---:|---:|']
for row in future:
    contexts = row['prefixes']['32']['contexts']
    values = [contexts[c][str(h)]['donors']['nearest']['correct_rank_fraction']
              for c, h in [('clean', 0), ('clean', 4), ('clean', 12),
                           ('generated_write', 4), ('generated_write', 12)]]
    lines.append('| '+row['name']+' | '+str(row['horizon_conditioned'])+' | '+
                 ' | '.join(f'{v:.5f}' for v in values)+' |')
lines += ['', 'Old references have no horizon conditioning; their h4/h12 scores use the immediate head.',
    'All models supply their own full-write corruption. Improved future classification is not',
    'a guarantee of useful immediate gradients or repeated memory retention.',
    '', 'MSE is evaluation only. Ranking uses noisy next points and fixed donor identities;',
    'mismatched continuations are not guaranteed impossible under observation noise.',
    'Neither classification accuracy nor timed-target gradient alignment establishes stable autonomous dynamics.']
(report/'diagnostics.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(execution, indent=2))
