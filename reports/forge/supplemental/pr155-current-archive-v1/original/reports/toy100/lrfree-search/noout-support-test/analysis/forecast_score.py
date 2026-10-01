"""Brier scores of the pass/fail forecasts written to forecast-*.json before the runs of this pass (E15-E20). Outcomes come from runs/<label>-<task>/result.json.
usage: python forecast_score.py"""
import json, os
F = [('forecast-E15.json', 'E15a', [.85, .93, .88]), ('forecast-E16.json', 'E16a', [.75, .55, .65]), ('forecast-E16-batch.json', 'E16b (gate off)', [.80, .55, .70]),
     ('forecast-E17.json', 'E17a', [.80, .80, .80]), ('forecast-E17b.json', 'E17b (gate off)', [.85, .85, .85]), ('forecast-E17b.json', 'E17blr075', [.80, .80, .75]),
     ('forecast-E17-S4.json', 'E17lr075', [.80, .85, .80]), ('forecast-E17-S4.json', 'E17lr133', [.55, .75, .70]), ('forecast-E18.json', 'E18a', [.80, .80, .80]),
     ('forecast-E19.json', 'E19a', [.85, .85, .85]), ('forecast-E19.json', 'E19lr075', [.80, .85, .80]), ('forecast-E19.json', 'E19lr133', [.70, .80, .75]),
     ('forecast-E19-horizon.json', 'E19a14k', [.80, .85, .80]), ('forecast-E19-horizon.json', 'E19a28k', [.65, .75, .65]),
     ('forecast-E20.json', 'E20a', [.85, .85, .85]), ('forecast-E20-S4.json', 'E20lr075', [.85, .85, .80]), ('forecast-E20-S4.json', 'E20lr133', [.70, .80, .75]),
     ('forecast-E20-horizon.json', 'E20a14k', [.80, .85, .80])]
label_of = {'E17b (gate off)': 'E17b', 'E16b (gate off)': 'E16b'}
tot = []
print('| forecast file | run | P(pass) grid / rotated / staggered | outcome | Brier |\n|---|---|---|---|---:|')
for f, lab, ps in F:
    run = label_of.get(lab, lab); outs = []
    for t in ('grid100', 'rotated100', 'staggered100'):
        try: outs.append(1 if json.load(open(f'runs/{run}-{t}/result.json'))['status'] == 'PASS' else 0)
        except Exception: outs.append(None)
    if any(o is None for o in outs):
        print(f'| {f} | {lab} | {ps} | pending | - |'); continue
    b = sum((p - o) ** 2 for p, o in zip(ps, outs)) / 3; tot += [(p - o) ** 2 for p, o in zip(ps, outs)]
    print(f'| {f} | {lab} | {ps} | {outs} | {b:.3f} |')
print(f'\nall scored cells: {len(tot)}, Brier {sum(tot) / len(tot):.3f} (coin flip .250)')
