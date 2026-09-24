"""Render a completed, audited GPU matrix with full and limited coverage separated."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

root = Path(__file__).resolve().parent
audit = json.loads((root / 'audit.json').read_text())
assert audit['status'] == 'PASS' and audit['complete']
rows = json.loads((root / 'candidates.json').read_text())
protocol = json.loads((root / 'protocol.json').read_text())
events = {(x['candidate'], x['task']): x for x in json.loads((root / 'latest.json').read_text())}
toys = protocol['toys']
summaries = []
for row in rows:
    counts = Counter(events[row['name'], task]['status'] for task in toys)
    conv = events[row['name'], 'convergence']['convergence']
    good = conv['hold_checks'] - int(conv['first_hold_failure'] is not None)
    summaries.append(dict(candidate=row['name'], scope=row['supported'], passed=counts['PASS'],
                          failed=counts['FAIL'], unsupported=counts['UNSUPPORTED'],
                          converged_step=conv['converged_step'], hold_passes=good,
                          hold_status=conv['status'], failure_step=conv['first_hold_failure'],
                          longest_settling_streak=conv['longest_settling_streak']))
full = sorted((x for x in summaries if x['scope'] == 'all'), key=lambda x: (-x['passed'], -x['hold_passes'], x['candidate']))
limited = sorted((x for x in summaries if x['scope'] != 'all'), key=lambda x: (-x['hold_passes'], x['candidate']))
lines = ['# GPU leaderboard — cuda_fp32_v1', '',
         '**Complete: 97 GPU toy runs and eight GPU convergence runs; all saved verdicts audited.**',
         'The remaining 79 matrix cells are unsupported by the published adapters and receive no credit.',
         'CPU scores are historical and do not enter this table. [Protocol and replay](README.md).', '',
         '## Full 22-toy coverage', '',
         '| Candidate | Toy PASS | Toy FAIL | Confirmed at | Good hold checks / 1200 | Hold result |',
         '|---|---:|---:|---:|---:|---|']

def convergence_text(item):
    if item['hold_status'] == 'NOT_CONVERGED':
        return 'Not confirmed by 6000'
    if item['hold_status'] == 'PASS':
        return 'PASS'
    return 'FAIL at ' + str(item['failure_step'])

for item in full:
    lines.append(f"| {item['candidate']} | {item['passed']}/22 | {item['failed']} | {item['converged_step'] or '—'} | {item['hold_passes']} | {convergence_text(item)} |")
lines += ['', 'Coverage rank is by toy pass count; equal counts are tied. Hold results are shown separately.', '',
          '## Limited adapter coverage', '',
          '| Candidate | Toy PASS / supported | Toy FAIL | Unsupported | Confirmed at | Good hold checks / 1200 | Hold result |',
          '|---|---:|---:|---:|---:|---:|---|']
for item in limited:
    lines.append(f"| {item['candidate']} | {item['passed']}/{22-item['unsupported']} | {item['failed']} | {item['unsupported']} | {item['converged_step'] or '—'} | {item['hold_passes']} | {convergence_text(item)} |")
lines += ['', 'Limited rows cannot take an overall 22-toy slot. A supported-subset pass is not full qualification.', '',
          '## All 22 toys', '', '| Toy | ' + ' | '.join(r['name'] for r in rows) + ' |', '|---|' + '---|' * len(rows)]
for task in toys:
    lines.append('| ' + task + ' | ' + ' | '.join(events[r['name'], task]['status'] for r in rows) + ' |')
lines += ['', '## Native 100-mode results', '',
          '| Candidate | Toy | Final modes | Final HQ | Coverage | Accuracy |', '|---|---|---:|---:|---|---|']
for item in full:
    for task in ('grid100', 'rotated100', 'staggered100'):
        event = events[item['candidate'], task]
        result = json.loads((root / 'runs' / event['artifact'].split('/runs/', 1)[1]).read_text())
        coverage = result['coverage']['problems'][task]
        metrics = coverage['final_metrics']
        lines.append(f"| {item['candidate']} | {task} | {metrics['modes']}/100 | {metrics['hq']:.5f} | {coverage['status']} | {result['accuracy']['status']} |")
qualified = [x['candidate'] for x in full if x['failed'] == 0 and x['hold_status'] == 'PASS']
lines += ['', '**No release-qualified winner.**' if not qualified else '**Fully passing candidates: ' + ', '.join(qualified) + '.**', '',
          'All toy verdicts use their frozen sustained live-model criteria. The separate hold begins only after',
          '200 consecutive qualifying checks; learning-time dips do not themselves fail that hold.',
          'A hold failure stops the diagnostic at its first miss, so the count does not describe later recovery.', '',
          '[Audit](audit.json) · [Raw ledger](ledger.jsonl) · [Declarations](candidates.json) · [Device repairs](repairs.json)', '',
          'Completed ' + datetime.now(timezone.utc).isoformat()]
lines[0] = '# GPU leaderboard — continuous-learning variants (cuda_fp32_v1)'
lines.insert(lines.index('## Full 22-toy coverage'), '## Previously passing 22-toy baseline\n\nThe original `constraints_simple_regularization` recipe remains **22/22 PASS on\nits recorded CPU run**, independently regraded. Its preserved-recipe GPU control\nscores **16/22**, including **3/3 native 100-mode passes**.\n\n| Recipe / scope | Backend | Toy passes | Native 100-mode passes |\n|---|---|---:|---:|\n| Original recipe: decay and original auxiliary host terms | Recorded CPU, regraded | **22/22 PASS** | 3/3 |\n| Same original recipe | CUDA control | **16/22** | 3/3 |\n| Best fully tested continuous-learning variants below | CUDA | 11/22 | 0/3 |\n\n[Original CPU winner](../simpler22/README.md) ·\n[GPU control, failures, and protocol comparison](../gpu-known-winner-control/README.md)\n\nThe eight variants below are a separate continuous-learning cohort. They change\nthe optimizer/noise policies and rate schedule; H and its descendants also\nremove auxiliary losses from the AE and unused-token hosts. Their scores do not\nreplace the known 22/22 CPU result or constitute a same-recipe backend comparison.\n\n')
lines = [line.replace('**No release-qualified winner.**', '**No release-qualified winner in this continuous-learning cohort.**') for line in lines]
(root / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
(root / 'summary.json').write_text(json.dumps(dict(profile='cuda_fp32_v1', complete=True, candidates=summaries,
                                                release_qualified=qualified, toy_runs=97, convergence_runs=8,
                                                unsupported=79), indent=2) + '\n')
print(json.dumps(summaries, indent=2))
