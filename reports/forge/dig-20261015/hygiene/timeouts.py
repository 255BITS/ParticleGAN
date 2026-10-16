"""Classify TIMEOUT/INCOMPLETE verdicts in attempt-coverage.json from saved run logs (read-only)."""
import json, os, collections
WT = os.getcwd()  # run from the repository root
d = json.load(open(f'{WT}/reports/forge/attempt-coverage.json'))
R = d['records']
steps = {t: json.load(open(f'{WT}/configs/forge/tasks/{t}.json'))['execution'].get('steps')
         for t in {r['task'] for r in R if r['status'] == 'TIMEOUT'}}
rows = []
for r in R:
    if r['status'] != 'TIMEOUT':
        continue
    log = os.path.join(os.path.dirname(r['original_attempt']), 'run.log')
    events = []
    if os.path.exists(log):
        for line in open(log):
            try: events.append(json.loads(line))
            except Exception: pass
    obs = [e for e in events if 'step' in e]
    last = obs[-1] if obs else {}
    m = last.get('metrics', {})
    confirmed = [e['step'] for e in obs if e.get('confirmed_pass')]
    passed_checks = [e['step'] for e in obs if m and (e.get('metrics') or {}).get('passed')]
    # trailing run of passing checks (sustained gate tasks)
    trail = 0
    for e in reversed(obs):
        if (e.get('metrics') or {}).get('passed') or e.get('passed'): trail += 1
        else: break
    rows.append(dict(task=r['task'], recipe=r['candidate_id'], family=r['family'], seconds=round(r['seconds']),
                     limit=r['timeout_seconds'], last_step=last.get('step'), total=steps[r['task']],
                     frac=(last.get('step') or 0) / steps[r['task']], confirmed=confirmed[:1], n_obs=len(obs),
                     trail_pass=trail, precision=m.get('precision'), modes=m.get('modes'),
                     word_ok=(m.get('modes') == 5 and m.get('reconstruction_exact') == 1) if r['task'] == 'five_word_joint_smoke' else None,
                     keys=sorted(k for k in last if k not in ('metrics', 'confirmation_metrics'))))
for x in rows:
    print(json.dumps(x))
