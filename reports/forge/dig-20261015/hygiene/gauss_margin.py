"""Gaussian1d_smoke margin from saved evidence only (read-only)."""
import json, os, statistics, collections
WT = os.getcwd()  # run from the repository root
os.chdir(WT)
# (i) bcap-six saved-sample audits
d = json.load(open('reports/forge/bcap-six/readout.json'))
print('## bcap-six saved-sample audits (selected BCAP recipe, smoothing arms)')
for a in d['gaussian_saved_sample_audits']:
    f = a['first_confirmed_pair']
    print('  smoothing %-6g first_confirmed_step %4d primary_ks %.6f confirm_ks %.6f endpoint_ks %.6f confirmed_checks %d/24'
          % (a['optimizer_smoothing'], f['step'], f['primary']['cdf_ks'], f['confirmation']['cdf_ks'],
             a['endpoint']['cdf_ks'], a['confirmed_scheduled_checks']))
# (ii) passing_observations across PASS records in test-performance.json
tp = json.load(open('reports/forge/test-performance.json'))
recipes = {r['id']: r.get('label') for r in tp['recipes']}
seen = {}
for t in tp['tests']:
    if t['task_id'] != 'gaussian1d_smoke':
        continue
    for rid, ids in t['cells'].items():
        for i in ids:
            r = tp['records'][i]
            if r['status'] != 'PASS':
                continue
            res = r.get('result') or {}
            ev = res.get('evaluator_summary') or (res.get('original_grade') or {}).get('evaluator_result') or {}
            if 'passing_observations' not in ev:
                continue
            key = res.get('attempt_id') or (rid, json.dumps(res.get('metrics'), sort_keys=True))
            seen[key] = (recipes.get(rid), ev['passing_observations'], ev.get('first_confirmed_step'), t['lane'])
po = collections.Counter(v[1] for v in seen.values())
print('## test-performance.json: distinct PASS gaussian1d_smoke results with saved evaluator summary: %d' % len(seen))
print('  passing scheduled observations (of 24) -> count:', dict(sorted(po.items())))
print('  share passing on exactly 1 observation: %.0f%%; <=2: %.0f%%' % (
    100 * po[1] / len(seen), 100 * (po[1] + po[2]) / len(seen)))
fcs = sorted(v[2] for v in seen.values() if v[2] is not None)
print('  first confirmed step: min %d median %d max %d' % (fcs[0], statistics.median(fcs), fcs[-1]))
print('## single-observation passes (label, lane):')
c = collections.Counter((str(v[0])[:45], v[3]) for v in seen.values() if v[1] == 1)
for k, n in c.most_common(): print('  ', n, k)
sel = [v for v in seen.values() if v[0] in ('BCAP', 'BISPF', 'GAN v3 release 0.7', 'K3P', 'KA2', 'R1/R2', 'recipe-prior-refactor-bcap-pure-v1')]
print('## selected-recipe labels (label, passing_obs, first_step):', sorted((str(v[0]), v[1], v[2]) for v in sel))
# (iii) attempt-fill run logs: primary/confirmation KS at first confirmed observation
ac = json.load(open('reports/forge/attempt-coverage.json'))
rows = []
for r in ac['records']:
    if r['task'] != 'gaussian1d_smoke' or r['status'] not in ('PASS', 'FAIL') or not r.get('original_attempt', '').startswith('/'):
        continue
    log = os.path.join(os.path.dirname(r['original_attempt']), 'run.log')
    if not os.path.exists(log):
        continue
    ev = [json.loads(l) for l in open(log) if l.startswith('{')]
    conf = [e for e in ev if e.get('full_pass') or e.get('confirmed_pass')]
    best = min((e['metrics'].get('cdf_ks', 1) for e in ev if 'metrics' in e), default=None)
    first = conf[0] if conf else None
    rows.append((r['status'], r['candidate_id'], first and first['step'], first and first['metrics']['cdf_ks'],
                 len(conf), len(ev), best))
print('## attempt-fill run logs (new attempts, local evidence; primary draw only, no confirmation draw logged): %d runs' % len(rows))
for x in sorted(rows, key=lambda x: (x[0], x[3] or 9)):
    print('  %-4s %-40s first_full_pass_step %-5s ks %-7s full_pass_obs %s/%s min_ks %s' % (
        x[0], x[1][:40], x[2], x[3] and '%.4f' % x[3], x[4], x[5], x[6] and '%.4f' % x[6]))
