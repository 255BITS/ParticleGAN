"""Snapshot raw results without changing verdicts; derive explicit hold-extension checks."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parent
RUN = Path('/ml2/hypergan/gan-attempts/gap-fill-20260925')
manifest = json.loads((DOC/'manifest.json').read_text())
(DOC/'results').mkdir(exist_ok=True)
rows = []
for job in manifest['jobs']:
    path = Path(job['output'])/'result.json'
    if not path.exists():
        continue
    raw = path.read_bytes()
    result = json.loads(raw)
    saved = DOC/'results'/(job['id']+'.json.gz')
    compressed = gzip.compress(raw,mtime=0)
    saved.write_bytes(compressed)
    row = {k:job[k] for k in ['id','candidate','kind','task','floors']}
    row.update(raw_status=result['status'],artifact=str(path),artifact_sha256=hashlib.sha256(raw).hexdigest(),snapshot=str(saved.relative_to(DOC)),snapshot_sha256=hashlib.sha256(compressed).hexdigest(),seconds=result['seconds'],environment=result.get('environment'),error=result.get('error'))
    if job['kind'] == 'toy':
        row.update(verdict=result.get('verdict'),live=result.get('result',{}).get('live'))
    elif job['kind'] == 'native' and result['status'] != 'ERROR':
        task=job['task']
        c=result['coverage']['problems'][task]
        a=result['accuracy']['problems'][task]
        row.update(steps=result['steps'],coverage=result['coverage']['status'],accuracy=result['accuracy']['status'],final=c['final_metrics'],terminal_checks=a['terminal_checks'],seed=json.loads((DOC/'sources'/job['candidate']/'config.json').read_text())['seed'])
    elif job['kind'] == 'hold' and result['status'] != 'ERROR':
        gate=result['gate']
        end=(gate.get('converged_step') or 0)+gate.get('hold_budget',1200)
        points=[p for p in result['dense'] if p['step']>end]
        failures=[p for p in points if p['modes']!=8 or p['hq']<.9]
        row.update(gate=gate,hold_window=result['hold_window'],extension=dict(checks=len(points),passing=len(points)-len(failures),first_failure=failures[0] if failures else None,min_hq=min((p['hq'] for p in points),default=None),passed=len(points)==300 and not failures))
    elif job['kind'] in ('shift','shift_frozen'):
        row.update(continued_hold=result.get('continued_hold'),shift_recovery=result.get('shift_recovery'),shift_pair=result.get('shift_pair'),final=result.get('final'),optimizer_final=result.get('optimizer_final'))
    rows.append(row)
for source in manifest['sources']:
    assert hashlib.sha256((DOC/source['saved']).read_bytes()).hexdigest()==source['sha256'], source
summary=dict(declared_jobs=len(manifest['jobs']),completed=len(rows),statuses=dict(collections.Counter(r['raw_status'] for r in rows)),records=rows)
(DOC/'results-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
for r in rows:
    if r['raw_status'] not in ('PASS','UNCONFIRMED'):
        print(r['id'],r['raw_status'],json.dumps(r.get('verdict',{}).get('convergence',{})))
