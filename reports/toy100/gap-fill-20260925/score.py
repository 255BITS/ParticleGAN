"""Combine disjoint measured gates for identical saved formulations; preserve seed scope."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

DOC=Path(__file__).resolve().parent
OLD=DOC.parent/'overnight-20260925'
manifest=json.loads((DOC/'manifest.json').read_text())
old=json.loads((OLD/'evidence.json').read_text())
fresh=json.loads((DOC/'results-summary.json').read_text())
gates=[]
for record in old['records']:
    candidate={'p1':'k3','k3p':'k3p','k3g':'k3g'}.get(record['candidate']) or (record.get('arm') or '').split('+')[0]
    if candidate not in ('k3','k3p','k3g','rg5-bcap'): continue
    if record['kind'] not in ('focused_toy','full_native'): continue
    result=json.loads(gzip.decompress((OLD/record['snapshot']).read_bytes()))
    kind='native' if record['kind']=='full_native' else 'toy'
    seed=None
    if kind=='native':
        seed=record.get('seed') or result.get('seed')
        if seed is None:
            matches=re.findall(r'123[4-7]',str(record['gate']))
            assert len(matches)==1,record
            seed=int(matches[0])
        for name in ['mechanism.py','latent.py','response.py']:
            assert hashlib.sha256((DOC/'sources'/candidate/name).read_bytes()).hexdigest()==result['candidate_hashes'][name]
        assert result['steps']==7000
        assert (result['status']=='PASS')==(result['coverage']['status']=='PASS' and result['accuracy']['status']=='PASS')
    else:
        assert result['status']==result['verdict']['status']
        config=json.loads((DOC/'sources'/candidate/'config.json').read_text())
        assert {k:v for k,v in config.items() if k!='device'}=={k:v for k,v in result['config'].items() if k!='device'}
    gates.append(dict(candidate=candidate,task=result['task'],kind=kind,seed=seed,status=result['status'],source='overnight',snapshot='../overnight-20260925/'+record['snapshot'],artifact_sha256=record['artifact_sha256']))
for r in fresh['records']:
    if r['kind'] not in ('native','toy'):continue
    gates.append(dict(candidate=r['candidate'],task=r['task'],kind=r['kind'],seed=r.get('seed'),status=r['raw_status'],source='gap-fill',snapshot=r['snapshot'],artifact_sha256=r['artifact_sha256']))
tasks=list(manifest['fixtures'])+['grid100','rotated100','staggered100']
assert len(tasks)==22
summary={}
for candidate in ['k3p','k3g','k3','rg5-bcap']:
    own=[g for g in gates if g['candidate']==candidate]
    canonical=[g for g in own if g['kind']=='toy' or g['seed']==1234]
    assert len(canonical)==len({g['task'] for g in canonical}),candidate
    bytask={g['task']:g for g in canonical}
    canonical=[bytask.get(t,dict(candidate=candidate,task=t,status='NOT_RUN')) for t in tasks]
    native={task:dict(passing=sum(g['status']=='PASS' for g in own if g['task']==task),measured=sum(g['task']==task for g in own),seeds=[g['seed'] for g in own if g['task']==task]) for task in ['grid100','rotated100','staggered100']}
    summary[candidate]=dict(statuses=dict(collections.Counter(g['status'] for g in canonical)),transfer_pass=sum(g['status']=='PASS' for g in canonical if g['task'] in manifest['fixtures']),native=native,gates=canonical)
(DOC/'qualification-summary.json').write_text(json.dumps(dict(scope='22 declared problems; native canonical seed1234. Historical additional native seeds reported separately. No averaging across configurations or early screens.',candidates=summary),indent=2)+'\n')
for c,s in summary.items():print(c,s['statuses'],'transfer',s['transfer_pass'],s['native'])

# The separate RG5+A2 high-floor recovery claim gets its own matched negative control.
live_record=next(r for r in old['records'] if r.get('arm')=='rg5-a2' and r['kind']=='ring_shift')
frozen_record=next((r for r in fresh['records'] if r['candidate']=='rg5-a2' and r['kind']=='shift_frozen'),None)
if frozen_record:
    live=json.loads(gzip.decompress((OLD/live_record['snapshot']).read_bytes()))
    frozen=json.loads(gzip.decompress((DOC/frozen_record['snapshot']).read_bytes()))
    checks={k:live[k]==frozen[k] for k in ['config','initialization_fixture_sha256','stationary','continued_hold','shift_pair']}
    checks['identical_pre_shift_diagnostics']=[r for r in live['diagnostic'] if r['step']<=2400]==[r for r in frozen['diagnostic'] if r['step']<=2400]
    checks['frozen_no_additional_optimizer_updates']=all(r['updates']==2400 and r['moment_steps_min']==2400 and r['moment_steps_max']==2400 for r in frozen['optimizer_final'])
    checks['live_hold_pass']=live['continued_hold']['pass_all']
    checks['live_deadline_recovery_pass']=live['shift_recovery']['deadline_pass']
    checks['frozen_no_recovery']=frozen['shift_recovery']['passing_checks']==0
    pair=dict(status='PASS' if all(checks.values()) else 'UNCONFIRMED',checks=checks,live_snapshot='../overnight-20260925/'+live_record['snapshot'],frozen_snapshot=frozen_record['snapshot'],live_deadline_checks=live['shift_recovery']['deadline_window'],frozen_deadline_checks=frozen['shift_recovery']['deadline_window'],recovery_delay_updates=live['shift_recovery']['delay_updates'],scope='RG5+A2+a_r1r2 at .1/.1 floors, one declared ring seed. Raw live UNCONFIRMED and raw frozen FAIL remain unchanged; this is a derived paired verdict.')
    (DOC/'rg5-recovery-pair.json').write_text(json.dumps(pair,indent=2)+'\n')
    print('RG5+A2 .1/.1 paired recovery',pair['status'],checks)
