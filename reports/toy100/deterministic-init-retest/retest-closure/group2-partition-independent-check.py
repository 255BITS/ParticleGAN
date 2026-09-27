"""Independent stdlib partition review; never starts or stops a worker."""
from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parents[1];R=E/'retest-closure/group2-partition';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=read(R/'partition-receipt.json');orig=read(Path(r['original_spec']['path']));assert sha(Path(r['original_spec']['path']))==r['original_spec']['sha256']=='b72575c0be39cd0aae04d243d9ed6778b016a992ae278da43fcbf0a66402b211'
original_cases=orig['cases'];prefix=r['completed_prefix'];parts=[]
for x in r['partitions']:
 p=Path(x['path']);assert sha(p)==x['sha256'];child=read(p)
 assert child['sources']==orig['sources'] and child['independent_reviews']==orig['independent_reviews']
 assert {k:v for k,v in child.items() if k not in ('cases','lane','instructions')}=={k:v for k,v in orig.items() if k not in ('cases','lane','instructions')}
 assert child['instructions'].startswith(orig['instructions'])
 assert [c['id'] for c in child['cases']]==x['case_ids']
 parts.extend(child['cases'])
assert [c['id'] for c in original_cases[:len(prefix)]]==prefix
assert original_cases[len(prefix):]==parts
ids=prefix+[c['id'] for c in parts];assert len(ids)==len(set(ids))==31
assert [len(x['case_ids']) for x in r['partitions']]==[13,13]
for f in r['retained_prefix_receipts']:assert sha(Path(f['original']))==sha(Path(f['retained']))==f['sha256']
summary=read(R/'prefix-receipts/00-batch-summary.json');assert summary['state']=='STOPPED_BY_SUPERVISOR' and summary['active'] is None and summary['completed']==5
assert [c['candidate'] for c in summary['cases']]==prefix
commands=[json.loads(x) for x in (R/'prefix-receipts/01-executed-commands.jsonl').read_text().splitlines()];assert [x['candidate'] for x in commands]==prefix
for f in r['retained_prefix_receipts'][4:]:
 e=read(Path(f['retained']));assert 'finished_at' in e and 'returncode' in e
 assert e['candidate']==prefix[e['index']] and e['argv']==commands[e['index']]['argv']
paths=read(Path(r['original_attempt'])/'repo/reports/reviewed-batch-paths.json');O=Path(paths['output'])
for c in parts:
 assert not (O/c['id']).exists()
 assert not (O/'logs'/(c['id']+'.execution.json')).exists()
 assert not (O/'logs'/(c['id']+'.log')).exists()
for source in orig['sources'].values():
 for name,wanted in source['files'].items():assert sha(Path(source['root'])/name)==wanted
for proof in orig['independent_reviews']:
 assert sha(Path(proof['path']))==proof['sha256'] and read(Path(proof['path']))['status']==proof['required_status']
active=[];repo=Path(r['original_attempt'])/'repo'
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cwd=(p/'cwd').resolve();args=(p/'cmdline').read_bytes().decode().strip('\0').split('\0');state=(p/'stat').read_text().split(') ',1)[1].split()[0]
 except (OSError,UnicodeError):continue
 if cwd==repo and state!='Z' and any(Path(a).name in ('execute_reviewed_batch.py','run_research_mode_hold.py') for a in args[:3]):active.append(p.name)
assert not active
out=dict(status='PASS_EXACT_UNTOUCHED_TAIL_PARTITION',partition_receipt_sha256=sha(R/'partition-receipt.json'),original_spec_sha256=r['original_spec']['sha256'],tail_specs=[dict(path=x['path'],sha256=x['sha256']) for x in r['partitions']],completed_prefix=5,untouched_cases=26,partition_sizes=[13,13],checks=['Exact original case objects/argv/order retained; no duplicate or new case','All original source groups/proofs identical and freshly hash-verified','Boundary-stop acknowledgement and five terminal execution receipts retained','No original tail output/log/execution exists','Original dispatcher and learner processes have exited'],scope='Stdlib read-only orchestration review. No worker/source change, GPU use, launch or quality alteration.')
(E/'retest-closure/group2-partition-independent-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],sha256=sha(E/'retest-closure/group2-partition-independent-review.json'))))
