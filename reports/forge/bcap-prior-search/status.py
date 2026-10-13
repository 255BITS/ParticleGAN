"""Read-only operational summary; does not launch or grade any run."""
import argparse, datetime, json, pathlib, time
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=pathlib.Path,default=pathlib.Path('/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010'))
parser.add_argument('--logs',action='store_true',help='print active worker run.log paths for tail -F')
args=parser.parse_args()
base=args.archive
names=['sigreg','coulomb','proximal','covariance','mmd']
if args.logs:
 for name in names:
  state=json.loads((base/name/'queue/queue/state.json').read_text())
  for job in state['jobs'].values():
   if job['status']=='running':
    path=pathlib.Path(job['worker']['directory'])/'run.log'
    if path.is_file():
     print(path)
 raise SystemExit(0)
def last_step(job):
 path=pathlib.Path(job['worker']['directory'])/'run.log'
 if not path.exists():
  return ''
 with path.open('rb') as handle:
  handle.seek(max(0,path.stat().st_size-65536))
  tail=handle.read().decode('utf-8',errors='replace')
 for line in reversed(tail.splitlines()):
  try:
   event=json.loads(line)
  except (ValueError,TypeError):
   continue
  if isinstance(event,dict) and isinstance(event.get('step'),(int,float)):
   return f":step{event['step']}"
 return ''
print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
for name in names:
 state=json.loads((base/name/'queue/queue/state.json').read_text()); progress=json.loads((base/name/'progress.json').read_text())
 descriptions=[]
 for role,rid in [('control',progress['control_request'])]+[(str(i+1),rid) for i,rid in enumerate(progress['search_requests'])]:
  entry=state['submissions'][rid]; request=entry['request']; rows={r['task_id']:r for d in request['jobs'] for r in (state['jobs'][d['compatibility_key']].get('result') or {}).get('task_results',[])}
  if role != 'control':
   overrides=request['candidate']['recipe_overrides']
   axis='prior_proximal_strength' if name=='proximal' else 'prior_reg'
   role=f"{axis}={overrides[axis]:g}"
  counts=[]
  for tier in [1,2]:
   assigned=[a for a in request['view']['assignments'] if a['qualification_tier']==tier and a['importance']=='required']; actual=[rows[a['task']] for a in assigned if a['task'] in rows]
   counts.append(f"{sum(r['gate_status']=='PASS' for r in actual)}/{len(actual)}/{len(assigned)}")
  descriptions.append(f"{role}:{entry['status']}[{' '.join(counts)}]")
 live=[f"{j['definition']['task_id']}@{j['worker']['device']}:{round(time.time()-j['worker']['started_at'])}s{last_step(j)}" for j in state['jobs'].values() if j['status']=='running']
 print(name,progress['phase'],' '.join(descriptions),'LIVE',','.join(live))
