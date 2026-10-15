"""One explicitly authorized execution-repair retry of the frozen word job."""
import argparse, datetime, hashlib, json, os, subprocess, time
from pathlib import Path
from experiments.forge.queue import Queue, drain, process_identity, lease_held

p=argparse.ArgumentParser(); p.add_argument('--device', choices=('0','1'), required=True); a=p.parse_args()
root=Path('/ml2/hypergan/particlegan-selective-history-20261015')
out=Path('/ml2/hypergan/combination-training-20261015'); stage=out/'ordinary-tier1'
plan=json.loads((stage/'word-execution-repair-plan.json').read_text())
request_path=Path(plan['request_file']); frozen=request_path.read_bytes(); request=json.loads(frozen)
assert hashlib.sha256(frozen).hexdigest()==plan['request_sha256']
assert request['source']['digest']==plan['source_digest'] and request['through_tier']==1
queue=Queue(out/'queue',report_root=root/'reports/forge',on_completion=None)
state=queue.inspect(); key=plan['compatibility_key']; job=state['jobs'][key]
assert len(job['attempts'])==1 and job['status']=='terminal'
assert not lease_held(out/'queue/coordinator.lock'), 'Original coordinator still active'
original_dir=Path(job['attempts'][0]['path'])
assert not lease_held(original_dir/'execution.lock'), 'Original execution lease still held'
child=json.loads((original_dir/'child.json').read_text())
assert process_identity(child['pid'])!=child['process_identity'], 'Original child still running'
result=job['result']; assert result['raw']['attempt_status'] in {'timeout','error','cancelled'}
assert result['attempt_id']==plan['original_attempt_id'] and job['definition']['budget_seconds']==900
assert state['submissions'][request['request_id']]['status']!='cancelled', 'Public re-enqueue frozen request required first'
assert all(state['jobs'][j['key']]['status']=='terminal' for j in json.loads((stage/'launch-manifest.json').read_text())['roster'][0]['jobs'])
assert all(state['jobs'][j['key']]['result']['task_results'][0]['gate_status']=='PASS' for j in json.loads((stage/'launch-manifest.json').read_text())['roster'][0]['jobs'] if j['key']!=key)
campaign=state['campaigns'][request['campaign_id']]
assert campaign['reserved_seconds']==0 and campaign['spent_seconds']+900<=campaign['definition']['budget_seconds']
durable=root/'reports/forge/attempts'/result['attempt_id']
assert all((durable/n).exists() for n in ('request.json','result.json','evidence.json'))
# A transient GPU gap is not an execution repair. Reject a live refill feeder.
feeders=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit(): continue
 try:
  args=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
 except (OSError,PermissionError): continue
 if not any('attempt_coverage' in v for v in args) or '--watch' not in args: continue
 gpu_args=[]
 for i,value in enumerate(args):
  if value=='--gpus' and i+1<len(args): gpu_args=args[i+1].split(',')
  elif value.startswith('--gpus='): gpu_args=value.split('=',1)[1].split(',')
 if not gpu_args or a.device in gpu_args: feeders.append(int(proc.name))
assert not feeders, ('Live attempt_coverage watcher can refill selected GPU; scheduling repair required',feeders)
# These are known persistent servers; any other compute client blocks this repair.
allowed={'0':{98921,34929,12273,2415818}, '1':{3062428}}
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpu=[x.split(', ') for x in gpus.strip().splitlines() if x.split(', ')[0]==a.device][0]
assert int(gpu[2])<=5, ('GPU still busy',gpu)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
selected=[x.split(', ') for x in apps.strip().splitlines() if x.startswith(gpu[1]+',')]
assert all(int(x[1]) in allowed[a.device] for x in selected), ('Other compute clients still present',selected)
assert not (stage/'word-execution-repair-transition.json').exists(), 'One retry only; inspect existing transition'
command=[str(Path(os.sys.executable)),'-m','experiments.forge','--root',str(root),'--queue-root',str(out/'queue'),'retry',key,'--reason',plan['reason']]
transition={'status':'PREPARED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_pid':os.getpid(),'script':str(Path(__file__).resolve()),'request_id':request['request_id'],'request_sha256':plan['request_sha256'],'compatibility_key':key,'original_attempt_id':result['attempt_id'],'source_digest':plan['source_digest'],'original_receipt_sha256':{n:hashlib.sha256((durable/n).read_bytes()).hexdigest() for n in ('request.json','result.json','evidence.json')},'original_status':result['raw']['attempt_status'],'original_cost_seconds':result['raw']['elapsed_seconds'],'campaign_spend_before_retry':campaign['spent_seconds'],'device':a.device,'gpu_at_launch':gpus,'compute_apps_at_launch':apps,'command':command,'reason':plan['reason'],'observer_environment':{'PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS':os.environ.get('PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS')}}
assert os.environ.get('PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS') is None, 'Preserve observed original diagnostics unset'
def save(): (stage/'word-execution-repair-transition.json').write_text(json.dumps(transition,indent=2,sort_keys=True)+'\n')
save(); done=subprocess.run(command,cwd=root,capture_output=True,text=True)
(stage/'word-retry.stdout.json').write_text(done.stdout); (stage/'word-retry.stderr.log').write_text(done.stderr)
assert done.returncode==0, done.stderr
transition['status']='RETRY_QUEUED'; save(); print('RETRY QUEUED',request['request_id'],key,a.device,flush=True)
drain(queue,[a.device],workers_per_gpu=1,allow_sharing=False,campaign=request['campaign_id'],request_filter=request['request_id'])
transition['status']='DRAIN_COMPLETE'; transition['finished_unix']=time.time(); save(); print('DRAIN COMPLETE',flush=True)
