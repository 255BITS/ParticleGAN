"""Tail-friendly summaries of this independent campaign; exits after draining."""
from pathlib import Path
import json,time
BRIEF=Path('/tmp/bcap-physics-round5-20261009/mass_allocation/progress.json')
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/mass_allocation/queue')
def main():
 while True:
  s=json.loads((QUEUE/'queue/state.json').read_text());p=json.loads(BRIEF.read_text());rows=[]
  for role,rid in p['requests'].items():
   for j in s['jobs'].values():
    if rid not in j['subscribers']:continue
    row=dict(role=role,task=j['definition']['task_id'],status=j['status'])
    if j.get('result'):row['grade']=j['result']['task_results'][0]['gate_status']
    if j['status']=='running' and j['attempts']:
     log=Path(j['attempts'][-1]['path'])/'run.log';row['log']=str(log)
     if log.exists():
      observations=[x for x in log.read_text().splitlines() if '"observation"' in x]
      if observations:
       try:row['latest_observation']=json.loads(observations[-1])
       except ValueError:pass
    rows.append(row)
  print(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=p['phase'],tasks=rows,campaign=s['campaigns']['mass-allocation-round5-v1'])),flush=True)
  if p['phase'] in ['trained','published','done']:break
  time.sleep(30)
if __name__=='__main__':main()
