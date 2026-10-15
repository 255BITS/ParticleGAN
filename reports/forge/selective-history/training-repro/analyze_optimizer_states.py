"""Read final optimizer packets only; no models are constructed or updated."""
import hashlib
import json
from pathlib import Path

import torch

from analyze_curves import ARMS, ROOT, attempt_directories


def visit(value, path='root'):
    found = []
    if isinstance(value, dict):
        for key in ('constraint_geometry', 'direction_blend', 'component_history', 'critic_cap'):
            if key not in value:
                continue
            state = value[key]
            if key == 'component_history':
                packets = state['signal']['packets']
                summary = {role: {k:v for k,v in packet.items() if k not in ('means','second')}
                           for role,packet in packets.items()}
                state = {k:v for k,v in state.items() if k != 'signal'} | {'packets':summary}
            found.append(dict(path=path+'.'+key, state=state))
        if 'param_groups' in value and 'state' in value:
            found.append(dict(path=path+'.normalized_clock', groups=[
                {k:group.get(k) for k in ('role','algorithm','lr','smoothing','sampled_rows_required')}
                for group in value['param_groups']],
                parameter_steps=[int(slot['step']) for slot in value['state'].values() if 'step' in slot]))
        for key, child in value.items():
            if key not in ('models','streams','role_parameters','applied','recipe','initialization'):
                found.extend(visit(child,path+'.'+str(key)))
    elif isinstance(value,(list,tuple)):
        for index, child in enumerate(value):
            found.extend(visit(child,path+'.'+str(index)))
    return found


def main():
    torch.set_num_threads(1)
    rows=[]
    for directory in attempt_directories():
        request_path=directory/'request.json'
        state_path=directory/'provenance/provenance-state.pt'
        if not request_path.exists() or not state_path.exists():
            continue
        request=json.loads(request_path.read_text())
        arm=ARMS[request['request']['candidate']['id']]
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        rows.append(dict(arm=arm,task=request['job']['task_id'],attempt=directory.name,
                         campaign=directory.parent.name,
                         path=str(state_path),sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                         summaries=visit(state)))
    (ROOT/'optimizer-state-analysis.json').write_text(json.dumps(dict(
        kind='read_only_final_optimizer_packets',training_updates=0,new_model_evaluations=0,
        limitations='Final counters and last history receipt only; no per-update rho or finite-loss trace.',
        attempts=rows),indent=2,sort_keys=True)+'\n')
    for row in rows:
        print(row['arm'],row['task'],row['summaries'])


if __name__=='__main__':
    main()
