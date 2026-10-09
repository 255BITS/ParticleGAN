"""Derive temporal failures and controller activity from certified final receipts."""
from pathlib import Path
import json
from experiments.forge.contracts import read_json,atomic_json
from experiments.forge.gaussian_tasks import bounds
OUT=Path(__file__).resolve().parent

def failures(row,thresholds):
    return [k for k,op,b in thresholds if not (row[k]>=b if op=='>=' else row[k]<=b if op=='<=' else row[k]==b)]

def main():
    result=read_json(OUT/'results.json');requests=result['requests'];queue=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/queue');state=read_json(queue/'queue/state.json');rows=[]
    for compact in result['task_results']:
        role,tid=compact['role'],compact['task_id'];request=state['submissions'][requests[role]]['request'];task=request['tasks'][tid]
        local=next(Path(j['attempts'][-1]['path']) for j in state['jobs'].values() if j.get('result',{}).get('attempt_id')==compact['attempt_id'])
        saved=read_json(local/'result.json')['task_results'][0];ev=saved['evidence'];curve=ev['observations'];thresholds=task['evaluation']['thresholds'];passing=[not failures(r,thresholds) for r in curve]
        suffix=0
        for flag in reversed(passing):
            if not flag:break
            suffix+=1
        row=dict(role=role,task_id=tid,gate_status=saved['gate_status'],passing_observations=sum(passing),observations=len(curve),terminal_passing_suffix=suffix,
            final_failed_bounds=failures(curve[-1],thresholds),failure_check_counts={k:sum(k in failures(r,thresholds) for r in curve) for k,_,_ in thresholds},
            evaluator_result=saved.get('evaluator_result'),geometry=compact['strict_progress'])
        if tid=='gaussian1d_stability':
            row.update(passing_per_1000=[sum(passing[i:i+24]) for i in range(0,120,24)],
                stationary_failed_steps=[r['step'] for r,ok in zip(curve[:72],passing[:72]) if not ok],
                deadline_terminal_five_pass=passing[91:96],shift_hold_failed_steps=[r['step'] for r,ok in zip(curve[96:],passing[96:]) if not ok],
                continuity=ev['continuity'],restored_step1000_endstate_metrics=next(x['metrics'] for x in result['task_results'] if x['role']==role and x['task_id']=='gaussian1d_smoke'))
        if tid.startswith('vector'):
            m=compact['metrics'];row['served_endpoint_guardrails']={k:m[k] for k in ('component_counts','component_mass','mass_tv','min_mass_ratio','hq','component_covariance_error','component_covariance_errors','component_min_eigen_ratio','component_core_covariance_error','component_spill','max_component_spill')}
            row['missing_components_below_10_samples']=sum(n<10 for n in m['component_counts'])
        rows.append(row)
    atomic_json(OUT/'temporal-diagnostics.json',dict(schema_version=1,qualification_input=False,optimizer_updates_added=0,sampling_draws_added=0,
        note='Scheduled quality and endpoint controller counters. No per-update event-to-quality attribution; terminal suffix alone cannot grade Gaussian all-check retention.',rows=rows))
    print(json.dumps([{k:r[k] for k in ('role','task_id','gate_status','passing_observations','terminal_passing_suffix','final_failed_bounds')} for r in rows],indent=2))
if __name__=='__main__':main()
