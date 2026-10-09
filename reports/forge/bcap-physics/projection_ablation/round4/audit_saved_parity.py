"""Three-arm numeric parity on inactive tasks; no forwards or sampling."""
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json, read_json
from publish import QUEUE, REQUESTS, OUT
import importlib.util
helper=ROOT/'reports/forge/bcap-physics/constraint_geometry/round3/audit_saved_parity.py'
spec=importlib.util.spec_from_file_location('parity_helpers',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def strip(value):
    if isinstance(value,dict):
        return {k:strip(v) for k,v in value.items() if k not in {'constraint_geometry','constraint_geometry_mode','strict_progress','direction_blend','cpu_rng','cuda_rng'}}
    if isinstance(value,(list,tuple)):return type(value)(strip(v) for v in value)
    return value


def main():
    torch.set_num_threads(1)
    state=read_json(QUEUE/'queue/state.json');checks=[]
    for task in ('two_pole','gaussian1d_smoke','gaussian1d_stability'):
        group={}
        for role,rid in REQUESTS.items():
            job=next(j for j in state['jobs'].values() if rid in j['subscribers'] and j['definition']['task_id']==task)
            group[role]=h.load(job)
        original=group['nonascent']
        for role,(row,saved,samples) in group.items():
            fields=('trainer','streams','initialization','prior') if 'trainer' in saved else ('models','role_parameters','optimizers','streams')
            mismatches=h.compare(strip({k:saved[k] for k in fields}),strip({k:original[1][k] for k in fields}))
            observation=h.compare(row['evidence']['observations'],original[0]['evidence']['observations'])
            outputs=h.compare(h.numeric_outputs(samples),h.numeric_outputs(original[2]))
            ambient=None
            if 'trainer' in saved:
                root=Path(row['evidence']['artifact_root'])
                initial=torch.load(root/'initial-state.pt',map_location='cpu',weights_only=False)
                ambient=h.compare({k:initial['trainer'][k] for k in ('cpu_rng','cuda_rng')},
                                  {k:saved['trainer'][k] for k in ('cpu_rng','cuda_rng')})
                assert not ambient
            check=dict(task_id=task,role=role,attempt_id=row['attempt_id'],control_attempt_id=original[0]['attempt_id'],
                state_mismatches=mismatches,observation_mismatches=observation,scored_output_mismatches=outputs,
                ambient_within_run_mismatches=ambient,bitwise_equal=not(mismatches or observation or outputs))
            checks.append(check);assert check['bitwise_equal'],check
    active={}
    for role in ('direction_blend','strict_progress'):
        rid=REQUESTS[role]
        job=next(j for j in state['jobs'].values() if rid in j['subscribers'] and j['definition']['task_id']=='trajectory')
        active[role]=h.load(job)
    a,b=active['direction_blend'],active['strict_progress']
    fields=('models','role_parameters','optimizers','streams')
    differences=h.compare(strip({k:a[1][k] for k in fields}),strip({k:b[1][k] for k in fields}))
    observations=h.compare(a[0]['evidence']['observations'],b[0]['evidence']['observations'])
    samples=h.compare(h.numeric_outputs(a[2]),h.numeric_outputs(b[2]))
    active_parity=dict(task_id='trajectory',roles=['direction_blend','strict_progress'],
        attempt_ids=[a[0]['attempt_id'],b[0]['attempt_id']],state_mismatches=differences,
        observation_mismatches=observations,scored_output_mismatches=samples,
        bitwise_equal=not(differences or observations or samples))
    assert active_parity['bitwise_equal'],active_parity
    atomic_json(OUT/'inactive-trained-parity.json',dict(schema_version=1,qualification_input=False,checks=checks,trajectory_blend_finite_parity=active_parity,
        excluded_metadata=['constraint_geometry','constraint_geometry_mode','strict_progress','direction_blend'],
        separated_unused_ambient_rng=['trainer.cpu_rng','trainer.cuda_rng'],
        excluded_cross_arm_full_checkpoint_hashes=['training_state_sha256','primary_state_sha256','confirmed_state_sha256'],
        note='Original certificate/checkpoint/confirmation hashes retained. Models, base optimizer states, every consumed stream and numerical outputs compared bitwise; ambient globals checked initial-to-final per run.',
        optimizer_updates_added=0,sampling_draws_added=0))
    print('All nine inactive task/arm comparisons are bitwise equal.')

if __name__=='__main__':main()
