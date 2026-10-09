"""Read-only endpoint parameter forces and archived winner parity, no training."""
from pathlib import Path
import time

import torch
from experiments.forge.contracts import atomic_json, file_hash, read_json
from inspect_saved import audit_state

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kernel_witness/queue')
ARCHIVED=Path('/home/martyn/dev/ParticleGAN-bcap-physics-alchemy/reports/forge/bcap-physics/alchemy/results.json')


def main():
    started=time.monotonic();torch.set_num_threads(1)
    before=torch.get_rng_state().clone()
    state=read_json(QUEUE/'queue/state.json');diagnostics=[]
    for rid,submission in state['submissions'].items():
        request=submission['request'];arm='candidate' if request['candidate']['id']=='kernel_witness_r3_v1' else 'control'
        for job in state['jobs'].values():
            if rid not in job['subscribers'] or not job.get('result'):continue
            task_id=job['definition']['task_id']
            if not task_id.startswith('vector_'):continue
            row=job['result']['task_results'][0]
            descriptor=row['evidence']['provenance_checkpoint']
            binding=dict(path=str(Path(descriptor['artifact_root'])/descriptor['path']),sha256=descriptor['sha256'],
                         original_source_digest=request['source']['digest'],attempt_id=job['result']['attempt_id'])
            diagnostics.append(audit_state(request['tasks'][task_id],binding,f'round3_{arm}'))
            print(arm,task_id,diagnostics[-1]['loss'],diagnostics[-1]['parameter_forces'],flush=True)
    archived=read_json(ARCHIVED);current=read_json(OUT/'results.json');parity=[]
    previous={x['task_id']:x for x in archived['arms']['control']['tasks']}
    for row in current['arms']['control']['tasks']:
        old=previous[row['task_id']]
        equal={key:row.get(key)==old.get(key) for key in ('final','observations','passing_checks','terminal_passing_suffix','status')}
        assert all(equal.values()),(row['task_id'],equal)
        parity.append(dict(task_id=row['task_id'],equal=equal,original_attempt=old['attempt_id'],new_attempt=row['attempt_id']))
    atomic_json(OUT/'final-witness-diagnostics.json',dict(schema_version=1,qualification_input=False,
        optimizer_updates_added=0,new_random_sampling_draws=0,scope='posthoc_saved_endpoint_CPU_float64_cubature_parameter_forces',
        limitations='Raw derivatives on reconstructed last real minibatch and deterministic all-row antithetic cubature, not the actual last sampled G batch or a reconstructed normalized optimizer update; no causal guarantee.',
        diagnostics=diagnostics,archived_control_parity=parity,archived_report=str(ARCHIVED),archived_report_sha256=file_hash(ARCHIVED),
        global_rng_unchanged=torch.equal(before,torch.get_rng_state()),cpu_wall_seconds=time.monotonic()-started))


if __name__=='__main__':main()
