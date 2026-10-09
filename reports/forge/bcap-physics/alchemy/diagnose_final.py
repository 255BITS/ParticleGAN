"""Diagnostic output-space witness gradients at saved final samples, no training."""
import json
from pathlib import Path
import time
import torch
from particlegan.distillation import distillation_loss
from experiments.forge import tier1_media
from inspect_saved import teacher

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/alchemy/queue')


def main():
    started=time.monotonic(); torch.set_num_threads(1)
    before=torch.get_rng_state().clone()
    state=json.loads((QUEUE/'queue/state.json').read_text())
    diagnostics=[]
    for rid,submission in state['submissions'].items():
        request=submission['request'];arm='candidate' if request['candidate']['id']=='alchemy_distillation_r2_v1' else 'control'
        for job in state['jobs'].values():
            if rid not in job['subscribers'] or not job.get('result'):continue
            task_id=job['definition']['task_id']
            if not task_id.startswith('vector_'):continue
            task=request['tasks'][task_id];row=job['result']['task_results'][0]
            records,proof=tier1_media._scored_outputs(task,row['evidence'],Path(job['attempts'][-1]['path']))
            fake=records[-1]['samples'].detach().double().requires_grad_()
            real=teacher(task['execution']['host_definition'],len(fake))
            loss,parts=distillation_loss(fake,real,cells=8,return_parts=True)
            gradients={k:torch.autograd.grad(v,fake,retain_graph=True)[0] for k,v in parts.items()}
            norms={k:float(g.square().mean().sqrt()) for k,g in gradients.items()}
            total=torch.autograd.grad(loss,fake)[0]
            cos={f'{a}_{b}':float(torch.nn.functional.cosine_similarity(gradients[a].flatten(),gradients[b].flatten(),dim=0))
                 for a,b in [('mass','location'),('mass','shape'),('location','shape')]}
            diagnostics.append(dict(arm=arm,task_id=task_id,attempt_id=job['result']['attempt_id'],
                saved_sample_proof=proof,loss=float(loss.detach()),residuals={k:float(v.detach()) for k,v in parts.items()},
                output_gradient_rms=norms,output_gradient_cosines=cos,total_output_gradient_rms=float(total.square().mean().sqrt()),
                mass_over_location_shape_rms=norms['mass']/max(norms['location']+norms['shape'],1e-300),
                finite=bool(torch.isfinite(total).all())))
    result=dict(schema_version=1,qualification_input=False,scope='posthoc_output_space_diagnostic_not_actual_training_gradient',
        target_frame='4096 deterministic mixture quantiles; same fixed target law, a separate diagnostic batch',
        limitations='Does not reconstruct the actual last training minibatch or network/prior parameter gradients; cannot establish training causality.',
        optimizer_updates_added=0,random_draws_added=0,global_rng_unchanged=torch.equal(before,torch.get_rng_state()),
        cpu_wall_seconds=time.monotonic()-started,diagnostics=diagnostics)
    (OUT/'final-witness-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print([(x['arm'],x['task_id'],x['residuals'],x['output_gradient_rms']) for x in diagnostics],flush=True)

if __name__=='__main__':main()
