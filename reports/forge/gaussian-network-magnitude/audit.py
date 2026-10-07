"""Initial-only CUDA field audit with both optimizer corrections disabled."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.toy_audit import gaussian_network_magnitude as study
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.contracts import atomic_json,file_hash
from experiments.forge.state import state_digest
from particlegan.optim.dualnorm import spectral_capped_direction


@reproducible_execution
def audit(*,device):
    if torch.device(device).type!='cuda':raise ValueError('CUDA required')
    protocol=study.declaration();results=[]
    for tid in protocol['tasks']:
        context,trainer,task=study.build('simultaneous',tid,device)
        initial=context.state_dict()
        data=context.streams.generator('data',component='target',purpose='training',device='cpu')
        target,_=study.shared.scorer(tid)
        real=target(task['execution']['host_definition'],128,data,0).to(device)
        # Public trainer computes its usual initial field; neither optimizer
        # applies it or advances histories. The whole context is then restored.
        with patch.object(trainer.opt_g,'step',return_value=None),patch.object(trainer.opt_d,'step',return_value=None):
            trainer.step(real)
        names={p:'G.'+n for n,p in trainer.G.named_parameters()}
        names.update({p:'D.'+n for n,p in trainer.D.named_parameters()})
        params=[]
        for optimizer in (trainer.opt_g,trainer.opt_d):
            for group in optimizer.param_groups:
                if group['role']=='prior':continue
                for p in group['params']:
                    g=p.grad.detach();direction=spectral_capped_direction(g,.1)
                    singular=torch.linalg.svdvals(g) if g.ndim==2 else g.norm()[None]
                    capped=(singular/.1).clamp_max(1)
                    params.append(dict(parameter=names[p],shape=list(p.shape),gradient_norm=float(g.norm()),
                        largest_singular=float(singular.max()),smallest_singular=float(singular.min()),
                        singular_values_above_cap=int((singular>=.1).sum()),singular_values=len(singular),
                        mean_singular_motion_fraction=float(capped.mean()),direction_norm=float(direction.norm()),
                        nominal_rate=group['lr'],implied_parameter_motion=group['lr']*float(direction.norm())))
        current=context.state_dict()['trainer']
        if state_digest(current['models'])!=state_digest(initial['trainer']['models']):
            raise ValueError('disabled-correction audit mutated model parameters')
        # The public field records critic observations and pending sampled rows;
        # these transient bookkeeping changes are restored below. Parameter
        # histories and update counters must remain untouched.
        for actual,before in zip(current['optimizers'],initial['trainer']['optimizers']):
            if state_digest(actual['state'])!=state_digest(before['state']):
                raise ValueError('disabled-correction audit advanced optimizer histories')
        context.load_state_dict(initial)
        if state_digest(context.state_dict())!=state_digest(initial):raise ValueError('initial audit restore not exact')
        results.append(dict(task=tid,initial_state_restored_exactly=True,parameters=params))
    return dict(scope='separate_initial_field_diagnostic',training_updates=0,public_field_backward_calls=2,
                model_sampling_draws=0,scale=.1,scale_changed_after_audit=False,device=device,
                protocol_sha256=file_hash(study.PROTOCOL),results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(device=args.device);atomic_json(args.output,result);print(json.dumps(result))
