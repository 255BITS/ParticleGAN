"""Frozen native states: rowwise network/prior compensation and local width."""
from pathlib import Path
import sys, time
import torch
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import read_json, atomic_json, file_hash
from experiments.forge.state import state_digest
from experiments.forge.api import task_formulation_context
from experiments.forge.adapters import _models
from lib.toy_models import SimpleMLPGenerator


def width(model, locations, sigma):
    cols = []
    for axis in range(2):
        tangent = torch.zeros_like(locations); tangent[:, axis] = 1
        cols.append(jvp(model, (locations,), (tangent,))[1].detach())
    jac = torch.stack(cols, 2)
    variance = sigma ** 2 * jac.square().sum((1, 2))
    return dict(mean=float(variance.mean()), median=float(variance.median()))


def energy(delta):
    mean = delta.mean(0)
    return dict(total=float(delta.square().sum(1).mean()),
                shared=float(mean.square().sum()),
                centered=float((delta - mean).square().sum(1).mean()))


def main():
    began = time.monotonic(); torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    directory = Path(__file__).parent
    receipts = [r for r in read_json(directory.parent/'provenance.json')['receipts'] if r['task_id']=='grid100']
    models, tables, bindings = {}, {}, {}
    for r in receipts:
        label = 'hydraulic' if r['candidate_id']=='hydraulic-output-travel-v1' else 'winner'
        desc = r['provenance_checkpoint']; path=Path(desc['artifact_root'])/desc['path']
        assert file_hash(path)==desc['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state_digest(state)==desc['state_sha256']
        with torch.device('meta'):
            model=SimpleMLPGenerator(2,128,3,2)
        model.to_empty(device='cpu'); model.load_state_dict(state['trainer']['models']['G'])
        models[label]=model.double().requires_grad_(False).eval()
        tables[label]=state['trainer']['models']['prior']['z'].double()
        bindings[label]=dict(attempt_id=r['attempt_id'],source_digest=r['source_digest'],checkpoint_sha256=desc['sha256'])
    task=read_json(ROOT/'configs/forge/tasks/grid100.json')
    candidate=read_json(ROOT/'configs/forge/ideas/hydraulic-output-travel-v1.json')
    context=task_formulation_context(candidate,task,device='cpu',root=ROOT)
    g,d=_models(context,task['execution']['model']); trainer=context.build_trainer(g,d,max_steps=7000)
    models['initial']=g.double().requires_grad_(False).eval(); tables['initial']=trainer.prior.z.detach().double()
    with torch.no_grad():
        winner=models['winner'](tables['winner']); fixed=models['hydraulic'](tables['winner'])
        hydraulic=models['hydraulic'](tables['hydraulic'])
        net=fixed-winner; prior=hydraulic-fixed; joint=hydraulic-winner
        denominator=net.square().sum(1).mean().sqrt()*prior.square().sum(1).mean().sqrt()
        compensation=dict(network=energy(net),prior=energy(prior),joint=energy(joint),
                          rowwise_dot=float((net*prior).sum(1).mean()),
                          normalized_rowwise_dot=float((net*prior).sum(1).mean()/denominator))
    widths={name:{table:width(model,z,.025) for table,z in tables.items()} for name,model in models.items()}
    assert torch.equal(rng,torch.get_rng_state())
    result=dict(schema_version=1,qualification_input=False,optimizer_updates_added=0,training_or_evaluation_sampling_draws_added=0,
                original_bindings=bindings,width_total_variance=widths,endpoint_network_prior_decomposition=compensation,
                row_count=20000,analysis_seconds=time.monotonic()-began,
                limitations='Cross-endpoint row correspondence and swapped tables are sensitivity diagnostics, not causal ablations or per-update compensation. Initial state is reconstructed through the public deterministic initialization API. Linearized widths omit activation crossings. No target labels or centers are consumed.',
                torch_global_rng_unchanged=True)
    atomic_json(directory/'saved-deformation.json',result)
    print(result,flush=True)


if __name__=='__main__': main()
