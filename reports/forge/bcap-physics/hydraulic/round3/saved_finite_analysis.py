"""Finite saved-state probes; separate seed0 diagnostic draws, no training."""
from pathlib import Path
import sys,time
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import read_json,atomic_json,file_hash
from experiments.forge.state import state_digest
from lib.toy_models import SimpleMLPGenerator


def stats(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.9)))


def energy(x):
    return dict(total=float(x.square().sum(1).mean()),shared=float(x.mean(0).square().sum()),
                centered=float((x-x.mean(0)).square().sum(1).mean()))


def main():
    began=time.monotonic();torch.set_num_threads(1);rng=torch.get_rng_state().clone()
    directory=Path(__file__).parent;old=directory.parent/'round2'
    receipts=[r for r in read_json(old/'provenance.json')['receipts'] if r['task_id']=='grid100']
    models,tables,bindings={},{},{}
    for r in receipts:
        label='secant_v2' if r['candidate_id']=='hydraulic-secant-deformation-v2' else 'winner'
        desc=r['provenance_checkpoint'];path=Path(desc['artifact_root'])/desc['path']
        assert file_hash(path)==desc['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state_digest(state)==desc['state_sha256']
        with torch.device('meta'): model=SimpleMLPGenerator(2,128,3,2)
        model.to_empty(device='cpu');model.load_state_dict(state['trainer']['models']['G'])
        models[label]=model.double().requires_grad_(False).eval()
        tables[label]=state['trainer']['models']['prior']['z'].double()
        bindings[label]=dict(attempt_id=r['attempt_id'],source_digest=r['source_digest'],checkpoint_sha256=desc['sha256'])
    # Retained observer targets, not a new training batch or a qualification draw.
    media=read_json(old/'media/candidate/grid100.json')
    path=next(Path(k) for k in media['source_inputs'] if k.endswith('step_007000.npz'))
    assert file_hash(path)==media['source_inputs'][str(path)]
    with np.load(path,allow_pickle=False) as saved: real=torch.from_numpy(saved['target'][:2048].copy()).double()
    distances=torch.cdist(real,real,compute_mode='donot_use_mm_for_euclid_dist')
    distances.masked_fill_(distances==0,float('inf'))
    radius=float(distances.min(1).values.median())
    n,labels=connected_components((distances<=4*radius).numpy(),directed=False)
    labels=torch.as_tensor(labels,dtype=torch.long);count=torch.bincount(labels,minlength=n)
    sums=torch.zeros(n,2,dtype=real.dtype).index_add_(0,labels,real)
    means=sums/count[:,None];ss=torch.zeros(n,dtype=real.dtype).index_add_(0,labels,(real-means[labels]).square().sum(1))
    capacity=ss/(count-1).clamp_min(1)
    capacity=torch.where(count>=2,capacity,distances.min(1).values.square().median())
    noise=torch.randn((20000,2),generator=torch.Generator().manual_seed(0),dtype=torch.float64)*.025
    responses={}
    with torch.no_grad():
        for name,g in models.items():
            c=tables[name];center=g(c);plus=g(c+noise);minus=g(c-noise)
            odd=(plus-minus)*.5;even=(plus+minus)*.5-center
            assignments=torch.cdist(center,real).argmin(1);budgets=capacity[labels[assignments]]
            responses[name]=dict(finite_odd_energy=stats(odd.square().sum(1)),finite_even_center_shift_energy=stats(even.square().sum(1)),
                full_one_sided_response_energy=stats((plus-center).square().sum(1)),
                local_real_capacity=stats(budgets),fraction_odd_energy_over_local_capacity=float((odd.square().sum(1)>budgets).double().mean()))
        base=models['winner'](tables['winner']+noise)
        net=models['secant_v2'](tables['winner']+noise)-base
        prior=models['secant_v2'](tables['secant_v2']+noise)-models['secant_v2'](tables['winner']+noise)
        joint=net+prior
        dot=float((net*prior).sum(1).mean())
        compensation=dict(network=energy(net),prior=energy(prior),joint=energy(joint),rowwise_dot=dot,
                          normalized_rowwise_dot=dot/(energy(net)['total']*energy(prior)['total'])**.5)
    assert torch.equal(rng,torch.get_rng_state())
    result=dict(schema_version=1,qualification_input=False,optimizer_updates_added=0,diagnostic_gaussian_scalar_draws=40000,
        diagnostic_seed=0,random_scope='independent CPU diagnostic generator; original consumed streams untouched',
        original_bindings=bindings,saved_target_file=str(path),saved_target_sha256=file_hash(path),saved_target_count=len(real),
        real_graph=dict(link_distance_multiplier=4,nearest_distinct_median=radius,component_count=n,
                        component_counts=stats(count.numpy()),component_variance_capacity=stats(capacity.numpy())),
        finite_responses=responses,endpoint_compensation=compensation,torch_global_rng_unchanged=True,
        limitations='New matched diagnostic jitter probes at saved full prior tables, not original training draws or served-law grades. Real targets are retained observer samples. Graph components infer neighborhoods without labels; they are not ground-truth mixture components. Endpoint swaps are not causal per-update ablations.',
        analysis_seconds=time.monotonic()-began)
    atomic_json(directory/'saved-finite.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
