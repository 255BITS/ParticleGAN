"""Deterministic FP64 center Jacobians of retained trained native states only."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.func import jvp
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from lib.toy_models import SimpleMLPGenerator
from benchmarks.toy100.problems import DATA_STD
from experiments.forge.contracts import atomic_json,read_json,file_hash
from experiments.forge.state import state_digest


def stats(x):
    v=x.detach().numpy().reshape(-1)
    return dict(min=float(v.min()),median=float(np.median(v)),mean=float(v.mean()),p90=float(np.quantile(v,.9)),max=float(v.max()))


def main():
    began=time.monotonic();torch.set_num_threads(1)
    rng=torch.get_rng_state().clone()
    directory=Path(__file__).parent;provenance=read_json(directory/'provenance.json')
    rows=[]
    for receipt in provenance['receipts']:
        if receipt['task_id']!='grid100':continue
        descriptor=receipt['provenance_checkpoint']
        path=Path(descriptor['artifact_root'])/descriptor['path']
        assert file_hash(path)==descriptor['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state_digest(state)==descriptor['state_sha256']
        prior=state['trainer']['models']['prior']
        assert state['prior']['standardize'] is False
        sigma=float(prior['sigma'])
        assert abs(sigma-.025)<1e-8
        with torch.device('meta'):
            model=SimpleMLPGenerator(z_dim=2,hidden_dim=128,n_hidden=3,out_dim=2)
        model.to_empty(device='cpu');model.load_state_dict(state['trainer']['models']['G'])
        model.double().requires_grad_(False).eval()
        means=prior['z'].double()
        columns=[]
        for axis in range(2):
            tangent=torch.zeros_like(means);tangent[:,axis]=1
            _,column=jvp(model,(means,),(tangent,));columns.append(column.detach())
        jac=torch.stack(columns,dim=2)
        covariance=sigma**2*jac@jac.transpose(1,2)
        trace_ratio=covariance.diagonal(dim1=1,dim2=2).sum(1)/(2*DATA_STD**2)
        eigen_ratio=torch.linalg.eigvalsh(covariance)/(DATA_STD**2)
        # Compare tangent propagation with independent reverse-mode derivatives
        # at the first four retained locations; no random or fixture points.
        expected=torch.stack([torch.autograd.functional.jacobian(model,z) for z in means[:4]])
        error=float((jac[:4]-expected).abs().max())
        assert error<1e-10
        assert state_digest(state)==descriptor['state_sha256']
        rows.append(dict(candidate_id=receipt['candidate_id'],attempt_id=receipt['attempt_id'],
            checkpoint_file=str(path),checkpoint_sha256=file_hash(path),centers=len(means),
            prior_sigma=sigma,target_sigma=DATA_STD,
            local_total_variance_over_target_total_variance=stats(trace_ratio),
            fraction_local_variance_ratio_gt1=float((trace_ratio>1).double().mean()),
            local_minimum_covariance_eigen_ratio=stats(eigen_ratio[:,0]),
            local_maximum_covariance_eigen_ratio=stats(eigen_ratio[:,1]),
            independent_derivative_max_abs_error=error,checkpoint_unchanged=True))
    assert torch.equal(rng,torch.get_rng_state())
    result=dict(schema_version=1,kind='frozen_all_center_fp64_linearized_MoG_width_diagnostic',
        source_digest=provenance['receipts'][0]['source_digest'],qualification_input=False,
        optimizer_updates_added=0,random_sampling_draws_added=0,torch_global_rng_unchanged=True,
        limitations='sigma^2 J J^T is first-order within-kernel covariance at saved means; it omits activation-boundary crossings, nonlinear width and assignment truncation. Not served-law scoring or qualification.',
        scope='CPU FP64 derivatives on restored FP32 generator weights at all20000 saved latent locations, no fixture or sampling substitution',
        results=rows,analysis_seconds=time.monotonic()-began)
    atomic_json(directory/'native-width.json',result)
    print(result)

if __name__=='__main__':main()
