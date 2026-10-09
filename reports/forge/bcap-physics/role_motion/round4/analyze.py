"""Certified controller counters and saved-center Jacobians; no new updates/draws."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import torch
from torch.func import jvp
from experiments.forge.contracts import read_json,atomic_json,file_hash
from experiments.forge.state import state_digest
from lib.toy_models import SimpleMLPGenerator
from benchmarks.toy100.problems import evaluation_geometry

def stats(x):
    x=x.detach().double().flatten()
    return dict(mean=float(x.mean()),median=float(x.median()),p90=float(x.quantile(.9)),
                min=float(x.min()),max=float(x.max())) if len(x) else None

def main():
    began=time.monotonic();torch.set_num_threads(1);rng=torch.get_rng_state().clone()
    directory=Path(__file__).parent;results=read_json(directory/'results.json')
    provenance=read_json(directory/'provenance.json');rows=[]
    for receipt in provenance['receipts']:
        desc=receipt.get('provenance_checkpoint')
        if not desc:continue
        path=Path(desc['artifact_root'])/desc['path'];assert file_hash(path)==desc['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert state_digest(saved)==desc['state_sha256']
        trainer=saved.get('trainer')
        if not trainer:continue
        label='candidate' if receipt['candidate_id']=='bcap-role-motion-balance-round4-v1' else 'control'
        outcome=next(r for r in results['comparison'][label]['tasks'] if r['task_id']==receipt['task_id'])
        row=dict(arm=label,task_id=receipt['task_id'],checkpoint_sha256=desc['sha256'],
                 checkpoint_state_sha256=desc['state_sha256'],optimizer_updates_added=0,sampling_draws_added=0)
        controller=trainer.get('role_motion')
        if label=='candidate':
            assert controller and controller['summary']['updates']==trainer['completed_steps']
            s=controller['summary'];assert 0<=s['max_network_prior_ratio']<=1
            row['controller']=controller
            row['mean_controller_metrics']={k[:-4]:v/s['updates'] for k,v in s.items() if k.endswith('_sum')}
            row['finite_bound_every_update']=True
        else:assert controller is None
        if receipt['task_id'] not in ['grid100','vector_unequal_width','vector_two_broad']:
            rows.append(row);continue
        gs=trainer['models']['G'];hidden=gs['net.0.weight'].shape[0]
        layers=sum(k.endswith('.weight') for k in gs)-1
        with torch.device('meta'):g=SimpleMLPGenerator(gs['net.0.weight'].shape[1],hidden,layers,gs[f'net.{2*layers}.weight'].shape[0])
        g.to_empty(device='cpu');g.load_state_dict(gs);g.double().requires_grad_(False).eval()
        ps=trainer['models']['prior'];c=ps['z'].double();sigma=float(ps['sigma'])
        center=g(c).detach()
        if receipt['task_id']=='grid100':
            means,sd=evaluation_geometry('grid100',dtype=torch.float64)
            covariance=torch.eye(2,dtype=torch.float64).repeat(100,1,1)*sd**2
        else:
            spec=outcome['host']['definition'];means=torch.tensor(spec['means'],dtype=torch.float64)
            covariance=torch.tensor(spec['covariances'],dtype=torch.float64)
        ids=torch.cdist(center,means).argmin(1)
        cols=[]
        for axis in range(c.shape[1]):
            tangent=torch.zeros_like(c);tangent[:,axis]=1
            _,col=jvp(g,(c,),(tangent,));cols.append(col.detach())
        jac=torch.stack(cols,dim=2);local=sigma**2*jac@jac.transpose(1,2)
        target=covariance[ids];chol=torch.linalg.cholesky(target)
        whitened=torch.linalg.solve_triangular(chol,jac,upper=False)
        eigen=torch.linalg.eigvalsh(sigma**2*whitened@whitened.transpose(1,2))
        independent=torch.stack([torch.autograd.functional.jacobian(g,z) for z in c[:4]])
        error=float((jac[:4]-independent).abs().max());assert error<1e-10
        row['local_jacobian_diagnostic']=dict(scope='uncensored_saved_centers_not_served_law',
            locations=len(c),prior_sigma=sigma,nearest_component_center_counts=torch.bincount(ids,minlength=len(means)).tolist(),
            local_total_variance_over_assigned_target_total_variance=stats(local.diagonal(dim1=1,dim2=2).sum(1)/target.diagonal(dim1=1,dim2=2).sum(1)),
            local_min_eigen_ratio=stats(eigen[:,0]),local_max_eigen_ratio=stats(eigen[:,-1]),
            independent_derivative_max_abs_error=error)
        cloud=[]
        for k in range(len(means)):
            selected=ids==k;n=int(selected.sum())
            if n<2:continue
            values=center[selected];delta=values-values.mean(0)
            between=delta.T@delta/n;within=local[selected].mean(0);target=covariance[k]
            cloud.append(dict(component=k,locations=n,
                center_only_covariance_relative_error=float((between-target).norm()/target.norm()),
                center_covariance_trace_over_target_trace=float(between.trace()/target.trace()),
                average_linearized_kernel_trace_over_target_trace=float(within.trace()/target.trace()),
                fixed_membership_linearized_mixture_covariance_relative_error=float((between+within-target).norm()/target.norm())))
        row['center_cloud_diagnostic']=dict(scope='saved_center_nearest_membership_not_served_law',
            component_summary=cloud if len(means)<=4 else None,
            average_center_only_covariance_relative_error=sum(v['center_only_covariance_relative_error'] for v in cloud)/len(cloud),
            average_linearized_mixture_covariance_relative_error=sum(v['fixed_membership_linearized_mixture_covariance_relative_error'] for v in cloud)/len(cloud),
            limitations='Finite uniform center population with fixed posthoc assignments. Linearized kernel covariance omits nonlinear jitter and assignment changes; this is not a replacement density score.')
        row['actual_served_density_metrics']={k:outcome['metrics'].get(k) for k in
            ['precision','hq','mass_tv','component_covariance_error','component_covariance_errors',
             'component_core_covariance_error','component_spill','component_counts','component_min_eigen_ratio']}
        assert state_digest(saved)==desc['state_sha256']
        rows.append(row)
    assert torch.equal(rng,torch.get_rng_state())
    result=dict(schema_version=1,qualification_input=False,optimizer_updates_added=0,sampling_draws_added=0,
        results=rows,global_rng_unchanged=True,analysis_seconds=time.monotonic()-began,
        limitations='sigma^2 J J^T at saved centers is first order and excludes nonlinear kernel motion, spill and assignments. Component labels enter this posthoc analysis only. Certified actual served-law grades remain authoritative.')
    atomic_json(directory/'controller-and-density.json',result)
    print({'seconds':result['analysis_seconds'],'rows':len(rows)},flush=True)

if __name__=='__main__':main()
