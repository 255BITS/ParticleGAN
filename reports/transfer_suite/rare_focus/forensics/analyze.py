"""Post hoc diagnostics; no output feeds training or changes suite scoring."""
import gzip,json,hashlib
from pathlib import Path
import torch
root=Path('/tmp/pr36-rare-forensics');torch.set_num_threads(1)
read=lambda p:json.loads(gzip.decompress((root/p).read_bytes()))
plan=read('plan.json.gz');spec=plan['spec'];snapshots=read('snapshots.json.gz')
means=torch.tensor(spec['means'],dtype=torch.float64);targetcov=torch.tensor(spec['covariances'],dtype=torch.float64)
summary=[]
def T(value):return torch.tensor(value,dtype=torch.float64)
def geometry(points,cov):
 center=points.mean(0);delta=points-center;emp=delta.T@delta/len(points);eig,vec=torch.linalg.eigh(emp);inv=torch.linalg.inv(torch.linalg.cholesky(cov));return center,delta,emp,eig,vec,torch.linalg.eigvalsh(inv@emp@inv.T)
for snap in snapshots:
 x,z=T(snap['x']),T(snap['z']);jac=T(snap['g_latent_jacobian']);grad=T(snap['d_input_gradient']);hess=T(snap['d_input_hessian']);velocity=T(snap['expected_rp_ascent']);vprior=T(snap['projected_prior_ascent']);idx=torch.tensor(snap['eval_indices']);weights=torch.bincount(idx,minlength=len(x));assignment=torch.cdist(x,means).argmin(1);components=[]
 for k in range(4):
  mask=assignment==k;ids=torch.where(mask)[0];points=x[mask];zz=z[mask];n=len(ids)
  if n<2:components.append(dict(component=k,count=n));continue
  center,delta,emp,eig,vec,ratio=geometry(points,targetcov[k]);minor=vec[:,0]
  samplepoints=x[idx][assignment[idx]==k];*_,sample_ratio=geometry(samplepoints,targetcov[k])
  zdelta=zz-zz.mean(0);zeig=torch.linalg.eigvalsh(zdelta.T@zdelta/n)
  gram=jac[mask]@jac[mask].transpose(-1,-2);jeig=torch.linalg.eigvalsh(gram)
  mahal=torch.einsum('ni,ij,nj->n',points-means[k],torch.linalg.inv(targetcov[k]),points-means[k])
  leave=[]
  if n>2:
   for q in range(n):
    rest=torch.cat([points[:q],points[q+1:]])
    *_,rr=geometry(rest,targetcov[k]);leave.append(dict(removed_atom=int(ids[q]),min_ratio=float(rr.min())))
  def rate(v):
   local=v[mask];return dict(radial=float(2*(delta*local).sum(1).mean()),minor=float(2*((delta@minor)*(local@minor)).mean()),mean_velocity=local.mean(0).tolist())
  components.append(dict(component=k,count=n,ids=ids.tolist(),eval_count=int(weights[mask].sum()),eval_weights=weights[mask].tolist(),center=center.tolist(),covariance=emp.tolist(),whitened_eigen_ratios=ratio.tolist(),sampled_whitened_eigen_ratios=sample_ratio.tolist(),covariance_error=float((emp-targetcov[k]).norm()/targetcov[k].norm()),hq_atoms=int((mahal<=9).sum()),max_mahalanobis_squared=float(mahal.max()),mahalanobis_squared=mahal.tolist(),latent_covariance_eigenvalues=zeig.tolist(),g_jacobian_singular_values=jeig.sqrt().tolist(),g_jacobian_minor_gain=(torch.einsum('i,nij,j->n',minor,gram,minor)).tolist(),d_gradient_norms=grad[mask].norm(dim=1).tolist(),d_hessian_eigenvalues=torch.linalg.eigvalsh(hess[mask]).tolist(),d_hessian_minor_curvature=torch.einsum('i,nij,j->n',minor,hess[mask],minor).tolist(),d_ascent_spread_rate=rate(velocity),projected_prior_ascent_spread_rate=rate(vprior),leave_one_out=leave))
 summary.append(dict(step=snap['step'],components=components))
raw=(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode();(root/'geometry.json.gz').write_bytes(gzip.compress(raw,mtime=0))
for row in summary[-9:]:
 print('STEP',row['step'])
 for c in row['components']:
  print(c['component'],'n',c['count'],'exact_eigs',[round(v,4) for v in c['whitened_eigen_ratios']],'sample_eigs',[round(v,4) for v in c['sampled_whitened_eigen_ratios']],'maxmahal',round(c['max_mahalanobis_squared'],2),'Jmins',[round(v[0],3) for v in c['g_jacobian_singular_values']] if c['component']==3 else '', 'Drate',round(c['d_ascent_spread_rate']['minor'],5),'JDrate',round(c['projected_prior_ascent_spread_rate']['minor'],5))
