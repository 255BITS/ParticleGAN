"""One declared real-local MMD width; exact free-output donor and descent gate.

The common first native cold real bank sets both the fixed empirical target
and a single frozen nearest-neighbor bandwidth. The generated law is the
equal-weight Gaussian mixture at the clean particles, integrated analytically
at each saved state's actual output-noise scale. No GAN, neural, or optimizer
step occurs. Target means enter the external fixed-draw grade only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import minimize
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.locked_shared import mode_hold
from reports.toy100.sample_anchor_free1200 import initial_support,load_states,sha
from reports.toy100.sample_anchor_mmd_filter import (
    gaussian_mmd_emitted,quality,same_law_rest,squared_distances)

SOURCES=(
    'reports/toy100/sample_anchor_local_mmd_filter.py',
    'reports/toy100/sample_anchor_mmd_filter.py',
    'reports/toy100/sample_anchor_free1200.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_early_geometry.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
)


def local_width(real):
    x=real.double()
    distances=squared_distances(x,x).sqrt()
    distances.fill_diagonal_(float('inf'))
    nearest=distances.min(1).values
    positive=nearest[(nearest>0)&torch.isfinite(nearest)]
    if not len(positive):
        raise RuntimeError('first real bank has zero/invalid nearest-neighbor spacing')
    h=float(positive.median())
    if not np.isfinite(h) or h<=0:raise RuntimeError('invalid local kernel width')
    return h,nearest


def replacement_deltas(real,points,width,sigma):
    """Exact MMD² change for every (real-target, generated-donor) pair."""
    x=real.double();y=points.double()
    b,n=len(x),len(y)
    h2=width**2;s2=sigma**2;d=x.shape[1]
    pref_pq=(h2/(h2+s2))**(d/2)
    pref_qq=(h2/(h2+2*s2))**(d/2)
    kpq_xy=pref_pq*torch.exp(-squared_distances(x,y)/(2*(h2+s2)))
    kpq_xx=pref_pq*torch.exp(-squared_distances(x,x)/(2*(h2+s2)))
    kqq_xy=pref_qq*torch.exp(-squared_distances(x,y)/(2*(h2+2*s2)))
    kqq_yy=pref_qq*torch.exp(-squared_distances(y,y)/(2*(h2+2*s2)))
    new_excluding=kqq_xy.sum(1)[:,None]-kqq_xy
    old_excluding=kqq_yy.sum(1)-torch.diagonal(kqq_yy)
    delta_qq=2*(new_excluding-old_excluding[None,:])/(n*n)
    delta_cross=-2*(kpq_xx.sum(0)[:,None]-kpq_xy.sum(0)[None,:])/(b*n)
    return delta_qq+delta_cross


def donor_descent(real,initial,width,sigma,means):
    points=initial.double().clone()
    rows=[]
    for move in range(len(points)):
        before=float(gaussian_mmd_emitted(real,points,width,sigma))
        deltas=replacement_deltas(real,points,width,sigma)
        flat=int(deltas.argmin())
        real_index,donor_index=divmod(flat,len(points))
        predicted=float(deltas[real_index,donor_index])
        tolerance=64*torch.finfo(torch.float64).eps*max(1.,abs(before))
        if predicted>=-tolerance:break
        proposal=points.clone();proposal[donor_index]=real[real_index].double()
        after=float(gaussian_mmd_emitted(real,proposal,width,sigma))
        if not np.isfinite(after) or abs((after-before)-predicted)>1e-10:
            raise RuntimeError('global donor delta disagrees with actual emitted MMD')
        if after>=before-tolerance:
            raise RuntimeError('selected donor replacement did not strictly decrease MMD')
        points=proposal
        rows.append(dict(move=move+1,real_index=real_index,donor_index=donor_index,
                         before=before,after=after,predicted_delta=predicted,
                         max_displacement=float((points-initial).norm(dim=1).max()),
                         quality=quality(points,means)))
    return points,rows


def local_lbfgs(real,initial,width,sigma,means):
    trajectory=[]
    shape=initial.shape
    initial_value=float(gaussian_mmd_emitted(real,initial,width,sigma))
    def value_gradient(flat):
        points=torch.from_numpy(np.asarray(flat,dtype=np.float64).copy()).reshape(shape)
        points.requires_grad_(True)
        with torch.enable_grad():
            value=gaussian_mmd_emitted(real,points,width,sigma)
            gradient=torch.autograd.grad(value,points)[0]
        if not torch.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite MMD field')
        return float(value.detach()),gradient.detach().numpy().ravel().copy()
    def callback(flat):
        points=torch.from_numpy(np.asarray(flat,dtype=np.float64).copy()).reshape(shape)
        trajectory.append(dict(iteration=len(trajectory)+1,
            mmd2=float(gaussian_mmd_emitted(real,points,width,sigma)),
            quality=quality(points,means)))
    result=minimize(value_gradient,initial.numpy().ravel().copy(),method='L-BFGS-B',jac=True,
        callback=callback,options=dict(maxiter=20,maxfun=100,ftol=1e-12,gtol=1e-8))
    proposal=torch.from_numpy(result.x.copy()).reshape(shape)
    proposed_value=float(gaussian_mmd_emitted(real,proposal,width,sigma))
    tolerance=64*torch.finfo(torch.float64).eps*max(1.,abs(initial_value))
    accepted=np.isfinite(proposed_value) and proposed_value<initial_value-tolerance
    selected=proposal if accepted else initial
    return selected,dict(status='ACCEPTED' if accepted else 'REST',
        solver='L-BFGS-B',maxiter=20,maxfun=100,ftol=1e-12,gtol=1e-8,
        success=bool(result.success),message=str(result.message),
        iterations=int(result.nit),function_evaluations=int(result.nfev),
        before=initial_value,proposed=proposed_value,
        final=float(gaussian_mmd_emitted(real,selected,width,sigma)),
        trajectory=trajectory,final_quality=quality(selected,means))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    cold,warm,input_hashes=load_states()
    means=mode_hold.ring_means()
    stream=torch.Generator().set_state(cold['rng']['data'])
    real=mode_hold.sample_ring(means,128,mode_hold.SIGMA,stream)
    width,nearest=local_width(real)
    source={}
    for name in SOURCES:
        raw=(ROOT/name).read_bytes();source[name]=sha(raw)
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
    declaration=dict(scope='same first cold real128 bank and frozen local bandwidth for two free-output saved clouds; no NN/GAN/optimizer update',
        inputs=input_hashes,source_sha256=source,
        common_real_bank_sha256=sha(real.contiguous().numpy().tobytes()),
        common_real_bank_source='saved cold1 pre_step first native D draw',
        common_real_bank_size=128,width_rule='median strictly positive nearest-neighbor distance within that first real bank',
        frozen_width=width,kernel='Gaussian exp(-||x-y||²/(2h²))',
        target='empirical real bank; additional banks would accumulate into its mean embedding with the same h',
        generated='equal-weight Gaussian mixture at clean particles; analytic p-q and q-q output-noise convolution',
        actual_emitted_sigma=dict(cold1=float(cold['noise_policy']['output_sigma']),
                                  warm1324=float(warm['noise']['output_sigma'])),
        donor='globally best exactly MMD-decreasing donor-to-observed-real replacement, at most N=12 moves',
        local_descent='at most20 L-BFGS-B iterations/100 calls after donor; accept only actual strict MMD decrease',
        fixed_grade='4096 late-noise draws at diagnostic clock240, true means only in external grade',
        cheap_gate='warm final 8 modes/HQ>=.9; cold final at least one mode and HQ>initial, both strict MMD decrease',
        no_width_or_seed_grid=True,shared_gate_eligible=False,
        theorem_scope='equality of full laws implies rest; finite particle zero gradient does not imply law equality')
    (a.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',width=width,source_sha256=source,
        cold_sigma=declaration['actual_emitted_sigma']['cold1'],
        warm_sigma=declaration['actual_emitted_sigma']['warm1324'])),flush=True)
    result=dict(status='COMPLETE',declaration=declaration,
        common_real_bank=real.tolist(),nearest_neighbor_distances=nearest.tolist(),cases={},
        data_rng_after_sha256=sha(stream.get_state().numpy().tobytes()),
        runtime=dict(torch=torch.__version__,scipy=scipy.__version__,
                     cpu_capability=torch.backends.cpu.get_cpu_capability()))
    rng_before=torch.random.get_rng_state().clone()
    for case,state in [('cold1',cold),('warm1324',warm)]:
        sigma=declaration['actual_emitted_sigma'][case]
        initial=initial_support(state).double().detach()
        before=float(gaussian_mmd_emitted(real,initial,width,sigma))
        initial_quality=quality(initial,means)
        donor,donor_rows=donor_descent(real,initial,width,sigma,means)
        donor_value=float(gaussian_mmd_emitted(real,donor,width,sigma))
        final,solver=local_lbfgs(real,donor,width,sigma,means)
        final_value=float(gaussian_mmd_emitted(real,final,width,sigma))
        final_quality=quality(final,means)
        strict=final_value<before-64*torch.finfo(torch.float64).eps*max(1.,abs(before))
        grade=(final_quality['modes']==8 and final_quality['hq']>=.9 if case=='warm1324'
               else final_quality['modes']>=1 and final_quality['hq']>initial_quality['hq'])
        row=dict(actual_output_sigma=sigma,initial_mmd2=before,initial_quality=initial_quality,
            donor_mmd2=donor_value,donor_moves=donor_rows,donor_quality=quality(donor,means),
            solver=solver,final_mmd2=final_value,final_quality=final_quality,
            max_output_displacement=float((final-initial).norm(dim=1).max()),
            strict_objective_decrease=strict,grade_gate=grade,cheap_gate_pass=strict and grade,
            final_points=final.tolist())
        result['cases'][case]=row
        print(json.dumps(dict(event='CASE_DONE',case=case,width=width,
            donor_moves=len(donor_rows),before=before,donor=donor_value,final=final_value,
            initial_grade={k:initial_quality[k] for k in ('modes','hq')},
            donor_grade={k:row['donor_quality'][k] for k in ('modes','hq')},
            final_grade={k:final_quality[k] for k in ('modes','hq')},
            cheap_gate_pass=row['cheap_gate_pass'])),flush=True)
    result['all_cheap_gates_pass']=all(x['cheap_gate_pass'] for x in result['cases'].values())
    result['global_torch_rng_unchanged']=torch.equal(torch.random.get_rng_state(),rng_before)
    assert result['global_torch_rng_unchanged']
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',all_cheap_gates_pass=result['all_cheap_gates_pass'])),flush=True)


if __name__=='__main__':main()
