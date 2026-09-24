"""Fixed local-width emitted-MMD free-output continuation and stress cases.

The width is fixed from cold step 1's first native D bank. Every later D bank
enters the cumulative empirical mean embedding. Within one bank history,
J=E_qq k-2 E_pq k omits only E_pp k, which is constant in generated points;
therefore exact J differences equal exact full MMD² differences. No model,
critic, optimizer, or training score enters an update.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.models import linear_output_noise
from reports.toy100.coverage_fixed_eval import fixed_draw,score_support
from reports.toy100.sample_anchor_free1200 import initial_support,load_states,sha
from reports.toy100.sample_anchor_local_mmd_filter import local_width
from reports.toy100.sample_anchor_mmd_filter import gaussian_mmd_emitted,squared_distances,quality

N_PARTICLES=12
N_REAL=128
MAX_MOVES=N_PARTICLES
MAX_ITER=20
MAX_FUN=100
PREFIX=16
DISCOVERY_UPDATES=16
ERROR_RESPONSE_UPDATES=5
SOURCES=(
    'reports/toy100/sample_anchor_local_mmd_continuation.py',
    'reports/toy100/sample_anchor_local_mmd_filter.py',
    'reports/toy100/sample_anchor_mmd_filter.py',
    'reports/toy100/sample_anchor_free1200.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_early_geometry.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'benchmarks/toy100/models.py',
    'particlegan/particle_prior.py',
)


def j_emitted(real,points,width,sigma):
    x=real.double();y=points.double();h2=width**2;s2=sigma**2;d=x.shape[1]
    pq=(h2/(h2+s2))**(d/2)*torch.exp(-squared_distances(x,y)/(2*(h2+s2))).mean()
    qq=(h2/(h2+2*s2))**(d/2)*torch.exp(-squared_distances(y,y)/(2*(h2+2*s2))).mean()
    return qq-2*pq


def candidate_cross_sums(real,width,sigma):
    x=real.double();h2=width**2;s2=sigma**2
    pref=(h2/(h2+s2))**(x.shape[1]/2)
    totals=torch.empty(len(x),dtype=torch.float64)
    for start in range(0,len(x),128):
        right=x[start:start+128]
        totals[start:start+len(right)]=pref*torch.exp(
            -squared_distances(x,right)/(2*(h2+s2))).sum(0)
    return totals


def deltas(real,points,width,sigma,cross_new):
    x=real.double();y=points.double();b,n=len(x),len(y)
    h2=width**2;s2=sigma**2;d=x.shape[1]
    pq=(h2/(h2+s2))**(d/2)*torch.exp(-squared_distances(x,y)/(2*(h2+s2)))
    pref=(h2/(h2+2*s2))**(d/2)
    qxy=pref*torch.exp(-squared_distances(x,y)/(2*(h2+2*s2)))
    qyy=pref*torch.exp(-squared_distances(y,y)/(2*(h2+2*s2)))
    new_excluding=qxy.sum(1)[:,None]-qxy
    old_excluding=qyy.sum(1)-torch.diagonal(qyy)
    return 2*(new_excluding-old_excluding[None,:])/(n*n) - 2*(
        cross_new[:,None]-pq.sum(0)[None,:])/(b*n)


def grade(points,clock,means):
    index,noise=fixed_draw(clock,points.float())
    value=score_support(points.float(),index,noise,means)
    return dict(step=clock-240,modes=value['modes'],hq=value['hq'],
        particle_draws=value['particle_draws'],particle_hq_rate=value['particle_hq_rate'])


def optimize(real,initial,width,sigma,clock,means):
    x=real.double();points=initial.double().clone()
    before=float(j_emitted(x,points,width,sigma))
    cross_new=candidate_cross_sums(x,width,sigma)
    donor_rows=[]
    for move in range(MAX_MOVES):
        previous=float(j_emitted(x,points,width,sigma))
        table=deltas(x,points,width,sigma,cross_new)
        flat=int(table.argmin());i,j=divmod(flat,len(points))
        predicted=float(table[i,j])
        tolerance=64*torch.finfo(torch.float64).eps*max(1.,abs(previous))
        if predicted>=-tolerance:break
        proposal=points.clone();proposal[j]=x[i]
        actual=float(j_emitted(x,proposal,width,sigma))
        if not np.isfinite(actual) or abs((actual-previous)-predicted)>1e-10:
            raise RuntimeError('global donor J delta differs from actual full-MMD delta')
        if actual>=previous-tolerance:raise RuntimeError('global donor did not strictly descend')
        points=proposal
        donor_rows.append(dict(move=move+1,real_index=i,donor_index=j,
                               before=previous,after=actual,predicted_delta=predicted,
                               grade=grade(points,clock,means)))
    donor=float(j_emitted(x,points,width,sigma))
    shape=points.shape
    callback_rows=[]
    def field(flat):
        candidate=torch.from_numpy(np.asarray(flat,dtype=np.float64).copy()).reshape(shape)
        candidate.requires_grad_(True)
        with torch.enable_grad():
            value=j_emitted(x,candidate,width,sigma)
            gradient=torch.autograd.grad(value,candidate)[0]
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite cumulative emitted-MMD field')
        return float(value.detach()),gradient.detach().numpy().ravel().copy()
    def callback(flat):
        candidate=torch.from_numpy(np.asarray(flat,dtype=np.float64).copy()).reshape(shape)
        callback_rows.append(dict(iteration=len(callback_rows)+1,
            j=float(j_emitted(x,candidate,width,sigma)),grade=grade(candidate,clock,means)))
    fit=minimize(field,points.numpy().ravel().copy(),method='L-BFGS-B',jac=True,
        callback=callback,options=dict(maxiter=MAX_ITER,maxfun=MAX_FUN,ftol=1e-12,gtol=1e-8))
    candidate=torch.from_numpy(fit.x.copy()).reshape(shape)
    candidate_j=float(j_emitted(x,candidate,width,sigma))
    tolerance=64*torch.finfo(torch.float64).eps*max(1.,abs(donor))
    accept=np.isfinite(candidate_j) and candidate_j<donor-tolerance
    final=candidate if accept else points
    end=float(j_emitted(x,final,width,sigma))
    if end>before+1e-11:raise RuntimeError('whole update raised exact emitted-MMD objective')
    record=dict(before_j=before,donor_j=donor,after_j=end,
        donor_moves=donor_rows,lbfgs=dict(status='ACCEPTED' if accept else 'REST',
            success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
            function_evaluations=int(fit.nfev),proposed_j=candidate_j,
            trajectory=callback_rows),
        max_output_displacement=float((final-initial).norm(dim=1).max()),
        grade=grade(final,clock,means),points=final.tolist())
    return final,record


def native_bank(stream,means,*,omit_mode0=False):
    real=mode_hold.sample_ring(means[1:] if omit_mode0 else means,N_REAL,mode_hold.SIGMA,stream)
    torch.randint(0,N_PARTICLES,(N_REAL,),generator=stream)
    torch.randint(0,N_PARTICLES,(N_REAL,),generator=stream)
    mode_hold.sample_ring(means,N_REAL,mode_hold.SIGMA,stream)
    return real


def step(state,points,history,stream,absolute_step,width,means,*,omit=False):
    real=native_bank(stream,means,omit_mode0=omit)
    history=torch.cat((history,real.double()),dim=0)
    sigma=linear_output_noise(.029,absolute_step-1,1200,.2)
    selected,receipt=optimize(history,points,width,sigma,240+absolute_step,means)
    if absolute_step==1:
        # The exact saved cold state runs the first update with zero emitted noise.
        assert sigma==state['noise_policy']['output_sigma']
    receipt.update(absolute_step=absolute_step,output_sigma=sigma,
        real_bank_sha256=sha(real.contiguous().numpy().tobytes()),
        observed_real_points=len(history),omitted_d_bank=omit,
        true_mode0_count=int((torch.cdist(real,means).argmin(1)==0).sum()))
    return selected,history,receipt


def run_sequence(state,width,means,*,case,length,omitted_prefix=0):
    points=initial_support(state).double().detach()
    stream=torch.Generator().set_state(state['rng']['data'])
    history=torch.empty((0,2),dtype=torch.float64)
    base_step=1 if case=='cold1' else 1324
    initial=quality(points,means)
    rows=[]
    snapshot=None
    for offset in range(length):
        absolute=base_step+offset
        points,history,row=step(state,points,history,stream,absolute,width,means,
                                omit=offset<omitted_prefix)
        rows.append(row)
        if case=='warm1324' and offset==PREFIX-1:
            snapshot=dict(points=points.clone(),history=history.clone(),
                          rng=stream.get_state().clone())
        if (offset+1)%4==0:
            print(json.dumps(dict(event='PROGRESS',case=case,updates=offset+1,
                                  modes=row['grade']['modes'],hq=row['grade']['hq'],
                                  real_points=len(history))),flush=True)
    result=dict(case=case,length=length,omitted_prefix=omitted_prefix,
        initial_quality=initial,rows=rows,final_quality=quality(points,means),
        first_eight=next((r['absolute_step'] for r in rows if r['grade']['modes']==8
                          and r['grade']['hq']>=.9),None),
        passing=sum(r['grade']['modes']==8 and r['grade']['hq']>=.9 for r in rows),
        final_points=points.tolist(),history_points=len(history),
        final_data_rng_sha256=sha(stream.get_state().numpy().tobytes()))
    return result,snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    cold,warm,inputs=load_states();means=mode_hold.ring_means()
    common_stream=torch.Generator().set_state(cold['rng']['data'])
    common_real=mode_hold.sample_ring(means,N_REAL,mode_hold.SIGMA,common_stream)
    width,_=local_width(common_real)
    sources={}
    for name in SOURCES:
        raw=(ROOT/name).read_bytes();sources[name]=sha(raw)
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    declaration=dict(scope='free-output cumulative emitted-MMD continuation, no neural/optimizer update',
        inputs=inputs,source_sha256=sources,
        common_first_cold_bank_sha256=sha(common_real.contiguous().numpy().tobytes()),
        width_rule='median strictly positive real nearest-neighbor distance in first cold D bank',
        frozen_width=width,noise='host linear output warmup .029 over 0.2*1200 completed steps; warm late .029',
        data='native D real128, two prior-index draws, native G real128 per update',
        objective='J=qq−2pq, exact generated-dependent part of analytic emitted-law Gaussian MMD²; real-real pp cancels within each update',
        candidates='all accumulated observed real points; globally best donor pair each move, at most12 moves',
        solver='L-BFGS-B at most20 iterations/100 calls after donor; strict actual J decrease or rest',
        sequences=dict(cold_full=16,warm_full=16,fullprefix_then_omission=17,
                       two_omitted_then_full=16,model_error_response=5),
        evaluation='every update fixed4096 late-noise draws at clock240+absolute_step; target means only external grade',
        cheap_filters='cold terminal5 must8/HQ>=.9; warm all16 must8/HQ>=.9; omission must retain8; two-omitted must regain8 by final5; output-error must regain8/HQ>=.9 by5',
        no_parameter_or_seed_sweep=True,shared_gate_eligible=False)
    (a.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',width=width,source_sha256=sources)),flush=True)
    rng_before=torch.random.get_rng_state().clone()
    cold_result,_=run_sequence(cold,width,means,case='cold1',length=16)
    warm_result,warm_prefix=run_sequence(warm,width,means,case='warm1324',length=16)
    assert warm_prefix is not None
    # A paired seventeenth bank starts from the exact same 16-bank history.
    pairs={}
    for name,omit in [('ordinary',False),('omitted',True)]:
        points=warm_prefix['points'].clone();history=warm_prefix['history'].clone()
        stream=torch.Generator().set_state(warm_prefix['rng'])
        selected,new_history,row=step(warm,points,history,stream,1340,width,means,omit=omit)
        pairs[name]=dict(row=row,final_quality=quality(selected,means),
                         data_rng_sha256=sha(stream.get_state().numpy().tobytes()))
    assert pairs['ordinary']['data_rng_sha256']==pairs['omitted']['data_rng_sha256']
    missing_result,_=run_sequence(warm,width,means,case='warm1324',length=16,omitted_prefix=2)
    # Model error is injected into output coordinates, not neural parameters.
    moved=warm_prefix['points'].clone();moved[:,0]+=.35
    history=warm_prefix['history'].clone();stream=torch.Generator().set_state(warm_prefix['rng'])
    error_initial=quality(moved,means)
    error_rows=[]
    for absolute in range(1340,1340+ERROR_RESPONSE_UPDATES):
        moved,history,row=step(warm,moved,history,stream,absolute,width,means)
        error_rows.append(row)
    response=dict(initial_quality=error_initial,rows=error_rows,
        final_quality=quality(moved,means),
        first_recovered=next((r['absolute_step'] for r in error_rows
            if r['grade']['modes']==8 and r['grade']['hq']>=.9),None),
        max_initial_shift=.35,
        history_at_injection=16*N_REAL)
    result=dict(status='COMPLETE',declaration=declaration,
        cold_full=cold_result,warm_full=warm_result,
        fullprefix_then_omission=pairs,two_omitted_then_full=missing_result,
        output_error_response=response,
        global_torch_rng_unchanged=torch.equal(torch.random.get_rng_state(),rng_before))
    assert result['global_torch_rng_unchanged']
    gates=dict(
        cold_terminal5=all(r['grade']['modes']==8 and r['grade']['hq']>=.9
                           for r in cold_result['rows'][-5:]),
        warm_all16=warm_result['passing']==16,
        omission_retains8=pairs['omitted']['row']['grade']['modes']==8
                           and pairs['omitted']['row']['grade']['hq']>=.9,
        missing_reacquires_terminal5=all(r['grade']['modes']==8 and r['grade']['hq']>=.9
            for r in missing_result['rows'][-5:]),
        error_recovers_by5=response['first_recovered'] is not None)
    result['gates']=gates
    result['all_gates_pass']=all(gates.values())
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',gates=gates,
        cold_first_eight=cold_result['first_eight'],missing_first_eight=missing_result['first_eight'],
        warm_min_hq=min(r['grade']['hq'] for r in warm_result['rows']),
        omitted_pair={k:(v['row']['grade']['modes'],v['row']['grade']['hq']) for k,v in pairs.items()},
        error_first_recovered=response['first_recovered'])),flush=True)


if __name__=='__main__':main()
