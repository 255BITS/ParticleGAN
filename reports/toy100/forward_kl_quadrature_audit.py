"""Fixed-state numerical audit of frozen cold forward-KL sign reversals.

No optimizer decision uses this result. Replays only cold updates 7 and 14,
then measures the recorded GH9-reversed individual moves and whole updates
with progressively higher tensor Gauss-Hermite orders. An analytic bound
covers Gaussian mass outside [-8,8]^2; GH order convergence is an estimate,
not a rigorous bound for the integral inside that square.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.models import linear_output_noise
from reports.toy100.forward_kl_free_filter import quadrature, cross_entropy, log_kernels
from reports.toy100.sample_anchor_free1200 import initial_support, load_states
from reports.toy100.sample_anchor_local_mmd_continuation import native_bank


ROOT=Path(__file__).resolve().parents[2]
ORDERS=(9,13,17,25,33,41)
RADIUS=8.0
SOURCE_NAMES=(
    'reports/toy100/forward_kl_quadrature_audit.py',
    'reports/toy100/forward_kl_free_filter.py',
    'reports/toy100/forward_kl_cumulative_filter.py',
    'reports/toy100/forward_kl_cold_parallel.py',
    'reports/toy100/sample_anchor_local_mmd_continuation.py',
    'reports/toy100/sample_anchor_free1200.py',
    'reports/toy100/pr84_early_geometry.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'benchmarks/toy100/models.py',
)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def em_one(points,locations,weights,variance):
    logkernels=log_kernels(locations,points,variance)
    logresponsibilities=logkernels-torch.logsumexp(logkernels,dim=1,keepdim=True)
    normalized=torch.softmax(weights.log()[:,None]+logresponsibilities,dim=0)
    return normalized.T@locations


def replay(cold, rows, step, width, means):
    stream=torch.Generator().set_state(cold['rng']['data'])
    history=torch.empty((0,2),dtype=torch.float64)
    for absolute in range(1,step+1):
        bank=native_bank(stream,means)
        history=torch.cat((history,bank.double()),0)
    recorded=rows[step-1]
    if sha(bank.contiguous().numpy().tobytes())!=recorded['real_bank_sha256']:
        raise RuntimeError('native replay real-bank hash mismatch')
    initial=(initial_support(cold).double() if step==1 else
             torch.tensor(rows[step-2]['receipt']['final_points'],dtype=torch.float64))
    points=initial.clone()
    sigma=linear_output_noise(.029,step-1,1200,.2)
    if sigma!=recorded['output_sigma']:
        raise RuntimeError('native output-noise clock mismatch')
    variance=width**2+sigma**2
    update=quadrature(history,width,5)
    audit=quadrature(history,width,9)
    receipt=recorded['receipt']
    if abs(float(cross_entropy(*update,points,variance))-
           receipt['initial_cross_entropy'])>1e-10:
        raise RuntimeError('source initial GH5 objective mismatch')
    if abs(float(cross_entropy(*audit,points,variance))-
           receipt['initial_audit9'])>1e-10:
        raise RuntimeError('source initial GH9 objective mismatch')
    pairs=[]
    for record in receipt['accepted_donors']:
        before=points.clone()
        points[record['donor_index']]=history[record['real_index']]
        after5=float(cross_entropy(*update,points,variance))
        after9=float(cross_entropy(*audit,points,variance))
        if abs(after5-record['after'])>1e-10 or abs(after9-record['audit9_after'])>1e-10:
            raise RuntimeError('donor replay differs from archived objective')
        if record['audit9_sign_flip']:
            pairs.append(dict(label=f'donor{record["move"]}',before=before,after=points.clone(),
                              archived_gh5_delta=record['after']-record['before'],
                              archived_gh9_delta=record['audit9_delta']))
    for record in receipt['accepted_em']:
        before=points.clone()
        points=em_one(points,*update,variance)
        after5=float(cross_entropy(*update,points,variance))
        after9=float(cross_entropy(*audit,points,variance))
        if abs(after5-record['after'])>1e-10 or abs(after9-record['audit9_after'])>1e-10:
            raise RuntimeError('EM replay differs from archived objective')
        if record['audit9_sign_flip']:
            pairs.append(dict(label=f'EM{record["step"]}',before=before,after=points.clone(),
                              archived_gh5_delta=record['after']-record['before'],
                              archived_gh9_delta=record['audit9_delta']))
    expected=torch.tensor(receipt['final_points'],dtype=torch.float64)
    if not torch.equal(points,expected):
        raise RuntimeError('final cold clean support differs from archived output')
    pairs.append(dict(label='whole_update',before=initial,after=points.clone(),
        archived_gh5_delta=receipt['final_cross_entropy']-receipt['initial_cross_entropy'],
        archived_gh9_delta=receipt['final_audit9']-receipt['initial_audit9']))
    return history,variance,pairs


def delta_quadrature(real, old, new, width, variance, order):
    """E_target log(q_old/q_new); Gaussian normalizer and x² cancel."""
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    mesh=np.stack(np.meshgrid(nodes,nodes,indexing='ij'),axis=-1).reshape(-1,2)
    noise=torch.as_tensor(math.sqrt(2)*width*mesh,dtype=torch.float64)
    quadrature_weights=torch.as_tensor((weights[:,None]*weights[None,:]).reshape(-1)/
                                   math.pi,dtype=torch.float64)
    old_half=old.square().sum(1)/(2*variance)
    new_half=new.square().sum(1)/(2*variance)
    value=0.
    for block in real.split(16):
        locations=(block[:,None,:]+noise[None,:,:]).reshape(-1,2)
        old_logits=locations@old.T/variance-old_half[None,:]
        new_logits=locations@new.T/variance-new_half[None,:]
        ratio=(torch.logsumexp(old_logits,dim=1)-
               torch.logsumexp(new_logits,dim=1)).reshape(len(block),-1)
        value+=float((ratio@quadrature_weights).sum())
    return value/len(real)


def outside_square_bound(real,old,new,width,variance,radius=RADIUS):
    """Rigorous absolute contribution of |Z_1|>R or |Z_2|>R to CE delta."""
    diff=new-old
    slope=float(diff.norm(dim=1).max())/variance
    offset=float((new.square().sum(1)-old.square().sum(1)).abs().max())/(2*variance)
    tail=math.erfc(radius/math.sqrt(2)) / 2
    pdf=math.exp(-radius*radius/2)/math.sqrt(2*math.pi)
    prob_bound=4*tail
    first_moment_bound=4*pdf+4*math.sqrt(2/math.pi)*tail
    bound=((slope*float(real.norm(dim=1).mean())+offset)*prob_bound+
           slope*width*first_moment_bound)
    return dict(absolute_contribution_upper_bound=bound,
        gaussian_outside_probability_upper_bound=prob_bound,
        slope_bound=slope,offset_bound=offset,radius=radius)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True)
    parser.add_argument('--cold-result',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cold_raw=args.cold_result.read_bytes();cold_result=json.loads(cold_raw)
    if cold_result['status']!='COMPLETE' or cold_result['case']['attempted']!=16:
        raise RuntimeError('complete frozen cold16 receipt required')
    rows=cold_result['case']['rows']
    cold,_,inputs=load_states();means=mode_hold.ring_means()
    width=cold_result['declaration']['frozen_width']
    paths={n:(ROOT/n if (ROOT/n).exists() else args.code_root/n) for n in SOURCE_NAMES}
    sources={n:sha(p.read_bytes()) for n,p in paths.items()}
    for name in ('forward_kl_free_filter.py','forward_kl_cumulative_filter.py',
                 'forward_kl_cold_parallel.py'):
        key='reports/toy100/'+name
        if sources[key]!=cold_result['declaration']['source'][key]:
            raise RuntimeError(f'frozen cold source changed: {key}')
    args.output.mkdir(parents=True)
    for name,path in paths.items():
        target=args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    declaration=dict(scope='fixed-state numerical integral audit only; no optimizer update',
        source=sources,cold_raw_sha256=sha(cold_raw),inputs=inputs,
        steps=[7,14], transitions='every archived GH9-reversed accepted move plus whole update',
        orders=ORDERS,inside_estimate='successive tensor Gauss-Hermite difference, not a theorem',
        outside_square='rigorous aligned-equal-mixture Gaussian tail bound for CE delta',
        radius=RADIUS, objective='E empirical real*N_h log(q_old/q_new)',
        tolerance='report last absolute order difference and sign; no optimizer decision',
        no_seed_or_bandwidth_sweep=True)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='QUADRATURE_AUDIT_DECLARED',source_sha256=sources[
        'reports/toy100/forward_kl_quadrature_audit.py'])),flush=True)
    before_rng=torch.random.get_rng_state().clone()
    result=dict(status='INCOMPLETE',declaration=declaration,steps={})
    try:
        for step in (7,14):
            real,variance,pairs=replay(cold,rows,step,width,means)
            entries=[]
            for pair in pairs:
                start=time.monotonic()
                estimates={str(order):delta_quadrature(real,pair['before'],pair['after'],
                    width,variance,order) for order in ORDERS}
                if abs(estimates['9']-pair['archived_gh9_delta'])>1e-9:
                    raise RuntimeError('independent GH9 delta disagrees with archived receipt')
                orders=list(ORDERS)
                previous=estimates[str(orders[-2])]
                last=estimates[str(orders[-1])]
                row=dict(label=pair['label'],archived_gh5_delta=pair['archived_gh5_delta'],
                    archived_gh9_delta=pair['archived_gh9_delta'],
                    estimates=estimates,last_order_difference=last-previous,
                    numerical_sign_stable_last_two=(last>=0)==(previous>=0),
                    outside_square=outside_square_bound(real,pair['before'],pair['after'],
                                                        width,variance),
                    elapsed_seconds=time.monotonic()-start,
                    old_points=pair['before'].tolist(),new_points=pair['after'].tolist())
                entries.append(row)
                result['steps'][str(step)]=entries
                (args.output/'partial.json').write_text(json.dumps(result,indent=2,
                    allow_nan=False)+'\n')
                print(json.dumps(dict(event='QUADRATURE_PAIR',step=step,label=pair['label'],
                    gh9=estimates['9'],gh41=last,last_order_difference=last-previous,
                    tail_bound=row['outside_square']['absolute_contribution_upper_bound'],
                    seconds=row['elapsed_seconds'])),flush=True)
        result['status']='COMPLETE'
        result['global_torch_rng_unchanged']=torch.equal(torch.random.get_rng_state(),before_rng)
        if not result['global_torch_rng_unchanged']:
            raise RuntimeError('read-only numerical audit changed global RNG')
        (args.output/'result.json').write_text(json.dumps(result,indent=2,
            allow_nan=False)+'\n')
        print(json.dumps(dict(event='QUADRATURE_AUDIT_DONE',
            pairs={step:len(rows) for step,rows in result['steps'].items()})),flush=True)
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE',
            error=repr(error),completed={key:len(value) for key,value in
                                    result['steps'].items()}))+'\n')
        raise


if __name__=='__main__':main()
