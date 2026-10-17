"""Actual even-real anchors with the previously specified four-row covariance prior."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import inspect
import json
from pathlib import Path
import torch
from scipy.stats import f as f_distribution
from diagnose import ROOT,common,geometry,cost,setup,conformal,summary,sha
HIGH=ROOT.parent/'geometry'/'support-highdim'
sys.path.insert(0,str(HIGH))
from fisher_rank import fit_score


def anchored_posterior(snapshot,ref):
    # Preserve the converted-double Fisher transform from the diagnosed raw
    # anchor comparison; actual source head dtype is recorded by the caller.
    original,metadata=fit_score(snapshot,ref)
    fitted=inspect.getclosurevars(original).nonlocals
    transform,means=(fitted[k] for k in ('transform','zcenters'))
    z=((ref.double()-snapshot.mean)/snapshot.scale)@transform
    ids,_=snapshot.assign(ref)
    count=snapshot.reference_counts
    anchors=z[(snapshot.real_representative_rows//2).long()]
    residual=z-means[ids]
    anchored_residual=z-anchors[ids]
    pooled=residual.T@residual/len(z)
    rank=transform.shape[1]
    scatter=torch.stack([residual[ids==c].T@residual[ids==c] for c in range(snapshot.cells)])
    offset=means-anchors
    residual_sum=torch.stack([residual[ids==c].sum(0) for c in range(snapshot.cells)])
    # Mean-then-transform and transform-then-mean need not give an exactly
    # zero residual sum after a conditioned projection. Keep both cross terms
    # in the algebraic identity rather than silently assuming that sum is zero.
    shifted=(scatter+count[:,None,None]*offset[:,:,None]*offset[:,None,:]
             +residual_sum[:,:,None]*offset[:,None,:]+offset[:,:,None]*residual_sum[:,None,:])
    actual=torch.stack([anchored_residual[ids==c].T@anchored_residual[ids==c]
                        for c in range(snapshot.cells)])
    eps=torch.finfo(torch.float64).eps
    bounds=[]
    for c in range(snapshot.cells):
        rows=ids==c
        a=anchored_residual[rows].abs()
        r=residual[rows].abs()
        o=offset[c].abs()
        # A conservative subtraction/addition error for (z-mean)+(mean-anchor)
        # versus z-anchor, followed by dot-product and sum forward error.
        delta_error=8*eps*(z[rows].abs()+2*means[c].abs()[None]+anchors[c].abs()[None])
        gram_error=a.T@delta_error+delta_error.T@a+delta_error.T@delta_error
        magnitude=(a.T@a+r.T@r+int(count[c])*o[:,None]*o[None]
                   +residual_sum[c].abs()[:,None]*o[None]+o[:,None]*residual_sum[c].abs()[None])
        gamma=eps*(len(a)+16)/(1-eps*(len(a)+16))
        bounds.append(gram_error+8*gamma*magnitude)
    bounds=torch.stack(bounds)
    error=(shifted-actual).abs()
    assert bool((error<=bounds).all()),'expanded anchor scatter exceeds its scale-conditioned forward-error bound'
    n=count.clamp_min(1).double()
    degrees=n+4.+2.
    posterior_scale=(actual+4.*pooled[None])*(1.+1./n[:,None,None])/degrees[:,None,None]
    eig,vec=torch.linalg.eigh(posterior_scale)
    floor=torch.finfo(torch.float64).eps*max(1,rank)*eig.max(1).values.clamp_min(torch.finfo(torch.float64).tiny)
    inverse=(vec/eig.clamp_min(floor[:,None])[:,None,:])@vec.transpose(1,2)
    def arrays(features):
        query=((features.double()-snapshot.mean)/snapshot.scale)@transform
        scores=[]
        for block in query.split(snapshot.chunk):
            delta=block[:,None]-anchors[None]
            mahal=torch.einsum('nkr,krh,nkh->nk',delta,inverse,delta).clamp_min(0.)
            statistic=mahal/max(1,rank)
            value=torch.from_numpy(-f_distribution.logsf(statistic.numpy(),max(1,rank),degrees[None].numpy()))
            value[:,count==0]=float('inf')
            scores.append(value)
        return torch.cat(scores)
    def score(features):
        return arrays(features).min(1).values
    metadata.update(prior_covariance=pooled.tolist(),posterior_covariance_eigenvalues=eig.tolist(),
                    posterior_numerical_floor=floor.tolist(),posterior_degrees=degrees.tolist(),
                    representative_rows=snapshot.real_representative_rows.tolist(),
                    representative_minus_cell_mean=offset.neg().tolist(),
                    scatter_offset_identity_max_error=float(error.max()),
                    scatter_offset_identity_max_forward_bound=float(bounds.max()),
                    scatter_offset_identity_max_error_to_bound=float((error/bounds.clamp_min(torch.finfo(torch.float64).tiny)).max()),
                    cell_residual_sum_max_abs=float(residual_sum.abs().max()))
    return score,arrays,metadata


@torch.no_grad()
def main():
    shared=setup()
    files=(Path(__file__),ROOT/'diagnose.py',HIGH/'fisher_rank.py')
    hashes={str(p):sha(p) for p in files}
    result=dict(scope='CPU single bounded actual-real-anchor covariance posterior',
                source_sha256=hashes,covariance_prior_rows=4,degrees='n_even+4+2',
                fisher_precision='converted-double arithmetic from matched anchor diagnostic',cases=[])
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    cases += [('cost',n,m,128 if m=='highdim' else 8,'frozen_toy_head')
              for m in cost.SCENARIOS for n in cost.SIZES]
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        score,arrays,meta=anchored_posterior(snap,R[0::2])
        qs,rs=score(q),score(R[1::2])
        assert bool(torch.isfinite(qs).all()) and bool(torch.isfinite(rs).all())
        flags,p=conformal(snap,qs,rs)
        qbest=arrays(q).argmin(1);rbest=arrays(R[1::2]).argmin(1)
        fp=flags&(ev.initial_modes!=ev.unsupported_bin)
        details=[]
        rows=set(fp.nonzero().flatten().tolist())
        if ev.family=='geometry' and ev.n==2048:
            rows.update((1745,1773,1924))
        for rowid in sorted(rows):
            c=int(qbest[rowid]);n=int(snap.reference_counts[c])
            details.append(dict(row=rowid,flagged=bool(flags[rowid]),rare=bool(ev.initial_modes[rowid]==ev.rare_bin),
                best_support_cell=c,even_cell_rows=n,odd_partition_rows=int(snap.real_calibration_counts[c]),
                odd_support_best_cell_rows=int((rbest==c).sum()),degrees=n+6,
                score=float(qs[rowid]),p=float(p[rowid]),global_null_max=float(rs.max()),
                margin_above_global_null_max=float(qs[rowid]-rs.max())))
        row=dict(fixture=case,detector=ev.detector(flags),source_head_dtype=str(next(ev.D.parameters()).dtype),metadata=meta,
                 false_positive_details=details,unsupported_scores=summary(qs,ev.initial_modes==ev.unsupported_bin),
                 calibration_scores=summary(rs,torch.ones(len(rs),dtype=torch.bool)),
                 unsupported_min_margin_above_null_max=float(qs[ev.initial_modes==ev.unsupported_bin].min()-rs.max()))
        result['cases'].append(row)
        print(json.dumps(dict(event='case',fixture=case,
            metrics={k:v for k,v in row['detector'].items() if k!='flag_ids'},
            rare_details=[d for d in details if d['rare']])),flush=True)
    assert hashes=={str(p):sha(p) for p in files}
    assert not torch.cuda.is_initialized()
    result['cuda_initialized']=False
    result['quality_cases_passed']=sum(r['detector']['passes'] for r in result['cases'])
    result['quality_cases_total']=len(result['cases'])
    (ROOT/'diagnosis-anchor-posterior.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
