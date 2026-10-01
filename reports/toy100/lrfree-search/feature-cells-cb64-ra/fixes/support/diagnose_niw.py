"""One full covariance posterior-predictive model in bounded Fisher coordinates."""
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


def niw_score(snapshot,ref,source_dtype):
    original,metadata=fit_score(snapshot,ref,source_dtype=source_dtype)
    fitted=inspect.getclosurevars(original).nonlocals
    transform,centers=(fitted[k] for k in ('transform','zcenters'))
    standardized=(ref.double()-snapshot.mean)/snapshot.scale
    z=standardized@transform
    ids,_=snapshot.assign(ref)
    count=snapshot.reference_counts
    residual=z-centers[ids]
    pooled=residual.T@residual/len(z)
    rank=transform.shape[1]
    scatter=torch.stack([residual[ids==c].T@residual[ids==c] for c in range(snapshot.cells)])
    # NIW prior E[Sigma]=pooled and variance pseudo-count4. The unknown mean
    # has flat prior; posterior predictive df=n+4+2 and scale differs from its
    # covariance by df/(df-2). No sample/calibration/oracle labels enter it.
    n=count.clamp_min(1).double()
    degrees=n+4.+2.
    posterior_scale=(scatter+4.*pooled[None])*(1.+1./n[:,None,None])/degrees[:,None,None]
    eig,vec=torch.linalg.eigh(posterior_scale)
    floor=torch.finfo(source_dtype).eps*max(1,rank)*eig.max(1).values.clamp_min(torch.finfo(torch.float64).tiny)
    inverse=(vec/eig.clamp_min(floor[:,None])[:,None,:])@vec.transpose(1,2)
    def arrays(features):
        query=((features.double()-snapshot.mean)/snapshot.scale)@transform
        scores=[]
        for block in query.split(snapshot.chunk):
            delta=block[:,None]-centers[None]
            mahal=torch.einsum('nkr,krh,nkh->nk',delta,inverse,delta).clamp_min(0.)
            statistic=mahal/max(1,rank)
            logsf=f_distribution.logsf(statistic.numpy(),max(1,rank),degrees[None].numpy())
            value=torch.from_numpy(-logsf)
            value[:,count==0]=float('inf')
            scores.append(value)
        return torch.cat(scores)
    def score(features):
        return arrays(features).min(1).values
    metadata.update(prior_covariance=pooled.tolist(),posterior_covariance_eigenvalues=eig.tolist(),
                    posterior_numerical_floor=floor.tolist(),posterior_degrees=degrees.tolist())
    return score,arrays,metadata


@torch.no_grad()
def main():
    shared=setup()
    files=(Path(__file__),ROOT/'diagnose.py',HIGH/'fisher_rank.py')
    hashes={str(p):sha(p) for p in files}
    result=dict(scope='CPU single bounded full covariance NIW support model',
                source_sha256=hashes,covariance_prior_rows=4,center_prior='flat',degrees='n_even+4+2',cases=[])
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    cases += [('cost',n,'highdim',128,'frozen_toy_head') for n in cost.SIZES]
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        source_dtype=next(ev.D.parameters()).dtype
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        score,arrays,meta=niw_score(snap,R[0::2],source_dtype)
        qs,rs=score(q),score(R[1::2])
        assert bool(torch.isfinite(qs).all()) and bool(torch.isfinite(rs).all())
        flags,p=conformal(snap,qs,rs)
        qbest=arrays(q).argmin(1);rbest=arrays(R[1::2]).argmin(1)
        fp=flags&(ev.initial_modes!=ev.unsupported_bin)
        rows=set(fp.nonzero().flatten().tolist())
        if ev.family=='geometry' and ev.n==2048:
            rows.update((1745,1773,1924))
        details=[]
        for rowid in sorted(rows):
            c=int(qbest[rowid]);n=int(snap.reference_counts[c])
            details.append(dict(row=rowid,flagged=bool(flags[rowid]),rare=bool(ev.initial_modes[rowid]==ev.rare_bin),
                best_support_cell=c,even_cell_rows=n,odd_partition_rows=int(snap.real_calibration_counts[c]),
                odd_support_best_cell_rows=int((rbest==c).sum()),degrees=n+6,
                score=float(qs[rowid]),p=float(p[rowid]),global_null_max=float(rs.max()),
                margin_above_global_null_max=float(qs[rowid]-rs.max())))
        row=dict(fixture=case,detector=ev.detector(flags),source_dtype=str(source_dtype),metadata=meta,
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
    (ROOT/'diagnosis-niw.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
