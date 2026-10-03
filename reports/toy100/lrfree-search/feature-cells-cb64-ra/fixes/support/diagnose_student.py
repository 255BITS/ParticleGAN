"""Bounded Fisher metric plus one coherent finite-reference Student tail model."""
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
from diagnose_fisher_precision import precision_score,HIGH
from diagnose_highdim import standardized


def student_log_tail(fvalue,rank,counts):
    """-log SF(F_rank, rank*(n-1+prior4)), closed form for even rank."""
    assert rank>0 and rank%2==0
    degrees=rank*(counts.clamp_min(1).double()-1.+4.)
    b=degrees/2.
    logz=-torch.log1p(rank*fvalue/degrees[None])
    logone_minus_z=torch.log(-torch.expm1(logz))
    terms=[]
    for j in range(rank//2):
        constant=torch.lgamma(b+j)-torch.lgamma(b)-torch.lgamma(b.new_tensor(j+1.))
        terms.append(constant[None]+(j*logone_minus_z if j else torch.zeros_like(logz)))
    return -(b[None]*logz+torch.logsumexp(torch.stack(terms),0)).clamp_max(0.)


def predictive_score(snapshot,ref,source_dtype):
    fisher,metadata=precision_score(snapshot,ref,bounded=True,source_dtype=source_dtype)
    fitted=inspect.getclosurevars(fisher).nonlocals
    transform,centers,scale2=(fitted[k] for k in ('transform','zcenters','scale2'))
    rank=transform.shape[1]
    def arrays(features):
        query=standardized(snapshot,features)@transform
        out=[]
        for block in query.split(256):
            fvalue=(block[:,None]-centers[None]).square().sum(2)/scale2[None]
            out.append(student_log_tail(fvalue,rank,snapshot.reference_counts))
        return torch.cat(out)
    def score(features):
        return arrays(features).min(1).values
    return score,arrays,metadata


@torch.no_grad()
def main():
    count=torch.tensor([1,2,5,20,100,500,4000],dtype=torch.long)
    values=torch.tensor([0.,1e-10,.01,.1,1.,2.,4.,8.,16.,100.,1e4],dtype=torch.float64)
    errors=[]
    for rank in (2,4,6,8):
        observed=student_log_tail(values[:,None].expand(-1,len(count)),rank,count)
        expected=torch.from_numpy(-f_distribution.logsf(values[:,None].numpy(),rank,(rank*(count-1+4))[None].numpy()))
        finite=torch.isfinite(expected)
        torch.testing.assert_close(observed[finite],expected[finite],rtol=1e-10,atol=5e-12)
        assert bool(torch.isfinite(observed).all()) and bool((observed>=0).all())
        errors.append(dict(rank=rank,finite_reference_cases=int(finite.sum()),
                           max_abs_error=float((observed[finite]-expected[finite]).abs().max())))
    shared=setup()
    files=(Path(__file__),ROOT/'diagnose.py',ROOT/'diagnose_fisher_precision.py',HIGH/'diagnose_highdim.py')
    hashes={str(p):sha(p) for p in files}
    result=dict(scope='CPU bounded Fisher posterior-predictive Student diagnostic',
                source_sha256=hashes,law='isotropic Gaussian cell, unknown mean/variance, four-row scale prior',
                center_variance_factor='1+1/max(n_even_cell,1)',degrees='rank*(max(n_even_cell,1)-1+4)',
                cdf_checks=errors,cases=[])
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    cases += [('cost',n,'highdim',128,'frozen_toy_head') for n in cost.SIZES]
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        source_dtype=next(ev.D.parameters()).dtype
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        score,arrays,meta=predictive_score(snap,R[0::2],source_dtype)
        qs,rs=score(q),score(R[1::2])
        flags,p=conformal(snap,qs,rs)
        qbest=arrays(q).argmin(1);rbest=arrays(R[1::2]).argmin(1)
        fp=flags&(ev.initial_modes!=ev.unsupported_bin)
        details=[]
        flagged_or_previous=set(fp.nonzero().flatten().tolist())
        if ev.family=='geometry' and ev.n==2048:
            flagged_or_previous.update((1745,1773,1924))
        for rowid in sorted(flagged_or_previous):
            cell=int(qbest[rowid]);n=int(snap.reference_counts[cell])
            details.append(dict(row=rowid,flagged=bool(flags[rowid]),rare=bool(ev.initial_modes[rowid]==ev.rare_bin),
                best_support_cell=cell,even_cell_rows=n,odd_partition_rows=int(snap.real_calibration_counts[cell]),
                odd_support_best_cell_rows=int((rbest==cell).sum()),degrees=meta['rank']*(max(n,1)-1+4),
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
    (ROOT/'diagnosis-student.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
