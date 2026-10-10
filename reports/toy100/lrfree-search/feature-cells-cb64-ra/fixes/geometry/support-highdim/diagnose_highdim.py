"""Fixed highdim support: bounded anchors and real-only discriminant geometry."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parent
SUPPORT=ROOT.parent.parent/'support'
sys.path.insert(0,str(SUPPORT))
from diagnose import setup, common, cost, PREV, OLD, conformal, summary, sha


def standardized(snapshot, x):
    return (x.double()-snapshot.mean)/snapshot.scale


@torch.no_grad()
def anchor_scores(snapshot, ref):
    x=standardized(snapshot,ref)
    representatives=x[(snapshot.real_representative_rows//2).long()]
    chosen=[]
    near=x.new_full((len(x),),float('inf'))
    index=int((x-x.mean(0)).square().sum(1).argmin())
    for _ in range(min(64*4,len(x))):
        chosen.append(index)
        near=torch.minimum(near,(x-x[index]).square().sum(1))
        index=int(near.argmax())
    anchors=x[chosen]
    def metric(points):
        def score(features):
            return torch.cat([torch.cdist(b,points).min(1).values
                              for b in standardized(snapshot,features).split(256)])
        return score
    return dict(full_representative64=metric(representatives),full_anchor256=metric(anchors)),len(anchors)


@torch.no_grad()
def fisher_score(snapshot, ref, *, bounded):
    x=standardized(snapshot,ref)
    ids,_=snapshot.assign(ref)
    means=torch.stack([x[ids==c].mean(0) if bool((ids==c).any()) else x.mean(0)
                       for c in range(snapshot.cells)])
    weights=snapshot.reference_counts.double()/len(x)
    grand=(means*weights[:,None]).sum(0)
    if bounded:
        # Cell-mean directions are a fixed-size real-only dictionary; no H².
        dictionary,_=torch.linalg.qr((means-grand).T,mode='reduced')
    else:
        dictionary=torch.eye(x.shape[1],dtype=x.dtype)
    y=x@dictionary
    centers=means@dictionary
    residual=y-centers[ids]
    within=residual.T@residual/len(x)
    delta=centers-grand@dictionary
    between=(delta*weights[:,None]).T@delta
    eig,vec=torch.linalg.eigh(within)
    floor=torch.finfo(torch.float64).eps*within.shape[0]*eig.max().clamp_min(1e-30)
    whitening=vec/eig.clamp_min(floor).sqrt()[None]
    separation=whitening.T@between@whitening
    signal,basis=torch.linalg.eigh((separation+separation.T)/2)
    rank=min(snapshot.requested_rank,int((signal>signal.max()*torch.finfo(torch.float64).eps*len(signal)).sum()))
    transform=dictionary@whitening@basis[:,-rank:] if rank else dictionary[:,:0]
    z=x@transform
    zcenters=means@transform
    distance=(z-zcenters[ids]).square().sum(1)
    sse=torch.stack([distance[ids==c].sum() for c in range(snapshot.cells)])
    radius=(sse/snapshot.reference_counts.clamp_min(1)).sqrt()
    positive=radius[(snapshot.reference_counts>0)&(radius>0)]
    global_radius=positive.median() if len(positive) else x.new_tensor(1.)
    scale2=(sse+4*global_radius.square())/(snapshot.reference_counts+4.)
    scale2=scale2.clamp_min(global_radius.square()*1e-6)
    scale2*=1+snapshot.reference_counts.clamp_min(1).double().reciprocal()
    def score(features):
        query=standardized(snapshot,features)@transform
        return torch.cat([(((b[:,None]-zcenters[None]).square().sum(2))/scale2[None]).min(1).values.sqrt()
                          for b in query.split(256)])
    metadata=dict(dictionary_width=dictionary.shape[1],rank=rank,
                  generalized_eigenvalues=signal.tolist(),within_eigenvalues=eig.tolist(),
                  covariance_bytes=within.untyped_storage().nbytes(),transform_bytes=transform.untyped_storage().nbytes(),
                  transform_cosines_with_original=(torch.linalg.qr(transform,mode='reduced').Q.T@snapshot.basis).square().sum(1).tolist()
                    if rank else [])
    return score,metadata


@torch.no_grad()
def main():
    shared=setup()
    paths=[Path(__file__),ROOT/'PROTOCOL.md',SUPPORT/'diagnose.py',PREV/'scaling_a'/'shared_toy.py',
           OLD/'geometry'/'run_validation.py',OLD/'pkg-CB64-RA'/'particlegan'/'feature_cells.py',common.CONFIG]
    hashes={str(p):sha(p) for p in paths}
    result=dict(scope='CPU fixed-seed highdim support diagnosis; no quality acceptance substitution',
                seed=cost.SEED,source_sha256=hashes,cases=[])
    for n in cost.SIZES:
        start=time.perf_counter()
        ev=common.Evaluator('cost',n,'highdim',128,'frozen_toy_head',shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        snapshot=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        scores,anchor_count=anchor_scores(snapshot,R[0::2])
        scores['original_pca8']=lambda features:snapshot._scores_metric(snapshot.transform(features))
        metadata={}
        for name,bounded in (('full_fisher8',False),('bounded_dictionary_fisher8',True)):
            scores[name],metadata[name]=fisher_score(snapshot,R[0::2],bounded=bounded)
        row=dict(n=n,anchors=anchor_count,metadata=metadata,methods={})
        for name,score in scores.items():
            qs,rs=score(q),score(R[1::2])
            flags,p=conformal(snapshot,qs,rs)
            row['methods'][name]=dict(detector=ev.detector(flags),
                 bad_scores=summary(qs,ev.initial_modes==ev.unsupported_bin),
                 bad_p=summary(p,ev.initial_modes==ev.unsupported_bin),
                 legitimate_p=summary(p,ev.initial_modes!=ev.unsupported_bin),
                 calibration_scores=summary(rs,torch.ones(len(rs),dtype=torch.bool)))
        row['seconds']=time.perf_counter()-start
        result['cases'].append(row)
        print(json.dumps(dict(event='case',n=n,seconds=row['seconds'],metrics={
            k:{f:v for f,v in value['detector'].items() if f!='flag_ids'} for k,value in row['methods'].items()})),flush=True)
        assert ev.state_hash()==ev.initial_state_hash
    result.update(cuda_initialized=torch.cuda.is_initialized(),sources_unchanged=hashes=={str(p):sha(p) for p in paths})
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (ROOT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='complete',cases=len(result['cases']),cuda_initialized=False)),flush=True)


if __name__=='__main__':
    main()
