"""CPU folded support-shape test: exact local four-anchor convex hull distance."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import itertools
import json
from pathlib import Path
import torch
from diagnose import ROOT,common,geometry,setup,conformal,sha
from diagnose_anchors import farthest_first


def hull_score(query,anchors,chunk=256):
    scores=[]
    width=min(4,len(anchors))
    for block in query.split(chunk):
        distances=(block.square().sum(1)[:,None]+anchors.square().sum(1)[None]-2*block@anchors.T).clamp_min(0.)
        nearest=distances.topk(width,dim=1,largest=False,sorted=True).indices
        points=anchors[nearest]
        best=(block-points[:,0]).square().sum(1)
        for size in range(2,width+1):
            for face in itertools.combinations(range(width),size):
                selected=points[:,face,:]
                origin=selected[:,0]
                directions=selected[:,1:]-origin[:,None]
                gram=directions@directions.transpose(1,2)
                target=block-origin
                rhs=(directions@target[:,:,None]).squeeze(2)
                coefficient=(torch.linalg.pinv(gram,hermitian=True)@rhs[:,:,None]).squeeze(2)
                weights=torch.cat((1.-coefficient.sum(1,keepdim=True),coefficient),1)
                viable=(weights>=-1e-12).all(1)
                weights=weights.clamp_min(0.)
                weights=weights/weights.sum(1,keepdim=True)
                closest=(weights[:,:,None]*selected).sum(1)
                candidate=(block-closest).square().sum(1)
                best=torch.minimum(best,torch.where(viable,candidate,float('inf')))
        scores.append(best.sqrt())
    return torch.cat(scores)


@torch.no_grad()
def main():
    shared=setup()
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    result=dict(scope='CPU fixed four-anchor convex support diagnostic only',cases=[],
                source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'diagnose.py',ROOT/'diagnose_anchors.py')})
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw)
        q=bd._features(trainer,ev.G(ev.z))
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        fullR=(R.double()-snap.mean)/snap.scale
        fullq=(q.double()-snap.mean)/snap.scale
        ref=fullR[0::2]
        anchors=ref[farthest_first(ref,snap.cells*4)]
        qs,rs=hull_score(fullq,anchors),hull_score(fullR[1::2],anchors)
        flags,p=conformal(snap,qs,rs)
        fp=flags&(ev.initial_modes!=ev.unsupported_bin)
        row=dict(fixture=case,detector=ev.detector(flags),
                 false_positive_ids=fp.nonzero().flatten().tolist(),
                 rare_false_positive_ids=(fp&(ev.initial_modes==ev.rare_bin)).nonzero().flatten().tolist(),
                 p1745=float(p[1745]) if len(p)>1745 else None,
                 p1924=float(p[1924]) if len(p)>1924 else None,
                 calibration_max_score=float(rs.max()),unsupported_min_score=float(qs[ev.initial_modes==ev.unsupported_bin].min()))
        result['cases'].append(row)
        print(json.dumps(dict(event='case',**{k:v for k,v in row.items() if k!='detector'},
                             detector={k:v for k,v in row['detector'].items() if k!='flag_ids'})),flush=True)
    assert not torch.cuda.is_initialized()
    result['cuda_initialized']=False
    (ROOT/'diagnosis-hull.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
