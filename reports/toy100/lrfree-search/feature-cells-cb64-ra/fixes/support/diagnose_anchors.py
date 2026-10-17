"""Folded geometry: bounded even-real anchors versus estimated radial cells."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import json
from pathlib import Path
import torch
from diagnose import ROOT,common,geometry,setup,conformal,sha


def farthest_first(real,budget):
    count=min(budget,len(real))
    ids=torch.empty(count,dtype=torch.long)
    selected=(real-real.mean(0)).square().sum(1).argmin()
    nearest=real.new_full((len(real),),float('inf'))
    for i in range(count):
        ids[i]=selected
        distance=(real-real.index_select(0,selected.reshape(1))).square().sum(1)
        nearest=torch.minimum(nearest,distance)
        nearest[ids[:i+1]]=-1.
        selected=nearest.argmax()
    return ids


def nearest_score(query,anchors,chunk=256):
    return torch.cat([(block.square().sum(1)[:,None]+anchors.square().sum(1)[None]
                      -2*block@anchors.T).clamp_min(0.).min(1).values.sqrt()
                      for block in query.split(chunk)])


@torch.no_grad()
def main():
    shared=setup()
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    result=dict(scope='CPU bounded-anchor geometry causal diagnostic only',cases=[],
                source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'diagnose.py')})
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw)
        q=bd._features(trainer,ev.G(ev.z))
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        fullR=(R.double()-snap.mean)/snap.scale
        fullq=(q.double()-snap.mean)/snap.scale
        ref=fullR[0::2]
        row=dict(fixture=case,methods={})
        # A fixed bound from existing K64 and four real anchors, no score sweep.
        anchors=farthest_first(ref,snap.cells*4)
        cellids,_=snap.assign(R[0::2])
        local=[]
        for cell in range(snap.cells):
            rows=(cellids==cell).nonzero().flatten()
            if len(rows):
                local.extend(rows[farthest_first(ref[rows],4)].tolist())
        local=torch.tensor(local,dtype=torch.long)
        representative=(snap.real_representative_rows//2).unique()
        for name,anchorids in (('representative64',representative),('global256',anchors),('percell4',local)):
            qs=nearest_score(fullq,ref[anchorids])
            rs=nearest_score(fullR[1::2],ref[anchorids])
            flags,p=conformal(snap,qs,rs)
            fp=flags&(ev.initial_modes!=ev.unsupported_bin)
            row['methods'][name]=dict(detector=ev.detector(flags),anchors=len(anchorids),
                 false_positive_ids=fp.nonzero().flatten().tolist(),
                 rare_false_positive_ids=(fp&(ev.initial_modes==ev.rare_bin)).nonzero().flatten().tolist(),
                 p1745=float(p[1745]) if len(p)>1745 else None,
                 p1924=float(p[1924]) if len(p)>1924 else None)
        result['cases'].append(row)
        print(json.dumps(dict(event='case',fixture=case,
             metrics={k:(m['detector']['recall'],m['detector']['fp'],m['detector']['rare_false_positive'])
                      for k,m in row['methods'].items()})),flush=True)
    assert not torch.cuda.is_initialized()
    result['cuda_initialized']=False
    (ROOT/'diagnosis-anchors.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
