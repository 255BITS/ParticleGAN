"""Causal precision check: do not whiten subprecision captured float32 noise."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import inspect
import json
from pathlib import Path
import torch
from diagnose import ROOT,common,geometry,setup,conformal,sha
HIGH = ROOT.parent/'geometry'/'support-highdim'
sys.path.insert(0,str(HIGH))
import diagnose_highdim

# Keep the independent implementation byte-for-byte except for the numerical
# floor's input precision. The edited text and its original file hash are saved.
original=inspect.getsource(diagnose_highdim.fisher_score)
old='floor=torch.finfo(torch.float64).eps*within.shape[0]*eig.max().clamp_min(1e-30)'
new='floor=torch.finfo(source_dtype).eps*within.shape[0]*eig.max().clamp_min(1e-30)'
assert original.count(old)==1
modified=original.replace('def fisher_score(snapshot, ref, *, bounded):',
                          'def fisher_score(snapshot, ref, *, bounded, source_dtype):').replace(old,new)
scope=dict(torch=torch,standardized=diagnose_highdim.standardized)
exec(modified,scope)
precision_score=scope['fisher_score']


@torch.no_grad()
def main():
    shared=setup()
    files=(Path(__file__),ROOT/'diagnose.py',HIGH/'diagnose_highdim.py')
    hashes={str(p):sha(p) for p in files}
    result=dict(scope='CPU captured precision floor check',source_sha256=hashes,
                original_floor=old,tested_floor=new,cases=[])
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        source_dtype=next(ev.D.parameters()).dtype
        score,meta=precision_score(snap,R[0::2],bounded=True,source_dtype=source_dtype)
        qs,rs=score(q),score(R[1::2])
        flags,p=conformal(snap,qs,rs)
        fp=flags&(ev.initial_modes!=ev.unsupported_bin)
        row=dict(fixture=case,detector=ev.detector(flags),metadata=meta,
                 converted_feature_dtype=str(R.dtype),source_feature_dtype=str(source_dtype),
                 false_positive_ids=fp.nonzero().flatten().tolist(),
                 rare_false_positive_ids=(fp&(ev.initial_modes==ev.rare_bin)).nonzero().flatten().tolist())
        result['cases'].append(row)
        print(json.dumps(dict(event='case',fixture=case,
            metrics={k:v for k,v in row['detector'].items() if k!='flag_ids'})),flush=True)
    assert hashes=={str(p):sha(p) for p in files}
    assert not torch.cuda.is_initialized()
    result['cuda_initialized']=False
    (ROOT/'diagnosis-fisher-source-precision.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
