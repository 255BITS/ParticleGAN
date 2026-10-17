"""Geometry cross-check of independently diagnosed bounded Fisher directions."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import json
from pathlib import Path
import torch
from diagnose import ROOT,common,geometry,setup,conformal,sha
HIGH = ROOT.parent/'geometry'/'support-highdim'
sys.path.insert(0,str(HIGH))
from diagnose_highdim import fisher_score


@torch.no_grad()
def main():
    shared=setup()
    files=(Path(__file__),ROOT/'diagnose.py',HIGH/'diagnose_highdim.py')
    hashes={str(p):sha(p) for p in files}
    result=dict(scope='CPU geometry cross-check of bounded Fisher support',source_sha256=hashes,cases=[])
    cases=[('geometry',n,'fold',128,fmap) for fmap in ('trained600','frozen_initialization')
           for n in (1024,2048,4096)]
    for case in cases:
        ev=common.Evaluator(*case,shared)
        trainer,bd=common.make_trainer(ev,'cb64_ra',shared)
        R=bd._features(trainer,ev.real_raw);q=bd._features(trainer,ev.G(ev.z))
        snap=shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        row=dict(fixture=case,methods={})
        for name,bounded in (('bounded_dictionary_fisher8',True),('full_fisher8',False)):
            score,meta=fisher_score(snap,R[0::2],bounded=bounded)
            qs,rs=score(q),score(R[1::2])
            flags,p=conformal(snap,qs,rs)
            fp=flags&(ev.initial_modes!=ev.unsupported_bin)
            row['methods'][name]=dict(detector=ev.detector(flags),metadata=meta,
                 false_positive_ids=fp.nonzero().flatten().tolist(),
                 rare_false_positive_ids=(fp&(ev.initial_modes==ev.rare_bin)).nonzero().flatten().tolist())
        result['cases'].append(row)
        print(json.dumps(dict(event='case',fixture=case,
             metrics={k:(m['detector']['recall'],m['detector']['fp'],m['detector']['rare_false_positive'])
                      for k,m in row['methods'].items()})),flush=True)
    assert hashes=={str(p):sha(p) for p in files}
    assert not torch.cuda.is_initialized()
    result['cuda_initialized']=False
    (ROOT/'diagnosis-fisher.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
