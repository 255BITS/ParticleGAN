"""CPU test of estimated cell-center uncertainty and diagonal cell shape."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import json
from pathlib import Path
import torch
from diagnose import ROOT, common, cost, geometry, setup, conformal, sha


def scores(snapshot, ref):
    x = snapshot.transform(ref)
    ids, _ = snapshot._assign_metric(x)
    delta = x-snapshot.centers[ids]
    pooled = delta.square().mean(0)
    variance = torch.stack([(delta[ids == c].square().sum(0)+4.*pooled)/(int((ids == c).sum())+4.)
                            for c in range(snapshot.cells)])
    variance = variance.clamp_min(torch.finfo(torch.float64).eps*pooled.max())
    uncertainty = 1.+1./snapshot.reference_counts.clamp_min(1).double()
    def radial(features):
        return torch.cat([(snapshot._distance(block,snapshot.centers)
                          /(snapshot.cell_scale.square()*uncertainty)[None]).min(1).values.sqrt()
                          for block in snapshot.transform(features).split(snapshot.chunk)])
    def diagonal(features):
        return torch.cat([((block[:,None]-snapshot.centers[None]).square()/variance[None]).sum(2).min(1).values.sqrt()
                          for block in snapshot.transform(features).split(snapshot.chunk)])
    def diagonal_uncertainty(features):
        return torch.cat([((block[:,None]-snapshot.centers[None]).square()
                          /(variance*uncertainty[:,None])[None]).sum(2).min(1).values.sqrt()
                          for block in snapshot.transform(features).split(snapshot.chunk)])
    return dict(radial_predictive_variance=radial, projected_diagonal=diagonal,
                projected_diagonal_predictive_variance=diagonal_uncertainty)


@torch.no_grad()
def main():
    shared = setup()
    cases = [('geometry', n, 'fold', 128, fmap) for fmap in ('trained600','frozen_initialization')
             for n in (1024,2048,4096)]
    cases += [('cost',n,m,128 if m=='highdim' else 8,'frozen_toy_head')
              for m in cost.SCENARIOS for n in cost.SIZES]
    result = dict(scope='CPU estimated-center variance diagnostic only', cases=[],
                  source_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'diagnose.py')})
    for case in cases:
        ev = common.Evaluator(*case, shared)
        trainer, bd = common.make_trainer(ev,'cb64_ra',shared)
        R,q = bd._features(trainer,ev.real_raw),bd._features(trainer,ev.G(ev.z))
        snap = shared.cb.FeatureCellSnapshot.fit(R,generator=bd.stream,cells=64,rank=8,chunk=256)
        row = dict(fixture=case, methods={})
        for name, score in scores(snap,R[0::2]).items():
            flags,p = conformal(snap,score(q),score(R[1::2]))
            fp = flags & (ev.initial_modes != ev.unsupported_bin)
            row['methods'][name] = dict(detector=ev.detector(flags),
                 false_positive_ids=fp.nonzero().flatten().tolist(),
                 rare_false_positive_ids=(fp & (ev.initial_modes == ev.rare_bin)).nonzero().flatten().tolist())
        result['cases'].append(row)
        print(json.dumps(dict(event='case',fixture=case,
             metrics={k:(m['detector']['recall'],m['detector']['fp'],m['detector']['rare_false_positive'])
                      for k,m in row['methods'].items()})),flush=True)
    assert not torch.cuda.is_initialized()
    result['cuda_initialized'] = False
    (ROOT/'diagnosis-sparse.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
