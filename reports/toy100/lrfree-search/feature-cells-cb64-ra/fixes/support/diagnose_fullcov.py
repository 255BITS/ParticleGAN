"""CPU causal test of correlated support directions retained by the critic."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import json
from pathlib import Path

import torch
from diagnose import ROOT, OLD, PREV, common, cost, geometry, setup, conformal, summary, sha


@torch.no_grad()
def metrics(snapshot, ref):
    t = (ref.double()-snapshot.mean)/snapshot.scale
    ids, _ = snapshot.assign(ref)
    centers = torch.stack([t[ids == c].mean(0) if bool((ids == c).any()) else t.mean(0)
                           for c in range(snapshot.cells)])
    residual = t-centers[ids]
    pooled = residual.T@residual/len(t)
    # A float64 numerical floor, not a tunable statistical ridge.
    floor = torch.finfo(torch.float64).eps*len(pooled)*pooled.trace().clamp_min(1e-30)
    ridge = torch.eye(len(pooled), dtype=pooled.dtype)*floor
    shared_precision = torch.linalg.inv(pooled+ridge)
    covariance = torch.stack([(residual[ids == c].T@residual[ids == c]+4.*pooled)/(int((ids == c).sum())+4.)
                              for c in range(snapshot.cells)])
    local_precision = torch.linalg.inv(covariance+ridge[None])
    radial2 = torch.stack([torch.einsum('nr,rh,nh->n', residual[ids == c], shared_precision,
                                        residual[ids == c]).sum() for c in range(snapshot.cells)])
    positive = (radial2/snapshot.reference_counts.clamp_min(1)).sqrt()
    globalrad = positive[positive > 0].median()
    scale2 = (radial2+4.*globalrad.square())/(snapshot.reference_counts+4.)
    def shared_score(features):
        query = (features.double()-snapshot.mean)/snapshot.scale
        return torch.cat([torch.einsum('nkr,rh,nkh->nk', block[:, None]-centers[None], shared_precision,
                                      block[:, None]-centers[None]).clamp_min(0.).div(scale2[None]).min(1).values.sqrt()
                          for block in query.split(snapshot.chunk)])
    def local_score(features):
        query = (features.double()-snapshot.mean)/snapshot.scale
        return torch.cat([torch.einsum('nkr,krh,nkh->nk', block[:, None]-centers[None], local_precision,
                                      block[:, None]-centers[None]).clamp_min(0.).min(1).values.sqrt()
                          for block in query.split(snapshot.chunk)])
    rfull = t-t@snapshot.basis@snapshot.basis.T
    rcov = rfull.T@rfull/len(t)
    vals, vecs = torch.linalg.eigh(rcov)
    usable = vals > floor
    whitening = vecs[:, usable]/vals[usable].sqrt()[None]
    def residual_score(features):
        query = (features.double()-snapshot.mean)/snapshot.scale
        remainder = query-query@snapshot.basis@snapshot.basis.T
        return (remainder@whitening).norm(dim=1)
    return {'pooled_full_covariance': shared_score, 'local_full_covariance': local_score,
            'orthogonal_covariance': residual_score}, torch.linalg.eigvalsh(pooled), int(usable.sum())


@torch.no_grad()
def main():
    shared = setup()
    cases = [('geometry', n, 'fold', 128, fmap) for fmap in ('trained600', 'frozen_initialization')
             for n in (1024, 2048, 4096)]
    cases += [('cost', n, 'highdim', 128, 'frozen_toy_head') for n in cost.SIZES]
    result = dict(scope='CPU full-covariance information diagnostic only', cases=[],
                  source_sha256={str(p):sha(p) for p in (Path(__file__), ROOT/'diagnose.py',
                       PREV/'geometry_a'/'bundle.pt', PREV/'scaling_a'/'shared_toy.py')})
    for case in cases:
        ev = common.Evaluator(*case, shared)
        trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
        R = bd._features(trainer, ev.real_raw)
        q = bd._features(trainer, ev.G(ev.z))
        snapshot = shared.cb.FeatureCellSnapshot.fit(R, generator=bd.stream, cells=64, rank=8, chunk=256)
        scores, eig, residual_rank = metrics(snapshot, R[0::2])
        row = dict(fixture=case, within_cell_covariance_eigenvalues=eig.tolist(),
                   residual_rank=residual_rank, methods={})
        for name, score in scores.items():
            qs, rs = score(q), score(R[1::2])
            flags, p = conformal(snapshot, qs, rs)
            row['methods'][name] = dict(detector=ev.detector(flags),
                 bad_p=summary(p, ev.initial_modes == ev.unsupported_bin),
                 legitimate_p=summary(p, ev.initial_modes != ev.unsupported_bin),
                 row1745_p=float(p[1745]) if len(p)>1745 else None)
        result['cases'].append(row)
        print(json.dumps(dict(event='case', fixture=case,
            detectors={k:{f:v for f,v in m['detector'].items() if f!='flag_ids'} for k,m in row['methods'].items()})), flush=True)
    result['cuda_initialized'] = torch.cuda.is_initialized()
    assert not result['cuda_initialized']
    (ROOT/'diagnosis-fullcov.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(event='complete', cases=len(cases), cuda_initialized=False)), flush=True)


if __name__ == '__main__':
    main()
