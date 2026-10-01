"""One fixed reference-radius normalization test in bounded Fisher geometry.

Uses the unchanged 64 actual-real cell representatives.  Radius is the
reference backend's k-th even-real neighbor distance, excluding the anchor
itself.  No labels, odd rows or query rows choose geometry or radii.
"""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import inspect
import json
from pathlib import Path
import time
import torch
from diagnose_highdim import ROOT, SUPPORT, common, cost, PREV, OLD, setup, conformal, sha, summary
from fisher_rank import fit_score


@torch.no_grad()
def normalized_anchor_metric(snapshot, reference_features, fisher_score, k):
    # Diagnostic access to the already fitted transform, without re-fitting or
    # creating an alternative partition.  A package API would expose this as
    # an explicit fitted support projection.
    transform = inspect.getclosurevars(fisher_score).nonlocals['transform']
    ref = ((reference_features.double()-snapshot.mean)/snapshot.scale)@transform
    rows = (snapshot.real_representative_rows//2).unique(sorted=True)
    anchors = ref[rows]
    k = min(k, len(ref)-1)
    nearest = ref.new_full((len(anchors), k), float('inf'))
    for start in range(0, len(ref), snapshot.chunk):
        block = ref[start:start+snapshot.chunk]
        distance = (anchors[:, None]-block[None]).square().sum(2)
        self_rows = rows-start
        valid = (self_rows >= 0) & (self_rows < len(block))
        distance[valid, self_rows[valid]] = float('inf')
        nearest = torch.cat((nearest, distance), 1).topk(k, dim=1, largest=False).values
    radius = nearest[:, -1].sqrt()
    positive = radius[radius > 0]
    floor = positive.median()*1e-3 if len(positive) else ref.new_tensor(1.)
    radius = radius.clamp_min(floor)

    def raw(features):
        query = ((features.double()-snapshot.mean)/snapshot.scale)@transform
        return torch.cat([((b[:, None]-anchors[None]).square().sum(2)).min(1).values.sqrt()
                          for b in query.split(snapshot.chunk)])

    def normalized(features):
        query = ((features.double()-snapshot.mean)/snapshot.scale)@transform
        return torch.cat([((b[:, None]-anchors[None]).square().sum(2)/radius.square()[None]).min(1).values.sqrt()
                          for b in query.split(snapshot.chunk)])

    return raw, normalized, dict(anchors=len(rows), k=k, rows=rows.tolist(), radius=radius.tolist(),
                                 floor=float(floor), max_fit_rows=snapshot.chunk,
                                 max_fit_columns=len(rows), rank=transform.shape[1])


def evaluate(ev, snapshot, score, query, real):
    qs, rs = score(query), score(real[1::2])
    flags, p = conformal(snapshot, qs, rs)
    bad = ev.initial_modes == ev.unsupported_bin
    rare = ev.initial_modes == ev.rare_bin
    fp = flags & ~bad
    largest_rare = qs.masked_fill(~rare, -float('inf')).argsort(descending=True)[:5]
    return dict(detector=ev.detector(flags), bad_scores=summary(qs, bad),
                rare_scores=summary(qs, rare), calibration_scores=summary(rs, torch.ones(len(rs), dtype=torch.bool)),
                bad_p=summary(p, bad), rare_p=summary(p, rare),
                bad_score_below_null_max=int((qs[bad] <= rs.max()).sum()),
                false_positive_ids=fp.nonzero().flatten().tolist(),
                rare_false_positive_ids=(fp & rare).nonzero().flatten().tolist(),
                largest_rare=[dict(row=int(i), score=float(qs[i]), p=float(p[i])) for i in largest_rare])


@torch.no_grad()
def main():
    shared = setup()
    files = [Path(__file__), ROOT/'fisher_rank.py', ROOT/'diagnose_highdim.py',
             SUPPORT/'diagnose.py', PREV/'geometry_a'/'bundle.pt', PREV/'geometry_a'/'toy_family.py',
             PREV/'scaling_a'/'shared_toy.py', OLD/'geometry'/'run_validation.py',
             OLD/'pkg-CB64-RA'/'particlegan'/'feature_cells.py', common.CONFIG]
    hashes = {str(p): sha(p) for p in files}
    result = dict(scope='CPU one local-anchor normalization test, unchanged partition/BH',
                  source_sha256=hashes, cases=[])
    cases = [('cost', n, 'highdim', 128, 'frozen_toy_head') for n in cost.SIZES]
    cases += [('geometry', n, 'fold', 128, fmap) for fmap in ('trained600', 'frozen_initialization')
              for n in (1024, 2048, 4096)]
    for case in cases:
        begin = time.perf_counter()
        ev = common.Evaluator(*case, shared)
        trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
        R = bd._features(trainer, ev.real_raw); q = bd._features(trainer, ev.G(ev.z))
        snap = shared.cb.FeatureCellSnapshot.fit(R, generator=bd.stream, cells=64, rank=8, chunk=256)
        fisher, meta = fit_score(snap, R[0::2])
        raw, normalized, anchors = normalized_anchor_metric(snap, R[0::2], fisher, bd.k)
        row = dict(fixture=case, metadata=meta, anchor_metadata=anchors,
                   methods={name:evaluate(ev, snap, score, q, R) for name,score in
                            (('fisher_centroid', fisher), ('fisher_anchor_raw', raw), ('fisher_anchor_local_radius', normalized))},
                   seconds=time.perf_counter()-begin)
        result['cases'].append(row)
        assert ev.state_hash() == ev.initial_state_hash
        print(json.dumps(dict(event='case', fixture=case, methods={name:{k:v for k,v in m['detector'].items()
                          if k != 'flag_ids'} for name,m in row['methods'].items()})), flush=True)
    result.update(cuda_initialized=torch.cuda.is_initialized(),
                  sources_unchanged=hashes == {str(p): sha(p) for p in files})
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (ROOT/'diagnosis-local-anchors.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
