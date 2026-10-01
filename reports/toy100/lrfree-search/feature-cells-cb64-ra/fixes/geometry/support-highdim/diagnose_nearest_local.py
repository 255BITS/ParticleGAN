"""Nearest-actual-anchor radius law: selection precedes normalization."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import inspect
import json
from pathlib import Path
import torch
from diagnose_local_anchors import (ROOT, SUPPORT, common, cost, PREV, OLD, setup, sha,
                                    fit_score, normalized_anchor_metric, evaluate)


def nearest_local_score(normalized):
    fitted = inspect.getclosurevars(normalized).nonlocals
    anchors, radius, transform, snapshot = [fitted[name] for name in ('anchors', 'radius', 'transform', 'snapshot')]
    def score(features):
        query = ((features.double()-snapshot.mean)/snapshot.scale)@transform
        result = []
        for block in query.split(snapshot.chunk):
            distance, nearest = (block[:, None]-anchors[None]).square().sum(2).min(1)
            result.append(distance.sqrt()/radius[nearest])
        return torch.cat(result)
    return score


@torch.no_grad()
def main():
    shared = setup()
    files = [Path(__file__), ROOT/'diagnose_local_anchors.py', ROOT/'fisher_rank.py',
             ROOT/'diagnose_highdim.py', SUPPORT/'diagnose.py', PREV/'geometry_a'/'bundle.pt',
             PREV/'geometry_a'/'toy_family.py', PREV/'scaling_a'/'shared_toy.py',
             OLD/'geometry'/'run_validation.py', OLD/'pkg-CB64-RA'/'particlegan'/'feature_cells.py', common.CONFIG]
    hashes = {str(p): sha(p) for p in files}
    result = dict(scope='CPU nearest-actual-anchor selection before local-radius normalization',
                  source_sha256=hashes, cases=[])
    cases = [('cost', n, 'highdim', 128, 'frozen_toy_head') for n in cost.SIZES]
    cases += [('geometry', n, 'fold', 128, fmap) for fmap in ('trained600', 'frozen_initialization')
              for n in (1024, 2048, 4096)]
    for case in cases:
        ev = common.Evaluator(*case, shared)
        trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
        R = bd._features(trainer, ev.real_raw); q = bd._features(trainer, ev.G(ev.z))
        snapshot = shared.cb.FeatureCellSnapshot.fit(R, generator=bd.stream, cells=64, rank=8, chunk=256)
        fisher, meta = fit_score(snapshot, R[0::2])
        _, normalized, anchors = normalized_anchor_metric(snapshot, R[0::2], fisher, bd.k)
        row = dict(fixture=case, metadata=meta, anchor_metadata=anchors,
                   method=evaluate(ev, snapshot, nearest_local_score(normalized), q, R))
        result['cases'].append(row)
        assert ev.state_hash() == ev.initial_state_hash
        print(json.dumps(dict(event='case', fixture=case,
                              detector={k:v for k,v in row['method']['detector'].items() if k != 'flag_ids'})), flush=True)
    result.update(cuda_initialized=torch.cuda.is_initialized(),
                  sources_unchanged=hashes == {str(p): sha(p) for p in files})
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (ROOT/'diagnosis-nearest-local.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
