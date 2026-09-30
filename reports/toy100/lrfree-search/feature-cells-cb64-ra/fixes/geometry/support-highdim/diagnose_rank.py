"""Compare bounded dictionary rank handling on unchanged fixed fixtures."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import json
from pathlib import Path
import time
import torch
from diagnose_highdim import ROOT, SUPPORT, common, cost, PREV, OLD, setup, conformal, sha, fisher_score
from fisher_rank import fit_score


@torch.no_grad()
def main():
    shared = setup()
    files = [Path(__file__), ROOT/'fisher_rank.py', ROOT/'diagnose_highdim.py',
             SUPPORT/'diagnose.py', PREV/'geometry_a'/'bundle.pt', PREV/'geometry_a'/'toy_family.py',
             PREV/'scaling_a'/'shared_toy.py', OLD/'geometry'/'run_validation.py',
             OLD/'pkg-CB64-RA'/'particlegan'/'feature_cells.py', common.CONFIG]
    hashes = {str(p): sha(p) for p in files}
    result = dict(scope='CPU bounded dictionary rank diagnosis, unchanged cells and BH',
                  source_sha256=hashes, cases=[])
    cases = [('cost', n, 'highdim', 128, 'frozen_toy_head') for n in cost.SIZES]
    cases += [('geometry', n, 'fold', 128, fmap) for fmap in ('trained600', 'frozen_initialization')
              for n in (1024, 2048)]
    for case in cases:
        begin = time.perf_counter()
        ev = common.Evaluator(*case, shared)
        trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
        R = bd._features(trainer, ev.real_raw); q = bd._features(trainer, ev.G(ev.z))
        snap = shared.cb.FeatureCellSnapshot.fit(R, generator=bd.stream, cells=64, rank=8, chunk=256)
        score, meta = fit_score(snap, R[0::2])
        qs, rs = score(q), score(R[1::2])
        flags, p = conformal(snap, qs, rs)
        fp = flags & (ev.initial_modes != ev.unsupported_bin)
        row = dict(fixture=case, detector=ev.detector(flags), metadata=meta,
                   false_positive_ids=fp.nonzero().flatten().tolist(),
                   rare_false_positive_ids=(fp & (ev.initial_modes == ev.rare_bin)).nonzero().flatten().tolist(),
                   seconds=time.perf_counter()-begin)
        result['cases'].append(row)
        assert ev.state_hash() == ev.initial_state_hash
        print(json.dumps(dict(event='case', fixture=case,
                              detector={k:v for k,v in row['detector'].items() if k != 'flag_ids'},
                              dictionary_rank=meta['dictionary_rank'], seconds=row['seconds'])), flush=True)
    result.update(cuda_initialized=torch.cuda.is_initialized(),
                  sources_unchanged=hashes == {str(p): sha(p) for p in files})
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (ROOT/'diagnosis-rank.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
