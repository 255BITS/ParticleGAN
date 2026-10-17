"""CPU properties of the proposed real-only metric; no quality seed runs."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import inspect
import json
import math
from pathlib import Path
import time
import torch
from diagnose_highdim import ROOT, OLD, setup, conformal, sha
from fisher_rank import fit_score


def structured(n, width):
    t = torch.linspace(-1., 1., n, dtype=torch.float64)[:, None]
    a = torch.arange(1, width+1, dtype=torch.float64)[None]
    return (t*a.sqrt()).sin()+.2*(t*a.remainder(5)).cos()+.01*t.square()*a.remainder(3)


@torch.no_grad()
def main():
    shared = setup()
    files = [Path(__file__), ROOT/'fisher_rank.py', ROOT/'diagnose_highdim.py',
             OLD/'pkg-CB64-RA'/'particlegan'/'feature_cells.py']
    hashes = {str(p):sha(p) for p in files}
    result = dict(scope='CPU metric contracts, deterministic arrays; no quality seed runs',
                  source_sha256=hashes, tests=[])

    def snapshot(features):
        stream = torch.Generator().manual_seed(90229)
        return shared.cb.FeatureCellSnapshot.fit(features, generator=stream, cells=64, rank=8, chunk=256), stream

    def record(name, operation):
        begin = time.perf_counter()
        try:
            evidence = operation()
            row = dict(name=name, status='PASS', evidence=evidence)
        except Exception as error:
            row = dict(name=name, status='FAIL', error=f'{type(error).__name__}: {error}')
        row['seconds'] = time.perf_counter()-begin
        result['tests'].append(row)
        print(json.dumps(row), flush=True)

    def degenerate(features):
        snap, stream = snapshot(features)
        before = stream.get_state().clone()
        score, meta = fit_score(snap, features[0::2])
        assert all(math.isfinite(v) for key in ('dictionary_singular_values', 'within_eigenvalues',
                                               'generalized_eigenvalues') for v in meta[key])
        assert math.isfinite(meta['within_floor'])
        values = score(features)
        flags, p = conformal(snap, values, values[1::2])
        assert bool(torch.isfinite(values).all()) and bool(torch.isfinite(p).all())
        assert torch.equal(before, stream.get_state())
        if not snap.valid_metric or snap.duplicate_fraction > .05:
            assert not bool(flags.any())
        return dict(rows=len(features), width=features.shape[1], rank=meta['rank'],
                    dictionary_rank=meta['dictionary_rank'], maximum_score=float(values.max()),
                    flagged=int(flags.sum()), duplicate_fraction=snap.duplicate_fraction)

    record('constant captured head', lambda:degenerate(torch.ones(64, 16, dtype=torch.float64)))
    record('three even rows each in its own cell', lambda:degenerate(structured(6, 3)))
    t = torch.linspace(-1., 1., 1024, dtype=torch.float64)[:, None]
    record('collinear 128-feature head', lambda:degenerate(t*torch.arange(1, 129, dtype=torch.float64)[None]))
    record('exact duplicate rows preserve original guard', lambda:degenerate(structured(64, 16).repeat(4, 1)))
    record('one captured feature', lambda:degenerate(structured(1024, 1)))

    def isolation():
        real = structured(1024, 16)
        changed = real.clone()
        changed[1::2] += 100*torch.arange(1, 17, dtype=torch.float64)[None]
        a, stream_a = snapshot(real); b, stream_b = snapshot(changed)
        for key in ('mean', 'scale', 'basis', 'centers', 'reference_counts', 'real_representative_rows'):
            assert torch.equal(getattr(a, key), getattr(b, key)), key
        fields_before = {key:value.clone() for key,value in vars(a).items() if isinstance(value, torch.Tensor)}
        rng_before = torch.get_rng_state().clone(); private_before = stream_a.get_state().clone()
        fa, ma = fit_score(a, real[0::2]); fb, mb = fit_score(b, changed[0::2])
        query = structured(333, 16)*1.03
        sa, sb = fa(query), fb(query)
        assert torch.equal(sa, sb)
        assert ma == mb
        assert torch.equal(rng_before, torch.get_rng_state()) and torch.equal(private_before, stream_a.get_state())
        for key, value in fields_before.items():
            assert torch.equal(value, getattr(a, key)), key
        assert not torch.equal(fa(real[1::2]), fb(changed[1::2]))
        permutation = torch.arange(len(query)-1, -1, -1)
        torch.testing.assert_close(fa(query[permutation]), sa[permutation], rtol=1e-10, atol=1e-10)
        chunked = torch.cat([fa(block) for block in query.split(17)])
        torch.testing.assert_close(chunked, sa, rtol=1e-10, atol=1e-10)
        null = fa(real[1::2])
        flags, p = conformal(a, sa, null)
        manual = (1.+len(null)-torch.searchsorted(null.sort().values, sa))/(1.+len(null))
        assert torch.equal(p, manual)
        return dict(even_fit_bit_exact=True, odd_changes_only_calibration=True,
                    original_partition_unchanged=True, rng_unchanged=True,
                    row_and_chunk_invariance=True, pvalue_formula_exact=True,
                    dictionary_rank=ma['dictionary_rank'], rank=ma['rank'])
    record('held-out isolation and unchanged partition/RNG', isolation)

    def dropped():
        real = torch.cat((t, torch.ones_like(t), 1+1e-12*t.sin()), 1)
        snap, _ = snapshot(real)
        score, meta = fit_score(snap, real[0::2])
        query = real[::3].clone()
        changed = query.clone(); changed[:, 1:] += 100.
        assert torch.equal(score(query), score(changed))
        rejected = 0
        for invalid in (query[:, :1], torch.full_like(query, float('nan')), query.long()):
            try:
                score(invalid)
            except ValueError:
                rejected += 1
        assert rejected == 3
        return dict(dropped_dimensions=int((~torch.isfinite(snap.scale)).sum()), invalid_inputs_rejected=rejected)
    record('constant-feature policy and invalid-input rejection', dropped)

    scaling = []
    def scale_case(n, width):
        real = structured(n, width)
        snap, _ = snapshot(real)
        begin = time.perf_counter(); score, meta = fit_score(snap, real[0::2])
        fit_seconds = time.perf_counter()-begin
        begin = time.perf_counter(); values = score(real[1::2])
        query_seconds = time.perf_counter()-begin
        retained = inspect.getclosurevars(score).nonlocals
        arrays = {key:dict(shape=list(value.shape), bytes=value.untyped_storage().nbytes())
                  for key,value in retained.items() if isinstance(value, torch.Tensor)}
        assert meta['dictionary_rank'] <= 64 and meta['rank'] <= 8
        assert meta['covariance_bytes'] <= 64*64*8
        assert all(len(v['shape']) <= 2 for v in arrays.values())
        assert all(v['shape'][0] <= max(width, 64) for v in arrays.values())
        assert bool(torch.isfinite(values).all())
        row = dict(n=n, width=width, fit_seconds=fit_seconds, query_seconds=query_seconds,
                   arrays=arrays, covariance_bytes=meta['covariance_bytes'], rank=meta['rank'],
                   dictionary_rank=meta['dictionary_rank'])
        scaling.append(row)
        return row
    for n, width in ((1024, 128), (8192, 128), (1024, 1024)):
        record(f'bounded storage N{n}/H{width}', lambda n=n,width=width:scale_case(n,width))
    result.update(scaling=scaling, cuda_initialized=torch.cuda.is_initialized(),
                  sources_unchanged=hashes == {str(p):sha(p) for p in files},
                  status='PASS' if all(row['status'] == 'PASS' for row in result['tests']) else 'FAIL')
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (ROOT/'metric-contract.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
