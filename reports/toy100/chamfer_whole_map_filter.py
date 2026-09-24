"""Free-output filter: replace the whole G move by the existing C+Q target.

This separates data-target geometry from the failed post-GAN latent pullback.
No GAN, neural optimizer, oracle means, seed sweep or hyperparameter search is
inside the update. Means enter only the host's data sampler and read-only grade.
Passing this necessary geometry check would not validate a neural controller.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared import mode_hold
from reports.toy100.chamfer_pullback import chamfer_targets, chamfer_terms
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_early_geometry import _model


def run(state, *, label):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        points = _model(state, 'g')(state['prior']['z']).detach()
    stream = torch.Generator().set_state(state['rng']['data'])
    initial = points.tolist()
    rows = []
    for step in range(1, 101):
        real = mode_hold.sample_ring(mode_hold.ring_means(), 128, mode_hold.SIGMA, stream)
        before = sum(chamfer_terms(real, points))
        target, counts, _, _ = chamfer_targets(real, points)
        points = target.to(points.dtype)
        after = sum(chamfer_terms(real, points))
        if float(after) > float(before) + 1e-6:
            raise RuntimeError('exact fixed-assignment C+Q target increased actual C+Q')
        indices, noise = fixed_draw(240 + step, points)
        grade = score_support(points, indices, noise, mode_hold.ring_means())
        rows.append(dict(step=step, grade=grade, before=float(before), after=float(after),
                         counts=counts.tolist(), points=points.tolist()))
    passing = [row['grade']['modes'] == 8 and row['grade']['hq'] >= .9 for row in rows]
    return dict(label=label, initial=initial, records=rows, passing_checks=sum(passing),
                all100=all(passing), terminal5=all(passing[-5:]),
                first_pass=next((i+1 for i,v in enumerate(passing) if v), None),
                final=rows[-1]['grade'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cold_dir = ROOT/'reports/toy100/continuous-evidence/pr84-finite-cold-prefix100'
    raw = gzip.decompress((cold_dir/'prefix-states.pt.gz').read_bytes())
    receipt = json.loads((cold_dir/'result.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == receipt['state_file_sha256']
    cold = torch.load(io.BytesIO(raw), weights_only=True)['selected']
    warm_raw = (args.capture/'selected-states.pt').read_bytes()
    warm_hash = hashlib.sha256(warm_raw).hexdigest()
    assert warm_hash == '37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47'
    warm = torch.load(io.BytesIO(warm_raw), weights_only=True)
    sources = ('reports/toy100/chamfer_whole_map_filter.py',
        'reports/toy100/chamfer_pullback.py', 'reports/toy100/pr84_early_geometry.py',
        'reports/toy100/coverage_fixed_eval.py', 'benchmarks/locked_shared/mode_hold.py',
        'benchmarks/locked_shared/mlp.py')
    hashes = {}
    for name in sources:
        value = (ROOT/name).read_bytes()
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        hashes[name] = hashlib.sha256(value).hexdigest()
    declaration = dict(scope='free-output geometry rejection filter, not neural training',
        source=hashes, cold_states_sha256=receipt['state_file_sha256'],
        warm_states_sha256=warm_hash, updates=100, batch=128,
        update='existing unit-mean C+Q exact fixed-assignment target; replaces entire output move',
        cases=['cold1 pre_step', 'cold100 post_bounded_g', 'warm1324 pre_step'],
        streams='each saved data stream copied independently; only real batch sampled per toy update',
        evaluation='4096 fixed draws with late .029 output noise, evaluation clock 241..340',
        gate='warm all100 and both cold terminal5:8 modes/HQ>=.9',
        caveat='100-update failure rejects this short filter, not global asymptotic acquisition')
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    cases = [('cold1',cold[1]['pre_step']), ('cold100',cold[100]['post_bounded_g']),
             ('warm1324',warm[1324]['pre_step'])]
    results = [run(state,label=label) for label,state in cases]
    okay = results[-1]['all100'] and all(row['terminal5'] for row in results[:-1])
    result = dict(status='PASS' if okay else 'FAIL', declaration=declaration, results=results)
    (args.output/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], cases=[{key:value for key,value in row.items()
        if key not in ('records','initial')} for row in results])),flush=True)


if __name__ == '__main__':
    main()
