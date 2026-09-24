"""Saved-state neural landing filter for one fixed sample-derived group bank.

Eight fixed-budget MM rounds, each with the same bounded joint G/prior solver.
This is a numerical prerequisite, not host training or stochastic stability.
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
from reports.toy100.pr84_early_geometry import _model
from reports.toy100.sample_group_anchor import mst_groups, output_mm_step, field
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_critic_refinement_capture import _sha


def run(saved, label):
    state_before = _sha(saved)
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        generator = _model(saved, 'g')
    prior = torch.nn.Parameter(saved['prior']['z'].detach().clone())
    stream = torch.Generator().set_state(saved['rng']['data'])
    real = mode_hold.sample_ring(mode_hold.ring_means(), 128, mode_hold.SIGMA, stream)
    centers, grouping = mst_groups(real)
    if not len(centers) < len(prior):
        raise RuntimeError('distinct-anchor surplus premise failed')
    initial = generator(prior).detach()
    rows = []
    for round_index in range(1, 9):
        points = generator(prior).detach()
        mm = output_mm_step(points.double(), centers)
        target = torch.tensor(mm['target'], dtype=points.dtype)
        fit = fit_output_targets(generator, prior, target)
        after = generator(prior).detach()
        actual_objective = field(after, centers)['total']
        index, noise = fixed_draw(240+round_index, after)
        grade = score_support(after, index, noise, mode_hold.ring_means())
        rows.append(dict(round=round_index, mm=mm, fit=fit,
                         actual_objective=actual_objective, grade=grade, points=after.tolist()))
    if _sha(saved) != state_before or not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError('read-only saved state or global randomness changed')
    passing = [row['grade']['modes'] == 8 and row['grade']['hq'] >= .9 for row in rows]
    return dict(label=label, saved_sha256=state_before, initial=initial.tolist(),
        real=real.tolist(), empirical_centers=centers.tolist(), grouping=grouping, records=rows,
        all_landed=all(row['fit']['status'] == 'CONVERGED' for row in rows),
        all_quality=all(passing), final_quality=passing[-1], final=rows[-1]['grade'],
        first_quality_pass=next((i+1 for i,v in enumerate(passing) if v),None),
        final_objective=rows[-1]['actual_objective'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    archive=ROOT/'reports/toy100/continuous-evidence/pr84-finite-cold-prefix100'
    cold_raw=gzip.decompress((archive/'prefix-states.pt.gz').read_bytes())
    expected=json.loads((archive/'result.json').read_text())['state_file_sha256']
    assert hashlib.sha256(cold_raw).hexdigest()==expected
    cold=torch.load(io.BytesIO(cold_raw),weights_only=True)['selected']
    warm_raw=(args.capture/'selected-states.pt').read_bytes()
    warm_sha=hashlib.sha256(warm_raw).hexdigest()
    assert warm_sha=='37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47'
    warm=torch.load(io.BytesIO(warm_raw),weights_only=True)
    names=('reports/toy100/anchor_joint_landing_filter.py',
        'reports/toy100/joint_output_pullback.py','reports/toy100/sample_group_anchor.py',
        'reports/toy100/pr84_early_geometry.py','reports/toy100/coverage_fixed_eval.py',
        'reports/toy100/pr84_critic_refinement_capture.py','benchmarks/locked_shared/mode_hold.py',
        'benchmarks/locked_shared/mlp.py','tests/test_joint_output_pullback.py')
    hashes={}
    for name in names:
        raw=(ROOT/name).read_bytes()
        target=args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    declaration=dict(scope='fixed-bank saved-state joint neural landing, not a host candidate',
        sources=hashes,cold_sha256=expected,warm_sha256=warm_sha,
        cases=['cold1 pre_step','cold100 post_bounded_g','warm1324 pre_step'],
        samples=128,groups='largest additive MST gap; no configured group count',
        objective='unit-mean injective anchor squared error plus nearest-group squared error',
        rounds=8,solver=dict(max_iterations=20,max_halves=12,svd_rtol=1e-6,
                           relative_landing_tolerance=1e-5,minimum_reduction_ratio=.1),
        gate='all fixed targets converge; cold final and all warm rounds8modes/HQ>=.9',
        evaluation='4096 late-noise .029 draws at declared diagnostic clock241..248',
        updates='G and prior only; copied saved models, no optimizer or training continuation',
        warning='fixed correctly inferred centers and free-output proof do not establish stochastic GAN stability')
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    rows=[]
    for label,saved in [('cold1',cold[1]['pre_step']),('cold100',cold[100]['post_bounded_g']),
                        ('warm1324',warm[1324]['pre_step'])]:
        row=run(saved,label)
        rows.append(row)
        (args.output/f'{label}.json').write_text(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(event='CASE_DONE',**{key:value for key,value in row.items()
            if key not in ('initial','real','empirical_centers','grouping','records')})),flush=True)
    okay=all(row['all_landed'] and row['final_quality'] for row in rows) and rows[-1]['all_quality']
    (args.output/'summary.json').write_text(json.dumps(dict(status='PASS' if okay else 'FAIL',
        declaration=declaration,results=rows),allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE',status='PASS' if okay else 'FAIL')),flush=True)


if __name__=='__main__':
    main()
