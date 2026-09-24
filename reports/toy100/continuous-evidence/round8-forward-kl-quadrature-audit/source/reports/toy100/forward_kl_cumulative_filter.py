"""Predeclared pure-output continuation of the frozen forward-KL one-bank filter.

Each native D real128 bank joins a cumulative empirical target. The first
cold-bank bandwidth never changes. No model, optimizer, or target-mode label
enters an update. Warm is screened first, then cold only if warm passes.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.models import linear_output_noise
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.forward_kl_free_filter import optimize
from reports.toy100.sample_anchor_free1200 import initial_support, load_states
from reports.toy100.sample_anchor_local_mmd_continuation import native_bank
from reports.toy100.sample_anchor_local_mmd_filter import local_width
from reports.toy100.sample_anchor_mmd_filter import quality


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = (
    'reports/toy100/forward_kl_cumulative_filter.py',
    'reports/toy100/forward_kl_free_filter.py',
    'reports/toy100/sample_anchor_local_mmd_continuation.py',
    'reports/toy100/sample_anchor_local_mmd_filter.py',
    'reports/toy100/sample_anchor_mmd_filter.py',
    'reports/toy100/sample_anchor_free1200.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_early_geometry.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'benchmarks/toy100/models.py',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def observed_grade(points, absolute_step, means):
    index, noise = fixed_draw(240+absolute_step, points.float())
    score = score_support(points.float(), index, noise, means)
    return dict(modes=score['modes'], hq=score['hq'])


def run_case(state, case, width, means, first_record, output):
    start = 1324 if case == 'warm1324' else 1
    points = initial_support(state).double().detach()
    stream = torch.Generator().set_state(state['rng']['data'])
    history = torch.empty((0, 2), dtype=torch.float64)
    rows = []
    for offset in range(16):
        absolute = start+offset
        real = native_bank(stream, means)
        history = torch.cat((history, real.double()), 0)
        sigma = linear_output_noise(.029, absolute-1, 1200, .2)
        selected, receipt = optimize(history, points, width, sigma, means=means)
        if offset == 0:
            reference = first_record['cases'][case]
            if (abs(receipt['final_cross_entropy']-reference['final_cross_entropy']) > 1e-12
                    or not torch.equal(selected, torch.tensor(reference['final_points'],
                                                               dtype=torch.float64))
                    or sha(real.contiguous().numpy().tobytes()) !=
                    first_record['declaration']['native_real_bank_sha256'][case]):
                raise RuntimeError('first cumulative update differs from frozen one-bank filter')
        points = selected
        grade = observed_grade(points, absolute, means)
        row = dict(absolute_step=absolute, output_sigma=sigma,
                   observed_real_points=len(history),
                   real_bank_sha256=sha(real.contiguous().numpy().tobytes()),
                   training_data_rng_sha256=sha(stream.get_state().numpy().tobytes()),
                   grade=grade, receipt=receipt)
        rows.append(row)
        (output/f'{case}.partial.json').write_text(json.dumps(dict(
            status='INCOMPLETE', case=case, rows=rows), allow_nan=False)+'\n')
        print(json.dumps(dict(event='FORWARD_KL_CUMULATIVE_STEP',case=case,
            update=offset+1, absolute_step=absolute, grade=grade,
            donors=len(receipt['accepted_donors']), em=len(receipt['accepted_em']),
            audit9_sign_flips=receipt['audit9_sign_flips'])),flush=True)
        if case == 'warm1324' and (grade['modes'] != 8 or grade['hq'] < .9):
            break
    return dict(case=case, attempted=len(rows), rows=rows,
        initial_quality=quality(initial_support(state), means),
        final_quality=quality(points, means), final_points=points.tolist(),
        final_data_rng_sha256=sha(stream.get_state().numpy().tobytes()),
        warm_all16=(case == 'warm1324' and len(rows)==16 and all(
            r['grade']['modes']==8 and r['grade']['hq']>=.9 for r in rows)),
        cold_terminal5=(case == 'cold1' and len(rows)==16 and all(
            r['grade']['modes']==8 and r['grade']['hq']>=.9 for r in rows[-5:])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--first-bank', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    first_raw = args.first_bank.read_bytes()
    first = json.loads(first_raw)
    if not first['all_cheap_gates_pass']:
        raise RuntimeError('predeclared one-bank screens did not pass')
    cold, warm, inputs = load_states()
    means = mode_hold.ring_means()
    first_stream = torch.Generator().set_state(cold['rng']['data'])
    first_real = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, first_stream)
    width, _ = local_width(first_real)
    if width != first['declaration']['frozen_width']:
        raise RuntimeError('frozen first-bank width differs')
    paths = {n: (ROOT/n if (ROOT/n).exists() else args.code_root/n) for n in SOURCE_NAMES}
    sources = {n: sha(p.read_bytes()) for n,p in paths.items()}
    if sources['reports/toy100/forward_kl_free_filter.py'] != first['declaration']['source'][
            'reports/toy100/forward_kl_free_filter.py']:
        raise RuntimeError('first-bank filter source changed after result')
    args.output.mkdir(parents=True)
    for name,path in paths.items():
        target = args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    declaration = dict(status='FROZEN_BEFORE_RUN', scope='pure-output cumulative16 only',
        source=sources, first_bank_sha256=sha(first_raw), inputs=inputs,
        frozen_width=width, actual_output_sigma='native linear output-noise ramp over original1200 clock',
        real_stream='native D128, two prior index draws, native G128 per absolute update',
        target='all D real banks seen so far convolved with fixed N(0,h²I)',
        update='global donor among all observed real points, <=12 moves; <=20 EM steps',
        quadrature='5x5 fixed positive GH selects; 9x9 audits all accepted states',
        evaluation='fixed4096 late-noise diagnostic draws clock240+absolute; labels grade only',
        order='warm16 first; stop on first warm quality failure; cold16 only after warm pass',
        gates='warm all16 8/HQ>=.9, cold last5 8/HQ>=.9; no 9x9 sign flips',
        no_neural_update=True, no_seed_or_bandwidth_sweep=True,
        shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='FORWARD_KL_CUMULATIVE_DECLARED',h=width,
        source_sha256=sources['reports/toy100/forward_kl_cumulative_filter.py'])),flush=True)
    before_rng = torch.random.get_rng_state().clone()
    results = {}
    try:
        results['warm1324'] = run_case(warm, 'warm1324', width, means, first, args.output)
        if results['warm1324']['warm_all16']:
            results['cold1'] = run_case(cold, 'cold1', width, means, first, args.output)
        all_no_flips=all(row['receipt']['audit9_sign_flips']==0 for result in results.values()
                         for row in result['rows'])
        result = dict(status='COMPLETE', declaration=declaration, cases=results,
            global_torch_rng_unchanged=torch.equal(torch.random.get_rng_state(),before_rng),
            all_audit9_signs_consistent=all_no_flips,
            all_gates_pass=results['warm1324']['warm_all16'] and
                results.get('cold1',{}).get('cold_terminal5',False) and all_no_flips)
        if not result['global_torch_rng_unchanged']:
            raise RuntimeError('pure cumulative filter changed global training RNG')
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(event='FORWARD_KL_CUMULATIVE_DONE',
            attempted={name:value['attempted'] for name,value in results.items()},
            all_gates_pass=result['all_gates_pass'])),flush=True)
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE',
            error=repr(error), attempted={name:len(value['rows']) for name,value in results.items()}))+'\n')
        raise


if __name__ == '__main__':
    main()
