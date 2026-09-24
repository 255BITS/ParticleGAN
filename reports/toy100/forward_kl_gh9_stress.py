"""Pure-output fixed-GH9 forward-KL whole-map screen.

GH5 current-bank donors and EM are a proposal only. The completed proposal
is accepted iff it strictly decreases cumulative finite-GH9 cross-entropy.
Otherwise one fixed-weight GH9 EM step from the original output is accepted
on strict GH9 decrease, or the output rests exactly. Conditional-bank and
output-error cases are diagnostics, not production distribution-shift gates.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.models import linear_output_noise
from reports.toy100.forward_kl_free_filter import (
    quadrature, cross_entropy, em_centroids, gradient,
)
from reports.toy100.forward_kl_chunked import global_donor
from reports.toy100.sample_anchor_free1200 import initial_support, load_states
from reports.toy100.sample_anchor_local_mmd_continuation import native_bank, grade
from reports.toy100.sample_anchor_local_mmd_filter import local_width
from reports.toy100.sample_anchor_mmd_filter import quality


ROOT = Path(__file__).resolve().parents[2]
N = 12
SOURCE_NAMES = (
    'reports/toy100/forward_kl_gh9_stress.py',
    'reports/toy100/forward_kl_free_filter.py',
    'reports/toy100/forward_kl_chunked.py',
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


def optimize(history, bank, initial, width, sigma, *, means):
    """GH5 proposes; the whole finite-GH9 target decides accept/fallback/rest."""
    target = history.detach().double()
    candidates = bank.detach().double()
    points = initial.detach().double().clone()
    if len(points) != N or len(candidates) != 128:
        raise ValueError('native twelve outputs and current real128 bank required')
    variance = width**2+sigma**2
    update = quadrature(target, width, 5)
    audit = quadrature(target, width, 9)
    before = float(cross_entropy(*update, points, variance))
    before9 = float(cross_entropy(*audit, points, variance))
    donated, donors = global_donor(candidates, points, *update, variance,
                                   limit=N, chunk_rows=2048)
    after_donor = float(cross_entropy(*update, donated, variance))
    proposed, em_rows = em_centroids(donated, *update, variance, limit=20,
                                     audit=audit)
    proposed5 = float(cross_entropy(*update, proposed, variance))
    proposed9 = float(cross_entropy(*audit, proposed, variance))
    if proposed5 > before+1e-11:
        raise RuntimeError('GH5 proposal search raised its declared finite objective')
    tolerance = 64*torch.finfo(torch.float64).eps*max(1.,abs(before9))
    fallback_rows=[]
    fallback9=None
    if proposed9 < before9-tolerance:
        final=proposed
        selected='GH5_WHOLE_ACCEPTED_BY_GH9'
    else:
        fallback,fallback_rows=em_centroids(points,*audit,variance,limit=1,
                                            audit=audit)
        fallback9=float(cross_entropy(*audit,fallback,variance))
        if fallback_rows and fallback9 < before9-tolerance:
            final=fallback
            selected='GH9_ONE_EM_FALLBACK'
        else:
            final=points.clone()
            selected='EXACT_REST'
    after = float(cross_entropy(*update, final, variance))
    after9 = float(cross_entropy(*audit, final, variance))
    if selected=='EXACT_REST':
        if not torch.equal(final,points) or abs(after9-before9)>tolerance:
            raise RuntimeError('exact rest failed GH9 or output parity')
    elif after9 >= before9-tolerance:
        raise RuntimeError('selected whole map did not strictly lower finite GH9')
    grad5 = gradient(*update, final, variance)
    grad9 = gradient(*audit, final, variance)
    return final, dict(initial_cross_entropy=before,
        donor_cross_entropy=after_donor, final_cross_entropy=after,
        initial_audit9=before9, final_audit9=after9,
        proposal_gh5=proposed5,proposal_gh9=proposed9,
        proposal_donors=donors,proposal_em=em_rows,
        proposal_inner_gh9_sign_flips=sum(row['audit9_sign_flip'] for row in em_rows),
        fallback_gh9=fallback9,fallback_em=fallback_rows,
        selected=selected,gh9_strict_tolerance=tolerance,
        final_gradient_5_l2=float(grad5.norm()),
        final_gradient_9_l2=float(grad9.norm()),
        final_gradient_max_absolute_discrepancy=float((grad5-grad9).abs().max()),
        initial_quality=quality(points, means), final_quality=quality(final, means),
        final_points=final.tolist(),
        max_output_displacement=float((final-points).norm(dim=1).max()),
        actual_sigma=sigma, frozen_width=width,
        observed_real_points=len(target), donor_candidate_points=len(candidates),
        quadrature_proposal=5, quadrature_acceptance=9)


def one_step(points, history, stream, absolute, width, means, *, omit=False):
    bank = native_bank(stream, means, omit_mode0=omit)
    history = torch.cat((history, bank.double()), 0)
    sigma = linear_output_noise(.029, absolute-1, 1200, .2)
    selected, receipt = optimize(history, bank, points, width, sigma, means=means)
    observed = grade(selected, 240+absolute, means)
    row = dict(absolute_step=absolute, output_sigma=sigma,
        omitted_d_bank=omit, real_bank_sha256=sha(bank.contiguous().numpy().tobytes()),
        data_rng_sha256=sha(stream.get_state().numpy().tobytes()),
        observed_real_points=len(history), grade=observed, receipt=receipt)
    return selected, history, row


def run_sequence(state, case, width, means, first, *, length=16,
                 omitted_prefix=0, fail_fast=False, output=None):
    points = initial_support(state).double().detach()
    stream = torch.Generator().set_state(state['rng']['data'])
    history = torch.empty((0,2), dtype=torch.float64)
    start = 1 if case=='cold1' else 1324
    rows = []
    for offset in range(length):
        absolute = start+offset
        points, history, row = one_step(points,history,stream,absolute,width,means,
                                        omit=offset<omitted_prefix)
        if offset==0 and omitted_prefix==0:
            reference=first['cases'][case]
            if (abs(row['receipt']['initial_cross_entropy']-
                    reference['initial_cross_entropy'])>1e-12 or
                    abs(row['receipt']['initial_audit9']-
                    reference['initial_audit9'])>1e-12 or
                    row['real_bank_sha256'] != first['declaration'][
                        'native_real_bank_sha256'][case]):
                raise RuntimeError('first-bank state/objective/native-real control differs')
        rows.append(row)
        if output is not None:
            (output/f'{case}-{omitted_prefix}.partial.json').write_text(json.dumps(
                dict(status='INCOMPLETE',case=case,omitted_prefix=omitted_prefix,
                     rows=rows),allow_nan=False)+'\n')
        print(json.dumps(dict(event='FORWARD_KL_GH9_STEP',case=case,
            omitted_prefix=omitted_prefix,offset=offset+1,absolute_step=absolute,
            modes=row['grade']['modes'],hq=row['grade']['hq'],
            donors=len(row['receipt']['proposal_donors']),
            em=len(row['receipt']['proposal_em']),
            selected=row['receipt']['selected'],
            finite_gh9_delta=row['receipt']['final_audit9']-
                             row['receipt']['initial_audit9'])),flush=True)
        if fail_fast and (row['grade']['modes']!=8 or row['grade']['hq']<.9):
            break
    result = dict(case=case, omitted_prefix=omitted_prefix, attempted=len(rows),
        initial_quality=quality(initial_support(state),means), rows=rows,
        final_quality=quality(points,means), final_points=points.tolist(),
        final_data_rng_sha256=sha(stream.get_state().numpy().tobytes()),
        passing=sum(r['grade']['modes']==8 and r['grade']['hq']>=.9 for r in rows))
    return result, dict(points=points,history=history,rng=stream.get_state().clone())


def paired_omission(endpoint, width, means):
    answer={}
    for label,omit in (('ordinary',False),('omitted',True)):
        points=endpoint['points'].clone();history=endpoint['history'].clone()
        stream=torch.Generator().set_state(endpoint['rng'])
        selected,_,row=one_step(points,history,stream,1340,width,means,omit=omit)
        answer[label]=dict(row=row,final_quality=quality(selected,means),
                           data_rng_sha256=sha(stream.get_state().numpy().tobytes()))
    if answer['ordinary']['data_rng_sha256'] != answer['omitted']['data_rng_sha256']:
        raise RuntimeError('paired omitted native bank changed RNG draw count')
    return answer


def model_error_response(endpoint,width,means):
    points=endpoint['points'].clone();points[:,0]+=.35
    frozen=points.clone()
    history=endpoint['history'].clone()
    stream=torch.Generator().set_state(endpoint['rng'])
    frozen_stream=torch.Generator().set_state(endpoint['rng'])
    rows=[]
    for absolute in range(1340,1345):
        selected,history,row=one_step(points,history,stream,absolute,width,means)
        native_bank(frozen_stream,means)
        row['frozen_grade']=grade(frozen,240+absolute,means)
        rows.append(row)
        points=selected
    if not torch.equal(stream.get_state(),frozen_stream.get_state()):
        raise RuntimeError('paired frozen response data stream differs')
    return dict(injected_output_bias_x=.35,
        initial_quality=quality(frozen,means),rows=rows,
        final_quality=quality(points,means),
        frozen_final_quality=quality(frozen,means),
        first_recovered=next((r['absolute_step'] for r in rows
            if r['grade']['modes']==8 and r['grade']['hq']>=.9),None),
        paired_data_rng_sha256=sha(stream.get_state().numpy().tobytes()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True)
    parser.add_argument('--first-bank',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    first_raw=args.first_bank.read_bytes();first=json.loads(first_raw)
    if not first['all_cheap_gates_pass']:
        raise RuntimeError('one-bank gate did not pass')
    cold,warm,inputs=load_states();means=mode_hold.ring_means()
    initial_stream=torch.Generator().set_state(cold['rng']['data'])
    bank=mode_hold.sample_ring(means,128,mode_hold.SIGMA,initial_stream)
    width,_=local_width(bank)
    if width!=first['declaration']['frozen_width']:
        raise RuntimeError('frozen width differs')
    paths={n:(ROOT/n if (ROOT/n).exists() else args.code_root/n) for n in SOURCE_NAMES}
    hashes={n:sha(path.read_bytes()) for n,path in paths.items()}
    if hashes['reports/toy100/forward_kl_free_filter.py']!=first['declaration'][
            'source']['reports/toy100/forward_kl_free_filter.py']:
        raise RuntimeError('source epoch differs from one-bank source')
    args.output.mkdir(parents=True)
    for name,path in paths.items():
        target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    declaration=dict(scope='pure free-output finite-GH9 whole-map stress; no neural training',
        source=hashes,first_bank_sha256=sha(first_raw),inputs=inputs,
        frozen_width=width,donor_candidates='current native D real128 only',
        target='all cumulative D real banks seen so far, same fixed h and output noise',
        proposal='GH5, at most12 current-bank donor replacements then at most20 EM steps',
        selection='accept whole proposal only if finite GH9 decreases strictly by 64eps scale; otherwise one GH9 EM step from original cloud, or exact rest',
        firstbank_control='same saved state, own native D real128 SHA, GH5/GH9 initial objectives; final output can differ because method changed',
        warm='16 updates, every update 8/HQ>=.9; stop at first failure',
        cold='16 updates, final5 8/HQ>=.9 only if warm passes',
        stage_order='warm16 -> cold16 -> paired omission -> false bootstrap16 -> error response5; stop after each failed stage',
        omission='paired native bank17 ordinary vs conditionally omit mode0; fixed target law elsewhere',
        false_bootstrap='first two conditional D banks omit mode0, then14 ordinary; final5 recover',
        response='bias clean output x by +.35 after warm16; 5 ordinary updates vs frozen same data stream',
        quality='fixed4096 late-noise external grade at clock240+absolute update',
        no_oracle_update=True,no_seed_or_bandwidth_sweep=True,shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='FORWARD_KL_GH9_DECLARED',h=width,
        source_sha256=hashes['reports/toy100/forward_kl_gh9_stress.py'])),flush=True)
    before_rng=torch.random.get_rng_state().clone();cases={}
    try:
        cases['warm_full'],warm_endpoint=run_sequence(warm,'warm1324',width,means,first,
            fail_fast=True,output=args.output)
        if cases['warm_full']['passing']==16:
            cases['cold_full'],_=run_sequence(cold,'cold1',width,means,first,
                output=args.output)
        if 'cold_full' in cases and all(r['grade']['modes']==8 and
                r['grade']['hq']>=.9 for r in cases['cold_full']['rows'][-5:]):
            cases['paired_omission']=paired_omission(warm_endpoint,width,means)
        if 'paired_omission' in cases and all(v['row']['grade']['modes']==8 and
                v['row']['grade']['hq']>=.9 for v in cases['paired_omission'].values()):
            cases['false_bootstrap'],_=run_sequence(warm,'warm1324',width,means,first,
                omitted_prefix=2,output=args.output)
        if 'false_bootstrap' in cases and all(r['grade']['modes']==8 and
                r['grade']['hq']>=.9 for r in cases['false_bootstrap']['rows'][-5:]):
            cases['model_error_response']=model_error_response(warm_endpoint,width,means)
        all_rows=[row for name in ('warm_full','cold_full','false_bootstrap')
                  if name in cases for row in cases[name]['rows']]
        if 'paired_omission' in cases:
            all_rows += [v['row'] for v in cases['paired_omission'].values()]
        if 'model_error_response' in cases:
            all_rows += cases['model_error_response']['rows']
        whole_gh9_valid=all((row['receipt']['final_audit9']<
            row['receipt']['initial_audit9']-row['receipt']['gh9_strict_tolerance']
            if row['receipt']['selected']!='EXACT_REST' else
            row['receipt']['final_audit9']==row['receipt']['initial_audit9'])
            for row in all_rows)
        gates=dict(warm_all16=cases['warm_full']['passing']==16,
            cold_terminal5=('cold_full' in cases and all(r['grade']['modes']==8 and
                r['grade']['hq']>=.9 for r in cases['cold_full']['rows'][-5:])),
            omitted_bank_retains=('paired_omission' in cases and all(
                v['row']['grade']['modes']==8 and v['row']['grade']['hq']>=.9
                for v in cases['paired_omission'].values())),
            false_bootstrap_reacquires=('false_bootstrap' in cases and all(
                r['grade']['modes']==8 and r['grade']['hq']>=.9
                for r in cases['false_bootstrap']['rows'][-5:])),
            output_error_recovers=('model_error_response' in cases and
                cases['model_error_response']['first_recovered'] is not None),
            finite_gh9_whole_map_valid=whole_gh9_valid)
        result=dict(status='COMPLETE',declaration=declaration,cases=cases,gates=gates,
            all_gates_pass=all(gates.values()),
            global_torch_rng_unchanged=torch.equal(torch.random.get_rng_state(),before_rng))
        if not result['global_torch_rng_unchanged']:
            raise RuntimeError('pure filter changed global RNG')
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(event='FORWARD_KL_GH9_DONE',gates=gates)),flush=True)
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE',
            error=repr(error),complete_cases=list(cases)),indent=2)+'\n')
        raise


if __name__=='__main__':main()
