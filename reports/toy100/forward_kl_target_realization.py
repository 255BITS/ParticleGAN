"""Cheapest isolated neural realization of the passed first output targets.

This is not native GAN training: copied pre-step G/prior only, no Adam step,
and no input/critic/noise-clock mutation. It checks whether the fixed GN budget
can realize the actual passed first finite-GH9 target from cold and warm.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.forward_kl_free_filter import quadrature, cross_entropy
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_early_geometry import _model
from reports.toy100.pr84_critic_refinement_capture import _sha
from reports.toy100.sample_anchor_free1200 import load_states
from reports.toy100.sample_anchor_local_mmd_continuation import native_bank
from reports.toy100.sample_anchor_mmd_filter import quality

ROOT=Path(__file__).resolve().parents[2]
FILES=('reports/toy100/forward_kl_target_realization.py',
       'reports/toy100/forward_kl_free_filter.py',
       'reports/toy100/joint_output_pullback.py',
       'reports/toy100/pr84_early_geometry.py',
       'reports/toy100/pr84_critic_refinement_capture.py',
       'reports/toy100/sample_anchor_free1200.py',
       'reports/toy100/sample_anchor_local_mmd_continuation.py',
       'reports/toy100/sample_anchor_mmd_filter.py',
       'benchmarks/locked_shared/mode_hold.py','benchmarks/locked_shared/mlp.py')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--free-result',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    raw=args.free_result.read_bytes()
    source=json.loads(raw)
    if not source['all_gates_pass']:
        raise ValueError('requires the passed finite-GH9 pure-output diagnostic')
    source_hashes={}
    for name in FILES:
        data=(ROOT/name).read_bytes()
        dest=args.output/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        source_hashes[name]=hashlib.sha256(data).hexdigest()
    cold,warm,inputs=load_states()
    declaration=dict(scope='copied pre-G model/prior target-realization only; no training',
        source=source_hashes,inputs=inputs,
        free_result_sha256=hashlib.sha256(raw).hexdigest(),
        fixed_budget=dict(gn_iterations=20,halvings=12,svd_rtol=1e-6,relative_tolerance=1e-5),
        required='converged fit, actual finite-GH9 decrease, both8/HQ>=.9 in late-noise diagnostic',
        shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    (args.output/'input-free-result.json').write_bytes(raw)
    print(json.dumps(dict(event='REALIZATION_DECLARED',**declaration)),flush=True)
    results={}
    for name,state,case in [('cold1',cold,'cold_full'),('warm1324',warm,'warm_full')]:
        input_hash=_sha(state)
        before_rng=torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            model=_model(state,'g')
        prior=torch.nn.Parameter(state['prior']['z'].detach().clone())
        old_params=[p.detach().clone() for p in model.parameters()]+[prior.detach().clone()]
        with torch.no_grad():initial=model(prior).detach().clone()
        first=source['cases'][case]['rows'][0]
        rec=first['receipt']
        target=torch.tensor(rec['final_points'],dtype=torch.float64)
        stream=torch.Generator().set_state(state['rng']['data'])
        real=native_bank(stream,mode_hold.ring_means())
        if hashlib.sha256(real.numpy().tobytes()).hexdigest()!=first['real_bank_sha256']:
            raise RuntimeError('native real bank differs from passed free-output fixture')
        locations,weights=quadrature(real,rec['frozen_width'],9)
        variance=rec['frozen_width']**2+rec['actual_sigma']**2
        before=float(cross_entropy(locations,weights,initial,variance))
        if abs(before-rec['initial_audit9'])>1e-12:
            raise RuntimeError('pre-G actual objective differs from pure fixture')
        target_cost=float(cross_entropy(locations,weights,target,variance))
        fit=fit_output_targets(model,prior,target)
        with torch.no_grad():fitted=model(prior).detach().clone()
        after=float(cross_entropy(locations,weights,fitted,variance))
        grade=quality(fitted,mode_hold.ring_means())
        if _sha(state)!=input_hash or not torch.equal(torch.random.get_rng_state(),before_rng):
            raise RuntimeError('isolated fit mutated saved input or caller RNG')
        okay=fit['status']=='CONVERGED' and after<before and grade['modes']==8 and grade['hq']>=.9
        results[name]=dict(status='PASS' if okay else 'FAIL',fit=fit,
            pre_gh9=before,target_gh9=target_cost,fitted_gh9=after,
            gh9_realization_error=after-target_cost,late_noise_quality=grade,
            actual_objective_output_sigma=rec['actual_sigma'],
            prior_displacement=float((prior.detach()-old_params[-1]).double().norm()),
            generator_displacement=float(sum((p.detach()-old).double().square().sum()
                for p,old in zip(model.parameters(),old_params[:-1])).sqrt()),
            final_outputs=fitted.tolist(),input_and_rng_unchanged=True)
        torch.save(dict(generator=model.state_dict(),prior_z=prior.detach(),
                        input_state=state,target=target),args.output/f'{name}-state.pt')
        print(json.dumps(dict(event='REALIZATION_CASE',case=name,status=results[name]['status'],
            fit_status=fit['status'],iterations=len(fit['records']),
            pre=before,target=target_cost,after=after,modes=grade['modes'],hq=grade['hq'])),flush=True)
        (args.output/'partial.json').write_text(json.dumps(results,allow_nan=False)+'\n')
    result=dict(status='PASS' if all(x['status']=='PASS' for x in results.values()) else 'FAIL',
                declaration=declaration,cases=results,native_training_qualified=False)
    (args.output/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(event='REALIZATION_DONE',status=result['status'])),flush=True)


if __name__=='__main__':main()
