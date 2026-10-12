"""Read-only AMSGrad role-memory diagnosis from exact retained checkpoints.

No model/optimizer construction, gradient call, sampling, update or rescoring.
The reported counterfactual changes only the last denominator on an actual
AMSGrad state; it does not invent historical maxima for a plain Adam state.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import random

import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
TASKS = ('ring16_acquisition', 'gaussian1d_smoke', 'five_word_joint_smoke')
PUBLICATIONS = {
    'role': 'reports/forge/r1r2-role-repair/publication/results.json',
    'five': 'reports/forge/r1r2-five-track-repair/publication/results.json',
}
CASES = (
    ('m000', 'role', 'r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9'),
    ('m110', 'role', 'r1r2-role-m110-v1'),
    ('m111', 'role', 'r1r2-five-track-momentum-v1'),
    ('ams_m111', 'role', 'r1r2-role-ams-m111-v1'),
    ('ams_m000', 'five', 'r1r2-amsgrad-repair-v1'),
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(value):
    return torch.quantile(value.double().flatten(), torch.tensor([0., .1, .5, .9, 1.], dtype=torch.float64)).tolist()


def role_memory(role, optimizer, group, binding):
    first, native, retained, ratios, steps = [], [], [], [], []
    b1, b2 = group['betas']
    assert group['eps'] == 1e-8 and group['lr'] == .0085
    assert group['weight_decay'] == 0 and not group.get('maximize', False)
    for index in group['params']:
        item = optimizer['state'][index]
        step = int(item['step'])
        m = item['exp_avg'].double()/(1-b1**step)
        q = (item['exp_avg_sq'].double()/(1-b2**step)).sqrt()+group['eps']
        qm = (item.get('max_exp_avg_sq', item['exp_avg_sq']).double()/(1-b2**step)).sqrt()+group['eps']
        assert torch.all(qm >= q)
        first.append(m.flatten())
        native.append((-group['lr']*m/q).flatten())
        retained.append((-group['lr']*m/qm).flatten())
        ratios.append((qm/q).flatten())
        steps.append(step)
    m, native, actual, ratio = map(torch.cat, (first, native, retained, ratios))
    active = m != 0
    row = dict(role=role, epsilon_binding=binding, betas=list(group['betas']),
        amsgrad=group['amsgrad'], steps_min_max=[min(steps),max(steps)],
        first_moment_l2=float(m.norm()), zero_first_moment_fraction=float((~active).double().mean()),
        reconstructed_actual_displacement_l2=float(actual.norm()),
        actual_displacement_max_abs=float(actual.abs().max()),
        same_saved_moments_without_max_displacement_l2=float(native.norm()),
        actual_over_no_max_displacement_l2=float(actual.norm()/native.norm()) if native.norm()>0 else None,
        retained_over_current_denominator_min_p10_median_p90_max=quantiles(ratio),
        active_denominator_ratio_min_p10_median_p90_max=quantiles(ratio[active]) if active.any() else None,
        active_ratio_ge2_fraction=float((ratio[active]>=2).double().mean()) if active.any() else None,
        active_ratio_ge10_fraction=float((ratio[active]>=10).double().mean()) if active.any() else None,
        first_moment_kind='actual last gradient' if b1==0 else 'EMA; actual last raw gradient unavailable')
    if role=='P':
        item=optimizer['state'][group['params'][0]]
        row['rows_with_nonzero_first_moment']=int((item['exp_avg'].norm(dim=1)>0).sum())
        row['row_capacity']=item['exp_avg'].shape[0]
    return row


def forward(x, weights, indices=(0,2,4)):
    for index in indices:
        x=x@weights[f'net.{index}.weight'].double().T+weights[f'net.{index}.bias'].double()
        if index!=indices[-1]:
            x=torch.nn.functional.leaky_relu(x,.2)
    return x


def word_panel(models):
    chars='abcdefghijklmnopqrstuvwxyz_ '
    words=('apple','grape','lemon','melon','berry')
    indices=torch.tensor([[chars.index(c) for c in w+'_'] for w in words])
    bank=torch.nn.functional.one_hot(indices,len(chars)).permute(0,2,1).double()
    prior=models['prior']['z'].double()
    encoded=forward(bank.flatten(1),models['encoder'],(1,3,5))
    generated_logits=forward(prior,models['generator']).reshape(5,len(chars),6)
    reconstruction_logits=forward(encoded,models['generator']).reshape(5,len(chars),6)
    generated=generated_logits.softmax(1)
    reconstructed=reconstruction_logits.softmax(1)
    names=lambda p:[''.join(chars[i] for i in row) for row in p.argmax(1).tolist()]
    correct=reconstructed.gather(1,indices.unsqueeze(1)).squeeze(1)
    gnorm=sum(float(v.double().square().sum()) for v in models['generator'].values())**.5
    return dict(cohort='fixed saved prior means and five canonical inputs; no draws or gate scoring',
        generator_parameter_l2=gnorm,
        generated_decoder_logit_max_abs=float(generated_logits.abs().max()),
        reconstruction_decoder_logit_max_abs=float(reconstruction_logits.abs().max()),
        generated_by_saved_row=names(generated),reconstructed_by_canonical_input=names(reconstructed),
        reconstruction_min_correct_token_probability=float(correct.min()),
        prior_pair_distance_min=float(torch.pdist(prior).min()),
        encoded_pair_distance_min=float(torch.pdist(encoded).min()))


def observations(raw,task):
    observations=raw['evidence']['observations']
    if task=='ring16_acquisition':
        keys=('step','component_covariance_error','component_min_eigen_ratio','hq','modes','mass_tv')
        best=min(observations,key=lambda x:x['component_covariance_error'])
        final=[{key:o[key] for key in keys} for o in observations[-5:]]
        cov=torch.tensor([o['component_covariance_error'] for o in observations[-16:]],dtype=torch.float64)
        return dict(saved_observations=len(observations),best_covariance_observation={k:best[k] for k in keys},
            last_five=final,last16_covariance_mean=float(cov.mean()),last16_covariance_std=float(cov.std(unbiased=False)),
            last16_covariance_min_max=[float(cov.min()),float(cov.max())],
            covariance_bound_passing_observations=sum(o['component_covariance_error']<=.85 for o in observations),
            last16_min_eigen_bound_failures=sum(o['component_min_eigen_ratio']<.15 for o in observations[-16:]))
    if task=='gaussian1d_smoke':
        keys=('step','cdf_ks','mean_error_sigma','std_ratio')
        best=min(observations,key=lambda x:x['cdf_ks'])
        return dict(saved_observations=len(observations),best_cdf_observation={k:best[k] for k in keys},
            endpoint={k:observations[-1][k] for k in keys})
    return dict(saved_observations=len(observations),endpoint=observations[-1])


def prospective_bindings(publisher, renderer):
    """Existing public preflight only; the prospective recipe stays in memory."""
    from experiments.forge.api import task_formulation_context
    from experiments.forge.contracts import stable_hash
    from experiments.forge.preflight import task_preflight
    from particlegan.coulomb_prior import CoulombPriorRegularizer
    parent=publisher.certified_attempt(ROOT,'20aca4f16e8a4c31900b652e2c71ee3b')['request']
    candidate=deepcopy(parent['candidate'])
    assert candidate['recipe_overrides']['amsgrad'] is False
    candidate['recipe_overrides']['amsgrad']=True
    if 'resolved_recipe' in candidate:
        candidate['resolved_recipe']['amsgrad']=True
    declaration=json.loads((ROOT/'reports/forge/r1r2-role-repair/registration.json').read_text())
    contracts=declaration['candidates']['m110']['task_contracts']
    tasks=deepcopy(parent['tasks'])
    rows=[]
    original_init=torch.nn.Module.__init__
    def validator_only_init(self,*args,**kwargs):
        # Recipe.__post_init__ validates inactive Coulomb options through this
        # exact parameter-free object. Forward and all other modules stay blocked.
        if type(self) is not CoulombPriorRegularizer:
            raise RuntimeError('model construction prohibited')
        return original_init(self,*args,**kwargs)
    with renderer.forbid_live_execution():
        torch.nn.Module.__init__=validator_only_init
        for task_id,contract in contracts.items():
            task=tasks[task_id]
            before=stable_hash({k:v for k,v in task.items() if k not in ('field_ownership','preflight_blockers')})
            assert before==contract
            blockers=task_preflight(task,candidate,parent['protocol'],root=ROOT,tasks=tasks)
            context=task_formulation_context(candidate,task,parent['protocol'],root=ROOT)
            recipe=context.recipe
            assert not blockers,(task_id,blockers)
            assert recipe.optimizer_family=='adam' and recipe.adam_variant=='pytorch' and recipe.amsgrad
            assert recipe.betas==(.5,.99) and (recipe.d_betas or recipe.betas)==(.5,.99)
            assert recipe.prior_betas==(0.,.99)
            assert recipe.lr==.0085 and recipe.reg_arm=='a_r1r2' and recipe.reg_coeff==.1 and recipe.reg_every==1
            assert recipe.prior_update=='learned' and recipe.prior_reg==0 and recipe.lr_schedule=='constant'
            assert recipe.eps==1e-8 and recipe.d_eps is None and recipe.prior_eps is None
            assert recipe.beta2_end is None and recipe.ema_decay==0 and recipe.output_noise_std==0
            after=stable_hash({k:v for k,v in task.items() if k not in ('field_ownership','preflight_blockers')})
            assert before==after
            rows.append(dict(task_id=task_id,task_contract_sha256=contract,blockers=blockers,
                family=recipe.optimizer_family,adam_variant=recipe.adam_variant,amsgrad=recipe.amsgrad,
                effective_betas=dict(G_and_E=list(recipe.betas),D=list(recipe.d_betas or recipe.betas),
                    P=list(recipe.prior_betas) if task_id not in ('two_pole','unused_token_hold') else 'not consumed'),
                direct_pole_binding='G beta1=.5, global AMSGrad; no prior beta override' if task_id=='two_pole' else None,
                prior_binding=context.receipt()['prior']))
    # Exact recipe comparison is local recall, not a new declaration.
    recipe=candidate['recipe_overrides']
    identical=[]
    for directory in ('configs/forge/ideas','configs/forge/configurations'):
        for path in (ROOT/directory).glob('*.json'):
            existing=json.loads(path.read_text())
            if existing.get('recipe_overrides')==recipe:
                identical.append(str(path.relative_to(ROOT)))
    return dict(prospective_only=True,one_field_delta={'amsgrad':[False,True]},
        parent_candidate=parent['candidate']['id'],parent_revision=parent['candidate_revision'],
        all_seven_existing_public_bindings_PASS=True,bindings=rows,identical_declared_recipe_matches=identical,
        validation_only_constructor_exception='Exact parameter-free CoulombPriorRegularizer for Recipe inactive-option validation; no forward; all other modules blocked')


def main():
    from experiments.forge.contracts import stable_hash
    before_torch,before_python=torch.get_rng_state().clone(),random.getstate()
    helper=load('reports/forge/recipe-prior-refactor/workflow.py','ams_role_workflow')
    publisher=helper.load_publisher(ROOT)
    renderer=publisher._saved_renderer(ROOT)
    mapper=load('reports/forge/r1r2-role-repair/prior/analyze_denominators.py','ams_role_mapper')
    pubs={k:json.loads((ROOT/path).read_text()) for k,path in PUBLICATIONS.items()}
    results=[]
    with renderer.forbid_live_execution(),torch.no_grad():
        for label,publication,candidate in CASES:
            for task in TASKS:
                compact=next(r for r in pubs[publication]['certified_results'] if r['candidate_id']==candidate and r['task_id']==task)
                attempt=publisher.certified_attempt(ROOT,compact['attempt_id'])
                assert attempt['request']['source']['digest']==compact['source_digest']==SOURCE
                assert attempt['request']['candidate_revision']==compact['candidate_revision']
                row=attempt['result']['task_results'][0]
                saved,proof=publisher.checkpoint(row)
                assert proof['sha256']==compact['checkpoint']['sha256']
                raw=json.loads((attempt['local']/'raw-result.json').read_text())
                recipe=saved['trainer']['recipe'] if 'trainer' in saved else saved['fixture']['api_state']['recipe']
                groups=mapper.roles(saved,task)
                detail=dict(role=label,task_id=task,candidate_id=candidate,candidate_revision=compact['candidate_revision'],
                    attempt_id=compact['attempt_id'],source_digest=SOURCE,source_commit=attempt['request']['source']['origin_commit'],
                    runtime_sha256=stable_hash(attempt['request']['runtime']),checkpoint=proof,raw_result_sha256=sha(attempt['local']/'raw-result.json'),
                    publication_sha256=sha(ROOT/PUBLICATIONS[publication]),
                    recorded_gate_status=compact['gate_status'],recorded_metrics=compact['metrics'],
                    optimizer_roles=[role_memory(*g) for g in groups],observations=observations(raw,task),
                    recipe_delta_fields={key:recipe.get(key) for key in ('optimizer_family','betas','d_betas','prior_betas','amsgrad','lr','reg_arm','reg_coeff','prior_update','prior_reg')})
                if task=='five_word_joint_smoke':
                    detail['fixed_word_panel']=word_panel(saved['fixture']['api_state']['models'])
                results.append(detail)
        # Verify the public optimizer/host equations against every original source.
        files=('particlegan/recipes.py','particlegan/recipe_schedules.py','benchmarks/toy_audit/api_images.py','experiments/forge/behavior_adapters.py')
        hashes={path:sha(ROOT/path) for path in files}
        for label,publication,candidate in CASES:
            row=next(r for r in pubs[publication]['certified_results'] if r['candidate_id']==candidate and r['task_id']==TASKS[0])
            request=publisher.certified_attempt(ROOT,row['attempt_id'])['request']
            assert all(request['source']['files'][path]==digest for path,digest in hashes.items())
    bindings=prospective_bindings(publisher,renderer)
    assert torch.equal(before_torch,torch.get_rng_state()) and before_python==random.getstate()
    print(json.dumps(dict(schema_version=1,qualification_input=False,optimizer_updates_added=0,sampling_draws_added=0,
        models_constructed=0,gradient_calls=0,python_and_torch_cpu_rng_unchanged=True,
        reproducer_sha256=sha(Path(__file__)),verified_public_source_files=hashes,results=results,prospective_bindings=bindings,
        limitations=[
            'Only terminal optimizer moments are saved. Plain Adam has no max-exp-avg history to recover an AMSGrad counterfactual.',
            'Beta1=.5 stores EMA, not the last raw gradient. No raw-to-momentum contraction is measurable for m110.',
            'AMSGrad same-moment last-step contraction is algebraic, not a matched feedback trajectory or observed causal effect.',
            'Saved observation changes combine learned feedback with evaluation draws. They do not prove per-update instability.',
            'Fixed word panels characterize saturation but do not replace sampled numerical acquisition gates.',
            'A new ams_m110 changes only global amsgrad relative to m110; global flag cannot isolate denominator memory by role.',
        ]),indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
