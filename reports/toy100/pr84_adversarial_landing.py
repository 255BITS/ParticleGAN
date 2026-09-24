"""Land the three frozen V2 adversarial proposals on copied G/prior states.

Joint GN is only a numerical proposal realization. The unchanged actual
Rp generator loss decides whether to retain its result; neither target
distance, heldout loss nor quality can substitute for that acceptance.
No host continuation, D fit, Adam moment update or resampling occurs.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_adversarial_reallocation_assay as assay
from reports.toy100.pr84_adversarial_reallocation_assay_v2 import METHOD as PROPOSAL_METHOD
from reports.toy100.joint_output_pullback import fit_output_targets

METHOD = 'original_rp_G_loss_fixed_nonlocal_proposal_joint_neural_landing'


def independent_loss(critic, fake, real, gan):
    """Direct paired GANLoss calls; independent of selector reduction code."""
    def score(points):
        values = [critic(points)]
        for dim in (0, 1):
            offset = torch.zeros_like(points)
            offset[:, dim] = .15
            values += [critic(points+offset), critic(points-offset)]
        return torch.stack(values).mean(0).reshape(-1)
    fake_scores, real_scores = score(fake), score(real)
    assert fake_scores.shape == real_scores.shape == (len(fake),)
    return gan.g_loss(fake_scores, real_scores)


def load_cases():
    values = {name: assay.read(ROOT/path) for name, path in assay.INPUTS.items()}
    warm = torch.load(BytesIO(values['warm']), weights_only=True)
    cold = torch.load(BytesIO(values['cold']), weights_only=True)
    refined = torch.load(BytesIO(values['refined']), weights_only=True)[1530]
    return [('warm1530_native_D', 1530, warm, None),
            ('warm1530_archived_refined_D', 1530, warm, refined['critic']),
            ('cold472_best_finite_D', 472, cold, None)]


def verify_proposal(proposal, cases, recipe):
    """Actual-state V1/V2 repair audit precedes every landing."""
    if proposal['declaration']['method'] != PROPOSAL_METHOD:
        raise ValueError('requires V2 paired-baseline proposal')
    for name, expected in proposal['declaration']['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('proposal source changed: '+name)
    for name, expected in proposal['declaration']['input_sha256'].items():
        if hashlib.sha256((ROOT/assay.INPUTS[name]).read_bytes()).hexdigest() != expected:
            raise RuntimeError('proposal saved input changed: '+name)
    old_path = ROOT/'reports/toy100/continuous-evidence/gan-nonlocal-output/result.json.gz'
    old = json.loads(assay.read(old_path))
    audit = []
    for (name, _, tensors, refined), row, previous in zip(cases, proposal['rows'], old['rows']):
        if row['name'] != previous['name'] or row['name'] != name:
            raise RuntimeError('proposal/control order changed')
        state = tensors['captured']['post_accepted_d']
        generator, critic, prior = assay.fit.modules(state)
        if refined is not None:
            critic.load_state_dict(refined)
        batch = tensors['grows'][0]
        with torch.no_grad():
            fake = generator(prior.z[batch['indices']]) + batch['sigma']*batch['noise']
            donor, point = row['selection']['donor'], row['selection']['target']
            changed = assay.replace_fake(fake, batch, donor, torch.tensor(point, dtype=fake.dtype))
            before = float(independent_loss(critic, fake, batch['real'], recipe.make_loss()))
            after = float(independent_loss(critic, changed, batch['real'], recipe.make_loss()))
        checks = dict(before_exact=before == row['selection']['native_loss_before'],
                      before_native_receipt_exact=before == row['original_native_G_update']['loss'],
                      after_exact=after == row['selection']['native_loss_after'],
                      objective_decreases=after < before,
                      paired_baseline_guard=row['selection'].get('paired_baseline_exact') is True,
                      same_heldout=row['heldout'] == previous['heldout'],
                      same_quality=row['grade_after'] == previous['grade_after'])
        for key in ('donor', 'candidate', 'target', 'selected', 'native_loss_after'):
            checks['v1_same_'+key] = row['selection'][key] == previous['selection'][key]
        if not all(checks.values()):
            raise RuntimeError(f'independent paired repair audit failed: {name}: {checks}')
        audit.append(dict(name=name, checks=checks, before=before, after=after,
                          invalid_v1_before=previous['selection']['native_loss_before']))
    return dict(status='PASS', v1_scope='INVALID baseline and acceptance accounting; preserved',
                v1_result_gzip_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(), rows=audit)


def run_case(case, proposal, recipe):
    name, step, tensors, refined = case
    state = tensors['captured']['post_accepted_d']
    generator, critic, prior = assay.fit.modules(state)
    if refined is not None:
        critic.load_state_dict(refined)
    gan = recipe.make_loss()
    before_g, before_z = deepcopy(generator.state_dict()), prior.z.detach().clone()
    critic_hash = assay._sha(critic.state_dict())
    # Materialize copied Adam state without advancing it. It is an ownership
    # witness, not the metric of this Euclidean minimum-increment GN solver.
    optimizer_g = assay.fit.g_optimizer(generator, prior, state['optimizer_g'])
    adam_hash = assay._sha((optimizer_g.state_dict(), state['optimizer_d']))
    initial = generator(prior.z).detach().clone()
    if not torch.equal(initial, torch.tensor(proposal['supports']['before'], dtype=initial.dtype)):
        raise RuntimeError('landing does not start at the proposal source cloud')
    target = torch.tensor(proposal['supports']['after'], dtype=initial.dtype)
    g_before = [p.detach().clone() for p in generator.parameters()]
    rng = torch.get_rng_state().clone()
    fit = fit_output_targets(generator, prior.z, target)
    fitted = generator(prior.z).detach().clone()
    losses = []
    for index, batch in enumerate(tensors['grows'][:9]):
        with torch.no_grad():
            fake = generator(prior.z[batch['indices']]) + batch['sigma']*batch['noise']
            actual = float(independent_loss(critic, fake, batch['real'], gan))
        before = proposal['selection']['native_loss_before'] if index == 0 else proposal['heldout'][index-1]['before']
        losses.append(dict(bank=index, before=before, after=actual, decreased=actual<before))
    accepted = losses[0]['decreased'] and all(torch.isfinite(p).all() for p in generator.parameters()) \
        and bool(torch.isfinite(prior.z).all())
    if not accepted:
        generator.load_state_dict(before_g)
        with torch.no_grad():
            prior.z.copy_(before_z)
    accepted_support = generator(prior.z).detach()
    indices, noise = assay.fixed_draw(step, initial)
    means = assay.mode_hold.ring_means()
    g_delta = sum((p.detach()-old).double().square().sum() for p,old in zip(generator.parameters(), g_before)).sqrt()
    result = dict(name=name, step=step, fit=fit, accepted_by_original_G_loss=accepted,
        losses=losses, all_eight_heldout_decrease=all(row['decreased'] for row in losses[1:]),
        grade_before=assay.score_support(initial, indices, noise, means),
        grade_fitted=assay.score_support(fitted, indices, noise, means),
        grade_accepted=assay.score_support(accepted_support, indices, noise, means),
        generator_parameter_displacement=float(g_delta),
        prior_parameter_displacement=float((prior.z.detach()-before_z).double().norm()),
        final_generator_parameter_norm=float(sum(p.detach().double().square().sum() for p in generator.parameters()).sqrt()),
        final_prior_parameter_norm=float(prior.z.detach().double().norm()),
        target=target.tolist(), before=initial.tolist(), after=accepted_support.tolist(),
        critic_unchanged=critic_hash==assay._sha(critic.state_dict()),
        both_Adam_states_unchanged=adam_hash==assay._sha((optimizer_g.state_dict(), state['optimizer_d'])),
        RNG_unchanged=torch.equal(rng, torch.get_rng_state()),
        gradients_remain_absent=all(p.grad is None for p in list(generator.parameters())+list(prior.parameters())))
    if not all(result[key] for key in ('critic_unchanged','both_Adam_states_unchanged','RNG_unchanged','gradients_remain_absent')):
        raise RuntimeError('landing changed non-owner state')
    endpoint = dict(generator=deepcopy(generator.state_dict()), prior=deepcopy(prior.state_dict()),
                    critic=deepcopy(critic.state_dict()), optimizer_g=deepcopy(optimizer_g.state_dict()),
                    optimizer_d=deepcopy(state['optimizer_d']))
    return result, endpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    raw = args.proposal.read_bytes()
    proposal = json.loads(raw)
    sources = dict(proposal['declaration']['sources'])
    for name in ('reports/toy100/pr84_adversarial_landing.py','reports/toy100/joint_output_pullback.py',
                 'tests/test_joint_output_pullback.py'):
        sources[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    declaration = dict(method=METHOD, proposal_sha256=hashlib.sha256(raw).hexdigest(), sources=sources,
        input_sha256=proposal['declaration']['input_sha256'],
        start='exact same pre-G parameter state as each frozen free-output proposal',
        solver=dict(max_iterations=20, max_halves=12, svd_rtol=1e-6, relative_tolerance=1e-5),
        acceptance='finite actual native paired Rp G loss strictly decreases; same D/stencil/noise',
        heldout='eight fixed postselection G banks, never selection or tuning',
        quality='postselection diagnostic only', optimizer_updates=0, new_critic_fits=0,
        shared_gate_eligible=False, scope='copied neural landing only; no native host continuation')
    for name in sources:
        path = args.output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    started = time.perf_counter()
    cases = load_cases()
    payload_hash = assay._sha(cases)
    rng = torch.get_rng_state().clone()
    recipe, _, _ = assay.declared_recipe(json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text()))
    rows, endpoints = [], {}
    with torch.random.fork_rng(devices=[]):
        repair = verify_proposal(proposal, cases, recipe)
        (args.output/'paired-repair-audit.json').write_text(json.dumps(repair, indent=2)+'\n')
        print(json.dumps(dict(event='PAIRED_REPAIR_AUDIT', status=repair['status'])), flush=True)
        for case, frozen in zip(cases, proposal['rows']):
            row, endpoint = run_case(case, frozen, recipe)
            rows.append(row); endpoints[case[0]] = endpoint
            print(json.dumps(dict(event='LANDING_DONE', name=row['name'], fit=row['fit']['status'],
                iterations=len(row['fit']['records']), max_error=row['fit']['final_max_row_error'],
                accepted=row['accepted_by_original_G_loss'], grade=row['grade_accepted'],
                all_eight_heldout_decrease=row['all_eight_heldout_decrease'])), flush=True)
    assert payload_hash == assay._sha(cases) and torch.equal(rng, torch.get_rng_state())
    result = dict(declaration=declaration, repair=repair, rows=rows,
                  input_and_caller_RNG_unchanged=True, seconds=time.perf_counter()-started,
                  all_converged_and_adversarial_decrease=all(row['fit']['status']=='CONVERGED'
                    and row['accepted_by_original_G_loss'] and row['all_eight_heldout_decrease'] for row in rows),
                  shared_gate_eligible=False)
    torch.save(endpoints, args.output/'endpoints.pt')
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
