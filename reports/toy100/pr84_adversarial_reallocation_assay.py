"""Two-state free-output falsifier of original-G-loss nonlocal search.

One donor and one native D-real sample are selected by the unchanged paired
Rp generator loss.  There is no neural landing, critic fit, target geometry,
optimizer update or training continuation. Heldout losses and quality never
select the move. A frozen critic can reward collapse; this assay tests that
failure before proposing an actual GAN update.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_critic_refinement_capture import _sha

METHOD = 'original_rp_G_loss_single_nonlocal_output_replacement_assay'
INPUTS = {
    'warm': 'reports/toy100/continuous-evidence/convex-profiled-value1530/v2/tensors.pt.gz',
    'cold': 'reports/toy100/continuous-evidence/convex-profiled-value-independent/472/tensors.pt.gz',
    'refined': 'reports/toy100/continuous-evidence/pr84-critic-relaxation/fit/fitted-critics-and-banks.pt.gz',
    'refined_receipt': 'reports/toy100/continuous-evidence/pr84-critic-relaxation/fit/step-1530.json.gz',
}
SOURCES = (
    'reports/toy100/pr84_adversarial_reallocation_assay.py',
    'reports/toy100/pr84_critic_relaxation.py',
    'reports/toy100/alternating_curvature_scratch.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_critic_refinement_capture.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'benchmarks/transfer_suite/toy100_compatibility.py',
    'particlegan/gan_loss.py', 'particlegan/grad_regularizers.py',
    'particlegan/particle_prior.py',
    'configs/toy100/constraints_simple_regularization.json',
    'tests/test_pr84_adversarial_reallocation_assay.py',
)


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def loss(critic, fake, real, gan):
    return gan.g_loss(fit.smooth(critic, fake), fit.smooth(critic, real))


def replace_fake(fake, batch, donor, target):
    """Keep every non-donor native value bitwise; retain paired noise."""
    result = fake.clone()
    selected = batch['indices'] == donor
    result[selected] = target + batch['sigma'] * batch['noise'][selected]
    return result


@torch.no_grad()
def select(critic, fake, batch, candidates, particles, gan):
    if gan.mode != 'rp' or gan.loss_type != 'logistic':
        raise ValueError('the separable enumeration is only the declared Rp logistic G loss')
    if not all(torch.isfinite(x).all() for x in (fake, batch['real'], candidates)):
        raise FloatingPointError('nonfinite selection input')
    real_logits = fit.smooth(critic, batch['real']).reshape(-1)
    original_terms = F.softplus(real_logits - fit.smooth(critic, fake).reshape(-1))
    base = gan.g_loss(fit.smooth(critic, fake), real_logits[:, None])
    scores = torch.empty((particles, len(candidates)), dtype=torch.float64)
    for donor in range(particles):
        mask = batch['indices'] == donor
        if not bool(mask.any()):
            scores[donor].fill_(float(base))
            continue
        points = candidates[:, None, :] + batch['sigma'] * batch['noise'][mask][None, :, :]
        logits = fit.smooth(critic, points.reshape(-1, points.shape[-1]))
        logits = logits.reshape(len(candidates), -1)
        terms = F.softplus(real_logits[mask][None, :] - logits)
        scores[donor] = (original_terms[~mask].double().sum() + terms.double().sum(1)) / len(fake)
    chosen = int(scores.argmin())
    donor, candidate = divmod(chosen, len(candidates))
    target = candidates[candidate].detach().clone()
    trial = replace_fake(fake, batch, donor, target)
    actual = loss(critic, trial, batch['real'], gan)
    # This is merely a numerical/native-full-loss check on the unique selected
    # trial. We do not scan again if its full reevaluation does not decrease.
    accepted = bool(torch.isfinite(actual) and actual < base)
    return dict(donor=donor, candidate=candidate, target=target.tolist(),
                selected=accepted, native_loss_before=float(base),
                native_loss_after=float(actual), enumerated_minimum=float(scores.min()),
                enumeration_full_loss_error=abs(float(actual)-float(scores.min())),
                proposals=particles*len(candidates)), target, scores


def run_case(name, step, tensors, gan, cap, *, refined=None, refined_receipt=None):
    captured = tensors['captured']
    state = captured['post_accepted_d']
    if state['noise']['input_sigma'] != 0 or state['noise']['output_sigma'] != .029:
        raise ValueError('this assay is restricted to the declared late-noise captures')
    generator, critic, prior = fit.modules(state)
    if refined is not None:
        critic.load_state_dict(refined)
    grows, drows = tensors['grows'], tensors['drows']
    # This control includes a cloned ordinary Adam proposal and the original
    # own-field bound, but it never supplies the nonlocal direction or score.
    native, native_endpoint, _ = fit.g_proposal(state, critic, grows[0], gan, step)
    if refined_receipt is not None:
        parity = native == refined_receipt['g_after']
    else:
        expected = captured['post_bounded_g']
        reference = dict(generator=fit.unwrapped(expected['generator']),
                         prior=expected['prior'], optimizer=expected['optimizer_g'])
        parity = _sha(native_endpoint) == _sha(reference)
    if not parity:
        raise RuntimeError(f'{name}: declared native G control differs')
    model_hash = _sha((generator.state_dict(), critic.state_dict(), prior.state_dict()))
    with torch.no_grad():
        support = generator(prior.z).detach().clone()
        fakes = [generator(prior.z[row['indices']]) + row['sigma']*row['noise']
                 for row in grows[:9]]
        baseline = loss(critic, fakes[0], grows[0]['real'], gan)
        if float(baseline) != native['loss']:
            raise RuntimeError('native G loss is not bitwise equal')
        selection, target, scores = select(critic, fakes[0], grows[0],
                                           drows[0]['real'], len(support), gan)
        donor = selection['donor']
        after = support.clone()
        if selection['selected']:
            after[donor] = target
        heldout = []
        for index, (batch, fake) in enumerate(zip(grows[1:9], fakes[1:9]), start=1):
            proposed = replace_fake(fake, batch, donor, target)
            old = float(loss(critic, fake, batch['real'], gan))
            new = float(loss(critic, proposed, batch['real'], gan))
            heldout.append(dict(bank=index, before=old, after=new, decreased=new<old))
    d_losses = []
    for index in (0, 8):
        row = drows[index]
        with torch.no_grad():
            before_fake = generator(prior.z[row['indices']]) + .029*row['noise']
            new_fake = replace_fake(before_fake, dict(row, sigma=.029), donor, target)
        values = {}
        for label, fake in (('before', before_fake), ('after_selected_proposal', new_fake)):
            total, logistic, penalty = fit.d_loss(critic, dict(real=row['real'], fake=fake),
                                                gan, cap, step)
            values[label] = dict(total=float(total.detach()), logistic=float(logistic.detach()),
                                 penalty=float(penalty.detach()))
        d_losses.append(dict(bank=index, **values))
    # All center/quality reads happen after the sole proposal is selected.
    means = mode_hold.ring_means()
    indices, noise = fixed_draw(step, support)
    before_grade = score_support(support, indices, noise, means)
    after_grade = score_support(after, indices, noise, means)
    allocation = lambda cloud: torch.bincount(torch.cdist(cloud, means).argmin(1), minlength=8).tolist()
    result = dict(name=name, step=step, native_control_exact=parity,
        native_G_loss_exact=True, critic_state_sha256=_sha(critic.state_dict()),
        data_bank_sha256=_sha(drows[0]['real']), selection=selection,
        heldout=heldout, all_eight_heldout_decrease=all(row['decreased'] for row in heldout),
        heldout_mean_before=sum(row['before'] for row in heldout)/8,
        heldout_mean_after=sum(row['after'] for row in heldout)/8,
        D_losses=d_losses, original_native_G_update=native,
        grade_before=before_grade, grade_after=after_grade,
        allocation_before=allocation(support), allocation_after=allocation(after),
        output_displacement=float((after-support).norm()),
        supports=dict(before=support.tolist(), after=after.tolist()),
        input_models_unchanged=model_hash==_sha((generator.state_dict(), critic.state_dict(), prior.state_dict())),
        classifier='diagnostic only; no neural landing or acquisition claim')
    if not result['input_models_unchanged']:
        raise RuntimeError('read-only assay changed its model copies')
    return result, dict(scores=scores, before=support, after=after, target=target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs = {name: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for name, path in INPUTS.items()}
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    declaration = dict(method=METHOD, inputs=dict(zip(INPUTS, INPUTS.values())), input_sha256=inputs,
        sources=sources, states=[1530, 472], proposal='one of12 donors to one of128 native D-real samples',
        selection='minimum unchanged native-paired Rp-logistic G loss, fixed .15 stencil and output noise',
        heldout='next eight separate saved-stream G banks; evaluation only, no reselection or tuning',
        warm_D='captured PR84 accepted D plus separately archived1024-pair40/80 local D fit',
        cold_D='captured best finite D used by repaired G472; no new fit; original fit was nonconverged',
        frozen_stencil_width=fit.WIDTH, parameter_updates=0, outer_training=False,
        quality_controls_proposal=False, shared_gate_eligible=False,
        warm_rejection='reject frozen-critic global search if the selected move reduces warm mode coverage')
    for name in SOURCES:
        path = args.output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    started = time.perf_counter()
    warm = torch.load(BytesIO(read(ROOT/INPUTS['warm'])), weights_only=True)
    cold = torch.load(BytesIO(read(ROOT/INPUTS['cold'])), weights_only=True)
    refined = torch.load(BytesIO(read(ROOT/INPUTS['refined'])), weights_only=True)[1530]
    refined_receipt = json.loads(read(ROOT/INPUTS['refined_receipt']))
    input_hash = _sha((warm, cold, refined, refined_receipt))
    caller_rng = torch.get_rng_state().clone()
    config = json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    recipe, _, _ = declared_recipe(config)
    rows, tensors = [], {}
    with torch.random.fork_rng(devices=[]):
        for name, step, data, extra in (
            ('warm1530_native_D', 1530, warm, {}),
            ('warm1530_archived_refined_D', 1530, warm,
             dict(refined=refined['critic'], refined_receipt=refined_receipt)),
            ('cold472_best_finite_D', 472, cold, {}),
        ):
            row, tensor = run_case(name, step, data, recipe.make_loss(), recipe.make_gradient_penalty(), **extra)
            rows.append(row); tensors[name] = tensor
            print(json.dumps(dict(event='CASE_DONE', name=name, selection=row['selection'],
                before=row['grade_before'], after=row['grade_after'],
                all_eight_heldout_decrease=row['all_eight_heldout_decrease'])), flush=True)
    if input_hash != _sha((warm, cold, refined, refined_receipt)) or not torch.equal(caller_rng, torch.get_rng_state()):
        raise RuntimeError('assay changed saved input or caller RNG')
    result = dict(declaration=declaration, rows=rows, seconds=time.perf_counter()-started,
                  global_rng_unchanged=True, input_payloads_unchanged=True, shared_gate_eligible=False)
    torch.save(tensors, args.output/'tensors.pt')
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
