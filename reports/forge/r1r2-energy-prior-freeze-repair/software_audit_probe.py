"""Pure prior-policy auditor fixtures; no model/optimizer/Recipe construction or draw.

Explicit synthetic tensors test acceptance and rejection laws, not qualification.
Run in the declared venv with PYTHONPATH=., no bytecode and one CPU thread.
"""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def load(name):
    path = OUT / name
    ast.parse(path.read_text(), filename=str(path))
    compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('fixture_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self):
        self.names = []

    def accept(self, name, function):
        function()
        self.names.append(name)

    def reject(self, name, function):
        try:
            function()
        except (ValueError, RuntimeError, AssertionError, KeyError):
            self.names.append(name)
        else:
            raise AssertionError(name)


def recipe(policy):
    return dict(prior_update=policy, energy_distance_weight=1., loss='non_saturating',
        amsgrad=True, optimizer_family='adam', lr=.0085, lr_schedule='constant',
        betas=(.5, .99), prior_betas=(0., .99), reg_arm='a_r1r2', reg_coeff=.1, reg_every=1,
        prior_reg=0., prior_l2=0., prior_proximal_strength=0., latent_damping_max_rate=0.)


def optimizer(players, *, steps=2, critic=False):
    groups, state = [], {}
    for index, player in enumerate(players):
        groups.append(dict(params=[index], lr=.0085, betas=[0. if player == 'P' else .5, .99],
                           eps=1e-8, amsgrad=True))
        state[index] = dict(step=torch.tensor(float(steps)), exp_avg=torch.tensor([.1]),
                            exp_avg_sq=torch.tensor([.2]), max_exp_avg_sq=torch.tensor([.3]))
    result = dict(state=state, param_groups=groups)
    if critic:
        result['regularizer'] = dict(record=dict(calls=steps, observed_steps=steps, anchor_started=False),
                                     ema=None, guard=None)
    return result


def measured_policy(a, policy):
    r = a.complete_recipe(recipe(policy))
    return dict(contract='recipe_owned_v1', update=policy, regularizer=r['prior_regularizer'],
                weight=0., target_std=r['prior_reg_target_std'], eps=r['prior_reg_eps'], l2=0.,
                applicability='latent_prior', trainable_locations=policy == 'learned')


def native(a, policy):
    frozen = policy == 'frozen'
    table = dict(z=torch.tensor([[.2]]))
    saved = dict(recipe=recipe(policy), trainer=dict(recipe=recipe(policy),
        completed_steps=2, models=dict(G=dict(weight=torch.tensor([.3])), D=dict(weight=torch.tensor([.4])), prior=table),
        requires_grad=dict(G=dict(weight=True), D=dict(weight=True), prior={} if frozen else dict(z=True)),
        optimizers=[optimizer(['G'] if frozen else ['G', 'P']), optimizer(['D'], critic=True)]))
    row = dict(evidence=dict(guards=dict(optimizer_updates=dict(generator=2, discriminator=2, prior=0 if frozen else 2)),
                             prior_policy=measured_policy(a, policy)))
    comparison = dict(initial_tensors_sha256=dict(prior=a.state_digest(table)))
    return saved, row, comparison


def word(a, policy):
    frozen = policy == 'frozen'
    saved, row, comparison = native(a, policy)
    state = saved['trainer']
    state['models'] = dict(generator=dict(weight=torch.tensor([.3])), critic=dict(weight=torch.tensor([.4])),
                          encoder=dict(weight=torch.tensor([.5])), prior=state['models']['prior'])
    state['requires_grad'] = dict(generator=dict(weight=True), critic=dict(weight=True), encoder=dict(weight=True),
                                 prior={} if frozen else dict(z=True))
    state.update(table_requires_grad=not frozen, table=state['models']['prior']['z'],
                 roles=[['generator', 'encoder'] + ([] if frozen else ['table']), ['critic']],
                 optimizers=[optimizer(['G', 'E'] if frozen else ['G', 'E', 'P']), optimizer(['D'], critic=True)])
    row['evidence']['guards']['optimizer_updates']['encoder'] = 2
    return dict(fixture=dict(recipe=recipe(policy), api_state=state)), row, comparison


def behavior(a, policy, task):
    frozen, sampled = policy == 'frozen', task == 'ae_gan_hold'
    g = optimizer(['G'])
    parameters = dict(discriminator=[dict(requires_grad=True, representation='network')])
    models, bindings = {}, []
    if sampled:
        parameters.update(generator=[dict(requires_grad=True, representation='network')],
                          encoder=[dict(requires_grad=True, representation='network')],
                          prior=[] if frozen else [dict(requires_grad=True, representation='latent_prior_locations')])
        g['param_groups'][0]['params'].append(1)
        g['state'][1] = deepcopy(g['state'][0])
        models['prior0'] = dict(z=torch.tensor([[.2]]))
    elif task == 'two_pole':
        parameters['prior'] = [dict(requires_grad=True, representation='direct_sample_coordinates')]
    else:
        parameters['generator'] = [dict(requires_grad=True, representation='network')]
    parts = [g] + ([optimizer(['P'])] if sampled and not frozen else [])
    for i, opt in enumerate(parts + [optimizer(['D'], critic=True)]):
        direct = task == 'two_pole' and i == 0
        latent = sampled and i == 1 and not frozen
        bindings.append(dict(current_group_betas=list(opt['param_groups'][0]['betas']),
            prior_betas_consumed=latent, prior_lr_mult_consumed=latent,
            representation='direct_sample_coordinates' if direct else 'latent_prior_locations' if latent else 'network',
            counter_role='prior' if direct or latent else 'generator' if i == 0 else 'discriminator',
            direct_particle_response=dict(installed=False)))
    saved = dict(models=models, role_parameters=parameters,
        optimizers=dict(generator=parts, discriminator=optimizer(['D'], critic=True)),
        applied=dict(recipe=recipe(policy), optimizer_group_bindings=bindings))
    counts = dict(discriminator=2)
    if sampled:
        counts.update(generator=2, encoder=2, prior=0 if frozen else 2)
    elif task == 'two_pole':
        counts['prior'] = 2
    else:
        counts['generator'] = 2
    row = dict(evidence=dict(guards=dict(optimizer_updates=counts)))
    if sampled:
        row['evidence']['prior_policy'] = measured_policy(a, policy)
    comparison = dict(initial_tensors_sha256={name: a.state_digest(value) for name, value in models.items()})
    return saved, row, comparison


def policy_fixtures(a, helper, checks):
    factories = [('gaussian1d_smoke', native), ('ring16_acquisition', native), ('five_word_joint_smoke', word),
                 *[(task, lambda a, policy, task=task: behavior(a, policy, task))
                   for task in ('ae_gan_hold', 'two_pole', 'unused_token_hold')]]
    for task, factory in factories:
        for policy, role in [('learned', 'energy_u'), ('frozen', 'frozen_prior')]:
            saved, row, comparison = factory(a, policy)
            check = lambda: a.prior_policy_binding(saved, row, task, role, helper, comparison=comparison)
            checks.accept(task + '_' + policy + '_actual_ownership', check)
            checks.accept(task + '_' + policy + '_native_ams_histories', lambda:
                a.optimizer_roles(saved, task, role, helper))
            checks.accept(task + '_' + policy + '_positive_every_update_R1_R2', lambda:
                a.score_and_penalty_binding(saved, task, role, helper, ordinary_steps=2))
            if task in {'gaussian1d_smoke', 'ring16_acquisition'}:
                checks.accept(task + '_' + policy + '_stateless_energy_retained', lambda:
                    a.energy_receipt(saved, row, task, role, helper, steps=2))
            original = check()
            if task in {'two_pole', 'unused_token_hold'}:
                assert original['prior_optimizer_groups'] == 0 and original['full_G_D_E_or_direct_generator_updates'] == 2
            elif policy == 'frozen':
                assert original['prior_optimizer_updates'] == 0 and original['frozen_table_matches_actual_initial_state']
        saved, row, comparison = factory(a, 'frozen')
        def binding(s=saved, r=row, c=comparison):
            return a.prior_policy_binding(s, r, task, 'frozen_prior', helper, comparison=c)
        changed = deepcopy(saved)
        if 'trainer' in changed:
            changed['trainer']['requires_grad']['prior'] = dict(z=True)
        elif 'fixture' in changed:
            changed['fixture']['api_state']['table_requires_grad'] = True
        elif task == 'ae_gan_hold':
            changed['role_parameters']['prior'] = [dict(requires_grad=True, representation='latent_prior_locations')]
        elif task == 'two_pole':
            changed['role_parameters']['prior'][0]['requires_grad'] = False
        else:
            changed['role_parameters']['generator'][0]['requires_grad'] = False
        checks.reject(task + '_wrong_actual_trainability', lambda: binding(s=changed))
        if task not in {'two_pole', 'unused_token_hold'}:
            bad = deepcopy(comparison)
            bad['initial_tensors_sha256'][next(iter(bad['initial_tensors_sha256']))] = '0' * 64
            checks.reject(task + '_moved_frozen_table', lambda: binding(c=bad))
            badrow = deepcopy(row); badrow['evidence']['guards']['optimizer_updates']['prior'] = 2
            checks.reject(task + '_paid_prior_updates', lambda: binding(r=badrow))
            badrow = deepcopy(row); badrow['evidence']['prior_policy']['trainable_locations'] = True
            checks.reject(task + '_false_policy_receipt', lambda: binding(r=badrow))
            changed = deepcopy(saved)
            if 'trainer' in changed:
                changed['trainer']['optimizers'][0]['param_groups'].append(optimizer(['P'])['param_groups'][0])
            elif 'fixture' in changed:
                changed['fixture']['api_state']['optimizers'][0]['param_groups'].append(optimizer(['P'])['param_groups'][0])
            else:
                changed['optimizers']['generator'].append(optimizer(['P']))
            checks.reject(task + '_prior_optimizer_group_retained', lambda: binding(s=changed))
        checks.reject(task + '_truncated_D_updates', lambda: a.score_and_penalty_binding(saved, task, 'frozen_prior', helper, ordinary_steps=3))
    aliased, _, _ = native(a, 'learned')
    aliased['trainer']['optimizers'][0]['param_groups'][1]['params'] = [0]
    checks.reject('aliased_G_P_optimizer_parameter', lambda: a.optimizer_roles(aliased, 'ring16_acquisition', 'energy_u', helper))
    signed = dict(schema_version=1, consumer='output_energy_u_v1', weight=1., calls=2, active_calls=2,
                  loss_sum=-.2, minimum_loss=-.2, maximum_loss=0.)
    for task, factory in [('ae_gan_hold', lambda: behavior(a, 'frozen', 'ae_gan_hold')),
                          ('five_word_joint_smoke', lambda: word(a, 'frozen'))]:
        saved, row, _ = factory()
        binding = dict(version=1, consumer='output_energy_u_v1',
            native_loop='benchmarks.toy_audit.api_images.WordFixture.step' if 'fixture' in saved else
                        'experiments.forge.behavior_adapters.run_behavior',
            source='experiments/forge/energy_adapters.py', source_sha256=a.file_hash(ROOT / 'experiments/forge/energy_adapters.py'),
            objective_source='particlegan/energy_distance.py', objective_source_sha256=a.file_hash(ROOT / 'particlegan/energy_distance.py'))
        row['evidence'].update(component_energy=deepcopy(signed))
        row['evidence']['guards'].update(component_energy_requested=True, component_energy_active_calls=2, hooks_exercised=True)
        if 'fixture' in saved:
            saved['fixture']['component_energy'] = deepcopy(signed)
            row['evidence']['energy_binding'] = binding
        else:
            saved['component_energy'] = deepcopy(signed)
            saved['applied'].update(component_energy=deepcopy(signed), energy_binding=binding)
        checks.accept(task + '_frozen_signed_energy_full_G_calls', lambda:
            a.energy_receipt(saved, row, task, 'frozen_prior', helper, steps=2))
        checks.reject(task + '_frozen_energy_truncation', lambda:
            a.energy_receipt(saved, row, task, 'frozen_prior', helper, steps=3))
    initial, row, comparison = native(a, 'frozen')
    clock_initial = dict(initialization=dict(prior=dict(initial_state_sha256=comparison['initial_tensors_sha256']['prior'])))
    checks.accept('clock_frozen_warmup_prior_matches_constructor_bytes', lambda:
        a.prior_policy_binding(initial, row, a.CLOCK, 'frozen_prior', helper, clock_initial=clock_initial))
    bad = deepcopy(clock_initial); bad['initialization']['prior']['initial_state_sha256'] = '0' * 64
    checks.reject('clock_changed_constructor_prior_rejected', lambda:
        a.prior_policy_binding(initial, row, a.CLOCK, 'frozen_prior', helper, clock_initial=bad))


def identity_fixtures(a, density, checks):
    parent, child = a.complete_recipe(recipe('learned')), a.complete_recipe(recipe('frozen'))
    checks.accept('global_and_task_only_prior_delta', lambda: a.prior_recipe_delta(parent, child))
    for name, field, value in [('energy', 'energy_distance_weight', 0.), ('gamma', 'reg_coeff', .01),
            ('prior_regularizer', 'prior_reg', .1), ('architecture', 'num_particles', 8), ('loss', 'loss', 'relativistic'),
            ('prior_state', 'prior_update', 'learned')]:
        changed = deepcopy(child); changed[field] = value
        checks.reject('additional_' + name + '_delta', lambda changed=changed: a.prior_recipe_delta(parent, changed))
    assert density.complete_recipe(recipe('frozen')) == child
    checks.names.append('density_recipe_completion_agrees_without_constructor')
    old = dict(candidates=dict(energy_u=dict(candidate='fixed-energy-card', candidate_revision='r', source_digest=a.SOURCE,
        card=dict(study='original'))), declarations={'configs/forge/ideas/fixed-energy-card.json': 'card-bytes'})
    new = deepcopy(old); new['candidates']['energy_u']['card']['study'] = 'new-cached-study'
    with patch.object(a, 'file_hash', lambda _: 'card-bytes'):
        checks.accept('new_study_same_cached_parent_projection', lambda: a.cached_parent_identity(ROOT, new, old))
        bad = deepcopy(new); bad['candidates']['energy_u']['candidate_revision'] = 'changed'
        checks.reject('cached_revision_changed', lambda: a.cached_parent_identity(ROOT, bad, old))
    with patch.object(a, 'file_hash', lambda _: 'changed-card-bytes'):
        checks.reject('cached_card_changed', lambda: a.cached_parent_identity(ROOT, new, old))
    execution = 'e' * 40
    checks.accept('cached_original_B_origin', lambda: a.certificate_origin('energy_u', a.PARENT_EXECUTION, execution))
    checks.accept('new_treatment_E_origin', lambda: a.certificate_origin('frozen_prior', execution, execution))
    checks.reject('cached_parent_relabelled_E', lambda: a.certificate_origin('energy_u', execution, execution))
    checks.reject('child_relabelled_B', lambda: a.certificate_origin('frozen_prior', a.PARENT_EXECUTION, execution))
    receipts = {role: dict(constructor_initialization_sha256='a', initial_streams_sha256='b',
        initial_trainer_streams_sha256='c', own_warmed_tensors_sha256=role) for role in a.ROLES}
    checks.accept('clock_distinct_own_warm_states_accepted', lambda: a.matched_clock_receipts(receipts))
    for field in ('constructor_initialization_sha256', 'initial_streams_sha256', 'initial_trainer_streams_sha256'):
        bad = deepcopy(receipts); bad['frozen_prior'][field] = 'changed'
        checks.reject('clock_' + field + '_changed', lambda bad=bad: a.matched_clock_receipts(bad))
    checks.reject('clock_missing_peer', lambda: a.matched_clock_receipts(dict(energy_u=receipts['energy_u'])))
    ordinary = dict(initial_tensors_sha256=dict(prior='a' * 64), actual_target_batches=dict(
        critic=dict(sha256='b' * 64, batches=2, samples=4), generator=dict(sha256='c' * 64, batches=2, samples=4)),
        initial_tensor_metadata=dict(prior=dict(z=dict(shape=[1, 1], dtype='torch.float32', device='cuda:0'))))
    peer = deepcopy(ordinary); peer['initial_tensor_metadata']['prior']['z']['device'] = 'cuda:1'
    checks.accept('cuda_worker_ordinal_only_normalized', lambda: a.matched_ordinary_comparisons(ordinary, peer))
    for field, value in [('device', 'cpu'), ('shape', [2, 1]), ('dtype', 'torch.float64')]:
        bad = deepcopy(peer); bad['initial_tensor_metadata']['prior']['z'][field] = value
        checks.reject('changed_initial_' + field, lambda bad=bad: a.matched_ordinary_comparisons(ordinary, bad))
    bad = deepcopy(peer); bad['actual_target_batches']['generator']['sha256'] = '0' * 64
    checks.reject('changed_actual_G_target_sequence', lambda: a.matched_ordinary_comparisons(ordinary, bad))
    signed = dict(schema_version=1, consumer='output_energy_u_v1', weight=1., calls=2, active_calls=2,
                  loss_sum=-.2, minimum_loss=-.2, maximum_loss=0.)
    checks.accept('negative_signed_energy_retained', lambda: a.validate_energy_state(signed, steps=2))
    checks.reject('truncated_signed_energy_coverage', lambda: a.validate_energy_state(signed, steps=3))
    for name, function in [('recipe', lambda: a.Recipe()), ('optimizer', lambda: torch.optim.Optimizer([], {})),
            ('torch_draw', lambda: torch.rand(1)), ('numpy_draw', lambda: np.random.normal()), ('python_draw', lambda: random.random())]:
        checks.reject('forbidden_' + name, function)


def main():
    a, density = load('audit_collection.py'), load('density_audit.py')
    helper, checks = a.load_helper(), Checks()
    torch.set_num_threads(1)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with a.forbid_live_execution(helper), torch.no_grad():
        policy_fixtures(a, helper, checks)
        identity_fixtures(a, density, checks)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0] and before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:] and torch.equal(before[2], after[2])
    print(json.dumps(dict(status='PASS', scope='explicit_synthetic_software_fixtures_not_qualification',
        checks=len(checks.names), names=checks.names, source_hashes={str((OUT / name).relative_to(ROOT)): a.file_hash(OUT / name)
            for name in ('audit_collection.py', 'density_audit.py', 'software_audit_probe.py')},
        imported_helper_sha256=dict(objective=a.HELPER_SHA, energy=a.ENERGY_HELPER_SHA, density=density.PINS),
        python_numpy_torch_cpu_rng_unchanged=True, model_optimizer_recipe_construction_training_sampling_draws=0,
        archived_grades_used=0), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
