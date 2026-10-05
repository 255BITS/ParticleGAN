"""Fixed-input NoisyParticlePrior compatibility around the ordinary host laws.

The six-task declaration is owned by noisy_prior_tier1.  This module certifies
only implemented owners; it never substitutes a selected-state observer for
the original live measurement or changes a numeric bound.
"""
from copy import deepcopy
import math
import os
import json
from pathlib import Path

from .contracts import atomic_json, stable_hash

SUPPORTED = frozenset({'gaussian1d_acquisition', 'two_pole', 'ae_gan_hold', 'ring16_acquisition'})
BLOCKED = {
    'unused_token_hold': 'fixed SharedSlotStudent owns shared2/slot2x2 and feeds only concept1 to the GAN plus protected unused hold; no latent N12 bank is sampled, and N2 slot alias fails unchanged independent Atlas birth minimum; population/semantics/control changes are outside this prior-only task',
    'five_word_joint_acquisition': 'the original N5 joint host declares conditional rows rejected by active independent Atlas controls, and reference birth/death requires k>=4 while N5 gives k<=3',
}


def _parent(task, root=None):
    from .noisy_prior_tier1 import validate
    binding = validate(task, root=root)
    parent = deepcopy(task)
    for key in ('task_cohort', 'prior_substitution_parent', 'preflight_blockers', 'field_ownership'):
        parent.pop(key, None)
    parent['id'] = binding['parent_task_id']
    from .noisy_prior_tier1 import PARENTS
    parent['execution']['prior']['kind'] = PARENTS[parent['id']]['prior_kind']
    return parent


def supports_candidate(candidate):
    return (isinstance(candidate, dict) and candidate.get('recipe_preset') == 'atlas'
            and candidate.get('recipe_overrides', {}) == {}
            and candidate.get('extensions', {}) == {}
            and candidate.get('host_adaptation') is None
            and not candidate.get('implementation')
            and candidate.get('initializer', 'deterministic_orthogonal') == 'deterministic_orthogonal')


def blockers(task, candidate, root=None):
    try:
        parent = _parent(task, root)
        if not supports_candidate(candidate):
            return [task['id'] + ': the unchanged Atlas preset and fixed task inputs are required']
        if parent['id'] == 'ae_gan_hold':
            from .atlas_noisy_ae import blockers as ae_blockers
            return ae_blockers(task, candidate, root=root)
        reason = BLOCKED.get(parent['id'])
        return [task['id'] + ': ' + reason] if reason else []
    except (ValueError, KeyError, TypeError, OSError) as error:
        return [task.get('id', '<task>') + ': ' + str(error)]


def task_resources(task):
    parent = _parent(task)
    if parent['id'] == 'two_pole':
        return {'num_particles': 12, 'z_dim': 1, 'batch_size': 12}
    if parent['id'] == 'ae_gan_hold':
        return {'num_particles': 12, 'z_dim': 2, 'batch_size': 64}
    spec = parent['execution'].get('host_definition', {})
    if parent['adapter'] == 'transfer_vector':
        return {'num_particles': spec['particles'], 'z_dim': spec['z_dim'], 'batch_size': spec['batch']}
    # These paths remain BLOCKED before any constructor/reservation.
    return {}


def supporting_source_paths(task):
    from .noisy_prior_tier1 import is_noisy_task, validate, VARIANT_DIRECTORY, PROTOCOL_PATH, VIEW_PATH
    if not is_noisy_task(task):
        return ()
    parent = validate(task)['parent_task_id']
    return (str(VARIANT_DIRECTORY / (task['id'] + '.json')),
            'configs/forge/tasks/' + parent + '.json', str(PROTOCOL_PATH), str(VIEW_PATH))


def live_evaluation_state(context, trainer):
    """Cover optimizer/policy/RNG, actual gradients and module training modes.

Only the original external evaluation streams are excluded by the maintained
projector. No sampler, selected-state snapshot or owner mutation runs here.
"""
    from .policy_adapters import evaluation_state
    modules = {'generator': trainer.G, 'discriminator': trainer.D, 'prior': trainer.prior,
               'ema_generator': trainer.policy.ema_G, 'ema_prior': trainer.policy.ema_prior}
    return {'state': evaluation_state(context.state_dict()),
            'modes': {name: module.training for name, module in modules.items() if module is not None},
            'gradients': {name: [p.grad for p in module.parameters()] for name, module in modules.items()
                          if module is not None}}


def restore_live_owner(trainer):
    """Restore the public fast storage before the original live observation.

GANTrainer.step already finished its unchanged full policy lifecycle and may
have installed served averages in the live module storage.  The same public
owner's release is normally done at the next begin_step.  Doing that release
at this completed boundary supplies the explicitly declared live reader;
the averages, kernel, selector, clocks and optimizer state stay owned.
"""
    from particlegan.noisy_particle_prior import NoisyParticlePrior
    if (type(trainer.prior) is not NoisyParticlePrior
            or trainer.policy.prior is not trainer.prior
            or trainer.policy.table is not trainer.prior.z
            or trainer.policy._phase != 'ready'
            or trainer.completed_steps != trainer.policy.completed_steps):
        raise ValueError('the actual ready Noisy prior/table/public lifecycle must own the live reader')
    trainer.policy._serve_release()
    if trainer.policy._fast is not None:
        raise ValueError('original live reader still contains swapped served parameters')


def initialize_vector_receipts(context, trainer, output, task):
    """Record the real constructed zero-clock owner before the first update."""
    from dataclasses import asdict
    from particlegan.noisy_particle_prior import NoisyParticlePrior
    from .policy_adapters import typed_state_digest
    parent = _parent(task)
    if parent['id'] not in {'gaussian1d_acquisition', 'ring16_acquisition'}:
        raise ValueError('only the original scalar vector constructors use this receipt')
    if (type(trainer.prior) is not NoisyParticlePrior
            or trainer.recipe is not context.recipe
            or trainer.policy.prior is not trainer.prior
            or trainer.policy.table is not trainer.prior.z
            or trainer.completed_steps != 0 or trainer.policy.completed_steps != 0
            or trainer.policy._phase != 'ready'
            or trainer.opt_g.state or trainer.opt_d.state):
        raise ValueError('fresh real zero-update Noisy vector owner required')
    kernel = trainer.prior.kernel_contract()
    if kernel['sigma'] != float(trainer.prior.sigma) or not math.isclose(
            kernel['sigma'], task['execution']['prior']['sigma'], rel_tol=1e-7, abs_tol=0.):
        raise ValueError('actual fixed kernel differs from the parent-width prior declaration')
    source = {'task_id': task['id'], 'parent_task_id': parent['id'],
              'pid': os.getpid(), 'recipe': asdict(context.recipe),
              'initialization': deepcopy(context.initialization),
              'prior': kernel, 'rng': context.streams.manifest(),
              'completed_steps': 0, 'phase': 'ready',
              'optimizer_state_entries': {'generator': 0, 'discriminator': 0},
              'initial_state_sha256': typed_state_digest(context.state_dict())}
    output = Path(output)
    atomic_json(output / 'INITIALIZATION.json', {'schema': 'forge_noisy_prior686_initialization_v1', **source})
    start = {'schema': 'forge_noisy_prior686_model_started_v1', 'task_id': task['id'],
             'pid': os.getpid(), 'completed_steps': 0, 'initialization_sha256': stable_hash(source)}
    atomic_json(output / 'MODEL_STARTED.json', start)
    print(json.dumps({'event': 'actual_model_started', **start}, sort_keys=True), flush=True)


def observation_receipt(task, policy):
    """Describe the actual original live reader without another draw."""
    from particlegan.noisy_particle_prior import NoisyParticlePrior
    parent = _parent(task)
    if (type(policy.prior) is not NoisyParticlePrior
            or policy.table is not policy.prior.z or policy._phase != 'ready'
            or policy._fast is not None):
        raise ValueError('the original live reader must own the actual ready Noisy table')
    return {'task_id': task['id'], 'parent_task_id': parent['id'],
            'completed_steps': policy.completed_steps, 'observed': True,
            'policy_owner': 'particlegan.UpdatePolicy', 'weights': 'live',
            'sampling_law': task['evaluation']['sampling_law'],
            'eval_output_noise': task['evaluation']['eval_output_noise'],
            'prior': policy.prior.kernel_contract(), 'table_alias_preserved': True,
            'sampling_calls_added_by_receipt': 0}


def validate_evidence(task, evidence):
    """Fail closed on ownership gaps, then leave every numerical gate intact."""
    try:
        parent = _parent(task)
        if parent['id'] == 'ae_gan_hold':
            from .atlas_noisy_ae import validate_evidence as ae_validate_evidence
            return ae_validate_evidence(task, evidence)
        if parent['id'] not in SUPPORTED:
            return {'status': 'BLOCKED', 'reason': BLOCKED[parent['id']]}
        if not isinstance(evidence, dict):
            raise ValueError('Noisy evidence must be an object')
        steps = task['execution']['steps']
        clocks = sorted({math.ceil(i * steps / 24) for i in range(1, 25)})
        controls = evidence.get('policy_controls', {})
        if (controls.get('cohort') != task['task_cohort']
                or controls.get('completed_steps') != steps
                or not controls.get('implementation_observed')
                or not controls.get('requested_owners_bound')
                or not controls.get('lifecycle', {}).get('complete')):
            return {'status': 'INCOMPLETE', 'reason': 'missing complete actual Noisy public lifecycle/owners'}
        rows, purity = evidence.get('policy_observations', []), evidence.get('policy_purity', [])
        if ([row['completed_steps'] for row in rows] != clocks
                or [row['completed_steps'] for row in purity] != clocks):
            return {'status': 'INCOMPLETE', 'reason': 'missing original24 actual observation/purity clocks'}
        expected = task['evaluation']
        for row in rows:
            kernel = row['prior']
            if (row.get('observed') is not True or row.get('table_alias_preserved') is not True
                    or row.get('task_id') != task['id'] or row.get('parent_task_id') != parent['id']
                    or row.get('policy_owner') != 'particlegan.UpdatePolicy'
                    or row.get('weights') != expected['scoring_weights']
                    or row.get('sampling_law') != expected['sampling_law']
                    or row.get('eval_output_noise') != expected['eval_output_noise']
                    or row.get('sampling_calls_added_by_receipt') != 0
                    or kernel.get('kind') != 'noisy_particle_cloud'
                    or kernel.get('code_path') != 'particlegan.noisy_particle_prior.NoisyParticlePrior'
                    or kernel.get('standardize') is not False or kernel.get('learned_width') is not False
                    or kernel.get('sigma_units') != 'raw_latent_coordinates'
                    or kernel.get('row_weights') != 'uniform'
                    or kernel.get('zero_sigma_consumes_no_kernel_rng') is not True
                    or not math.isclose(kernel['sigma'], task['execution']['prior']['sigma'], rel_tol=1e-7, abs_tol=0.)):
                return {'status': 'INVALID', 'reason': 'actual Noisy prior or original live sampling law differs'}
        if any(row.get('pure') is not True or row.get('before_sha256') != row.get('after_sha256') for row in purity):
            return {'status': 'INVALID', 'reason': 'original live observation changed full training state'}
        return None
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        return {'status': 'INVALID', 'reason': 'malformed fixed-input Noisy evidence: ' + str(error)}
