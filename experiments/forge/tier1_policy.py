"""Task-owned, source-bound Tier 1 policy cohorts; historical parents stay intact.

The public independent row mechanism requires an equal-mass cloud. This is a
scoped task exception, not a trainer-family identity or a MoG pass. Numerical
settings and the complete family Recipe are never tuned by this transformation.
"""
from copy import deepcopy
from pathlib import Path

from .contracts import file_hash, stable_hash

COHORT = 'tier1_policy_selected_cloud_v1'
SUFFIX = '_' + COHORT
PARENTS = ('gaussian1d_acquisition', 'two_pole', 'unused_token_hold',
           'ae_gan_hold', 'ring16_acquisition', 'five_word_joint_acquisition', 'clockfree_audit')
SOURCES = ('particlegan/recipes.py', 'particlegan/policy.py', 'particlegan/training.py',
           'experiments/forge/api.py', 'experiments/forge/adapters.py',
           'experiments/forge/policy_adapters.py', 'experiments/forge/policy_behavior_adapters.py',
           'experiments/forge/tier1_policy.py', 'experiments/forge/sampling.py',
           'experiments/forge/views.py', 'experiments/forge/clockfree.py', 'experiments/forge/boundaries.py',
           'experiments/forge/mechanisms.py')
BLOCKED_HOSTS = {
    'unused_token_hold': 'shared/slot parameters require a conditional RoutedRows mechanism; independent Atlas/E22 cannot claim its protected-context law',
    'ae_gan_hold': 'the frozen posterior/encoder MoG law needs a routed AE formulation; an independent cloud policy is not that law',
    'five_word_joint_acquisition': 'the frozen five-row joint BiGAN needs complete generator/encoder/policy ownership and a separately scoped minimum-population contract',
}


def is_policy_task(task):
    return task.get('task_cohort') == COHORT


def _execution(task):
    return stable_hash({key: task.get(key) for key in
                       ('schema_version', 'adapter', 'execution', 'requires_capabilities', 'dependencies')})


def _variant(parent, pin, sources):
    task = deepcopy(parent)
    task.update(id=parent['id'] + SUFFIX, task_cohort=COHORT, policy_parent=deepcopy(pin))
    execution = task['execution']
    execution['policy_parent_definition'] = deepcopy(parent)
    execution['prior'].update(kind='particle_cloud', sigma=0., standardize=False, learnable=True,
        exception_reason='Independent Atlas/E22 row controls require an explicit equal-mass cloud; this scoped selected-policy variant supplies no original MoG/live credit.')
    from .priors import recipe_owned_prior
    if recipe_owned_prior(parent):
        execution['prior'].pop('learnable')
    execution['policy_contract'] = {
        'schema_version': 1, 'cohort': COHORT, 'owner': 'particlegan.UpdatePolicy',
        'lifecycle': 'ordered_public_update', 'row_semantics': 'independent',
        'execution_path': execution.get('execution_path', 'public_trainer'),
        'external_limit': 'task.execution.steps', 'schedule': 'preserve_resolved_recipe_none',
        'checkpoint': 'complete_public_policy_models_optimizers_streams_and_cursor',
        'sources': deepcopy(sources), 'recipe_overrides': {}, 'original_parent_credit': False}
    if parent['id'] == 'clockfree_audit':
        execution['policy_clock_horizon'] = 'independent_GANTrainer_max_steps_preserving_Recipe_total_steps_None'
    if parent['id'] == 'two_pole':
        execution['resources'] = {'num_particles': 12, 'z_dim': 1, 'batch_size': 12}
    execution['device'] = 'cuda'
    task['resources'].update(gpus=1, gpu_memory_mb=max(2048, task['resources']['gpu_memory_mb']))
    if 'allow_cpu' in task['resources']:
        task['resources']['allow_cpu'] = False
    evaluation = task['evaluation']
    evaluation.setdefault('sources', {}).update(sources)
    evaluation['sources']['configs/forge/tasks/' + parent['id'] + '.json'] = pin['task_sha256']
    evaluation['scoring_weights'] = 'state_selected'
    parameter = parent['id'] in ('two_pole', 'unused_token_hold')
    evaluation['policy_observation'] = {
        'schema_version': 1, 'weight_selector': 'state_selected',
        'sampler': 'served_snapshot' if parameter else 'GANTrainer.sample',
        'output_noise': False, 'latent_policy': 'not_applied_to_parameter_measurement' if parameter else 'actual_selected_public_policy',
        'row_selection': 'original_task_strategy', 'eval_streams': 'forge-rng-v1_isolated',
        'diagnostic_credit': False}
    if parent['id'] == 'two_pole':
        evaluation['policy_observation']['parameter_measurement'] = 'selected_table_and_critic_gradient'
    # Different cohort law names prevent a selected policy from impersonating
    # a parent clean/live sampler, even when the output-noise flag is false.
    evaluation['sampling_law'] = 'tier1_selected_' + parent['evaluation']['sampling_law']
    caps = ['served_sampling' if name == 'live_sampling' else
            'particle_cloud' if name == 'mog_prior' else name for name in task['requires_capabilities']]
    task['requires_capabilities'] = list(dict.fromkeys(caps + ['policy_controls', 'policy_serving']))
    return task


def materialize(parent, root):
    if parent['id'] not in PARENTS:
        raise ValueError('only the seven declared Tier 1 questions may be transformed')
    root = Path(root)
    path = root / 'configs/forge/tasks' / (parent['id'] + '.json')
    pin = {'id': parent['id'], 'task_sha256': file_hash(path),
           'execution_fingerprint': _execution(parent), 'evaluation_fingerprint': stable_hash(parent['evaluation'])}
    sources = {name: file_hash(root / name) for name in SOURCES}
    return _variant(parent, pin, sources)


def validate(task, *, root=None):
    if not is_policy_task(task):
        raise ValueError('explicit Tier 1 policy cohort is required')
    pin = task['policy_parent']
    parent = task['execution']['policy_parent_definition']
    if (parent['id'] not in PARENTS or pin['id'] != parent['id'] or
        pin['execution_fingerprint'] != _execution(parent) or
        pin['evaluation_fingerprint'] != stable_hash(parent['evaluation'])):
        raise ValueError('policy parent definition identity differs')
    sources = task['execution']['policy_contract']['sources']
    if set(sources) != set(SOURCES) or any(not isinstance(x, str) or len(x) != 64 for x in sources.values()):
        raise ValueError('complete fixed public-policy source identities are required')
    scientific = {key: value for key, value in task.items() if key not in {'field_ownership', 'preflight_blockers'}}
    if stable_hash(scientific) != stable_hash(_variant(parent, pin, sources)):
        raise ValueError('policy variant changed parent settings beyond its declared transformation')
    if root is not None:
        root = Path(root)
        parent_path = root / 'configs/forge/tasks' / (parent['id'] + '.json')
        if file_hash(parent_path) != pin['task_sha256']:
            raise ValueError('on-disk policy parent identity differs')
        for name, digest in sources.items():
            if file_hash(root / name) != digest:
                raise ValueError('policy source binding drift: ' + name)
    return parent


def blockers(task, recipe, *, root=None):
    try:
        parent = validate(task, root=root)
        reasons = []
        if recipe.continuous_policy is None or recipe.total_steps is not None:
            reasons.append('Tier 1 selected policy cohort requires the actual schedule-free public policy recipe')
        if recipe.row_policy != 'independent':
            reasons.append('independent Tier 1 policy task cannot silently adopt routed semantics')
        if parent['id'] in BLOCKED_HOSTS:
            reasons.append(BLOCKED_HOSTS[parent['id']])
        return [task['id'] + ': ' + reason for reason in reasons]
    except (KeyError, ValueError, TypeError, OSError) as error:
        return [task.get('id', '<task>') + ': ' + str(error)]


def validate_evidence(task, evidence):
    """Require actual ordered hook, owner, health and observation attestation."""
    controls = evidence.get('policy_controls', {})
    lifecycle = controls.get('lifecycle', {})
    if controls.get('cohort') != COHORT:
        return {'status': 'INVALID', 'reason': 'observed policy cohort differs from the scoped task'}
    if not controls.get('implementation_observed') or not controls.get('requested_owners_bound') or not lifecycle.get('complete'):
        return {'status': 'INCOMPLETE', 'reason': 'missing observed ordered public policy controls'}
    observations = evidence.get('policy_observations', [])
    if not observations or any(row.get('observed') is not True or row.get('policy_owner') != 'particlegan.UpdatePolicy' for row in observations):
        return {'status': 'INCOMPLETE', 'reason': 'missing actual selected-policy observations'}
    if not evidence.get('policy_purity') or not all(row.get('pure') is True for row in evidence['policy_purity']):
        return {'status': 'INVALID', 'reason': 'selected-policy observation changed training state'}
    declaration = task['evaluation']['policy_observation']
    if any(any(row.get(name) != value for name, value in declaration.items()) for row in observations):
        return {'status': 'INVALID', 'reason': 'actual selected sampler declaration differs'}
    if controls.get('completed_steps') != task['execution']['steps']:
        return {'status': 'INCOMPLETE', 'reason': 'policy execution lacks the full task horizon'}
    return None


def clock_measurement(parent, root):
    """Execute a property audit without asserting the recipe is clock-free."""
    if parent['id'] != 'clockfree_audit':
        raise ValueError('clock measurement requires the exact clockfree parent')
    task = deepcopy(parent)
    task['id'] = 'clockfree_audit_measurement_v1'
    task['execution']['clock_audit_scope'] = 'measure_known_dependencies'
    task['execution']['clock_audit_parent'] = {'id': parent['id'],
        'task_sha256': file_hash(Path(root) / 'configs/forge/tasks/clockfree_audit.json'),
        'execution_fingerprint': _execution(parent),
        'evaluation_fingerprint': stable_hash(parent['evaluation'])}
    task['evaluation'].setdefault('sources', {})['configs/forge/tasks/clockfree_audit.json'] = task['execution']['clock_audit_parent']['task_sha256']
    task['description'] = 'Measure the original four clock/state parity conditions and zero-dependency condition for an explicitly scheduled recipe; diagnostic FAIL grants no clock-free claim or qualification.'
    return task


def load_variants(root, parents):
    root = Path(root)
    variants = {}
    for path in sorted((root / 'configs/forge/task-variants' / COHORT).glob('*.json')):
        import json
        task = json.loads(path.read_text())
        # Stale physical sources/parents block this variant at preflight; they
        # must not prevent an unrelated ordinary view from loading its tasks.
        parent = validate(task)
        if parent['id'] not in parents:
            raise ValueError('policy variant names an unknown parent')
        variants[task['id']] = task
    if variants and {task['policy_parent']['id'] for task in variants.values()} != set(PARENTS):
        raise ValueError('the policy coverage cohort needs all seven parent questions')
    return variants


def write_declarations(root):
    import json
    from .contracts import atomic_json
    root = Path(root)
    parents = {name: json.loads((root / 'configs/forge/tasks' / (name + '.json')).read_text()) for name in PARENTS}
    for parent in parents.values():
        variant = materialize(parent, root)
        atomic_json(root / 'configs/forge/task-variants' / COHORT / (variant['id'] + '.json'), variant)
    atomic_json(root / 'configs/forge/tasks/clockfree_audit_measurement_v1.json', clock_measurement(parents['clockfree_audit'], root))
    view = {'schema_version': 1, 'id': 'tier1_policy_coverage', 'goal': 'discriminator_stability',
        'revision': 1, 'description': 'Separate Atlas/E22 selected-policy cloud Tier 1 coverage. Original live/MoG parents receive no pass credit; incompatible independent objectives retain explicit blockers.',
        'assignments': [{'task': name + SUFFIX, 'qualification_tier': 1, 'importance': 'required', 'order': index} for index, name in enumerate(PARENTS)],
        'eligibility': {}, 'reporting': {'family_totals': False}}
    atomic_json(root / 'configs/forge/views/tier1_policy_coverage.json', view)
    return view
