"""Task-owned, source-bound policy cohorts; historical parents stay intact.

The public independent row mechanism requires an equal-mass cloud. This is a
scoped task exception, not a trainer-family identity or a MoG pass. Numerical
settings and the complete family Recipe are never tuned by this transformation.

Each cohort is one explicit, registered transformation of named parent
questions. The original Tier 1 cohort keeps its exact declarations; later
cohorts reuse the same transformation with their own identity, sources and
explicit host blockers (see ``policy_coverage``). A cohort's ``measurement``
names its evaluation output-noise law: ``clean`` removes the learned output
noise from measured samples; ``as_served`` keeps it exactly as GANTrainer
serves it. An as-served cohort differs from its clean partner only in that
declared measurement; law-independent parents stay with the partner.
"""
from copy import deepcopy
from dataclasses import dataclass
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
CLOCK_HORIZON = 'independent_GANTrainer_max_steps_preserving_Recipe_total_steps_None'
SCHEDULE_FREE = 'preserve_resolved_recipe_none'
SERVED_OUTPUT_SIGMA = ('GANTrainer.sample(output_noise=True): GANTrainer.output_sigma() at the measured state '
                       '(learned sigma with its policy floor), drawn from the isolated evaluation stream')


@dataclass(frozen=True)
class PolicyCohort:
    """One registered transformation; its fields are part of variant identity."""
    id: str
    parents: tuple
    sources: tuple
    blocked_hosts: dict
    law_prefix: str
    label: str
    parameter_parents: frozenset = frozenset({'two_pole', 'unused_token_hold'})
    clock_parents: frozenset = frozenset({'clockfree_audit'})
    rename_dependencies: bool = False
    # Evaluation output-noise law of every measured sample (see module doc).
    measurement: str = 'clean'
    # Law-independent parents owned by the clean ``partner`` cohort. They are
    # neither duplicated nor re-run here; a gate dependency names the
    # partner's variant and publication shows the partner's single verdict.
    shared: frozenset = frozenset()
    partner: str | None = None
    # A cohort registered with the per-draw ledger must attest the measured
    # draws of every sample-measuring observation, under either law (an
    # as-served cohort always does). The original Tier 1 receipts predate the
    # ledger and keep their format.
    attest_draws: bool = False
    # Parents allowed to continue their own scoped producer's saved state
    # after its quality gate failed (user-authorized diagnostic continuation).
    diagnostic_continuations: frozenset = frozenset()

    def __post_init__(self):
        if self.measurement not in ('clean', 'as_served'):
            raise ValueError('policy cohort measurement must be clean or as_served')
        if self.measurement == 'as_served' and (
                self.parameter_parents or self.clock_parents or self.blocked_hosts
                or not self.partner or set(self.parents) & set(self.shared)):
            raise ValueError('an as-served cohort measures output samples only; '
                             'law-independent parents stay with its clean partner')
        if self.measurement == 'clean' and (self.shared or self.partner):
            raise ValueError('only an as-served cohort shares law-independent parents')
        if not set(self.diagnostic_continuations) <= set(self.parents):
            raise ValueError('a diagnostic continuation must be a declared parent of its cohort')

    @property
    def suffix(self):
        return '_' + self.id

    @property
    def directory(self):
        return Path('configs/forge/task-variants') / self.id

    def dependency_task(self, name, kind):
        """Same-law scoped producer; a shared gate names the clean partner."""
        if name in self.shared:
            if kind != 'gate':
                raise ValueError(f'{self.id}: a checkpoint must come from its own same-law producer, not shared {name}')
            return name + '_' + self.partner
        return name + self.suffix

    def sampling_law(self, parent_law):
        """Cohort law name; it can never equal a parent or another cohort's law."""
        if self.measurement == 'clean':
            return self.law_prefix + parent_law
        from .sampling import AS_SERVED_LAWS
        if parent_law not in AS_SERVED_LAWS:
            raise ValueError(f'{self.id}: {parent_law} draws no output samples; share its clean verdict instead')
        return self.law_prefix + AS_SERVED_LAWS[parent_law]

    def eval_output_noise(self, parent_noise):
        if self.measurement == 'clean':
            return parent_noise
        from .sampling import SERVED_LEARNED_NOISE
        return SERVED_LEARNED_NOISE


TIER1 = PolicyCohort(id=COHORT, parents=PARENTS, sources=SOURCES, blocked_hosts=BLOCKED_HOSTS,
                     law_prefix='tier1_selected_', label='Tier 1 selected policy cohort')


def cohorts():
    """Registered cohorts by id; later modules import this one, never the reverse."""
    from .policy_coverage import COVERAGE, SERVED
    return {TIER1.id: TIER1, COVERAGE.id: COVERAGE, SERVED.id: SERVED}


def task_cohort(task):
    """The registered cohort a declaration claims, or None for ordinary tasks."""
    if not isinstance(task, dict):
        return None
    return cohorts().get(task.get('task_cohort'))


def is_policy_task(task):
    return task_cohort(task) is not None


def schedule_free_policy(task):
    """A registered policy task preserving the recipe's schedule-free horizon."""
    return (is_policy_task(task) and
            task.get('execution', {}).get('policy_contract', {}).get('schedule') == SCHEDULE_FREE)


def diagnostic_continuation_allowed(task):
    """Only a registered cohort's declared parents continue after a failed gate.

    Ordinary tasks (including other checkpoint-repair intakes) keep their own
    continuation rules; this never widens them.
    """
    cohort = task_cohort(task)
    parent = task.get('policy_parent', {}).get('id') if isinstance(task, dict) else None
    return cohort is not None and parent in cohort.diagnostic_continuations


def _execution(task):
    return stable_hash({key: task.get(key) for key in
                       ('schema_version', 'adapter', 'execution', 'requires_capabilities', 'dependencies')})


def _variant(parent, pin, sources, cohort=TIER1):
    task = deepcopy(parent)
    task.update(id=parent['id'] + cohort.suffix, task_cohort=cohort.id, policy_parent=deepcopy(pin))
    execution = task['execution']
    if cohort.rename_dependencies:
        # Dependents consume their own scoped parent's state and gates, never
        # an original (live/MoG) parent's evidence.
        for dependency in task['dependencies']:
            dependency['task'] = cohort.dependency_task(dependency['task'], dependency['kind'])
        if 'continuation_of' in execution:
            execution['continuation_of'] = cohort.dependency_task(execution['continuation_of'], 'checkpoint')
    execution['policy_parent_definition'] = deepcopy(parent)
    execution['prior'].update(kind='particle_cloud', sigma=0., standardize=False, learnable=True,
        exception_reason='Independent Atlas/E22 row controls require an explicit equal-mass cloud; this scoped selected-policy variant supplies no original MoG/live credit.')
    from .priors import recipe_owned_prior
    if recipe_owned_prior(parent):
        execution['prior'].pop('learnable')
    execution['policy_contract'] = {
        'schema_version': 1, 'cohort': cohort.id, 'owner': 'particlegan.UpdatePolicy',
        'lifecycle': 'ordered_public_update', 'row_semantics': 'independent',
        'execution_path': execution.get('execution_path', 'public_trainer'),
        'external_limit': 'task.execution.steps', 'schedule': SCHEDULE_FREE,
        'checkpoint': 'complete_public_policy_models_optimizers_streams_and_cursor',
        'sources': deepcopy(sources), 'recipe_overrides': {}, 'original_parent_credit': False}
    if parent['id'] in cohort.clock_parents:
        execution['policy_clock_horizon'] = CLOCK_HORIZON
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
    parameter = parent['id'] in cohort.parameter_parents
    served = cohort.measurement == 'as_served'
    evaluation['policy_observation'] = {
        'schema_version': 1, 'weight_selector': 'state_selected',
        'sampler': 'served_snapshot' if parameter else 'GANTrainer.sample',
        'output_noise': served, 'latent_policy': 'not_applied_to_parameter_measurement' if parameter else 'actual_selected_public_policy',
        'row_selection': 'original_task_strategy', 'eval_streams': 'forge-rng-v1_isolated',
        'diagnostic_credit': False}
    if parent['id'] == 'two_pole':
        evaluation['policy_observation']['parameter_measurement'] = 'selected_table_and_critic_gradient'
    if served:
        # The single declared difference from the clean partner: measured
        # samples keep the learned output noise exactly as GANTrainer serves it.
        evaluation['policy_observation']['output_sigma'] = SERVED_OUTPUT_SIGMA
        evaluation['eval_output_noise'] = cohort.eval_output_noise(parent['evaluation']['eval_output_noise'])
    # Different cohort law names prevent a selected policy from impersonating
    # a parent clean/live sampler, even when the output-noise flag is false.
    evaluation['sampling_law'] = cohort.sampling_law(parent['evaluation']['sampling_law'])
    caps = ['served_sampling' if name == 'live_sampling' else
            'particle_cloud' if name == 'mog_prior' else name for name in task['requires_capabilities']]
    task['requires_capabilities'] = list(dict.fromkeys(caps + ['policy_controls', 'policy_serving']))
    return task


def materialize(parent, root, cohort=TIER1):
    if parent['id'] not in cohort.parents:
        if cohort is TIER1:
            raise ValueError('only the seven declared Tier 1 questions may be transformed')
        if parent['id'] in cohort.shared:
            raise ValueError(f'{parent["id"]} is law-independent; {cohort.id} shares the {cohort.partner} '
                             'verdict instead of duplicating it')
        raise ValueError(f'{parent["id"]} is not a declared parent of {cohort.id}')
    root = Path(root)
    path = root / 'configs/forge/tasks' / (parent['id'] + '.json')
    pin = {'id': parent['id'], 'task_sha256': file_hash(path),
           'execution_fingerprint': _execution(parent), 'evaluation_fingerprint': stable_hash(parent['evaluation'])}
    sources = {name: file_hash(root / name) for name in cohort.sources}
    return _variant(parent, pin, sources, cohort)


def validate(task, *, root=None):
    cohort = task_cohort(task)
    if cohort is None:
        raise ValueError('explicit Tier 1 policy cohort is required')
    pin = task['policy_parent']
    parent = task['execution']['policy_parent_definition']
    if (parent['id'] not in cohort.parents or pin['id'] != parent['id'] or
        pin['execution_fingerprint'] != _execution(parent) or
        pin['evaluation_fingerprint'] != stable_hash(parent['evaluation'])):
        raise ValueError('policy parent definition identity differs')
    sources = task['execution']['policy_contract']['sources']
    if set(sources) != set(cohort.sources) or any(not isinstance(x, str) or len(x) != 64 for x in sources.values()):
        raise ValueError('complete fixed public-policy source identities are required')
    scientific = {key: value for key, value in task.items() if key not in {'field_ownership', 'preflight_blockers'}}
    if stable_hash(scientific) != stable_hash(_variant(parent, pin, sources, cohort)):
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


def blocked_host(task):
    """The cohort's explicit ownership blocker for this parent, if any."""
    cohort = task_cohort(task)
    try:
        parent = task['execution']['policy_parent_definition']['id']
    except (KeyError, TypeError):
        return None
    return None if cohort is None else cohort.blocked_hosts.get(parent)


def blockers(task, recipe, *, root=None):
    try:
        parent = validate(task, root=root)
        cohort = task_cohort(task)
        reasons = []
        if recipe.continuous_policy is None or recipe.total_steps is not None:
            reasons.append(cohort.label + ' requires the actual schedule-free public policy recipe')
        if recipe.row_policy != 'independent':
            reasons.append('independent ' + ('Tier 1 ' if cohort is TIER1 else '') +
                           'policy task cannot silently adopt routed semantics')
        if parent['id'] in cohort.blocked_hosts:
            reasons.append(cohort.blocked_hosts[parent['id']])
        return [task['id'] + ': ' + reason for reason in reasons]
    except (KeyError, ValueError, TypeError, OSError) as error:
        return [task.get('id', '<task>') + ': ' + str(error)]


def _expected_completed_steps(task, evidence):
    """Uninterrupted endurance stops when its gate decides; others run fully."""
    execution = task['execution']
    if execution.get('uninterrupted') is True:
        dense = evidence.get('dense') or []
        final = dense[-1].get('step') if dense and isinstance(dense[-1], dict) else None
        if type(final) is not int or final > execution.get('max_total_steps', execution['steps']):
            return None
        return final
    return execution['steps']


def _measured_draws_grade(task, evidence, observations):
    """The observed output-noise law of every measured draw.

    Each recorded draw states whether learned output noise was applied and the
    served sigma at that state. Clean draws must be noise-free; an as-served
    cohort must attest a served draw in every observation and a receipt. A
    cohort registered with the draw ledger (``attest_draws``) must also attest
    the draws and receipt of every clean sample-measuring task, and the frozen
    control of a paired adaptation is held to the same declared law.
    """
    cohort = task_cohort(task)
    served = cohort.measurement == 'as_served'
    declaration = task['evaluation']['policy_observation']
    declared = declaration['output_noise']
    required = served or (cohort.attest_draws and declaration.get('sampler') == 'GANTrainer.sample')
    for row in observations:
        draws = row.get('measured_draws')
        if draws is None and not required:
            continue  # Parameter measurements and archived Tier 1 receipts record no sampled draws.
        if not isinstance(draws, list) or not draws:
            return {'status': 'INCOMPLETE', 'reason': 'observation lacks its measured output-noise draws'}
        # The recorded sigma is a receipt; the scientific grader judges the samples.
        if any(not isinstance(draw, dict) or draw.get('output_noise') is not declared or 'output_sigma' not in draw
               for draw in draws):
            return {'status': 'INVALID', 'reason': 'measured draws differ from the declared output-noise law'}

    def attested(summary):
        return (isinstance(summary, dict) and summary.get('output_noise') is declared
                and summary.get('law') == task['evaluation']['eval_output_noise']
                and type(summary.get('draws')) is int and summary['draws'] >= 1
                and summary.get('draws_without_output_noise') == (0 if declared else summary['draws']))
    if required and not attested(evidence.get('served_output_noise')):
        return {'status': 'INCOMPLETE', 'reason': ('as-served' if served else 'clean') +
                ' receipt lacks its measured output-noise attestation'}
    if required and task.get('adapter') == 'paired_adaptation' and not attested(evidence.get('frozen_served_output_noise')):
        return {'status': 'INCOMPLETE', 'reason': 'frozen control lacks its measured output-noise attestation'}
    return None


def validate_evidence(task, evidence):
    """Require actual ordered hook, owner, health and observation attestation."""
    controls = evidence.get('policy_controls', {})
    lifecycle = controls.get('lifecycle', {})
    if controls.get('cohort') != task.get('task_cohort'):
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
    measured = _measured_draws_grade(task, evidence, observations)
    if measured is not None:
        return measured
    expected = _expected_completed_steps(task, evidence)
    if expected is None or controls.get('completed_steps') != expected:
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
    import json
    root = Path(root)
    variants = {}
    for cohort in cohorts().values():
        members = {}
        for path in sorted((root / cohort.directory).glob('*.json')):
            task = json.loads(path.read_text())
            if task.get('task_cohort') != cohort.id:
                raise ValueError('policy variant is stored outside its cohort directory')
            # Stale physical sources/parents block this variant at preflight; they
            # must not prevent an unrelated ordinary view from loading its tasks.
            parent = validate(task)
            if parent['id'] not in parents:
                raise ValueError('policy variant names an unknown parent')
            members[task['id']] = task
        if members and {task['policy_parent']['id'] for task in members.values()} != set(cohort.parents):
            if cohort is TIER1:
                raise ValueError('the policy coverage cohort needs all seven parent questions')
            raise ValueError(f'{cohort.id} needs all {len(cohort.parents)} declared parent questions')
        variants.update(members)
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
