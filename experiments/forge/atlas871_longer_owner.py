"""Atlas871 longer Gaussian: original MoG/live law with a 2000/48 external host limit.

Only the exact new candidate owns this compatibility bridge. Original TaskSpecs,
public MoG class/kernel, numerical gates, host loops and named streams remain
unchanged. Metadata and measured owner receipts do not grant scientific credit.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os

from .contracts import atomic_json, stable_hash

CANDIDATE_ID = 'atlas-existing-mog-longer871-v1'
VIEW_ID = 'atlas_existing_mog_longer871_v1'
COHORT = 'atlas_existing_mog_longer871_v1'
TRACK_ID = 'atlas871_existing_mog_longer'
PROTOCOL_PATH = 'configs/forge/protocols/screening.json'
PROTOCOL_SHA256 = '3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803'
PARENTS = {'gaussian1d_acquisition_longer871_v1': {'raw_sha256': '3b438187b658afadf98d49d42e2433613530d9d8587ab6a0b49db04741427242', 'bytes': 4175, 'payload_sha256': '09025c182ec7670f3346238ae2e95c2b92c5312cae8ae19a5f8bb4d52701e256', 'prior_kind': 'mog', 'sigma': 0.025}}
CORE_PINS = {'particlegan/__init__.py': {'bytes': 1886, 'sha256': '18506efd720327de46f4bd5f4f66dab6e1b0574c5388e5496795400a45aa4070'}, 'particlegan/_qr.py': {'bytes': 1951, 'sha256': 'b59cd62cc27ed557b41b2d17ed86ed578c762a3a8a392d63731a5a3b2b018916'}, 'particlegan/anchor_birth.py': {'bytes': 6401, 'sha256': 'aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f'}, 'particlegan/autoencoder.py': {'bytes': 6194, 'sha256': 'f3fbb9184e9f44112e15eccf0c0246dfba5ed490d3c7b81b0f837a5335916402'}, 'particlegan/birth_death.py': {'sha256': '7482b29f0065be446a6b5e087a4b55d0961c8a6f2602cf9919ca502edca78c91', 'bytes': 48741}, 'particlegan/birth_phase.py': {'bytes': 21144, 'sha256': '5ff929e5e9858aa24c4cc2f8999b5d0603e25e42ca3844d1da4cc28b1981d3d7'}, 'particlegan/capabilities.py': {'bytes': 2548, 'sha256': '74f57e8485bd72193e535c4f78978b5f908a6c965bb417c2fd57b2cf446720a0'}, 'particlegan/conditioning.py': {'bytes': 4878, 'sha256': '8734fe338868ace4b5851a8fdb1f071e46d47396f66a3765e6b1ef37f0481486'}, 'particlegan/continuous.py': {'bytes': 60875, 'sha256': '4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41'}, 'particlegan/diffusion.py': {'bytes': 5100, 'sha256': '19c3aa8772703891551b9122fde8e74c07996f779892b7cb6be09553691efa61'}, 'particlegan/discriminators.py': {'bytes': 5833, 'sha256': '0e2efb125ffb314577612ab7a2eba66b0a1a2d28ad25f403f42c18b6f6ee333f'}, 'particlegan/feature_cells.py': {'bytes': 125325, 'sha256': 'e0ea05f61abd7845c7437eed9a79c3a2511d5551ff04f4e619b6b9c8992e221a'}, 'particlegan/feature_policy.py': {'bytes': 14650, 'sha256': 'de1a50f70a37d4c9174c8850ce3eac2a2597ed01f19fc4fe1a2ce0ec3176a8b2'}, 'particlegan/feature_reference.py': {'bytes': 32368, 'sha256': '58e076e9e04600be25e28afafc537a340de727371f4acbbfb2b7dd8bebac18b1'}, 'particlegan/gan_loss.py': {'bytes': 3203, 'sha256': '2ba031aa3c162bc6c4666c7dd43a2aac295ed2a841f7855e6b0a28b29eca6b2f'}, 'particlegan/grad_regularizers.py': {'bytes': 14853, 'sha256': 'a540d05a4a992540b6a5f3b5ff7ccc11ebaf18592135b87c95638adbe68aa747'}, 'particlegan/init.py': {'bytes': 24559, 'sha256': 'e15d4de01eb49f7097bc249abacab907ff067385412346f54ab0d510cb8649d0'}, 'particlegan/k3p.py': {'bytes': 33012, 'sha256': '200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c'}, 'particlegan/ka2.py': {'bytes': 16551, 'sha256': '95fdaa58a2bdc229d5d146ef89548bc3fe07582d06181a149ed1bb4f5d632703'}, 'particlegan/mean_transport.py': {'bytes': 39633, 'sha256': '57a68e0a65c5424a4460f65002f3862a69e90af330a4d47c700dc773683a2c06'}, 'particlegan/noisy_particle_prior.py': {'bytes': 3328, 'sha256': 'a59eb7340cae67b0f8f40729bd89cd2071bba63c7ae66d516246554b6c621316'}, 'particlegan/output_moments.py': {'bytes': 8387, 'sha256': 'dcf3e27228d2c3c9738e473c5b22ee9e6d5047ddcd60c538b66526adb53ef188'}, 'particlegan/particle_prior.py': {'bytes': 17899, 'sha256': '0220878bebea227da63abbf9f5b1ebdabdc6aea54fd2332fd2cab02ed734238f'}, 'particlegan/policy.py': {'sha256': '7a6ac1410260d720f58b6903310443f3f8dc3a283c1f3b5abd921965835a4967', 'bytes': 81173}, 'particlegan/population_continuity.py': {'bytes': 20329, 'sha256': '378cd75cca45bc8da4da33e686a12cf99df853042687978314e37c532bc17fe6'}, 'particlegan/recipe_schedules.py': {'bytes': 6044, 'sha256': 'd05e459b3e9e5f938b01a854273b74343f2b6d382bec4ae5151dd1f1cd032500'}, 'particlegan/recipes.py': {'sha256': '79905bb948cb967f80a087063960ea72e72c04d3b572b4b90ff4fa1dda5f299f', 'bytes': 56022}, 'particlegan/routing.py': {'bytes': 64904, 'sha256': '63249d808fccb3d4e113eb1d638a245ee665c3cade20a1542bb706f316b4abcc'}, 'particlegan/row_evidence.py': {'bytes': 9397, 'sha256': '78be40141ef3946860458e9842159be9321f05c217a4800dd4134113b6eba93e'}, 'particlegan/training.py': {'bytes': 40093, 'sha256': '740c01b14ce4542f5fcef43c59f9eed09b122f349a2308fb17bcc87e97786eb2'}, 'particlegan/vicreg_loss.py': {'bytes': 2297, 'sha256': 'ab1c4dc266dec2c35337f449917240eb38afede7590d2154b63f6f471dc45a36'}}
EXPECTED_RECIPES = {'gaussian1d_acquisition_longer871_v1': {'name': 'atlas', 'critic_formulation': 'ka2', 'model': 'gan', 'z_dim': 2, 'num_particles': 256, 'prior_kind': 'mog', 'sigma_rel': 0.0, 'standardize': False, 'num_classes': None, 'conditioning': 'scalar', 'ucd_target': 'class', 'ucd_weight': 0.02, 'alpha_bar': [1.0, 0.9, 0.5, 0.05, 0.0001], 'batch_size': 128, 'total_steps': None, 'continuous_policy': 'dv12', 'lr': 0.00425, 'd_lr_mult': 1.0, 'prior_lr_mult': 2.0, 'betas': [0.0, 0.999], 'prior_betas': None, 'reg_arm': None, 'reg_coeff': 3.0, 'reg_kappa': 1.0, 'reg_every': 1, 'prior_reg': 0.0, 'ema_decay': 0.995, 'lr_anneal_start': 0.6, 'lr_floor': 0.05, 'network_lr_floor': 0.01, 'network_lr_horizon_cap': 1600, 'reg_anchor_min_decay': 0.9, 'reg_anchor_weight': 1.0, 'direct_particle_gain': True, 'd_guard_ratio': 5.0, 'd_guard_min_steps': 200, 'latent_damping_max_rate': 0.5, 'direct_particle_betas': [0.0, 0.9], 'input_noise_std': 0.0, 'input_noise_anneal_end': 0.1, 'output_noise_std': 0.029, 'output_noise_warmup': 0.0, 'encoder_mode': 'none', 'routing_temperature': 0.25, 'distance_reduction': 'sum', 'observation_sigma': 0.03, 'reconstruction_weight': 1.0, 'amsgrad': True, 'critic_r1_real': True, 'critic_payoff_damping': True, 'output_noise_mode': 'learnable', 'lr_control': 'stationarity', 'particle_birth_death': True, 'row_evidence_gate': True, 'table_release_rule': 'anchor', 'row_evidence_hot': True, 'row_evidence_exclude': True, 'row_evidence_hold': True, 'birth_death_space': 'critic', 'serve_average': 4.0, 'reopen_signal': 'optimizer', 'reopen_anchor': 'release', 'reopen_guard': 'settled', 'row_evidence_null': 'scaled', 'birth_death_isolation': True, 'birth_death_feature_scale': 'std', 'birth_death_backend': 'auto', 'birth_death_cells': 128, 'birth_death_metric_rank': 8, 'birth_death_chunk': 256, 'birth_death_parent_policy': 'real_anchor', 'row_policy': 'independent', 'optimizer_family': 'formulation', 'eps': 1e-08, 'beta2_end': None, 'beta2_anneal_end': 0.2, 'reg_coeff_end': None, 'reg_coeff_anneal_end': 0.2, 'loss': 'relativistic'}}
SUPPORTED = ['gaussian1d_acquisition_longer871_v1']
BLOCKED = {}
ALLOWANCES = {'gaussian1d_acquisition_longer871_v1': 240}


def is_candidate(candidate):
    return isinstance(candidate, dict) and candidate.get('id') == CANDIDATE_ID


def supports_candidate(candidate):
    return (is_candidate(candidate) and candidate.get('claim_contract', {}).get('experimental_track') == TRACK_ID
            and candidate.get('recipe_preset') == 'atlas'
            and candidate.get('recipe_overrides', {}) == {}
            and candidate.get('extensions', {}) == {} and candidate.get('host_adaptation') is None
            and not candidate.get('implementation')
            and candidate.get('initializer', 'deterministic_orthogonal') == 'deterministic_orthogonal')


def declaration(task):
    if not isinstance(task, dict):
        raise ValueError('original task declaration must be an object')
    result = deepcopy(task)
    if 'preflight_blockers' in result:
        if not isinstance(result.pop('preflight_blockers'), list):
            raise ValueError('malformed preflight cache')
    if 'field_ownership' in result and not isinstance(result.pop('field_ownership'), dict):
        raise ValueError('malformed ownership annotation')
    return result


def _pin(root, relative, expected):
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != path:
        raise ValueError('source alias forbidden: ' + relative)
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != (expected['sha256'], expected['bytes']):
        raise ValueError('source drift: ' + relative)
    return raw


def validate(task, root=None):
    from .atlas871_longer import validate as validate_duration
    result = validate_duration(task, root=root)
    if root is not None:
        for relative, expected in CORE_PINS.items():
            _pin(root, relative, expected)
    return result


def blockers(task, candidate, root=None):
    try:
        metadata = validate(task, root=root)
        if not supports_candidate(candidate):
            raise ValueError('the exact Atlas871 longer Gaussian candidate and unchanged Atlas preset are required')
        name = metadata['task_id']
        if name in BLOCKED:
            return [name + ': ' + BLOCKED[name]]
        if name == 'ae_gan_hold':
            from .atlas_existing_mog_ae import blockers as ae_blockers
            return ae_blockers(task, candidate, root=root)
        return []
    except (ValueError, KeyError, TypeError, OSError) as error:
        return [task.get('id', '<task>') + ': ' + str(error)]


def task_resources(task):
    validate(task)
    spec = task['execution']['host_definition']
    return dict(num_particles=spec['particles'], z_dim=spec['z_dim'], batch_size=spec['batch'])


def supporting_source_paths(task, candidate, root=None):
    from .atlas871_longer import TASK_ID, TASK_PATH, PARENT_PATH
    if not is_candidate(candidate) or not isinstance(task, dict) or task.get('id') != TASK_ID:
        return ()
    validate(task)
    catalog = ()
    if root is not None:
        base = Path(root).resolve()
        catalog = tuple(sorted({p.relative_to(base).as_posix()
            for directory in ('configs/forge/tasks', 'configs/forge/task-variants')
            for p in (base / directory).rglob('*.json') if p.is_file()}))
    return (*catalog, TASK_PATH, PARENT_PATH, PROTOCOL_PATH,
        'configs/forge/defaults.json','configs/forge/ideas/atlas.json',
        'configs/forge/legacy-ideas-v1.json','configs/forge/ideas/ka2.json',
        'configs/forge/views/' + VIEW_ID + '.json', 'configs/forge/ideas/' + CANDIDATE_ID + '.json',
        'configs/forge/studies/atlas-existing-mog-longer871-study-v1.json',
        'reports/forge/prior-evidence/atlas-type-only686.json',
        'tests/test_forge_atlas871_longer_gaussian.py')


def prior_contract(prior, *, kernel_stream=None):
    """Measure the existing public MoG without sampling or replacing a Parameter."""
    from particlegan.particle_prior import MoGParticlePrior
    from .policy_adapters import typed_state_digest
    if (type(prior) is not MoGParticlePrior or prior.standardize is not False
            or not prior.z.requires_grad or prior.sigma.requires_grad
            or prior.sigma.numel() != 1):
        raise ValueError('exact raw trainable MoG with fixed scalar width required')
    return dict(kind='mog', code_path='particlegan.particle_prior.MoGParticlePrior',
        sigma=float(prior.sigma), sigma_dtype=str(prior.sigma.dtype),
        sigma_buffer_sha256=typed_state_digest(prior.sigma), sigma_units='raw_latent_coordinates',
        learned_width=False, standardize=False, row_weights='uniform',
        sampling='uniform_indices_then_raw_z_plus_fixed_gaussian_before_generator',
        index_stream=['prior', 'latent', 'indices'], kernel_stream=(['noise', 'prior', 'gaussian'] if kernel_stream is None else list(kernel_stream)),
        same_location_parameter=True, original_public_sampler=True, sampling_calls_added=0)


def live_evaluation_state(context, trainer):
    from .noisy_prior_adapters import live_evaluation_state as maintained_live_state
    return maintained_live_state(context, trainer)


def restore_live_owner(trainer):
    """Release installed averages at the completed boundary for original live reads."""
    from particlegan.particle_prior import MoGParticlePrior
    if (type(trainer.prior) is not MoGParticlePrior or trainer.prior.standardize is not False
            or trainer.policy.prior is not trainer.prior or trainer.policy.table is not trainer.prior.z
            or trainer.policy.row_policy != 'independent' or trainer.policy._phase != 'ready'
            or trainer.completed_steps != trainer.policy.completed_steps):
        raise ValueError('the actual independent ready MoG table/public lifecycle must own the live reader')
    trainer.policy._serve_release()
    if trainer.policy._fast is not None:
        raise ValueError('original live reader still contains served storage')


def vector_admission_contract(request, task, output, device, resolved, *, cuda_visible_devices):
    """Validate the maintained runtime envelope; do not infer an injected worker."""
    output = Path(output).resolve()
    if (resolved.get('schema_version') != 1
            or stable_hash(resolved.get('request')) != stable_hash(request)):
        raise ValueError('ordinary immutable admitted vector request differs')
    worker, job = resolved['worker'], resolved['job']
    jobs = [member for member in request['jobs'] if member['compatibility_key'] == job['compatibility_key']]
    if len(jobs) != 1 or stable_hash(jobs[0]) != stable_hash(job):
        raise ValueError('one exact admitted ordinary vector job required')
    physical = worker['device']
    if (not isinstance(physical, str) or physical not in {'0', '1'}
            or str(device) != 'cuda:0' or cuda_visible_devices != physical
            or job['science']['compute']['backend'] != 'cuda'):
        raise ValueError('ordinary physical card/CUDA visibility and logical cuda:0 differ')
    resources = task['resources']
    compute = job['science']['compute']
    expected_resources = {'memory_mb': resources['gpu_memory_mb'], 'gpus': resources['gpus'],
        **({'host_memory_mb': resources['host_memory_mb']} if 'host_memory_mb' in resources else {}),
        'cpu_threads': resources['cpu_threads'], 'allow_cpu': False,
        'backend': 'cuda', 'gpu_model': compute.get('model')}
    if (job['task_id'] != task['id'] or job.get('task_ids', [task['id']]) != [task['id']]
            or job['budget_seconds'] != ALLOWANCES[task['id']]
            or job['budget_seconds'] != resources['timeout_seconds']
            or job['resources'] != expected_resources or job['resources']['gpus'] != 1
            or compute['threads'] != resources['cpu_threads']
            or job['resources']['cpu_threads'] != 1
            or Path(worker['directory']).resolve() != output
            or not isinstance(worker.get('token'), str) or not worker['token']
            or not isinstance(worker.get('attempt'), str) or not worker['attempt']
            or job['compatibility_key'] != stable_hash(job['science'])
            or job['science']['candidate_revision'] != request['candidate_revision']
            or stable_hash(job['science']['protocol']) != stable_hash(request['protocol'])
            or task['id'] not in request['tasks'] or stable_hash(task) != stable_hash(request['tasks'][task['id']])):
        raise ValueError('ordinary vector request/job/worker identity differs')
    return dict(candidate_id=CANDIDATE_ID, task_id=task['id'],
        attempt=worker['attempt'], compatibility_key=job['compatibility_key'],
        source_digest=request['source']['digest'], candidate_revision=request['candidate_revision'],
        budget_seconds=job['budget_seconds'], device=str(device), physical_device=physical,
        cuda_visible_devices=cuda_visible_devices)


def admitted_vector_context(request, task, context, output, device):
    """Join ordinary request.json and inherited lease before the first constructor."""
    if not is_candidate(request['candidate']):
        return
    from .atlas_two_pole import source_guard
    from .contracts import read_json
    root = Path(request['source']['snapshot_path']).resolve()
    source_guard(request, root)
    validate(task, root=root)
    errors = blockers(task, request['candidate'], root=root)
    if errors:
        raise ValueError('; '.join(errors))
    output = Path(output).resolve()
    path = output / 'request.json'
    if path.resolve() != path:
        raise ValueError('ordinary vector request alias forbidden')
    admission = vector_admission_contract(request, task, output, device, read_json(path),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    fd = int(os.environ['FORGE_LEASE_FD'])
    actual, declared = os.fstat(fd), (output / 'execution.lock').stat()
    if (actual.st_dev, actual.st_ino) != (declared.st_dev, declared.st_ino):
        raise ValueError('ordinary inherited lease does not own vector attempt')
    context.existing_mog_admission = dict(admission, inherited_lease_checked=True)


def initialize_vector_receipts(context, trainer, output, task):
    from dataclasses import asdict
    from particlegan.particle_prior import MoGParticlePrior
    from particlegan.policy import UpdatePolicy
    from .policy_adapters import typed_state_digest
    validate(task)
    if task['id'] not in {'gaussian1d_acquisition_longer871_v1'}:
        raise ValueError('only original vector constructors use this receipt')
    policy = trainer.policy
    params = [p for opt in (trainer.opt_g, trainer.opt_d) for group in opt.param_groups for p in group['params']]
    bindings = dict(policy_generator=policy.G is trainer.G, policy_critic=policy.D is trainer.D,
        policy_prior=policy.prior is trainer.prior, policy_table=policy.table is trainer.prior.z,
        table_optimizer=policy.table_optimizer is trainer.opt_g,
        unique_optimizer_parameters=len(params) == len({id(p) for p in params}),
        a2_enabled=trainer.prior_mechanisms['a2']['enabled'] is True,
        birth_death=policy.birth_death is not None, row_evidence=policy.row_evidence is not None)
    if (type(trainer.prior) is not MoGParticlePrior or type(policy) is not UpdatePolicy
            or trainer.recipe is not context.recipe or trainer.completed_steps != 0
            or policy.completed_steps != 0 or policy._phase != 'ready' or policy.row_policy != 'independent'
            or trainer.opt_g.state or trainer.opt_d.state or not all(bindings.values())
            or not hasattr(context, 'existing_mog_admission')):
        raise ValueError('fresh admitted actual MoG/public owner with empty optimizers and zero clocks required')
    kernel = prior_contract(trainer.prior)
    if not math.isclose(kernel['sigma'], task['execution']['prior']['sigma'], rel_tol=1e-7, abs_tol=0.):
        raise ValueError('actual fixed MoG kernel differs from original task')
    recipe = asdict(context.recipe)
    if stable_hash(recipe) != stable_hash(EXPECTED_RECIPES[task['id']]):
        raise ValueError('full current Atlas79 differs from fixed original task binding')
    source = dict(schema='forge_atlas871_existing_mog_longer_initialization_v1',
        candidate_id=CANDIDATE_ID, task_id=task['id'], pid=os.getpid(),
        recipe=recipe, recipe_sha256=stable_hash(recipe), prior=kernel, object_bindings=bindings,
        reaction_kernel=reaction_kernel_receipt(policy),
        admission=deepcopy(context.existing_mog_admission), initialization=deepcopy(context.initialization),
        rng=context.streams.manifest(), completed_steps=0, phase='ready', restored=False,
        optimizer_state_entries=dict(generator=0, discriminator=0),
        initial_state_sha256=typed_state_digest(context.state_dict()))
    context.existing_mog_initialization = deepcopy(source)
    atomic_json(Path(output) / 'INITIALIZATION.json', source)
    start = dict(schema='forge_atlas871_existing_mog_longer_model_started_v1', task_id=task['id'],
        candidate_id=CANDIDATE_ID, pid=os.getpid(), completed_steps=0, initialization_sha256=stable_hash(source))
    atomic_json(Path(output) / 'MODEL_STARTED.json', start)
    print(json.dumps(dict(event='actual_model_started', **start), sort_keys=True), flush=True)


def observation_receipt(task, policy):
    validate(task)
    if (policy.table is not policy.prior.z or policy._phase != 'ready'
            or policy.row_policy != 'independent' or policy._fast is not None):
        raise ValueError('original live reader requires actual independent ready MoG table')
    return dict(task_id=task['id'], completed_steps=policy.completed_steps, observed=True,
        candidate_id=CANDIDATE_ID, policy_owner='particlegan.UpdatePolicy', weights='live',
        sampling_law=task['evaluation']['sampling_law'], eval_output_noise=task['evaluation']['eval_output_noise'],
        prior=prior_contract(policy.prior), reaction_kernel=reaction_kernel_receipt(policy),
        table_alias_preserved=True, sampling_calls_added_by_receipt=0)


def vector_receipt(context, trainer, task):
    from dataclasses import asdict
    initial = context.existing_mog_initialization
    return dict(schema='forge_atlas871_existing_mog_longer_evidence_v1', candidate_id=CANDIDATE_ID,
        cohort=COHORT, task_id=task['id'], task_payload_sha256=validate(task)['task_payload_sha256'],
        full_recipe=asdict(trainer.recipe), full_recipe_sha256=stable_hash(asdict(trainer.recipe)),
        actual_prior=prior_contract(trainer.prior), initialization=deepcopy(initial),
        reaction_kernel=reaction_kernel_receipt(trainer.policy),
        admission=deepcopy(context.existing_mog_admission), original_task_credit_transferred=False,
        policy_table_alias=trainer.policy.table is trainer.prior.z,
        table_optimizer_alias=trainer.policy.table_optimizer is trainer.opt_g)


def validate_reaction_kernel_receipt(receipt, prior, *, completed_steps):
    """Validate measured implementation metadata, without sampling or grading."""
    if (type(completed_steps) is not int or completed_steps < 0
            or not isinstance(receipt, dict)
            or receipt.get('schema') != 'forge_atlas871_mog_longer_v1'
            or receipt.get('candidate_id') != CANDIDATE_ID
            or type(receipt.get('completed_steps')) is not int
            or receipt['completed_steps'] != completed_steps
            or receipt.get('core_pins') != {name: CORE_PINS[name] for name in
                ('particlegan/birth_death.py', 'particlegan/policy.py')}
            or receipt.get('birth_death_type') != 'particlegan.birth_death.ParticleBirthDeath'
            or receipt.get('reaction_prior_alias') is not True
            or receipt.get('reaction_table_alias') is not True
            or type(receipt.get('sampling_calls_added')) is not int
            or receipt['sampling_calls_added'] != 0):
        raise ValueError('corrected actual reaction owner/source/clock binding differs')
    backend = receipt.get('actual_backend')
    if (backend not in {'pending', 'knn'} or completed_steps > 0 and backend != 'knn'):
        raise ValueError('corrected sampled MoG proof requires the actual knn backend')
    fake = receipt.get('fake_pool_prior')
    keys = {'kind', 'code_path', 'sigma', 'sigma_dtype', 'sigma_units', 'standardize',
            'row_weights', 'same_table_parameter', 'sampler', 'stream'}
    if (not isinstance(fake, dict) or set(fake) != keys
            or fake['kind'] != 'mog'
            or fake['code_path'] != 'particlegan.particle_prior.MoGParticlePrior'
            or type(fake['sigma']) is not float or not math.isfinite(fake['sigma'])
            or not math.isclose(fake['sigma'], prior['sigma'], rel_tol=1e-7, abs_tol=0.)
            or fake['sigma_dtype'] != prior['sigma_dtype']
            or fake['sigma_units'] != 'raw_latent_coordinates'
            or fake['standardize'] is not False or fake['row_weights'] != 'uniform'
            or fake['same_table_parameter'] is not True
            or fake['sampler'] != 'original_public_sample_then_existing_dv12_and_output_noise'
            or fake['stream'] != 'private_birth_death'
            or receipt.get('state_config_fake_pool_prior') != fake):
        raise ValueError('corrected fake pool must retain exact public MoG kernel and checkpoint configuration')


def reaction_kernel_receipt(policy):
    """Read the actual reaction owner. No prior draw or state mutation is added."""
    from particlegan.birth_death import ParticleBirthDeath
    birth = policy.birth_death
    if (type(birth) is not ParticleBirthDeath or birth.rows.mog_prior is not policy.prior
            or birth.rows.table is not policy.prior.z):
        raise ValueError('corrected raw MoG reaction owner and same table are required')
    selection = policy._feature_selection.state_dict() if policy._feature_selection is not None else None
    actual = selection['actual_backend'] if selection is not None else 'knn'
    receipt = dict(schema='forge_atlas871_mog_longer_v1', candidate_id=CANDIDATE_ID,
        completed_steps=policy.completed_steps,
        core_pins={name: deepcopy(CORE_PINS[name]) for name in
            ('particlegan/birth_death.py', 'particlegan/policy.py')},
        birth_death_type='particlegan.birth_death.ParticleBirthDeath', actual_backend=actual,
        reaction_prior_alias=birth.rows.mog_prior is policy.prior,
        reaction_table_alias=birth.rows.table is policy.prior.z,
        fake_pool_prior=deepcopy(birth.diagnostics()['fake_pool_prior']),
        state_config_fake_pool_prior=deepcopy(birth._config()['fake_pool_prior']), sampling_calls_added=0)
    validate_reaction_kernel_receipt(receipt, prior_contract(policy.prior),
                                    completed_steps=policy.completed_steps)
    return receipt


def validate_evidence(task, evidence):
    """Ownership check only. Numerical grading uses the untouched original scorer."""
    try:
        validate(task)
        if task['id'] == 'ae_gan_hold':
            from .atlas_existing_mog_ae import validate_evidence as ae_validate
            return ae_validate(task, evidence)
        if task['id'] in BLOCKED:
            return dict(status='BLOCKED', reason=BLOCKED[task['id']])
        if task['id'] == 'two_pole':
            return validate_direct_evidence(task, evidence)
        marker = evidence['existing_mog717']
        if (marker['schema'] != 'forge_atlas871_existing_mog_longer_evidence_v1'
                or marker['candidate_id'] != CANDIDATE_ID or marker['cohort'] != COHORT
                or marker['task_id'] != task['id'] or marker['task_payload_sha256'] != validate(task)['task_payload_sha256']
                or marker['full_recipe_sha256'] != stable_hash(marker['full_recipe'])
                or stable_hash(marker['full_recipe']) != stable_hash(EXPECTED_RECIPES[task['id']])
                or marker['full_recipe']['prior_kind'] != 'mog' or marker['full_recipe']['standardize'] is not False
                or marker['full_recipe']['row_policy'] != 'independent' or marker['full_recipe']['total_steps'] is not None
                or marker['policy_table_alias'] is not True or marker['table_optimizer_alias'] is not True
                or marker['initialization']['completed_steps'] != 0
                or marker['initialization']['optimizer_state_entries'] != dict(generator=0, discriminator=0)):
            raise ValueError('actual existing MoG/Recipe/fresh table identity differs')
        steps = task['execution']['steps']
        validate_reaction_kernel_receipt(marker['reaction_kernel'], marker['actual_prior'], completed_steps=steps)
        validate_reaction_kernel_receipt(marker['initialization']['reaction_kernel'],
            marker['initialization']['prior'], completed_steps=0)
        from .atlas871_longer import checkpoints
        clocks = checkpoints()
        controls = evidence['policy_controls']
        if (controls.get('cohort') != COHORT or controls.get('completed_steps') != steps
                or controls.get('implementation_observed') is not True or controls.get('requested_owners_bound') is not True
                or controls.get('lifecycle', {}).get('complete') is not True):
            return dict(status='INCOMPLETE', reason='missing complete actual existing MoG lifecycle')
        rows, purity = evidence['policy_observations'], evidence['policy_purity']
        if [r['completed_steps'] for r in rows] != clocks or [r['completed_steps'] for r in purity] != clocks:
            return dict(status='INCOMPLETE', reason='missing diagnostic48 original-grid scored reads and full-state purity clocks')
        for row in [dict(prior=marker['actual_prior']), *rows]:
            kernel = row['prior']
            if (kernel['kind'] != 'mog' or kernel['code_path'] != 'particlegan.particle_prior.MoGParticlePrior'
                    or kernel['standardize'] is not False or kernel['learned_width'] is not False
                    or kernel['row_weights'] != 'uniform' or kernel['sampling_calls_added'] != 0
                    or kernel['same_location_parameter'] is not True
                    or not math.isclose(kernel['sigma'], task['execution']['prior']['sigma'], rel_tol=1e-7, abs_tol=0.)):
                raise ValueError('existing public MoG identity/kernel/parameter differs')
        for row in rows:
            validate_reaction_kernel_receipt(row['reaction_kernel'], row['prior'], completed_steps=row['completed_steps'])
            if (row['task_id'] != task['id'] or row['candidate_id'] != CANDIDATE_ID
                    or row['policy_owner'] != 'particlegan.UpdatePolicy' or row['observed'] is not True
                    or row['table_alias_preserved'] is not True or row['sampling_calls_added_by_receipt'] != 0
                    or row['weights'] != task['evaluation']['scoring_weights']
                    or row['sampling_law'] != task['evaluation']['sampling_law']
                    or row['eval_output_noise'] != task['evaluation']['eval_output_noise']):
                raise ValueError('original live observation law differs')
        if any(r['pure'] is not True or r['before_sha256'] != r['after_sha256'] for r in purity):
            raise ValueError('original observation changed complete training state')
        return None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, OSError) as error:
        return dict(status='INVALID', reason='malformed existing MoG evidence: ' + str(error))




