"""Verify the public GAN default through real user-facing construction paths.

Vector cases train on the shared toy runner under gan_v3_recipe() at each
task's shape; image cases through GANTrainer(gan_v3_recipe(), ...). The nine
legacy auxiliary hosts retain their required custom loops, using public GAN
primitives and the same unmodified global recipe. Frozen host data, model
initialization order, RNG streams, resources, steps, measurements, and gates
remain explicit in the output. This is a verification run, not a search.

For an installed-wheel proof, pass --require-installed-root SITE_PACKAGES. The
runner imports the wheel before benchmark modules and rejects source fallback
or any installed public module whose bytes differ from this checkout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch

from benchmarks.toy100.device import add_device_argument, apply_device_policy, experiment_generator
from benchmarks.gan_v3 import gan_v3_recipe, legacy_dict


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / 'reports/transfer_suite/unadjusted/leading_profile.json'
GLOBAL_RECIPE_FIELDS = (
    'lr', 'd_lr_mult', 'prior_lr_mult', 'betas', 'prior_betas', 'loss_type',
    'gan_mode', 'reg_arm', 'reg_coeff', 'reg_kappa', 'reg_every', 'reg_method',
    'prior_reg', 'ema_decay', 'lr_anneal_start', 'lr_floor',
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _required_package_dir(root):
    root = Path(root).resolve()
    package_dir = root / 'particlegan' if (root / 'particlegan/__init__.py').is_file() else root
    if package_dir.name != 'particlegan' or not (package_dir / '__init__.py').is_file():
        raise ValueError('--require-installed-root must contain an installed particlegan package')
    if package_dir == (ROOT / 'particlegan').resolve():
        raise ValueError('checkout source is not an installed package')
    return package_dir


def public_module_manifest(require_installed_root=None):
    """Import installed public code first; prove origin and byte parity."""
    required_dir = (_required_package_dir(require_installed_root)
                    if require_installed_root is not None else None)
    if required_dir is not None:
        original_path = sys.path[:]
        try:
            sys.path[:] = [str(required_dir.parent)] + [entry for entry in original_path
                if Path(entry or '.').resolve() != ROOT]
            package = importlib.import_module('particlegan')
        finally:
            sys.path[:] = original_path
    else:
        package = importlib.import_module('particlegan')
    package_dir = Path(package.__file__).resolve().parent
    if required_dir is not None and package_dir != required_dir:
        raise RuntimeError(f'public package loaded from {package_dir}, expected {required_dir}')
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name != 'particlegan' and not name.startswith('particlegan.'):
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            raise RuntimeError(f'public module has no file origin: {name}')
        path = Path(origin).resolve()
        if not path.is_relative_to(package_dir):
            raise RuntimeError(f'public module escaped installed package: {name}: {path}')
        relative = path.relative_to(package_dir)
        checkout = ROOT / 'particlegan' / relative
        if not checkout.is_file():
            raise RuntimeError(f'public module has no checkout counterpart: {name}: {relative}')
        installed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        checkout_hash = hashlib.sha256(checkout.read_bytes()).hexdigest()
        if installed_hash != checkout_hash:
            raise RuntimeError(f'installed public module differs from checkout: {name}')
        modules[name] = dict(origin=str(path), sha256=installed_hash,
                             checkout=str(checkout), checkout_sha256=checkout_hash)
    return dict(package_file=str(Path(package.__file__).resolve()),
                package_directory=str(package_dir),
                required_installed_root=None if required_dir is None else str(required_dir),
                verified_modules=modules)


def load_declaration():
    # This import follows public_module_manifest, so benchmark hosts receive
    # the already verified public package rather than a checkout fallback.
    from .compare_defaults import plan
    jobs = plan()
    profile = json.loads(PROFILE_PATH.read_text())
    if len(jobs) != 19 or profile['candidates'][0]['name'] != 'shared_c6':
        raise ValueError('frozen default declaration changed')
    if set(profile['discriminators']) != {
            'vector_unequal_mass', 'vector_unequal_width',
            'vector_overlap', 'vector_anisotropic'}:
        raise ValueError('frozen discriminator profile changed')
    return jobs, profile


def public_default(profile):
    from benchmarks.legacy.recipe import get_recipe
    base = gan_v3_recipe()
    if base.name != 'gan_v3':
        raise ValueError('gan_v3_recipe() does not resolve to the public gan_v3 default')
    for name, expected in profile['candidates'][0]['overrides'].items():
        actual = getattr(base, name)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise ValueError(f'public default differs from frozen profile: {name}')
    if (base.loss_type, base.gan_mode, base.reg_arm, base.lr_anneal_start,
            base.lr_floor, base.reg_every, base.reg_method) != (
            'logistic', 'rp', 'b_cap', .6, .05, 1, 'autograd'):
        raise ValueError('public loss, penalty, or schedule differs from frozen profile')
    return base


def host_recipe(base, spec):
    """Only resource dimensions/budget adapt; global recipe fields are fixed."""
    if spec['runner'] == 'vector':
        resources = dict(z_dim=spec['z_dim'], num_particles=spec['particles'],
                         batch_size=spec['batch'], total_steps=spec['steps'])
    elif spec['runner'] == 'image':
        resources = dict(z_dim=spec['z_dim'], num_particles=spec['particles'],
                         batch_size=spec['batch_size'], total_steps=spec['steps'])
    else:
        raise ValueError('only vector/image hosts use GANTrainer')
    resolved = base.replace(**resources)
    if any(getattr(resolved, key) != getattr(base, key) for key in GLOBAL_RECIPE_FIELDS):
        raise AssertionError('host changed a global recipe field')
    return resolved


def declared_spec(job, profile, base):
    from .compare_defaults import effective_spec
    from .shared_variants import architecture_spec
    original = job['spec']
    card = profile['discriminators'].get(original['name'])
    variant = None
    if card is not None:
        if original['runner'] != 'vector':
            raise ValueError('D variants are vector-only')
        width = card['width'] if card['implementation'] == 'shared_batch_feature_v1' else card['hidden']
        variant = dict(name=card['name'], overrides=dict(
            d_hidden=width, d_layers=card['layers'], fourier=card['fourier'],
            research_discriminator=deepcopy(card)))
    spec = effective_spec(architecture_spec(original, variant), base)
    return spec, card, variant


def optimizer_receipts(trainer):
    from particlegan import learning_rate_scale
    roles = (('g', trainer.opt_g.param_groups[0]),
             ('prior', trainer.opt_g.param_groups[1]),
             ('d', trainer.opt_d.param_groups[0]))
    expected = dict(g=trainer.recipe.lr,
                    prior=trainer.recipe.lr*trainer.recipe.prior_lr_mult,
                    d=trainer.recipe.lr*trainer.recipe.d_lr_mult)
    receipts = []
    for role, group in roles:
        betas = (trainer.recipe.prior_betas or trainer.recipe.betas
                 if role == 'prior' else trainer.recipe.betas)
        if group['lr'] != expected[role] or tuple(group['betas']) != tuple(betas):
            raise RuntimeError(f'optimizer does not implement public recipe: {role}')
        receipts.append(dict(role=role, lr=group['lr'], betas=list(group['betas']),
                             parameters=sum(p.numel() for p in group['params']),
                             optimizer='Adam'))
    if learning_rate_scale(0, trainer.recipe.total_steps,
            trainer.recipe.lr_anneal_start, trainer.recipe.lr_floor) != 1.:
        raise RuntimeError('unexpected public learning-rate schedule')
    return receipts


def shape_receipt(trainer, batch, data_shape):
    with torch.no_grad():
        fake = trainer.G(torch.zeros(2, trainer.recipe.z_dim, device=trainer.device,
                                     dtype=trainer.dtype))
        score = trainer.D(fake)
    if tuple(fake.shape) != (2, *data_shape) or tuple(score.shape) != (2,):
        raise RuntimeError('public trainer network shapes differ from frozen host')
    if tuple(trainer.prior.z.shape) != (trainer.recipe.num_particles, trainer.recipe.z_dim):
        raise RuntimeError('public trainer prior shape differs from frozen host')
    return dict(real_batch=[batch, *data_shape], latent_batch=[batch, trainer.recipe.z_dim],
                generator_output=list(fake.shape), discriminator_output=list(score.shape),
                prior=list(trainer.prior.z.shape),
                generator_parameters=sum(p.numel() for p in trainer.G.parameters()),
                discriminator_parameters=sum(p.numel() for p in trainer.D.parameters()))


def rate_action(trainer, completed):
    from particlegan import learning_rate_scale
    scale = learning_rate_scale(completed-1, trainer.recipe.total_steps,
                                trainer.recipe.lr_anneal_start, trainer.recipe.lr_floor)
    rates = [group['lr'] for opt in (trainer.opt_g, trainer.opt_d)
             for group in opt.param_groups]
    expected = [trainer.recipe.lr*scale, trainer.recipe.lr*trainer.recipe.prior_lr_mult*scale,
                trainer.recipe.lr*trainer.recipe.d_lr_mult*scale]
    if any(not math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15)
           for a, b in zip(rates, expected)):
        raise RuntimeError(f'actual public optimizer rates differ at step {completed}')
    return dict(step=completed, multiplier=scale,
                lr_g=rates[0], lr_prior=rates[1], lr_d=rates[2])


def run_vector(spec, card, base, *, log_path=None):
    """The declared default on the shared runner (``vector_tasks.VectorTask``).

    The problem is the frozen host's data, networks and metrics; ``base`` at
    the task's shape is the recipe whose factories build everything else.
    ``log_path`` gets one JSON line per observation (``tail -f``).
    """
    from . import vector_tasks
    torch.set_num_threads(1)
    recipe = host_recipe(base, spec)
    result = vector_tasks.train(vector_tasks.VectorTask(spec, card), recipe, log_path=log_path)
    context = dict(applied=result.pop('applied'), shapes=result.pop('shapes'), host_recipe=recipe)
    return result, context


def setup_image(spec, base):
    from particlegan import GANTrainer
    from . import image_tasks
    torch.set_num_threads(1)
    torch.manual_seed(0)
    centers = image_tasks.templates(spec)
    recipe = host_recipe(base, spec)
    # Frozen image host constructs G and D before a global-RNG prior draw.
    generator, discriminator = image_tasks.Generator(spec), image_tasks.Discriminator(spec)
    prior = recipe.make_prior()
    global_stream = experiment_generator()
    trainer = GANTrainer(recipe, generator, discriminator, prior=prior, seed=0,
                                  latent_generator=global_stream,
                                  penalty_generator=global_stream)
    shapes = shape_receipt(trainer, spec['batch_size'], (1, 8, 8))
    return dict(trainer=trainer, centers=centers, shapes=shapes,
                applied=optimizer_receipts(trainer), host_recipe=recipe)


def run_image(spec, base, *, max_steps=None):
    from benchmarks.locked_shared.observation import sustained
    from . import image_tasks
    started = time.perf_counter()
    context = setup_image(spec, base)
    trainer, centers = context['trainer'], context['centers']
    expected = image_tasks.evaluation_steps(spec)
    observations, actions, losses = [], [], []
    budget = spec['steps'] if max_steps is None else min(max_steps, spec['steps'])
    for index in range(budget):
        completed = index+1
        real = centers[torch.randint(len(centers), (spec['batch_size'],))]
        real = (real+spec['noise_std']*torch.randn_like(real)).clamp(0., 1.)
        stats = trainer.step(real, generator_real=real)
        if not all(torch.isfinite(value) for key, value in stats.items()
                   if key != 'step' and isinstance(value, torch.Tensor)):
            raise FloatingPointError('nonfinite public trainer loss')
        actions.append(rate_action(trainer, completed))
        if completed in expected:
            live = image_tasks.measure(trainer.G, trainer.prior, centers, spec['thresholds'])
            ema = image_tasks.measure(trainer.ema_G, trainer.ema_prior, centers, spec['thresholds'])
            observations.append(dict(step=completed, seconds=time.perf_counter()-started,
                                     **live, ema=ema))
            losses.append(dict(step=completed, d=float(stats['loss_d']),
                               g=float(stats['loss_g']), d_penalty=float(stats['penalty']),
                               prior=float(base.prior_reg*stats['prior_regularization'])))
    result = dict(live=observations[-1] if observations else {},
                  ema=observations[-1]['ema'] if observations else {},
                  observations=observations, losses=losses, actions=actions,
                  update_counts=dict(g=trainer.completed_steps, d=trainer.completed_steps),
                  seconds=time.perf_counter()-started)
    if len(observations) == 24:
        result['live'] = {k: v for k, v in observations[-1].items() if k not in ('ema', 'step', 'seconds')}
        result['convergence'] = sustained(observations,
            [('modes', '>=', spec['thresholds']['modes']),
             ('hq', '>=', spec['thresholds']['hq_min'])], expected_steps=expected,
            minimum=spec['thresholds']['minimum_stable_checks'])
    return result, context


def run_legacy(spec, base):
    from benchmarks.locked_shared import baseline
    from benchmarks.smart_descent import evaluate
    from .compare_defaults import candidate, optimizer_defaults
    from . import vector_tasks
    from benchmarks import learned_lr_evaluation as bridge
    started = time.perf_counter()
    applied = []
    policy = vector_tasks.fixed_policy('cosine')
    with optimizer_defaults(base, applied):
        control = evaluate.FixedControl(policy, spec['steps'])
        with bridge.control_host_schedules(control):
            result = baseline.run_toy(spec['name'], candidate(base))
    result['actions'] = control.trace
    result['seconds'] = time.perf_counter()-started
    return result, dict(applied=applied, shapes=dict(host='legacy auxiliary custom loop'),
                        host_recipe=base)


def run(output, *, tasks=None, require_installed_root=None):
    if output.exists():
        raise FileExistsError('use a new output directory')
    package = public_module_manifest(require_installed_root)
    from . import suite
    from .compare_defaults import ema_verdict
    from .protocol import test_verdict
    jobs, profile = load_declaration()
    base = public_default(profile)
    if tasks:
        unknown = set(tasks)-{job['spec']['name'] for job in jobs}
        if unknown:
            raise ValueError(f'unknown frozen tests: {sorted(unknown)}')
        jobs = [job for job in jobs if job['spec']['name'] in set(tasks)]
    output.mkdir(parents=True)
    (output/'episodes').mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(version='public-default-verification-v1',
                    public_package=public_module_manifest(require_installed_root),
                    base_get_recipe=legacy_dict(base),
                    global_recipe_fields=list(GLOBAL_RECIPE_FIELDS),
                    profile_path=str(PROFILE_PATH.relative_to(ROOT)),
                    profile_sha256=hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
                    frozen_profile=profile, jobs=jobs, seed=0,
                    routes=dict(vector='toy_runner',
                                image='GANTrainer',
                                legacy='public_primitives_custom_host'))
    write(output/'protocol.json', protocol)
    records = []
    for job in jobs:
        suite.verify_source(protocol)
        spec, card, variant = declared_spec(job, profile, base)
        route = ('toy_runner' if spec['runner'] == 'vector' else 'GANTrainer'
                 if spec['runner'] == 'image' else 'public_primitives_custom_host')
        print(f'START {spec["name"]} route={route} steps={spec["steps"]}', flush=True)
        started = time.perf_counter()
        try:
            if spec['runner'] == 'vector':
                result, context = run_vector(spec, card, base,
                                             log_path=output/'logs'/f"{spec['name']}.jsonl")
            elif spec['runner'] == 'image':
                result, context = run_image(spec, base)
            else:
                result, context = run_legacy(spec, base)
            observations = result.get('observations', result.get('curve', []))
            if len(observations) != 24:
                raise RuntimeError(f'frozen host did not produce 24 live observations: {spec["name"]}')
            if route in ('GANTrainer', 'toy_runner'):
                if any(not isinstance(point.get('ema'), dict) for point in observations):
                    raise RuntimeError(f'frozen host has an incomplete EMA curve: {spec["name"]}')
                if len(result['actions']) != spec['steps']:
                    raise RuntimeError(f'public trainer action trace is incomplete: {spec["name"]}')
            json.dumps(result, allow_nan=False)
        except Exception:
            result = dict(error=traceback.format_exc(), seconds=time.perf_counter()-started)
            context = dict(applied=[], shapes={}, host_recipe=base)
        verdict = test_verdict(spec, result)
        ema = ema_verdict(spec, result)
        record = dict(name=spec['name'], route=route, recipe=legacy_dict(base),
                      host_recipe=legacy_dict(context['host_recipe']),
                      original_spec=deepcopy(job['spec']), spec=spec,
                      discriminator_variant=variant,
                      architecture=variant['name'] if variant else job['architecture'],
                      reference=job['reference'], reference_sha256=job['reference_sha256'],
                      applied=context['applied'], shapes=context['shapes'],
                      public_package_file=package['package_file'],
                      verdict=verdict, ema_verdict=ema, result=result,
                      source_sha256=protocol['source_sha256'])
        raw = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
        artifact = f'episodes/gan_v3__{spec["name"]}.json.gz'
        (output/artifact).write_bytes(gzip.compress(raw, mtime=0))
        records.append({k: v for k, v in record.items() if k not in ('result', 'source_sha256')} |
                       dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                            live=result.get('live'), ema=result.get('ema'),
                            observations=len(result.get('observations', result.get('curve', []))),
                            seconds=result['seconds']))
        write(output/'index.json', dict(records=records))
        public_module_manifest(require_installed_root)
        print(json.dumps(dict(event='DONE', task=spec['name'], route=route,
                              status=verdict['status'], suffix=verdict.get('convergence', {}).get('passing_suffix'),
                              live=result.get('live'), error=result.get('error'))), flush=True)
    suite.verify_source(protocol)
    public_module_manifest(require_installed_root)
    complete = len(records) == 19
    passed = sum(row['verdict']['passed'] for row in records)
    write(output/'summary.json', dict(version='public-default-verification-v1',
         attempted=len(records), passed=passed, overall='PASS' if complete and passed == 19 else
         'FAIL' if complete else 'INCOMPLETE', routes=protocol['routes'],
         public_package=protocol['public_package']['package_file'],
         cases=[dict(name=row['name'], route=row['route'], live=row['verdict']['status'],
                     ema=row['ema_verdict']['status'], observations=row['observations'],
                     artifact=row['artifact']) for row in records]))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--require-installed-root', type=Path)
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.output, tasks=args.tasks, require_installed_root=args.require_installed_root)
