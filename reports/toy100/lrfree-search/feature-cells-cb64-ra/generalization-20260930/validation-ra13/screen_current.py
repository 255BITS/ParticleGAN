"""Fast multi-task screen for public-GANTrainer candidates.

Reuses the frozen PR155 new-init hosts (mode_hold, image, vector, ring) with their
exact construction order, data/latent streams, observation schedules, scorers and
pass rules.  The sealing/hash/declaration ceremony is dropped; stream identity is
still checked every update (strict by default).  See ../README.md.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import traceback
from current_api_fixtures import frozen_recipe, frozen_prior, frozen_trainer
sys.dont_write_bytecode = True
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
HOSTS = HARNESS / 'hosts'
TASKS = HARNESS / 'tasks'
IMAGE_TASKS = ('img_intensity2', 'img_blobs4', 'img_stripes2', 'img_bars4')
VECTOR_TASKS = tuple(json.loads((TASKS / 'vector_task_specs.json').read_text()))
RING_TASKS = ('ring_shift', 'stationary')
NATIVE_TASKS = ('grid100', 'rotated100', 'staggered100')
ALL_TASKS = ('mode_hold',) + IMAGE_TASKS + VECTOR_TASKS + RING_TASKS + NATIVE_TASKS
CUSTOM_TASKS = ('two_pole', 'trajectory', 'residual_student', 'unipolar', 'ae_gan_hold', 'cover_leftover', 'unused_token_hold', 'mid_scale_identity')
DEFAULT_OPTIONS = dict(evaluation_generate='auto', serial_backward_argument='auto', strict_streams=True, initialization='batch_feature_zero', diagnostics=True, save_final_state=False, ring_frozen_control=False, eval_output_noise=False, image_prior_perturb=False, image_steps=None, native_steps=None)
BEHAVIOR_OPTIONS = ('evaluation_generate', 'serial_backward_argument', 'strict_streams', 'initialization', 'eval_output_noise', 'image_prior_perturb')

class StreamError(RuntimeError):
    pass

def load_json_arg(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = value.strip()
    if text.startswith('{'):
        return json.loads(text)
    return json.loads(Path(value).read_text())

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def finite(value):
    if isinstance(value, float) and (not math.isfinite(value)):
        return str(value)
    return value

def jsonable(torch, value, depth=0):
    """Compact JSON view of diagnostics; never mutates learner state."""
    if depth > 6:
        return str(type(value).__name__)
    if isinstance(value, torch.Tensor):
        v = value.detach()
        if v.numel() == 1:
            return finite(v.item())
        if v.numel() <= 16:
            return [finite(x) for x in v.flatten().tolist()]
        f = v.float()
        return dict(shape=list(v.shape), mean=finite(float(f.mean())), min=finite(float(f.min())), max=finite(float(f.max())))
    if isinstance(value, dict):
        return {str(k): jsonable(torch, v, depth + 1) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return f'<{type(value).__name__} len={len(value)}>'
        return [jsonable(torch, v, depth + 1) for v in value]
    if isinstance(value, float):
        return finite(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)

def dump(path, value):
    path.write_text(json.dumps(value, indent=1, default=str) + '\n')

def package_digest(package_root):
    root = Path(package_root).resolve() / 'particlegan'
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(str(path.relative_to(root)).encode() + b'\x00' + path.read_bytes() + b'\x00')
    return h.hexdigest()

def tensor_receipt(torch, tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

class Context:

    def __init__(self, args, torch, package, overrides, options, out):
        (self.args, self.torch, self.package, self.out) = (args, torch, package, out)
        (self.overrides, self.options) = (overrides, options)
        self.observations = []
        self.alt_observations = []
        self.alt_key = 'clean' if options['eval_output_noise'] else 'noisy'
        self.warnings = []
        self.stream_deviations = 0
        self.started = time.monotonic()
        from particlegan import training
        self.input_noise_std = getattr(training, 'input_noise_std', None)
        self.output_noise_std = training.output_noise_std
        self.probe_real = None
        self.metrics = open(out / 'metrics.jsonl', 'w', buffering=1)
        self.rates = open(out / 'rates.jsonl', 'w', buffering=1)

    def deviation(self, step, what):
        self.stream_deviations += 1
        if self.options['strict_streams']:
            raise StreamError(f'step {step}: {what} (frozen stream transaction changed; pass candidate option strict_streams=false to continue with the frozen data order)')
        if self.stream_deviations <= 5:
            self.warnings.append(f'step {step}: {what}')

    def serial(self):
        torch = self.torch

        @contextmanager
        def scope():
            previous = torch.autograd.is_multithreading_enabled()
            with torch.autograd.set_multithreading_enabled(False):
                yield
            assert torch.autograd.is_multithreading_enabled() == previous
        return scope()

    def trainer_options(self, **options):
        options.setdefault('optimizer_options', {'foreach': False, 'fused': False})
        if self.options['serial_backward_argument']:
            options['serial_backward'] = True
        return options

    def sigma(self, trainer, completed=None):
        """Output-noise std the candidate currently samples with: GANTrainer.output_sigma() when the
        package has it (learnable / mobility output noise), else the module-level output_noise_std."""
        current = getattr(trainer, 'output_sigma', None)
        if callable(current):
            return current()
        return self.output_noise_std(trainer.recipe, trainer.completed_steps if completed is None else completed)

    def rate_row(self, trainer, step):
        row = dict(step=step, lr=[[g['lr'] for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)])
        recipe = trainer.recipe
        if self.input_noise_std is not None:
            row['in_noise'] = self.input_noise_std(recipe, step - 1)
        applied = getattr(trainer, 'last_output_sigma', None)
        row['out_noise'] = self.output_noise_std(recipe, step - 1) if applied is None else applied
        self.rates.write(json.dumps(row, default=float) + '\n')

    def diagnostics(self, trainer):
        if not self.options['diagnostics']:
            return {}
        (torch, out) = (self.torch, {})
        for name in ('controller', 'precision', 'penalty', 'lr_settle', 'birth_death'):
            obj = getattr(trainer, name, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, 'diagnostics'):
                    out[name] = jsonable(torch, obj.diagnostics())
                elif name == 'precision' and hasattr(obj, 'state'):
                    out[name] = jsonable(torch, obj.state)
            except Exception as error:
                out[name] = f'error: {error!r}'
        if hasattr(trainer, 'game_stats'):
            try:
                out['game_stats'] = jsonable(torch, trainer.game_stats)
            except Exception as error:
                out['game_stats'] = f'error: {error!r}'
        try:
            out['sched'] = self.schedule_diagnostics(trainer)
        except Exception as error:
            out['sched'] = f'error: {error!r}'
        return out

    def schedule_diagnostics(self, trainer):
        """Recorded only (no RNG, no parameter/grad/optimizer change): controller pe / m / gt, applied LR
        scales (group LR / initial group LR), current output sigma, and the critic's real-data input-gradient
        norm mean_i ||grad_x D(x_i)|| on the latest D-real batch (post-update critic)."""
        torch = self.torch
        out = {}
        c = getattr(trainer, 'controller', None)
        if c is not None:
            out.update(pe=getattr(c, 'payoff_error', None), m=getattr(c, 'mobility', None), gt=getattr(c, 'game_trust', None))
            if hasattr(c, 'critic_scale'):
                out['critic_payoff_factor'] = c.critic_scale()
        initial = getattr(trainer, 'initial_lrs', None)
        if initial:
            scales = [[g['lr'] / r for (g, r) in zip(o.param_groups, rates)] for (o, rates) in zip((trainer.opt_g, trainer.opt_d), initial)]
            out['g_lr_scale'] = scales[0]
            out['d_lr_scale'] = scales[1][0] if len(scales[1]) == 1 else scales[1]
        out['output_sigma'] = float(self.sigma(trainer))
        if getattr(trainer, 'log_output_sigma', None) is not None:
            out['log_output_sigma'] = float(trainer.log_output_sigma.detach())
        mode = getattr(getattr(trainer, 'recipe', None), 'output_noise_mode', None)
        if mode is not None:
            out['output_noise_mode'] = mode
        real = self.probe_real
        if real is not None:
            with torch.enable_grad():
                x = real.detach().clone().requires_grad_(True)
                logits = trainer.D(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                score = logits.flatten(1).mean(1).sum() if logits.ndim >= 2 else logits.sum()
                g = torch.autograd.grad(score, x)[0]
            out['real_grad_norm'] = float(g.flatten(1).norm(dim=1).mean())
        return out

    def pick(self, step, live, ema):
        """live/ema = (clean, noisy) metric pairs -> (primary point, secondary point)."""
        p = 1 if self.options['eval_output_noise'] else 0
        return (dict(step=step, **live[p], ema=ema[p]), dict(step=step, **live[1 - p], ema=ema[1 - p]))

    def observe(self, point, passed, trainer=None, extra=None, alt=None, alt_passed=None):
        point = dict(point)
        point['pass'] = bool(passed)
        if alt is not None:
            alt = dict(alt, **{'pass': bool(alt_passed)})
            self.alt_observations.append(alt)
            point[self.alt_key] = {k: v for (k, v) in alt.items() if k != 'step'}
        point['seconds'] = round(time.monotonic() - self.started, 3)
        if trainer is not None:
            point['lr'] = [[g['lr'] for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)]
            diag = self.diagnostics(trainer)
            if diag:
                point['diag'] = diag
        if extra:
            point.update(extra)
        self.observations.append(point)
        self.metrics.write(json.dumps(point, default=str) + '\n')
        keys = [k for k in ('modes', 'hq', 'sw1_normalized', 'mass_tv', 'min_mass_ratio', 'component_covariance_error', 'mean_error', 'covariance_error') if k in point]
        compact = dict(step=point['step'], ok=int(bool(passed)), **{k: round(point[k], 4) if isinstance(point[k], float) else point[k] for k in keys})
        if isinstance(point.get('ema'), dict):
            compact['ema'] = {k: round(point['ema'][k], 4) if isinstance(point['ema'][k], float) else point['ema'][k] for k in keys if k in point['ema']}
        print(json.dumps(compact), flush=True)

    def close(self):
        self.metrics.close()
        self.rates.close()

def deterministic(torch):
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0)

def summary_from_convergence(convergence, observations):
    final = {k: v for (k, v) in observations[-1].items() if k not in ('ema', 'diag', 'lr', 'noisy', 'clean')}
    return dict(passing_checks=convergence['passing_observations'], observations=convergence['observations'], first_arrival=convergence['first_pass_step'], final_streak=convergence['passing_suffix'], final=final, ema_final=observations[-1].get('ema'), final_lr=observations[-1].get('lr'))

def alt_summary(ctx, status, convergence):
    """Secondary verdict: noisy_* (clean primary, default) or clean_* (eval_output_noise=true)."""
    (alt, obs) = (ctx.alt_key, ctx.alt_observations)
    return {f'{alt}_status': status, f'{alt}_passing_checks': convergence['passing_observations'], f'{alt}_first_arrival': convergence['first_pass_step'], f'{alt}_final_streak': convergence['passing_suffix'], f'{alt}_final': {k: v for (k, v) in obs[-1].items() if k not in ('ema', 'pass')}, 'eval_output_noise': bool(ctx.options['eval_output_noise'])}

def sharp_nearest(torch, x, centres, std):
    """Recorded only: median over samples of (distance to nearest mode centre / that mode's data std).
    Ideal for an isotropic 2-D Gaussian mode: sqrt(2 ln 2) = 1.1774."""
    return float((torch.cdist(x.float(), centres.float()).min(1).values / std).median())

def sharp_rmse(images, centers):
    """Recorded only: median per-sample rmse to the nearest template."""
    return float((images[:, None] - centers[None]).square().mean((2, 3, 4)).sqrt().min(1).values.median())

@contextmanager
def host_device(torch, device):
    """Verbatim frozen mode_hold_contract.host_device."""
    previous = str(torch.get_default_device())
    original = torch.Generator

    class DefaultGenerator(original):

        def __new__(cls, device=None):
            return original.__new__(cls, target if device is None else device)

        def __init__(self, device=None):
            pass
    target = str(device)
    try:
        with torch.device(target):
            torch.Generator = DefaultGenerator
            yield
    finally:
        torch.Generator = original
        if str(torch.get_default_device()) != previous:
            raise RuntimeError('host device scope leaked into learner')

def run_mode_hold(ctx):
    (torch, package, options) = (ctx.torch, ctx.package, ctx.options)
    host = load_module('lrfree_mode_hold_host', HOSTS / 'mode_hold_host.py')
    protocol = json.loads((TASKS / 'mode_hold_protocol.json').read_text())
    fixture = json.loads((HOSTS / 'mode_hold_fixture_receipt.json').read_text())
    with gzip.open(HOSTS / 'mode_hold_batch_receipts.jsonl.gz', 'rt') as handle:
        expected_batches = [json.loads(line) for line in handle]
    device = 'cuda:0'
    deterministic(torch)
    stream = torch.Generator(device=device).manual_seed(0)
    overrides = dict(ctx.overrides)
    overrides.update(num_particles=12, z_dim=4, batch_size=128)
    recipe = frozen_recipe(package, **overrides)
    with host_device(torch, device):
        means = host.ring_means()
        prior = frozen_prior(recipe, init_std=0.5, generator=stream)
        generator = host.SimpleMLPGenerator(4, 96, 3, 2)
        critic = host.SimpleMLPDiscriminator(2, 96, 3, 3)
    trainer = frozen_trainer(package, recipe, generator, critic, **ctx.trainer_options(prior=prior, seed=0, latent_generator=stream))
    assert str(torch.get_default_device()) == 'cpu'
    raw = fixture['expected_initial']
    rng_match = {'data_rng': tensor_receipt(torch, stream.get_state()) == raw['data_rng'], 'cpu_rng': tensor_receipt(torch, torch.get_rng_state()) == raw['cpu_rng'], 'cuda_rng': tensor_receipt(torch, torch.cuda.get_rng_state()) == raw['cuda_rng']}
    if not rng_match['data_rng']:
        ctx.deviation(0, 'data stream position after prior construction differs from frozen fixture')
    if not all(rng_match.values()):
        ctx.warnings.append(f'construction RNG vs frozen fixture: {rng_match}')

    def peek_batch(step):
        """Verbatim frozen mode_hold_contract.peek_batch."""
        with host_device(torch, device):
            real = host.sample_ring(means, 128, host.SIGMA, stream)
        future = torch.Generator(device=device)
        future.set_state(stream.get_state())
        latent_d = trainer.prior.sample_indices(128, generator=future)
        latent_g = trainer.prior.sample_indices(128, generator=future)
        before_g_real = future.get_state()
        with host_device(torch, device):
            real_g = host.sample_ring(means, 128, host.SIGMA, future)
        after_g_real = future.get_state()
        receipt = dict(step=step, real_d=host.digest(real), latent_d=host.digest(latent_d), latent_g=host.digest(latent_g), real_g=host.digest(real_g), accepted_cursor=host.digest(after_g_real))
        return (real, real_g, before_g_real, after_g_real, receipt)

    def measure(ema=False):
        """Verbatim frozen mode_hold_contract.measure."""
        (model, table) = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
        modes = [(m, m.training) for root in (model, table) for m in root.modules()]
        dev = next(model.parameters()).device
        devices = [dev.index] if dev.type == 'cuda' else []
        try:
            model.eval()
            table.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                torch.manual_seed(402 + trainer.completed_steps)
                (latent, indices) = table.sample(4096, generator=torch.Generator(device=dev).manual_seed(9))
                local_noise = torch.Generator(device=dev).manual_seed(2303 + trainer.completed_steps)
                if options['evaluation_generate'] == 'indexed':
                    fake = trainer._generate(model, latent, 0.0, local_noise, indices)
                else:
                    fake = trainer._generate(model, latent, 0.0, local_noise)
                sigma = ctx.sigma(trainer)
                noisy_x = fake + sigma * torch.randn_like(fake) if sigma else fake
                noisy = host.diversity(noisy_x, means) if sigma else None
                clean = host.diversity(fake, means)
                sharp = dict(sharp_clean=sharp_nearest(torch, fake, means, host.SIGMA), sharp_noisy=sharp_nearest(torch, noisy_x, means, host.SIGMA))
                noisy = noisy if sigma else dict(clean)
                clean.update(sharp)
                noisy.update(sharp)
                return (clean, noisy)
        finally:
            for (module, flag) in modes:
                module.training = flag
    requirements = host.requirements(protocol['host'])
    for step in range(1, 1201):
        (real, real_g, before_g, after_g, receipt) = peek_batch(step)
        ctx.probe_real = real
        if receipt != expected_batches[step - 1]:
            ctx.deviation(step, 'batch/latent receipt differs from frozen fixture')
        with ctx.serial():
            trainer.step(real, generator_real=real_g, collect_stats=step % 50 == 0)
        assert trainer.completed_steps == step
        if not torch.equal(stream.get_state(), before_g):
            ctx.deviation(step, 'candidate consumed the shared latent stream differently (expected two index draws)')
        stream.set_state(after_g)
        ctx.rate_row(trainer, step)
        if step % 50 == 0:
            (point, alt) = ctx.pick(step, measure(), measure(True))
            ok = [all((c['status'] == 'PASS' for c in host.score_metrics(x, requirements))) for x in (point, alt)]
            ctx.observe(point, ok[0], trainer, alt=alt, alt_passed=ok[1])
    if tensor_receipt(torch, stream.get_state()) != fixture['expected_final_data_rng']:
        ctx.deviation(1200, 'final data cursor differs from frozen fixture')

    def judge(obs):
        verdict = host.test_verdict(protocol['host'], dict(live=obs[-1], observations=obs))
        return ('PASS' if verdict['passed'] and verdict['convergence']['passing_suffix'] >= 5 else 'FAIL', verdict)
    (status, verdict) = judge(ctx.observations)
    (alt_status, alt_verdict) = judge(ctx.alt_observations)
    return (trainer, dict(status=status, verdict=verdict, **summary_from_convergence(verdict['convergence'], ctx.observations), **alt_summary(ctx, alt_status, alt_verdict['convergence'])))

def run_image(ctx, task):
    (torch, package, options) = (ctx.torch, ctx.package, ctx.options)
    host = load_module('lrfree_image_host', HOSTS / 'image_host.py')
    spec = json.loads((TASKS / 'image_task_specs.json').read_text())[task]
    device = 'cuda:0'
    deterministic(torch)
    centers = host.templates(spec).to(device)
    overrides = dict(ctx.overrides)
    overrides.update(num_particles=spec['particles'], z_dim=spec['z_dim'], batch_size=spec['batch_size'])
    recipe = frozen_recipe(package, **overrides)
    (generator, critic) = (host.Generator(spec), host.Discriminator(spec))
    prior = frozen_prior(recipe)
    (generator, critic, prior) = (generator.to(device), critic.to(device), prior.to(device))
    shared = torch.cuda.default_generators[0]
    trainer = frozen_trainer(package, recipe, generator, critic, **ctx.trainer_options(prior=prior, seed=0, latent_generator=shared, penalty_generator=shared))
    assert trainer.completed_steps == 0
    untouched = torch.Generator(device=device).manual_seed(0)
    if not torch.equal(shared.get_state(), untouched.get_state()):
        ctx.deviation(0, 'construction consumed the shared global CUDA data/latent stream')
    expected = host.evaluation_steps(spec)
    assert expected == list(range(25, 601, 25))
    if options.get('image_steps') is not None:
        spec = dict(spec, steps=int(options['image_steps']))
        assert spec['steps'] >= 600 and spec['steps'] % 25 == 0, 'image_steps must be a multiple of 25, >= 600'
        expected = list(range(25, spec['steps'] + 1, 25))
    batch = spec['batch_size']

    def data():
        real = centers[torch.randint(len(centers), (batch,), device=device)]
        noise = torch.randn_like(real)
        return (real + spec['noise_std'] * noise).clamp(0.0, 1.0)

    def measure(ema=False):
        """Verbatim frozen image_screen.measure."""
        (model, prior) = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            sigma = ctx.sigma(trainer)
            (out, sharp) = ([], [])
            for s in (sigma, 0.0) if sigma else (0.0,):
                stream = torch.Generator(device=device).manual_seed(402 + trainer.completed_steps + 1901)
                latent = prior.perturb(prior.z, generator=stream) if options['image_prior_perturb'] else prior.z
                arguments = [model, latent, s, stream]
                if options['evaluation_generate'] == 'indexed':
                    arguments.append(torch.arange(prior.num_particles, device=prior.z.device))
                images = trainer._generate(*arguments)
                out.append(host.image_metrics(images, centers, spec['thresholds']))
                sharp.append(sharp_rmse(images, centers))
            sharp = dict(sharp_rmse_clean=sharp[-1], sharp_rmse_noisy=sharp[0])
            (clean, noisy) = (out[1], out[0]) if sigma else (out[0], dict(out[0]))
            clean.update(sharp)
            noisy.update(sharp)
            return (clean, noisy)
    requirements = [('modes', '>=', spec['thresholds']['modes']), ('hq', '>=', spec['thresholds']['hq_min'])]
    oracle = torch.Generator(device=device)
    for completed in range(1, spec['steps'] + 1):
        real = data()
        oracle.set_state(shared.get_state())
        torch.randint(spec['particles'], (batch,), device=device, generator=oracle)
        torch.randint(spec['particles'], (batch,), device=device, generator=oracle)
        cursor = oracle.get_state()
        ctx.probe_real = real
        with ctx.serial():
            trainer.step(real, generator_real=real, collect_stats=completed in expected)
        assert trainer.completed_steps == completed
        assert str(torch.get_default_device()) == 'cpu'
        if not torch.equal(shared.get_state(), cursor):
            ctx.deviation(completed, 'shared global CUDA cursor differs from real+2 latent draws')
            shared.set_state(cursor)
        ctx.rate_row(trainer, completed)
        if completed in expected:
            (point, alt) = ctx.pick(completed, measure(), measure(True))
            ok = [all((x[k] >= v for (k, _, v) in requirements)) for x in (point, alt)]
            ctx.observe(point, ok[0], trainer, alt=alt, alt_passed=ok[1])

    def judge(obs):
        convergence = host.sustained(obs, requirements, expected_steps=expected, minimum=spec['thresholds']['minimum_stable_checks'])
        return ('PASS' if convergence['complete'] and convergence['passing_suffix'] >= 5 else 'FAIL', convergence)
    (status, convergence) = judge(ctx.observations)
    (alt_status, alt_convergence) = judge(ctx.alt_observations)
    return (trainer, dict(status=status, convergence=convergence, **summary_from_convergence(convergence, ctx.observations), **alt_summary(ctx, alt_status, alt_convergence)))

def run_vector(ctx, task_name):
    (torch, package, options) = (ctx.torch, ctx.package, ctx.options)
    host = load_module('lrfree_vector_host', HOSTS / 'vector_host.py')
    task = json.loads((TASKS / 'vector_task_specs.json').read_text())[task_name]
    cfg = host.resolve(task['spec'])
    card = task['discriminator_card']
    device = 'cuda:0'
    deterministic(torch)
    recipe_args = dict(ctx.overrides)
    recipe_args.update(num_particles=cfg['particles'], z_dim=cfg['z_dim'], batch_size=cfg['batch'])
    recipe = frozen_recipe(package, **recipe_args)
    prior = frozen_prior(recipe, init_std=0.5, generator=torch.Generator(device='cpu').manual_seed(0))
    generator = host.SimpleMLPGenerator(cfg['z_dim'], cfg['hidden'], cfg['layers'], 2)
    if card is None:
        critic = host.SimpleMLPDiscriminator(2, cfg.get('d_hidden', cfg['hidden']), cfg.get('d_layers', cfg['layers']), cfg['fourier'])
    elif card['implementation'] == 'shared_batch_feature_v1':
        critic = package.BatchDistanceDiscriminator(in_dim=2, hidden_dim=card['width'], n_hidden=card['layers'], scales=tuple(card['kernel_scales']), beta=card['softplus_beta'], eps=card['eps'])
    elif card['implementation'] == 'shared_critic_v1':
        sys.path.insert(0, str(HOSTS))
        from frozen_shared_critic import constructor
        critic = constructor(card)(2, card['hidden'], card['layers'], card['fourier'])
    else:
        raise ValueError('unknown frozen discriminator card')
    data = torch.Generator(device=device).manual_seed(0)
    trainer = frozen_trainer(package, recipe, generator.to(device), critic.to(device), **ctx.trainer_options(prior=prior.to(device), seed=0, latent_generator=torch.Generator(device=device).manual_seed(1), penalty_generator=torch.Generator(device=device).manual_seed(2)))
    steps = cfg['steps']
    expected = [math.ceil(i * steps / 24) for i in range(1, 25)]

    def real(step):
        with torch.device(device):
            return host.sample_target(cfg, cfg['batch'], data, step)

    def measure(completed, ema=False):
        """Verbatim frozen vector_screen.measure."""
        (model, prior) = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            (latent, indices) = prior.sample(4096, generator=torch.Generator(device=device).manual_seed(990))
            sigma = ctx.sigma(trainer, completed)
            out = []
            for s in (sigma, 0.0) if sigma else (0.0,):
                stream = torch.Generator(device=device).manual_seed(402 + 1901)
                arguments = [model, latent, s, stream]
                if options['evaluation_generate'] == 'indexed':
                    arguments.append(indices)
                fake = trainer._generate(*arguments)
                with torch.device(device):
                    out.append(host.score_samples(fake, cfg, completed))
            return (out[1], out[0]) if sigma else (out[0], dict(out[0]))
    requirements = host.requirements(task['spec'])
    future = torch.Generator(device=device)
    for step in range(1, steps + 1):
        real_d = real(step)
        ctx.probe_real = real_d
        future.set_state(trainer.latent_generator.get_state())
        trainer.prior.sample_indices(cfg['batch'], generator=future)
        trainer.prior.sample_indices(cfg['batch'], generator=future)
        latent_cursor = future.get_state()
        cache = []

        def generator_real():
            if not cache:
                cache.append(real(step))
            else:
                ctx.warnings.append(f'step {step}: generator_real called more than once (cached tensor reused)') if len(ctx.warnings) < 20 else None
            return cache[0]
        with ctx.serial():
            trainer.step(real_d, generator_real=generator_real, collect_stats=step in expected)
        assert trainer.completed_steps == step
        assert str(torch.get_default_device()) == 'cpu'
        if not cache:
            ctx.deviation(step, 'candidate never called generator_real')
            generator_real()
        if not torch.equal(trainer.latent_generator.get_state(), latent_cursor):
            ctx.deviation(step, 'candidate latent stream consumption differs from two index draws')
        ctx.rate_row(trainer, step)
        if step in expected:
            (point, alt) = ctx.pick(step, measure(step), measure(step, True))
            ok = [all((c['status'] == 'PASS' for c in host.score_metrics(x, requirements))) for x in (point, alt)]
            ctx.observe(point, ok[0], trainer, alt=alt, alt_passed=ok[1])

    def judge(obs):
        verdict = host.test_verdict(task['spec'], dict(observations=obs, live=obs[-1]))
        return ('PASS' if verdict['passed'] and verdict['convergence']['passing_suffix'] >= 5 else 'FAIL', verdict)
    (status, verdict) = judge(ctx.observations)
    (alt_status, alt_verdict) = judge(ctx.alt_observations)
    return (trainer, dict(status=status, verdict=verdict, thresholds=requirements, **summary_from_convergence(verdict['convergence'], ctx.observations), **alt_summary(ctx, alt_status, alt_verdict['convergence'])))

def run_ring(ctx, task):
    """Original public recovery ring (DV2/DV3/RP12 new-init single-shift worker)."""
    (torch, package) = (ctx.torch, ctx.package)
    host = load_module('lrfree_ring_host', HOSTS / 'ring_host.py')
    reference = json.loads((TASKS / 'ring_reference_declaration_dv2.json').read_text())
    total = 4600 if task == 'ring_shift' else 7500
    change = 2400 if task == 'ring_shift' else None
    device = 'cuda:0'
    deterministic(torch)
    overrides = dict(ctx.overrides)
    overrides.update(num_particles=20000, z_dim=2, batch_size=2048)
    recipe = frozen_recipe(package, **overrides)

    def make_trainer():
        torch.manual_seed(0)
        generator = host.SimpleMLPGenerator(recipe.z_dim, 96, 3, 2).to(device)
        critic = host.SimpleMLPDiscriminator(2, 96, 3, 3).to(device)
        return frozen_trainer(package, recipe, generator, critic, **ctx.trainer_options(seed=0))
    trainer = make_trainer()
    stream = torch.Generator(device=device).manual_seed(0)
    means = host.ring_means().to(device)
    original_means = means.clone()
    try:
        receipt = host.state_receipt(trainer, stream, means)
        wanted = reference['expected_initial_fixture']
        match = {k: receipt.get(k) == v for (k, v) in wanted.items()}
        if not all(match.values()):
            ctx.warnings.append(f'ring construction RNG vs DV2 reference: {match}')
    except Exception as error:
        ctx.warnings.append(f'ring initial receipt unavailable: {error!r}')

    def measure(value, ema=False, both=False):
        with torch.random.fork_rng(devices=[0]):
            isolated = torch.Generator(device=device).manual_seed(9)
            noisy = host.diversity(value.sample(4096, ema=ema, generator=isolated, output_noise=True), means)
            sigma = ctx.sigma(value)
            if not sigma:
                pair = (noisy, dict(noisy))
            else:
                isolated = torch.Generator(device=device).manual_seed(9)
                clean = host.diversity(value.sample(4096, ema=ema, generator=isolated, output_noise=False), means)
                pair = (clean, noisy)
        if both:
            return pair
        return pair[1 if ctx.options['eval_output_noise'] else 0]
    frozen = None
    for step in range(1, total + 1):
        indices = torch.randint(0, 8, (2048,), device=device, generator=stream)
        real = means[indices] + 0.07 * torch.randn(2048, 2, device=device, generator=stream)
        ctx.probe_real = real
        with ctx.serial():
            trainer.step(real, generator_real=real, collect_stats=step % 10 == 0)
        assert trainer.completed_steps == step
        ctx.rate_row(trainer, step)
        if step % 10 == 0:
            (point, alt) = ctx.pick(step, measure(trainer, both=True), measure(trainer, True, both=True))
            extra = None
            if frozen is not None:
                extra = dict(frozen=measure(frozen))
            ctx.observe(point, host.good(point), trainer if step % 100 == 0 else None, extra, alt=alt, alt_passed=host.good(alt))
        if change is not None and step == change:
            if ctx.options['ring_frozen_control']:
                (cpu, cuda) = (torch.get_rng_state(), torch.cuda.get_rng_state())
                saved = trainer.state_dict()
                frozen = make_trainer()
                frozen.load_state_dict(saved)
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state(cuda)
            means.copy_(original_means + means.new_tensor([1.0, 0.0]))
            print(json.dumps(dict(event='shift', after_step=change, absolute_offset=[1.0, 0.0])), flush=True)

    def judge(obs):
        if change is None:
            segments = [host.segment(obs, 0, total)]
        else:
            segments = [host.segment(obs, 0, change), host.segment(obs, change, total)]
        ok = all((s['first_arrival'] is not None and s['stable_suffix_checks'] >= 5 for s in segments))
        return ('PASS' if ok else 'FAIL', segments)
    obs = ctx.observations
    (status, segments) = judge(obs)
    (alt_status, alt_segments) = judge(ctx.alt_observations)
    compact = []
    for s in segments:
        compact.append(dict(start=s['start'], end=s['end'], arrival=s['first_arrival'], delay=s['delay'], retained=s['passing_since_arrival'], checks_since_arrival=s['checks_since_arrival'], departures=len(s['departures']), first_departures=s['departures'][:5], final_suffix=s['stable_suffix_checks'], min_hq_since_arrival=s['minimum_hq_since_arrival'], min_modes_since_arrival=s['minimum_modes_since_arrival']))
    first = segments[0]
    final = {k: v for (k, v) in obs[-1].items() if k not in ('ema', 'diag', 'lr', 'frozen', 'noisy', 'clean')}
    alt = ctx.alt_key
    alt_obs = ctx.alt_observations
    result = dict(status=status, segments=compact, segments_full=segments, **{f'{alt}_status': alt_status, f'{alt}_passing_checks': sum((bool(p['pass']) for p in alt_obs)), f'{alt}_first_arrival': alt_segments[0]['first_arrival'], f'{alt}_final_streak': alt_segments[-1]['stable_suffix_checks'], f'{alt}_segments': [dict(arrival=s['first_arrival'], delay=s['delay'], retained=s['passing_since_arrival'], checks_since_arrival=s['checks_since_arrival'], departures=len(s['departures']), final_suffix=s['stable_suffix_checks']) for s in alt_segments], f'{alt}_final': {k: v for (k, v) in alt_obs[-1].items() if k not in ('ema', 'pass')}}, eval_output_noise=bool(ctx.options['eval_output_noise']), passing_checks=sum((bool(p['pass']) for p in obs)), observations=len(obs), first_arrival=first['first_arrival'], final_streak=segments[-1]['stable_suffix_checks'], final=final, ema_final=obs[-1].get('ema'), final_lr=obs[-1].get('lr'), pass_rule='harness rule: every segment arrives (8 modes, HQ>=.9) and ends with >=5 consecutive passing checks')
    if change is not None:
        result['comparison3600'] = host.segment(obs, change, 3600)
    return (trainer, result)
NATIVE_COVERAGE_THRESHOLDS = [('modes', '>=', 100), ('precision', '>=', 0.97), ('min_hq_mode_mass', '>=', 0.005), ('mass_tv', '<=', 0.1), ('max_mode_mass', '<=', 0.02), ('min_cov_eig_ratio', '>=', 0.4), ('max_cov_eig_ratio', '<=', 1.7), ('min_radial_median_ratio', '>=', 0.65), ('max_radial_median_ratio', '<=', 1.4)]
NATIVE_ACCURACY_THRESHOLDS = [('acc_mass_tv', '<=', 0.06), ('acc_center_rms_sigma', '<=', 0.2), ('acc_abs_cov_trace_bias', '<=', 0.1), ('acc_radial_ks', '<=', 0.04)]
NATIVE_ACC_KEYS = ('mass_tv', 'center_rms_sigma', 'abs_cov_trace_bias', 'cov_trace_bias', 'radial_ks', 'accuracy_score', 'accuracy_pass', 'frozen_pass', 'passed')

def native_evaluation_steps(budget, interval, early):
    """Verbatim benchmarks/toy100/train.py::evaluation_steps (validation stripped)."""
    return sorted({0, budget} | {s for s in early if s <= budget} | set(range(interval, budget + 1, interval)))

def run_native(ctx, task):
    """Frozen native100 host (grid100 / rotated100 / staggered100) with the candidate's public GANTrainer.

    Host = configs/toy100/constraints_simple_regularization.json resources (7000 updates, seed 1234, 20000 particles,
    z 2, batch 2048, affine_square_v1 G, SimpleMLPDiscriminator(2,128,3,fourier=3)) on the canonical native CUDA
    fixture (gpu-known-winner-control worker: construction under the CUDA default device, prior normal draw ->
    uniform[-5,5] redraw -> Linear(2,2) defaults -> identity/zero -> D defaults -> Xavier/zero; archived prior/G
    hashes asserted). Learner = candidate package + recipe (resources forced; batch_feature_zero keeps the
    identity G / zero biases / supplied prior and replaces D's Xavier weights, like every other harness task).
    Data: own CUDA stream seed 1234, D-real then a fresh G-real batch per update (generator_real).
    Evaluation (34 obs: 0,1,10,25,50,100, every 250 to 7000; live+EMA): 20000 prior draws with replacement
    (latent stream 1637) through the candidate's `_generate` with sigma 0 (= public `sample` with no output
    noise) -> clean cloud; noisy cloud = clean + sigma_candidate * randn under the forked global seed 1636
    (paired live/EMA, the frozen protocol). Target 1635; final-five quality clouds; 100k holdout 2835/2836/2837.
    Evidence for both clouds is written in the native run-directory format and graded by the UNCHANGED
    gate.score_run + accuracy_gate.score_run (harness/native100_score.py, separate process).
    """
    import subprocess
    import numpy as np
    import native100_diagnostics
    from torch import nn
    (torch, package, options) = (ctx.torch, ctx.package, ctx.options)
    fixture = json.loads((TASKS / 'native100_fixture.json').read_text())
    frozen_path = TASKS / 'native100_constraints_simple_regularization.json'
    local = {'configs/toy100/constraints_simple_regularization.json': frozen_path, 'benchmarks/toy100/problems.py': HOSTS / 'native100' / 'problems.py', 'benchmarks/toy100/metrics.py': HOSTS / 'native100' / 'metrics.py', 'benchmarks/toy100/accuracy.py': HOSTS / 'native100' / 'accuracy.py', 'lib/toy_models.py': HOSTS / 'native100' / 'toy_models.py'}
    for (name, path) in local.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != fixture['host_source_sha256'][name]:
            raise RuntimeError(f'native100 host copy differs from the frozen source: {name}')
    if str(HOSTS) not in sys.path:
        sys.path.insert(0, str(HOSTS))
    problems = importlib.import_module('native100.problems')
    metrics = importlib.import_module('native100.metrics')
    accuracy = importlib.import_module('native100.accuracy')
    toy_models = importlib.import_module('native100.toy_models')
    frozen = json.loads(frozen_path.read_text())
    config = {k: frozen[k] for k in fixture['host_fields']}
    config.update(problem=task, device='cuda:0', host='frozen native100 (constraints_simple_regularization.json resources, canonical CUDA fixture); learner = candidate public GANTrainer, see job-header.json')
    test_steps = os.environ.get('LRFREE_NATIVE_TEST_STEPS')
    if test_steps:
        config['steps'] = int(test_steps)
    native_steps = options.get('native_steps')
    if native_steps is not None and (not test_steps):
        assert config['steps'] == 7000 and native_steps >= 7000
        config['steps'] = int(native_steps)
    (budget, seed) = (config['steps'], config['seed'])
    (eval_n, snap_n) = (config['eval_samples'], config['snapshot_samples'])
    expected = native_evaluation_steps(budget, config['eval_interval'], config['early_eval_steps'])
    assert test_steps or (len(expected) == 34 and expected[:7] == [0, 1, 10, 25, 50, 100, 250] and (expected[-1] == 7000)) or (native_steps is not None and expected[-1] == budget and (expected[:7] == [0, 1, 10, 25, 50, 100, 250]))
    check_steps = expected[-5:]
    sub = {}
    if native_steps is not None:
        for b in range(7000, budget, 7000):
            b_expected = native_evaluation_steps(b, config['eval_interval'], config['early_eval_steps'])
            assert b_expected == [x for x in expected if x <= b]
            sub[b] = dict(expected=b_expected, check_steps=b_expected[-5:], config=dict(config, steps=b))
    (holdout_n, holdout_offsets) = (100000, {'target': 1601, 'noise': 1602, 'latent': 1603})
    device = 'cuda:0'
    deterministic(torch)
    overrides = dict(ctx.overrides)
    overrides.update(num_particles=config['num_particles'], z_dim=config['z_dim'], batch_size=config['batch_size'])
    recipe = frozen_recipe(package, **overrides)
    from particlegan.particle_prior import ParticlePrior
    raw = lambda x: hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def init_linear(module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.device(device):
            prior = ParticlePrior(config['num_particles'], config['z_dim'])
            with torch.no_grad():
                prior.z.uniform_(-5.0, 5.0)
            generator = nn.Linear(2, 2)
            with torch.no_grad():
                generator.weight.copy_(torch.eye(2, dtype=generator.weight.dtype))
                generator.bias.zero_()
            critic = toy_models.SimpleMLPDiscriminator(in_dim=2, hidden_dim=config['d_hidden'], n_hidden=config['n_hidden'], fourier=config['fourier'])
            init_linear(critic)
        initial = {role: {n: raw(p) for (n, p) in m.named_parameters()} for (role, m) in (('G', generator), ('D', critic), ('prior', prior))}
        for (role, parameters) in fixture['expected_parameters'].items():
            for (name, want) in parameters.items():
                if initial[role][name] != want:
                    raise RuntimeError(f'native100 fixture mismatch: {role}.{name} {initial[role][name]} != {want}')
        z_range = [float(prior.z.min()), float(prior.z.max())]
        if z_range != fixture['prior_range']:
            raise RuntimeError(f"native100 prior range {z_range} != {fixture['prior_range']}")
        assert str(torch.get_default_device()) == 'cpu'
        trainer = frozen_trainer(package, recipe, generator, critic, **ctx.trainer_options(prior=prior, seed=seed))
    torch.manual_seed(seed)
    after_init = {role: {n: raw(p) for (n, p) in m.named_parameters()} for (role, m) in (('G', trainer.G), ('D', trainer.D), ('prior', trainer.prior))}
    fixture_receipt = dict(initial=initial, after_trainer_init=after_init, prior_range=z_range, G_kept=after_init['G'] == initial['G'], prior_kept=after_init['prior'] == initial['prior'], D_changed=sorted((n for n in initial['D'] if initial['D'][n] != after_init['D'][n])))
    dump(ctx.out / 'native-fixture.json', fixture_receipt)
    stream = torch.Generator(device=device).manual_seed(seed)
    target = problems.sample_real(task, max(eval_n, snap_n), device=device, generator=torch.Generator(device=device).manual_seed(seed + 401))
    target_eval = target[:eval_n].detach().cpu().numpy()
    target_snap = target[:snap_n].detach().cpu().numpy()
    kinds = ('clean', 'noisy')
    dirs = {k: ctx.out / f'native-{k}' for k in kinds}
    for d in dirs.values():
        (d / 'snapshots').mkdir(parents=True, exist_ok=True)
        (d / 'quality_checks').mkdir(exist_ok=True)
        dump(d / 'config.json', config)
    for (b, info) in sub.items():
        info['dirs'] = {k: ctx.out / f'native-{k}-b{b}' for k in kinds}
        for d in info['dirs'].values():
            (d / 'snapshots').mkdir(parents=True, exist_ok=True)
            (d / 'quality_checks').mkdir(exist_ok=True)
            dump(d / 'config.json', info['config'])
    events = {k: open(dirs[k] / 'events.jsonl', 'w', buffering=1) for k in kinds}
    diagnostic_rows = open(ctx.out / 'native100-diagnostics.jsonl', 'w', buffering=1)
    previous_prior_centres = None
    previous_affine_state = None
    previous_row_displacement = None
    previous_bd_moves = 0
    previous_interval_moved = False
    started = time.monotonic()

    @torch.no_grad()
    def draw(n, ema, latent_seed, noise_seed):
        """(clean, noisy, sigma); training modes/RNGs untouched."""
        (model, table) = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
        modes = [(m, m.training) for root in (model, table) for m in root.modules()]
        try:
            model.eval()
            table.eval()
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(noise_seed)
                latent_stream = torch.Generator(device=device).manual_seed(latent_seed)
                (latent, indices) = table.sample(n, generator=latent_stream)
                arguments = [model, latent, 0.0, latent_stream]
                if options['evaluation_generate'] == 'indexed':
                    arguments.append(indices)
                clean = trainer._generate(*arguments)
                sigma = float(ctx.sigma(trainer))
                noisy = clean + sigma * torch.randn_like(clean) if sigma else clean
                return (clean, noisy, sigma)
        finally:
            for (module, flag) in modes:
                module.training = flag

    def flat(m, a):
        out = {k: m[k] for k in ('modes', 'hq', 'precision', 'min_hq_mode_mass', 'max_mode_mass', 'mass_tv', 'min_cov_eig_ratio', 'max_cov_eig_ratio', 'min_radial_median_ratio', 'max_radial_median_ratio') if k in m}
        out.update({f'acc_{k}': a[k] for k in NATIVE_ACC_KEYS if k in a})
        return out
    finals = {k: {} for k in kinds}
    (geometry_centres, geometry_std) = problems.evaluation_geometry(task, device=device)

    def observe(step):
        nonlocal previous_prior_centres, previous_affine_state, previous_row_displacement
        nonlocal previous_bd_moves, previous_interval_moved
        elapsed = time.monotonic() - started
        arrays = {k: {} for k in kinds}
        scored = {k: {} for k in kinds}
        sigma = None
        sharp = {}
        for model in ('live', 'ema'):
            (clean, noisy, sigma) = draw(eval_n, model == 'ema', seed + 403, seed + 402)
            sharp[model] = dict(sharp_clean=sharp_nearest(torch, clean[:eval_n], geometry_centres, geometry_std), sharp_noisy=sharp_nearest(torch, noisy[:eval_n], geometry_centres, geometry_std))
            for (kind, x) in (('clean', clean), ('noisy', noisy)):
                values = x[:eval_n]
                m = dict(metrics.evaluate_samples(values, task))
                a = accuracy.evaluate_accuracy(values, task, gate_metrics=m)
                events[kind].write(json.dumps(dict(event='eval', step=step, model=model, metrics=m, elapsed=elapsed, accuracy=a, output_sigma=sigma if kind == 'noisy' else 0.0)) + '\n')
                arrays[kind][model] = values.detach().cpu().numpy()
                scored[kind][model] = (m, a)
                finals[kind][model] = m
        for kind in kinds:
            d = dirs[kind]
            np.savez_compressed(d / 'snapshots' / f'step_{step:06d}.npz', live=arrays[kind]['live'][:snap_n], ema=arrays[kind]['ema'][:snap_n], target=target_snap)
            if step in check_steps:
                np.savez_compressed(d / 'quality_checks' / f'step_{step:06d}.npz', **arrays[kind], target=target_eval)
            if step == budget:
                np.savez_compressed(d / 'final_samples.npz', **arrays[kind], target=target_eval)
            for (b, info) in sub.items():
                if step > b:
                    continue
                sd = info['dirs'][kind]
                os.link(d / 'snapshots' / f'step_{step:06d}.npz', sd / 'snapshots' / f'step_{step:06d}.npz')
                if step in info['check_steps']:
                    np.savez_compressed(sd / 'quality_checks' / f'step_{step:06d}.npz', **arrays[kind], target=target_eval)
                if step == b:
                    np.savez_compressed(sd / 'final_samples.npz', **arrays[kind], target=target_eval)
                    info.setdefault('finals', {})[kind] = dict(finals[kind])
        (primary, alt) = ('noisy', 'clean') if options['eval_output_noise'] else ('clean', 'noisy')
        points = {}
        for kind in (primary, alt):
            ((m, a), (em, ea)) = (scored[kind]['live'], scored[kind]['ema'])
            points[kind] = dict(step=step, **flat(m, a), **sharp['live'], ema=dict(flat(em, ea), **sharp['ema'], **{'pass': bool(metrics.passes(task, em))}))
        ctx.observe(points[primary], metrics.passes(task, scored[primary]['live'][0]), trainer, extra=dict(output_sigma=sigma), alt=points[alt], alt_passed=metrics.passes(task, scored[alt]['live'][0]))
        try:
            diagnostic = dict(step=step, problem=task, primary_evaluation=primary, output_sigma=sigma, lr=[[group['lr'] for group in optimizer.param_groups] for optimizer in (trainer.opt_g, trainer.opt_d)], clouds={kind: {model: native100_diagnostics.cloud_diagnostics(arrays[kind][model], task) for model in ('live', 'ema')} for kind in kinds})
            settle = getattr(trainer, 'lr_settle', None)
            birth_death = getattr(trainer, 'birth_death', None)
            if settle is not None:
                diagnostic['settle'] = jsonable(torch, settle.diagnostics())
            if birth_death is not None:
                diagnostic['birth_death'] = jsonable(torch, birth_death.diagnostics())
            if isinstance(trainer.G, nn.Linear) and hasattr(trainer.prior, 'z'):
                with torch.no_grad():
                    prior_centres = torch.nn.functional.linear(trainer.prior.z.detach(), trainer.G.weight.detach(), trainer.G.bias.detach() if trainer.G.bias is not None else None).cpu().numpy()
                    affine_state = (trainer.prior.z.detach().cpu().numpy().copy(), trainer.G.weight.detach().cpu().numpy().copy(), trainer.G.bias.detach().cpu().numpy().copy() if trainer.G.bias is not None else np.zeros(trainer.G.out_features))
                diagnostic['prior_motion'] = native100_diagnostics.prior_motion_diagnostics(previous_prior_centres, prior_centres, task)
                bd_moves = int(birth_death.counters['moves']) if birth_death is not None else 0
                interval_moved = bd_moves != previous_bd_moves
                (diagnostic['affine_motion_v1'], row_displacement) = native100_diagnostics.affine_motion_diagnostics(previous_affine_state, affine_state, previous_row_displacement, task, lag_lineage_valid=not (interval_moved or previous_interval_moved))
                previous_prior_centres = prior_centres
                previous_affine_state = affine_state
                previous_row_displacement = row_displacement
                previous_bd_moves = bd_moves
                previous_interval_moved = interval_moved
            diagnostic_rows.write(json.dumps(diagnostic, allow_nan=False) + '\n')
        except Exception as error:
            diagnostic_rows.write(json.dumps(dict(step=step, problem=task, diagnostic_error=repr(error))) + '\n')

    def draw_holdout():
        out = {k: {} for k in kinds}
        for model in ('live', 'ema'):
            (clean, noisy, _) = draw(holdout_n, model == 'ema', seed + holdout_offsets['latent'], seed + holdout_offsets['noise'])
            (out['clean'][model], out['noisy'][model]) = (clean.cpu().numpy(), noisy.cpu().numpy())
        return out
    try:
        observe(0)
        for step in range(1, budget + 1):
            real_d = problems.sample_real(task, config['batch_size'], device=device, generator=stream)
            ctx.probe_real = real_d
            cache = []

            def generator_real():
                if not cache:
                    cache.append(problems.sample_real(task, config['batch_size'], device=device, generator=stream))
                elif len(ctx.warnings) < 20:
                    ctx.warnings.append(f'step {step}: generator_real called more than once (cached tensor reused)')
                return cache[0]
            with ctx.serial():
                trainer.step(real_d, generator_real=generator_real, collect_stats=step in expected)
            assert trainer.completed_steps == step
            assert str(torch.get_default_device()) == 'cpu'
            if not cache:
                ctx.deviation(step, 'candidate never called generator_real')
                generator_real()
            ctx.rate_row(trainer, step)
            if step in expected:
                observe(step)
            if step in sub:
                sub[step]['holdout'] = draw_holdout()
                sub[step]['completed_steps'] = trainer.completed_steps
        holdout = draw_holdout()
        holdout_target = problems.sample_real(task, holdout_n, device=device, generator=torch.Generator(device=device).manual_seed(seed + holdout_offsets['target'])).cpu().numpy()
    finally:
        for handle in events.values():
            handle.close()
        diagnostic_rows.close()

    def score_dir(kind, d, b_config, b_budget, b_expected, b_checks, b_holdout, b_finals, completed):
        arrays = dict(b_holdout[kind], target=holdout_target)
        np.savez_compressed(d / 'holdout_samples.npz', **arrays)
        summary = dict(status='complete', problem=task, budget_steps=b_budget, config=b_config, eval_steps=b_expected, snapshot_steps=b_expected, completed_steps=completed, final_samples_file='final_samples.npz', final=b_finals, accuracy=dict(protocol=accuracy.PROTOCOL, check_steps=b_checks, sample_count=eval_n, holdout_samples=holdout_n, holdout_seed_offsets=holdout_offsets), accuracy_check_steps=b_checks, holdout={name: accuracy.evaluate_accuracy(points, task) for (name, points) in arrays.items()}, holdout_samples_file='holdout_samples.npz', evaluation=kind)
        dump(d / 'summary.json', summary)
        proc = subprocess.run([sys.executable, str(HARNESS / 'native100_score.py'), str(d), task], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        if proc.returncode != 0:
            raise RuntimeError(f'native100 scorer failed ({kind}, budget {b_budget}): {proc.stderr[-1500:]}')
        verdict = json.loads(proc.stdout.strip().splitlines()[-1])
        dump(d / 'verdict.json', verdict)
        return verdict
    verdicts = {}
    for kind in kinds:
        verdicts[kind] = score_dir(kind, dirs[kind], config, budget, expected, check_steps, holdout, finals[kind], trainer.completed_steps)
    sub_verdicts = {}
    for (b, info) in sub.items():
        for kind in kinds:
            lines = [line for line in (dirs[kind] / 'events.jsonl').read_text().splitlines() if json.loads(line)['step'] <= b]
            (info['dirs'][kind] / 'events.jsonl').write_text(''.join((line + '\n' for line in lines)))
            sub_verdicts.setdefault(b, {})[kind] = score_dir(kind, info['dirs'][kind], info['config'], b, info['expected'], info['check_steps'], info['holdout'], info['finals'][kind], info['completed_steps'])

    def compact(kind, verdict_set=None):
        verdict_set = verdicts if verdict_set is None else verdict_set
        (cov, acc) = (verdict_set[kind]['coverage'], verdict_set[kind]['accuracy'])
        if (cov['status'] not in ('PASS', 'FAIL') or acc['status'] not in ('PASS', 'FAIL')) and (not test_steps):
            raise RuntimeError(f"native100 {kind} evidence INVALID: coverage {cov['status']} {cov.get('reason')}; accuracy {acc['status']} {acc.get('reason')}")
        hold = acc.get('holdout_metrics') or {}
        return dict(status=acc['status'], coverage_status=cov['status'], accuracy_status=acc['status'], coverage_stable_checks=cov['stable_checks'], first_full_coverage=cov.get('first_full_coverage_step'), first_arrival=cov.get('first_full_quality_step'), stable_from=cov.get('stable_from_step'), terminal_accuracy=[bool(c['passed']) for c in acc.get('terminal_checks', [])], holdout_pass=bool(hold.get('frozen_pass') and hold.get('accuracy_pass')), holdout={k: hold.get(k) for k in NATIVE_ACC_KEYS + ('modes', 'precision') if k in hold}, reason=acc.get('reason'))
    (primary, alt) = ('noisy', 'clean') if options['eval_output_noise'] else ('clean', 'noisy')
    (main_v, alt_v) = (compact(primary), compact(alt))
    (obs, alt_obs) = (ctx.observations, ctx.alt_observations)

    def acc_arrival(rows):
        return next((r['step'] for r in rows if r['step'] > 0 and r.get('acc_passed')), None)
    final = {k: v for (k, v) in obs[-1].items() if k not in ('ema', 'diag', 'lr', 'noisy', 'clean', 'pass', 'seconds')}
    final.update({f'holdout_{k}': v for (k, v) in main_v['holdout'].items()})
    return (trainer, dict(status=main_v['status'], passing_checks=sum((bool(p['pass']) for p in obs)), observations=len(obs), first_arrival=main_v['first_arrival'], final_streak=main_v['coverage_stable_checks'], final=final, ema_final=obs[-1].get('ema'), final_lr=obs[-1].get('lr'), thresholds=NATIVE_COVERAGE_THRESHOLDS + NATIVE_ACCURACY_THRESHOLDS, native=dict(main_v, accuracy_first_arrival=acc_arrival(obs), accuracy_passing_checks=sum((bool(p.get('acc_passed')) for p in obs))), **{f'{alt}_status': alt_v['status'], f'{alt}_passing_checks': sum((bool(p['pass']) for p in alt_obs)), f'{alt}_first_arrival': alt_v['first_arrival'], f'{alt}_final_streak': alt_v['coverage_stable_checks'], f'{alt}_final': {k: v for (k, v) in alt_obs[-1].items() if k not in ('ema', 'pass')}, f'{alt}_native': dict(alt_v, accuracy_first_arrival=acc_arrival(alt_obs))}, eval_output_noise=bool(options['eval_output_noise']), native_fixture=fixture_receipt, **{'native_budgets': {str(b): {k: compact(k, sub_verdicts[b]) for k in kinds} for b in sub}} if native_steps is not None else {}, pass_rule='frozen native gates: coverage gate (>=5 terminal passing live checks) AND accuracy gate (final five 20k clouds + independent 100k holdout pass fidelity limits)'))

def resolve_options(package, declared, given):
    options = dict(DEFAULT_OPTIONS)
    options.update({k: v for (k, v) in declared.items() if k in options})
    unknown = set(given) - set(options)
    if unknown:
        raise ValueError(f'unknown candidate options: {sorted(unknown)}')
    options.update(given)
    detected = dict(evaluation_generate='indexed' if 'indices' in inspect.signature(package.GANTrainer._generate).parameters else 'plain', serial_backward_argument='serial_backward' in inspect.signature(package.GANTrainer.__init__).parameters)
    for (key, value) in detected.items():
        if options[key] == 'auto':
            options[key] = value
    if options['evaluation_generate'] not in ('plain', 'indexed'):
        raise ValueError('evaluation_generate must be plain/indexed/auto')
    return (options, detected)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True, help='directory containing particlegan/')
    parser.add_argument('--overrides', default='{}', help='recipe overrides: inline JSON or file (a declaration.json with recipe_overrides is accepted)')
    parser.add_argument('--task', choices=ALL_TASKS + CUSTOM_TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--candidate-options', default='{}', help='inline JSON or file; see DEFAULT_OPTIONS')
    parser.add_argument('--cand', default=None, help='label recorded in result.json')
    parser.add_argument('--force', action='store_true', help='allow an existing non-empty output directory')
    args = parser.parse_args()
    started = time.monotonic()
    loaded = load_json_arg(args.overrides)
    declared = {}
    if 'recipe_overrides' in loaded:
        declared = {k: loaded[k] for k in ('evaluation_generate', 'serial_backward_argument') if k in loaded}
        loaded = loaded['recipe_overrides']
    given = load_json_arg(args.candidate_options)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists() and (not args.force):
        raise SystemExit(f'{args.output}/result.json exists; use --force or a new output directory')
    index = int(args.device.split(':')[1]) if ':' in args.device else 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        choices = [v for v in visible.split(',') if v.strip()]
        os.environ['CUDA_VISIBLE_DEVICES'] = choices[index]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    if any((n == 'particlegan' or n.startswith('particlegan.') for n in sys.modules)):
        raise RuntimeError('fresh process required')
    sys.path.insert(0, str(args.package_root.resolve()))
    import torch
    package = importlib.import_module('particlegan')
    package_file = Path(package.__file__).resolve()
    assert package_file.parent.parent == args.package_root.resolve(), f'imported wrong particlegan: {package_file}'
    (options, detected) = resolve_options(package, declared, given)
    if options.get('image_steps') is None:
        options.pop('image_steps', None)
    elif args.task not in IMAGE_TASKS:
        raise ValueError('image_steps applies to image tasks only')
    if options.get('native_steps') is None:
        options.pop('native_steps', None)
    elif args.task not in NATIVE_TASKS:
        raise ValueError('native_steps applies to native tasks only')
    elif type(options['native_steps']) is not int or options['native_steps'] < 7000:
        raise ValueError('native_steps must be an integer >= 7000')
    overrides = dict(loaded)
    if options['initialization'] is not None:
        overrides.setdefault('initialization', options['initialization'])
    header = dict(cand=args.cand, task=args.task, package_root=str(args.package_root.resolve()), package_sha256=package_digest(args.package_root), overrides=overrides, options=options, auto_detected=detected, declared_options=declared, device=args.device, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__), cuda=torch.version.cuda, gpu=None if args.task in CUSTOM_TASKS else torch.cuda.get_device_name(0), host=os.uname().nodename, python=sys.executable)
    dump(args.output / 'job-header.json', header)
    ctx = trainer = None
    try:
        assert torch.cuda.is_available(), 'CUDA required'
        assert str(torch.get_default_device()) == 'cpu'
        ctx = Context(args, torch, package, overrides, options, args.output)
        if args.task == 'mode_hold':
            (trainer, result) = run_mode_hold(ctx)
        elif args.task in IMAGE_TASKS:
            (trainer, result) = run_image(ctx, args.task)
        elif args.task in VECTOR_TASKS:
            (trainer, result) = run_vector(ctx, args.task)
        elif args.task in NATIVE_TASKS:
            (trainer, result) = run_native(ctx, args.task)
        elif args.task in CUSTOM_TASKS:
            import custom22
            (trainer, result) = custom22.run(ctx, args.task)
        else:
            (trainer, result) = run_ring(ctx, args.task)
        if options['save_final_state']:
            torch.save(dict(trainer=trainer.state_dict()), args.output / 'final-state.pt')
    except Exception as error:
        result = dict(status='ERROR', error=repr(error), traceback=traceback.format_exc(), observations=len(ctx.observations) if ctx else 0)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        if ctx is not None:
            ctx.close()
    out = dict(status=result.pop('status'), task=args.task, cand=args.cand)
    for key in ('passing_checks', 'observations', 'first_arrival', 'final_streak', 'final', 'ema_final', 'final_lr'):
        if key in result:
            out[key] = result.pop(key)
    out['seconds'] = round(time.monotonic() - started, 2)
    out['train_seconds'] = round(time.monotonic() - ctx.started, 2) if ctx else None
    if ctx is not None:
        out['stream_deviations'] = ctx.stream_deviations
        out['warnings'] = ctx.warnings[:20]
    if trainer is not None:
        out['completed_steps'] = trainer.completed_steps
        try:
            out['recipe'] = json.loads(json.dumps(trainer.recipe.to_dict(), default=str))
        except Exception:
            pass
    try:
        out['max_gpu_mib'] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    except Exception:
        pass
    out.update(result)
    out['header'] = header
    dump(args.output / 'result.json', out)
    brief = {k: out.get(k) for k in ('status', 'task', 'cand', 'passing_checks', 'observations', 'first_arrival', 'final_streak', 'seconds')}
    print('RESULT ' + json.dumps(brief), flush=True)
    if out['status'] == 'ERROR':
        raise SystemExit(1)
if __name__ == '__main__':
    main()
