"""Frozen host scopes and transaction. Importing this module imports no Torch."""
from contextlib import contextmanager
from pathlib import Path
import importlib.util


def load_host():
    path = Path(__file__).with_name('mode-hold-source') / 'frozen_host.py'
    spec = importlib.util.spec_from_file_location('det_retest_frozen_host', path)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    return host


@contextmanager
def host_device(torch, device):
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


@contextmanager
def serial_step(torch):
    previous = torch.autograd.is_multithreading_enabled()
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        if torch.autograd.is_multithreading_enabled() != previous:
            raise RuntimeError('serial context failed to restore caller mode')


def construct(torch, package, host, declaration, device, stream):
    """No seed reset here: caller owns the original constructor RNG boundary."""
    if str(torch.get_default_device()) != 'cpu':
        raise RuntimeError('learner factories require CPU default device')
    overrides = dict(declaration['recipe_overrides'])
    for name, value in dict(num_particles=12, z_dim=4, batch_size=128,
                            initialization='batch_feature_zero').items():
        if name in overrides and overrides[name] != value:
            raise ValueError(f'frozen resource/default initialization mismatch: {name}')
        overrides[name] = value
    recipe = package.get_recipe(**overrides)
    if recipe.initialization != 'batch_feature_zero':
        raise ValueError('legacy initialization=None must be explicitly opted in')
    with host_device(torch, device):
        means = host.ring_means()
        # Retain the old Gaussian constructor draw, then public R2 overwrite.
        # Candidate geometry/controller creation happens only AFTER this call.
        prior = recipe.make_prior(init_std=.5, generator=stream)
        generator = host.SimpleMLPGenerator(4, 96, 3, 2)
        critic = host.SimpleMLPDiscriminator(2, 96, 3, 3)
    options = dict(prior=prior, seed=0, latent_generator=stream,
                   optimizer_options={'foreach': False, 'fused': False})
    if declaration['serial_backward_argument']:
        options['serial_backward'] = True
    trainer = package.GANTrainer(recipe, generator, critic, **options)
    if 'resolved_recipe' in declaration and trainer.recipe.to_dict() != declaration['resolved_recipe']:
        # JSON tuples/lists are normalized by the caller's source declaration.
        import json
        if json.loads(json.dumps(trainer.recipe.to_dict())) != declaration['resolved_recipe']:
            raise ValueError('resolved recipe differs from sealed declaration')
    return trainer, means


def peek_batch(torch, host, trainer, stream, means, device, step):
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
    receipt = dict(step=step, real_d=host.digest(real), latent_d=host.digest(latent_d),
                   latent_g=host.digest(latent_g), real_g=host.digest(real_g),
                   accepted_cursor=host.digest(after_g_real))
    return real, real_g, before_g_real, after_g_real, receipt


def measure(torch, host, trainer, means, declaration, output_noise_std, ema=False):
    model, table = ((trainer.ema_G, trainer.ema_prior) if ema
                    else (trainer.G, trainer.prior))
    modes = [(m, m.training) for root in (model, table) for m in root.modules()]
    device = next(model.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    try:
        model.eval()
        table.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(402 + trainer.completed_steps)
            latent, indices = table.sample(4096, generator=torch.Generator(device=device).manual_seed(9))
            local_noise = torch.Generator(device=device).manual_seed(2303 + trainer.completed_steps)
            # Same package generation path as its own previous mode-hold gate;
            # output noise remains on the separate historical global stream.
            if declaration['evaluation_generate'] == 'indexed':
                fake = trainer._generate(model, latent, 0., local_noise, indices)
            elif declaration['evaluation_generate'] == 'plain':
                fake = trainer._generate(model, latent, 0., local_noise)
            else:
                raise ValueError('unreviewed generation binding')
            sigma = output_noise_std(trainer.recipe, trainer.completed_steps)
            if sigma:
                fake = fake + sigma * torch.randn_like(fake)
            return host.diversity(fake, means)
    finally:
        for module, flag in modes:
            module.training = flag
