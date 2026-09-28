#!/usr/bin/env python
"""L1 parity gate: the engine's scalar re-expression vs the candidate package's own ``GANTrainer.step``.

  components_parity.py --package-root DIR --overrides JSON|FILE [--steps 1000] [--device cpu]
                       [--configs c1,c2] [--out FILE]

Twin-constructed (identically seeded) G/D/prior; the same real batches; after EVERY update compares
bitwise: returned losses and penalty stats, G/D/prior/EMA-G/EMA-prior tensors, both optimizer
state_dicts (Adam/AMSGrad moments, KA2 record, spike guard, EMA critic, A2 history, group LRs/betas),
controller state, lr_settle state, learnable sigma, last_output_sigma, the four trainer streams, and the
global CPU (and CUDA) RNG. c1 = mode_hold resources (12 particles, z 4, batch 128, mode_hold MLPs,
separate generator_real batch via a callable); c2 = sparse table (512 particles, z 2, batch 64: A2's
sparse path). Fresh process per package (imports ``particlegan`` from --package-root).
"""
from pathlib import Path
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time

HARNESS = Path(__file__).resolve().parent
CONFIGS = {
    'c1': dict(resources=dict(num_particles=12, z_dim=4, batch_size=128), g=(96, 3), d=(96, 3, 3),
               init_std=.5, callable_real=True),
    'c2': dict(resources=dict(num_particles=512, z_dim=2, batch_size=64), g=(64, 2), d=(64, 2, 3),
               init_std=1., callable_real=False),
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(a, b, path='', out=None):
    """First differing path between two nested structures (tensors bitwise), or None."""
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return path + ' (tensor vs non-tensor)'
        if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
            return path + f' (dtype/shape/device {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)})'
        if a.is_floating_point():
            same = torch.equal(a.nan_to_num(nan=0.), b.nan_to_num(nan=0.)) and torch.equal(a.isnan(), b.isnan())
        else:
            same = torch.equal(a, b)
        return None if same else path
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return path + f' (keys {sorted(map(str, set(a) ^ set(b)))[:6]})'
        for key in a:
            found = compare(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return path + f' (len {len(a)} vs {len(b)})'
        for i, (x, y) in enumerate(zip(a, b)):
            found = compare(x, y, f'{path}[{i}]')
            if found:
                return found
        return None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return None
    return None if a == b and type(a) is type(b) else f'{path} ({a!r} vs {b!r})'


def trainer_state(trainer, torch):
    return dict(
        G=trainer.G.state_dict(), D=trainer.D.state_dict(), prior=trainer.prior.state_dict(),
        ema_G=trainer.ema_G.state_dict(), ema_prior=trainer.ema_prior.state_dict(),
        optimizers=[trainer.opt_g.state_dict(), trainer.opt_d.state_dict()],
        lrs=[[g['lr'] for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)],
        controller=None if getattr(trainer, 'controller', None) is None else trainer.controller.state_dict(),
        lr_settle=None if getattr(trainer, 'lr_settle', None) is None else trainer.lr_settle.state_dict(),
        birth_death=None if getattr(trainer, 'birth_death', None) is None
        else trainer.birth_death.state_dict(),
        log_sigma=None if getattr(trainer, 'log_output_sigma', None) is None else trainer.log_output_sigma.detach().clone(),
        last_sigma=getattr(trainer, 'last_output_sigma', None),
        streams=[getattr(trainer, n).get_state() for n in ('latent_generator', 'penalty_generator',
                                                          'eval_generator', 'noise_generator')],
        completed=trainer.completed_steps)


def engine_state(eng, torch):
    state = dict(
        G=eng.G.state_dict(), D=eng.D.state_dict(), prior=eng.tables[0].state_dict(),
        ema_G=eng.ema_G_side[0].state_dict(), ema_prior=eng.ema_tables[0].state_dict(),
        optimizers=[eng.opt_g.state_dict(), eng.opt_d.state_dict()],
        lrs=eng.current_lrs(),
        controller=None if eng.controller is None else eng.controller.state_dict(),
        lr_settle=None if eng.lr_settle is None else eng.lr_settle.state_dict(),
        birth_death=None if eng.birth_death is None else eng.birth_death.state_dict(),
        log_sigma=None if eng.log_output_sigma is None else eng.log_output_sigma.detach().clone(),
        last_sigma=eng.last_output_sigma if eng._sigma_api else None,
        streams=[getattr(eng, n).get_state() for n in ('latent_generator', 'penalty_generator',
                                                      'eval_generator', 'noise_generator')],
        completed=eng.completed_steps)
    return state


def run_config(package, components, overrides, name, steps, device, torch):
    cfg = CONFIGS[name]
    host = _load('lrfree_parity_mode_hold_host', HARNESS / 'hosts' / 'mode_hold_host.py')
    options = dict(overrides)
    options.update(cfg['resources'])
    recipe = package.get_recipe(**options)
    trainer_kwargs = dict(optimizer_options={'foreach': False, 'fused': False})
    serial = 'serial_backward' in __import__('inspect').signature(package.GANTrainer.__init__).parameters
    if serial:
        trainer_kwargs['serial_backward'] = True
    z_dim = cfg['resources']['z_dim']

    def construct():
        torch.manual_seed(0)
        stream = torch.Generator().manual_seed(0)
        prior = recipe.make_prior(init_std=cfg['init_std'], generator=stream)
        G = host.SimpleMLPGenerator(z_dim, cfg['g'][0], cfg['g'][1], 2)
        D = host.SimpleMLPDiscriminator(2, *cfg['d'])
        return prior.to(device), G.to(device), D.to(device)
    prior1, G1, D1 = construct()
    trainer = package.GANTrainer(recipe, G1, D1, prior=prior1, seed=0, **trainer_kwargs)
    prior2, G2, D2 = construct()
    eng = components.Engine(package, overrides, cfg['resources'], seed=0, serial_backward=serial)
    eng.build(generator=G2, critic=D2, tables=[prior2])
    first = compare(trainer_state(trainer, torch), engine_state(eng, torch), 'init')
    data = torch.Generator().manual_seed(1234)
    means = host.ring_means()
    batch = cfg['resources']['batch_size']
    cuda = torch.device(device).type == 'cuda'
    mismatch = first
    started = time.monotonic()
    done = 0
    if recipe.total_steps is not None:   # horizon recipes (scheduled controls) stop at their own budget
        steps = min(steps, recipe.total_steps)
    for step in range(1, steps + 1):
        if mismatch:
            break
        real = host.sample_ring(means, batch, host.SIGMA, data).to(device)
        real_g = host.sample_ring(means, batch, host.SIGMA, data).to(device)
        collect = step % 50 == 0
        cpu, gpu = torch.get_rng_state(), (torch.cuda.get_rng_state() if cuda else None)
        arg = (lambda: real_g) if cfg['callable_real'] else real_g
        out1 = trainer.step(real, generator_real=arg, collect_stats=collect)
        rng1 = (torch.get_rng_state(), torch.cuda.get_rng_state() if cuda else None)
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state(gpu)
        arg = (lambda: real_g) if cfg['callable_real'] else real_g
        out2 = components.scalar_step(eng, real, generator_real=arg, collect_stats=collect)
        rng2 = (torch.get_rng_state(), torch.cuda.get_rng_state() if cuda else None)
        mismatch = (compare(out1, out2, f'step{step}.result')
                    or compare(list(rng1), list(rng2), f'step{step}.global_rng')
                    or compare(trainer_state(trainer, torch), engine_state(eng, torch), f'step{step}'))
        done = step
    record = trainer.opt_d.record
    coverage = dict(ka2_calls=record.calls, blend_reached=record.calls >= 800, anchor_started=record.anchor_started,
                    ema_updates=getattr(record, 'ema_updates', None), ema_reseeds=getattr(record, 'ema_reseeds', None),
                    guard_clipped=getattr(trainer.opt_d.guard, 'clipped_tensors', None),
                    a2=None if trainer.opt_g.latent_damping is None else trainer.opt_g.latent_damping.state_dict(),
                    settle=None if getattr(trainer, 'lr_settle', None) is None else {
                        k: dict(windows=v['windows'], s=v['s'], counts=v['counts'])
                        for k, v in trainer.lr_settle.diagnostics().items()},
                    controller=None if getattr(trainer, 'controller', None) is None else dict(
                        mobility=trainer.controller.mobility, game_trust=trainer.controller.game_trust,
                        payoff_error=trainer.controller.payoff_error),
                    last_output_sigma=getattr(trainer, 'last_output_sigma', None))
    return dict(config=name, resources=cfg['resources'], device=str(device), steps_requested=steps,
                steps_compared=done, status='PASS' if mismatch is None and done == steps else 'FAIL',
                first_mismatch=mismatch, seconds=round(time.monotonic() - started, 2), coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--overrides', default='{}')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--configs', default='c1,c2')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.device.startswith('cuda'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(args.package_root.resolve()))
    sys.path.insert(0, str(HARNESS))
    import torch
    package = importlib.import_module('particlegan')
    assert Path(package.__file__).resolve().parent.parent == args.package_root.resolve()
    import components
    import lrlib
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    overrides = lrlib.load_json_arg(args.overrides)
    if 'recipe_overrides' in overrides:
        overrides = overrides['recipe_overrides']
    overrides = dict(overrides)   # used as given (screen.py has already injected its initialization option)
    results = []
    status = 'PASS'
    try:
        for name in [c for c in args.configs.split(',') if c]:
            row = run_config(package, components, overrides, name, args.steps, args.device, torch)
            results.append(row)
            print(json.dumps({k: row[k] for k in ('config', 'device', 'steps_compared', 'status', 'first_mismatch',
                                                  'seconds')}), flush=True)
            if row['status'] != 'PASS':
                status = 'FAIL'
    except components.EngineRefusal as error:
        status, results = 'REFUSED', results + [dict(error=repr(error))]
    out = dict(level='L1', status=status, package_root=str(args.package_root.resolve()),
               package_sha256=lrlib.package_digest(args.package_root), overrides=overrides,
               engine_sha256=components.source_sha256(), engine_version=components.ENGINE_VERSION,
               torch=str(torch.__version__), device=args.device, steps=args.steps, results=results,
               time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(out, indent=1, default=str) + '\n')
        os.replace(tmp, args.out)
    print('PARITY ' + json.dumps(dict(status=status, device=args.device, steps=args.steps)), flush=True)
    raise SystemExit(0 if status == 'PASS' else 1)


if __name__ == '__main__':
    main()
