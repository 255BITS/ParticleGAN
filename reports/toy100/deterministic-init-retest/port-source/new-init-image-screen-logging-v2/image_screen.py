"""Frozen image screen using a sealed candidate package; no execution on import."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def material(torch, value):
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), device=str(value.device), dtype=str(value.dtype),
                    sha256=sha(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()))
    if isinstance(value, dict):
        return {str(k): material(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [material(torch, v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unregistered checkpoint object: {type(value)}')


def verify(package_root, declaration_path):
    seal = json.loads((ROOT / 'bundle-sha256.json').read_text())
    actual_bundle = {f.name: sha(f.read_bytes()) for f in ROOT.iterdir()
                     if f.is_file() and f.name != 'bundle-sha256.json'}
    assert actual_bundle == seal, 'reviewed image source bundle changed'
    declaration = json.loads(declaration_path.read_text())
    if declaration.get('post_constructor_setup'):
        raise ValueError('historical diagnostic setup requires separate review')
    actual = {str(f.relative_to(package_root)): sha(f.read_bytes())
              for f in (package_root / 'particlegan').rglob('*.py')}
    assert actual == declaration['package_sha256'], 'candidate package changed'
    proof = json.loads((ROOT / 'host-source-proof.json').read_text())
    assert sha((ROOT / 'frozen_image_host.py').read_bytes()) == proof['frozen_host_sha256']
    assert sha((ROOT / 'task-specs.json').read_bytes()) == proof['task_specs_sha256']
    assert declaration['initializer_commit'] == 'c720645ecae6b648e9fc6034e9d6b48ccff06ed3'
    assert declaration['recipe_overrides']['initialization'] == 'batch_feature_zero'
    assert declaration['initial_optimizer_state'] in ('native_lazy', 'declared_eager')
    assert set(declaration['optimizer_step_devices']) == {'G', 'D'}
    if declaration['initial_optimizer_state'] == 'native_lazy':
        assert set(declaration['optimizer_step_devices'].values()) == {'cpu'}
    return declaration


def import_modules(package_root, declaration):
    assert not any(n == 'particlegan' or n.startswith('particlegan.') for n in sys.modules)
    sys.path.insert(0, str(package_root.resolve()))
    import particlegan
    spec = importlib.util.spec_from_file_location('frozen_image_host', ROOT / 'frozen_image_host.py')
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    verify_imports(package_root, declaration)
    return particlegan, host


def verify_imports(package_root, declaration):
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == 'particlegan' or name.startswith('particlegan.'):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(package_root.resolve()))
            assert sha(path.read_bytes()) == declaration['package_sha256'][relative]
            imports[name] = dict(path=str(path), sha256=sha(path.read_bytes()))
    return imports


def construct(torch, package, host, declaration, spec, device):
    assert str(torch.get_default_device()) == 'cpu'
    overrides = dict(declaration['recipe_overrides'])
    overrides.update(num_particles=spec['particles'], z_dim=spec['z_dim'], batch_size=spec['batch_size'])
    recipe = package.get_recipe(**overrides)
    # Exact historical host constructor order. Public factories perform R2 init.
    generator, critic = host.Generator(spec), host.Discriminator(spec)
    prior = recipe.make_prior()
    assert all(p.device.type == 'cpu' for m in (generator, critic, prior) for p in m.parameters())
    generator, critic, prior = generator.to(device), critic.to(device), prior.to(device)
    shared = (torch.cuda.default_generators[0] if str(device).startswith('cuda')
              else torch.default_generator)
    options = dict(prior=prior, seed=0, latent_generator=shared, penalty_generator=shared,
                   optimizer_options={'foreach': False, 'fused': False})
    if declaration['serial_backward_argument']:
        options['serial_backward'] = True
    trainer = package.GANTrainer(recipe, generator, critic, **options)
    expected = dict(declaration['resolved_recipe'])
    expected.update(num_particles=spec['particles'], z_dim=spec['z_dim'], batch_size=spec['batch_size'])
    assert json.loads(json.dumps(recipe.to_dict())) == expected
    assert str(torch.get_default_device()) == 'cpu'
    return trainer, shared


def optimizer_proof(torch, trainer, declaration, include_values=True):
    rows = {}
    for role, optimizer in (('G', trainer.opt_g), ('D', trainer.opt_d)):
        entries = []
        for group in optimizer.param_groups:
            assert group.get('capturable', False) is False
            assert group['foreach'] is False and group['fused'] is False
            for parameter in group['params']:
                state = optimizer.state.get(parameter)
                if not state:
                    assert trainer.completed_steps == 0 and declaration['initial_optimizer_state'] == 'native_lazy'
                    entries.append({'state_missing': True, 'parameter': material(torch, parameter)})
                    continue
                clock_device = (str(parameter.device) if declaration['optimizer_step_devices'][role] == 'parameter'
                                else 'cpu')
                assert str(state['step'].device) == clock_device
                assert float(state['step']) == trainer.completed_steps
                assert state['exp_avg'].device == state['exp_avg_sq'].device == parameter.device
                entry = dict(shape=list(parameter.shape), step=float(state['step']),
                             step_device=str(state['step'].device), parameter_device=str(parameter.device))
                if include_values:
                    entry['state'] = material(torch, state)
                entries.append(entry)
        rows[role] = entries
    return rows


def geometry(torch, trainer):
    controller = getattr(trainer, 'controller', None)
    if controller is None or not hasattr(controller, 'latent_bandwidth'):
        return None
    z = trainer.prior.z.detach()
    expected = z.std(0, unbiased=False) * len(z) ** (-1. / z.shape[1])
    assert torch.equal(controller.latent_bandwidth, expected)
    return material(torch, expected)


def cpu_check(torch, package, host, declaration, spec):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    from particlegan import initialization, qr_bz_pq_init
    assert initialization._external_init is None
    originals = {name: getattr(initialization, name) for name in ('_initialize', '_initialize_prior')}
    witnesses = []
    def wrap(name):
        def call(*args, **kwargs):
            before = torch.get_rng_state().clone()
            result = originals[name](*args, **kwargs)
            assert torch.equal(before, torch.get_rng_state()), 'initializer consumed RNG'
            witnesses.append(name)
            return result
        return call
    try:
        for name in originals:
            setattr(initialization, name, wrap(name))
        first, _ = construct(torch, package, host, declaration, spec, 'cpu')
    finally:
        for name, function in originals.items():
            setattr(initialization, name, function)
    def initial(trainer):
        state = trainer.state_dict()
        for key in ('streams', 'cpu_rng', 'cuda_rng'):
            state.pop(key, None)
        return material(torch, state)
    first_material = initial(first)
    torch.rand(97)  # Same seed, no reset: preceding draw count must not affect weights.
    second, _ = construct(torch, package, host, declaration, spec, 'cpu')
    assert initial(second) == first_material
    assert set(witnesses) == set(originals)
    expected = qr_bz_pq_init.qmc_draw(0, (spec['particles'], spec['z_dim']),
                                    ('normal', 0., 1.), rows_as_points=True).to(first.prior.z)
    assert torch.equal(first.prior.z, expected)
    assert first.completed_steps == second.completed_steps == 0
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_CPU_ZERO_STEP', cuda_initialized=False, initializer_rng_neutral=True,
                repeated_without_rng_reset=True, initial_material=first_material,
                geometry=geometry(torch, first), optimizer=optimizer_proof(torch, first, declaration))


def run_cuda(torch, package, host, declaration, spec, args, source_hashes):
    from particlegan.training import input_noise_std, output_noise_std
    receipt = json.loads(args.cpu_receipt.read_text())
    assert receipt['status'] == 'PASS_CPU_ZERO_STEP' and receipt['cuda_initialized'] is False
    assert receipt['candidate_declaration_sha256'] == sha(args.declaration.read_bytes())
    assert receipt['task'] == args.task and receipt['source_sha256'] == source_hashes
    assert str(torch.get_default_device()) == 'cpu'
    assert str(torch.__version__) == '2.13.0+cu126' and torch.version.cuda == '12.6'
    assert torch.cuda.get_device_name(0) == 'NVIDIA RTX A6000'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
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
    centers = host.templates(spec).to('cuda:0')
    trainer, shared = construct(torch, package, host, declaration, spec, 'cuda:0')
    assert trainer.completed_steps == 0
    geometry(torch, trainer)
    def without_tensor_devices(value):
        if isinstance(value, dict):
            tensor = {'shape', 'dtype', 'sha256', 'device'} <= value.keys()
            return {k: without_tensor_devices(v) for k, v in value.items()
                    if not (tensor and k == 'device')}
        if isinstance(value, list):
            return [without_tensor_devices(v) for v in value]
        return value
    actual_models = material(torch, trainer.state_dict()['models'])
    assert without_tensor_devices(actual_models) == without_tensor_devices(receipt['initial_material']['models'])
    dump(args.output / 'initial-model-cpu-cuda-proof.json', dict(status='EXACT_ALL_MODEL_BYTES',
         tensor_placement_only_ignored=True, models=actual_models))
    # Frozen constructors are CPU-only and the public initializer draws no RNG.
    untouched = torch.Generator(device='cuda:0').manual_seed(0)
    assert torch.equal(shared.get_state(), untouched.get_state())

    def checkpoint():
        return dict(trainer=trainer.state_dict(), global_cpu_rng=torch.get_rng_state(),
                    global_cuda_rng=torch.cuda.get_rng_state_all(), data_rng=shared.get_state(),
                    modes={name: [m.training for m in getattr(trainer, name).modules()]
                           for name in ('G', 'D', 'prior', 'ema_G', 'ema_prior')})

    def data(stream=None):
        real = centers[torch.randint(len(centers), (spec['batch_size'],),
                                    device='cuda:0', generator=stream)]
        noise = (torch.randn_like(real) if stream is None else
                 torch.randn(real.shape, device=real.device, dtype=real.dtype, generator=stream))
        return (real + spec['noise_std'] * noise).clamp(0., 1.)

    initial = checkpoint()
    torch.save(initial, args.output / 'initial-state.pt')
    dump(args.output / 'initial.json', material(torch, initial))
    dump(args.output / 'optimizer-initial.json', optimizer_proof(torch, trainer, declaration))
    # Full fixed-window sampling preflight, no model forward/backward/update.
    # Its cloned global stream follows real -> D indices -> G indices exactly.
    oracle = torch.Generator(device='cuda:0')
    oracle.set_state(shared.get_state())
    batches = []
    for completed in range(1, spec['steps'] + 1):
        real = data(oracle)
        idx_d = torch.randint(spec['particles'], (spec['batch_size'],), device='cuda:0', generator=oracle)
        idx_g = torch.randint(spec['particles'], (spec['batch_size'],), device='cuda:0', generator=oracle)
        batches.append(dict(step=completed, real=material(torch, real),
                            latent_d=material(torch, idx_d), latent_g=material(torch, idx_g),
                            accepted_cursor=material(torch, oracle.get_state())))
    assert material(torch, checkpoint()) == material(torch, initial), 'sampling preflight changed learner/caller state'
    dump(args.output / 'batch-receipts.json', batches)
    observations, started = [], time.monotonic()
    expected = host.evaluation_steps(spec)
    assert expected == list(range(25, 601, 25))

    def measure(ema=False):
        model, prior = ((trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior))
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            stream = torch.Generator(device='cuda:0').manual_seed(402 + trainer.completed_steps + 1901)
            arguments = [model, prior.z, output_noise_std(trainer.recipe, trainer.completed_steps), stream]
            if declaration['evaluation_generate'] == 'indexed':
                arguments.append(torch.arange(prior.num_particles, device=prior.z.device))
            elif declaration['evaluation_generate'] != 'plain':
                raise ValueError('unreviewed generation binding')
            return host.image_metrics(trainer._generate(*arguments), centers, spec['thresholds'])

    try:
        with (args.output / 'metrics.jsonl').open('w', buffering=1) as metrics, (args.output / 'learning-rates.jsonl').open('w', buffering=1) as rates:
            for completed, batch in enumerate(batches, 1):
                real = data()
                assert material(torch, real) == batch['real'], 'real stream differs from frozen host'
                future = torch.Generator(device='cuda:0')
                future.set_state(shared.get_state())
                assert material(torch, trainer.prior.sample_indices(spec['batch_size'], generator=future)) == batch['latent_d']
                assert material(torch, trainer.prior.sample_indices(spec['batch_size'], generator=future)) == batch['latent_g']
                previous = torch.autograd.is_multithreading_enabled()
                with torch.autograd.set_multithreading_enabled(False):
                    trainer.step(real, generator_real=real, collect_stats=completed in expected)
                assert torch.autograd.is_multithreading_enabled() == previous
                assert str(torch.get_default_device()) == 'cpu'
                assert material(torch, shared.get_state()) == batch['accepted_cursor'], 'accepted global data/latent cursor differs'
                assert trainer.completed_steps == completed
                # Check all declared native/eager clocks; never repair optimizer state.
                optimizer_proof(torch, trainer, declaration, include_values=False)
                row = dict(step=completed, rates=[[g['lr'] for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)],
                           input_noise=input_noise_std(trainer.recipe, completed - 1),
                           output_noise=output_noise_std(trainer.recipe, completed - 1),
                           evaluation_output_noise=output_noise_std(trainer.recipe, completed))
                if hasattr(trainer.opt_d, 'record'):
                    row['critic_record'] = material(torch, trainer.opt_d.record.state_dict())
                controller = getattr(trainer, 'controller', None)
                if controller is not None and hasattr(controller, 'diagnostics'):
                    row['policy'] = controller.diagnostics()
                if hasattr(trainer, 'game_stats'):
                    row['game_stats'] = material(torch, trainer.game_stats)
                precision = getattr(trainer, 'precision', None)
                if precision is not None:
                    if hasattr(precision, 'state'):
                        row['precision'] = material(torch, precision.state)
                    elif hasattr(precision, 'diagnostics'):
                        row['precision'] = material(torch, precision.diagnostics())
                rates.write(json.dumps(row, allow_nan=False) + '\n')
                if completed in expected:
                    before = material(torch, checkpoint())
                    point = dict(step=completed, seconds=time.monotonic() - started, **measure(), ema=measure(True))
                    assert material(torch, checkpoint()) == before, 'evaluation changed training/caller state'
                    observations.append(point)
                    metrics.write(json.dumps(point, allow_nan=False) + '\n')
                    print(json.dumps(point, allow_nan=False), flush=True)
        convergence = host.sustained(observations,
            [('modes', '>=', spec['thresholds']['modes']), ('hq', '>=', spec['thresholds']['hq_min'])],
            expected_steps=expected, minimum=spec['thresholds']['minimum_stable_checks'])
        status = 'PASS' if convergence['complete'] and convergence['passing_suffix'] >= 5 else 'FAIL'
        torch.save(checkpoint(), args.output / 'final-state.pt')
        dump(args.output / 'optimizer-final.json', optimizer_proof(torch, trainer, declaration))
        result = dict(status=status, convergence=convergence, final=observations[-1])
    except Exception:
        torch.save(checkpoint(), args.output / 'error-state.pt')
        result = dict(status='ERROR', traceback=traceback.format_exc(), observations=len(observations))
    result.update(candidate=declaration['candidate'], task=args.task, completed_steps=trainer.completed_steps,
                  seconds=time.monotonic() - started, recipe=trainer.recipe.to_dict(),
                  imports=verify_imports(args.package_root, declaration), cpu_receipt_sha256=sha(args.cpu_receipt.read_bytes()))
    dump(args.output / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('candidate', 'task', 'status', 'completed_steps', 'seconds')}), flush=True)
    if result['status'] == 'ERROR':
        raise RuntimeError(result['traceback'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--task', choices=('img_intensity2', 'img_bars4', 'img_blobs4', 'img_stripes2'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--cpu-receipt', type=Path)
    args = parser.parse_args()
    declaration = verify(args.package_root, args.declaration)
    spec = json.loads((ROOT / 'task-specs.json').read_text())[args.task]
    args.output.mkdir(parents=True, exist_ok=False)
    source = {name: args.package_root / name for name in declaration['package_sha256']}
    source.update({f.name: f for f in ROOT.iterdir() if f.is_file() and f.suffix in ('.py', '.json', '.md')})
    with zipfile.ZipFile(args.output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, file in source.items():
            archive.write(file, name)
        archive.write(args.declaration, 'candidate-declaration.json')
    source_hashes = {name: sha(file.read_bytes()) for name, file in source.items()}
    dump(args.output / 'declaration.json', dict(candidate=declaration, task=spec,
         image_policy='only num_particles/z_dim/batch_size adapted; all learner policy retained',
         source_sha256=source_hashes))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    import torch
    package, host = import_modules(args.package_root, declaration)
    if args.cpu_only:
        receipt = cpu_check(torch, package, host, declaration, spec)
        receipt.update(candidate_declaration_sha256=sha(args.declaration.read_bytes()),
                       source_zip_sha256=sha((args.output / 'source.zip').read_bytes()),
                       task=args.task, source_sha256=source_hashes,
                       imports=verify_imports(args.package_root, declaration))
        dump(args.output / 'cpu-preflight.json', receipt)
        print(json.dumps({'status': receipt['status'], 'task': args.task}), flush=True)
        return
    if args.cpu_receipt is None:
        raise ValueError('matching image-specific CPU zero-step receipt required')
    try:
        run_cuda(torch, package, host, declaration, spec, args, source_hashes)
    except Exception:
        if not (args.output / 'result.json').exists():
            dump(args.output / 'result.json', dict(status='ERROR', candidate=declaration['candidate'],
                 task=args.task, traceback=traceback.format_exc(), phase='pre_update_runtime_checks'))
        raise
    finally:
        dump(args.output / 'artifact-sha256.json', {f.name: sha(f.read_bytes()) for f in args.output.iterdir()
             if f.is_file() and f.name != 'artifact-sha256.json'})


if __name__ == '__main__':
    main()
