"""Exact frozen unequal_mass follow-up for the unchanged new-init DV12 package."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile
from receipts import sha, dump, material, optimizer_proof, geometry

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def verify(package_root, declaration_path):
    expected = json.loads((ROOT / 'bundle-sha256.json').read_text())
    actual = {f.name: sha(f.read_bytes()) for f in ROOT.iterdir()
              if f.is_file() and f.name != 'bundle-sha256.json'}
    assert actual == expected
    plan = json.loads((ROOT / 'followup-protocol.json').read_text())
    assert sha(declaration_path.read_bytes()) == plan['candidate_declaration_sha256']
    declaration = json.loads(declaration_path.read_text())
    assert declaration['candidate'] == 'API-DV12-new-init'
    assert declaration['initial_optimizer_state'] == 'native_lazy'
    assert declaration['optimizer_step_devices'] == {'G': 'cpu', 'D': 'cpu'}
    actual = {str(f.relative_to(package_root)): sha(f.read_bytes())
              for f in (package_root / 'particlegan').rglob('*.py')}
    assert actual == declaration['package_sha256']
    assert declaration['initializer_commit'] == 'c720645ecae6b648e9fc6034e9d6b48ccff06ed3'
    assert declaration['recipe_overrides']['initialization'] == 'batch_feature_zero'
    assert declaration['resolved_recipe']['continuous_policy'] == 'dv12'
    assert declaration['resolved_recipe']['total_steps'] is None
    task = json.loads((ROOT / 'task.json').read_text())
    proof = json.loads((ROOT / 'host-source-proof.json').read_text())
    assert sha((ROOT / 'frozen_vector_host.py').read_bytes()) == proof['frozen_host_sha256']
    assert sha((ROOT / 'task.json').read_bytes()) == proof['task_sha256']
    return declaration, task


def imports(package_root, declaration):
    rows = {}
    for name, module in list(sys.modules.items()):
        if name == 'particlegan' or name.startswith('particlegan.'):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(package_root.resolve()))
            assert sha(path.read_bytes()) == declaration['package_sha256'][relative]
            rows[name] = dict(path=str(path), sha256=sha(path.read_bytes()))
    return rows


def construct(torch, package, host, declaration, cfg, card, device):
    assert str(torch.get_default_device()) == 'cpu'
    recipe_args = dict(declaration['recipe_overrides'])
    recipe_args.update(num_particles=cfg['particles'], z_dim=cfg['z_dim'], batch_size=cfg['batch'])
    recipe = package.get_recipe(**recipe_args)
    # Same archived CPU prior-first host. Historical learned fixture is not loaded.
    prior = recipe.make_prior(init_std=.5, generator=torch.Generator(device='cpu').manual_seed(0))
    generator = host.SimpleMLPGenerator(cfg['z_dim'], cfg['hidden'], cfg['layers'], 2)
    # Exact selected branch of original vector_discriminator; reject other cards.
    assert card['implementation'] == 'shared_batch_feature_v1'
    assert (card['feature'], card['placement'], card['trunk_normalization'], card['name']) == (
        'distance', 'head', 'center', 'batchfeat_center6_distance_head')
    critic = package.BatchDistanceDiscriminator(in_dim=2, hidden_dim=card['width'],
        n_hidden=card['layers'], scales=tuple(card['kernel_scales']),
        beta=card['softplus_beta'], eps=card['eps'])
    assert all(p.device.type == 'cpu' for m in (prior, generator, critic) for p in m.parameters())
    data = torch.Generator(device=device).manual_seed(0)
    options = dict(prior=prior.to(device), seed=0,
                   latent_generator=torch.Generator(device=device).manual_seed(1),
                   penalty_generator=torch.Generator(device=device).manual_seed(2),
                   optimizer_options={'foreach': False, 'fused': False})
    if declaration['serial_backward_argument']:
        options['serial_backward'] = True
    trainer = package.GANTrainer(recipe, generator.to(device), critic.to(device), **options)
    expected = dict(declaration['resolved_recipe'])
    expected.update(num_particles=cfg['particles'], z_dim=cfg['z_dim'], batch_size=cfg['batch'])
    assert json.loads(json.dumps(recipe.to_dict())) == expected
    geometry(torch, trainer)
    return trainer, data


def cpu_check(torch, package, host, declaration, cfg, card):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    from particlegan import initialization, qr_bz_pq_init
    assert initialization._external_init is None
    originals = {n: getattr(initialization, n) for n in ('_initialize', '_initialize_prior')}
    witnesses = []
    def wrapper(name):
        def call(*args, **kwargs):
            before = torch.get_rng_state().clone()
            result = originals[name](*args, **kwargs)
            assert torch.equal(before, torch.get_rng_state())
            witnesses.append(name)
            return result
        return call
    try:
        for name in originals:
            setattr(initialization, name, wrapper(name))
        first, _ = construct(torch, package, host, declaration, cfg, card, 'cpu')
    finally:
        for name, function in originals.items():
            setattr(initialization, name, function)
    def initial(trainer):
        state = trainer.state_dict()
        for key in ('streams', 'cpu_rng', 'cuda_rng'):
            state.pop(key, None)
        return material(torch, state)
    first_state = initial(first)
    torch.rand(97)
    second, _ = construct(torch, package, host, declaration, cfg, card, 'cpu')
    assert initial(second) == first_state
    assert set(witnesses) == set(originals)
    expected = qr_bz_pq_init.qmc_draw(0, (cfg['particles'], cfg['z_dim']),
                                    ('normal', 0., .5), rows_as_points=True).to(first.prior.z)
    assert torch.equal(first.prior.z, expected)
    assert torch.count_nonzero(first.D.head.weight[:, -first.D.scales.numel():]) == 0
    assert first.completed_steps == second.completed_steps == 0
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_CPU_ZERO_STEP', cuda_initialized=False, initializer_rng_neutral=True,
                repeated_without_rng_reset=True, initial_material=first_state,
                geometry=geometry(torch, first), optimizer=optimizer_proof(torch, first, declaration))


def run_cuda(torch, package, host, declaration, task, cfg, args, source_hashes):
    from particlegan.training import input_noise_std, output_noise_std
    receipt = json.loads(args.cpu_receipt.read_text())
    assert receipt['status'] == 'PASS_CPU_ZERO_STEP' and receipt['cuda_initialized'] is False
    assert receipt['source_sha256'] == source_hashes
    assert receipt['candidate_declaration_sha256'] == sha(args.declaration.read_bytes())
    assert str(torch.__version__) == '2.13.0+cu126' and torch.version.cuda == '12.6'
    assert torch.cuda.get_device_name(0) == 'NVIDIA RTX A6000'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert str(torch.get_default_device()) == 'cpu'
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
    trainer, data = construct(torch, package, host, declaration, cfg, task['discriminator_card'], 'cuda:0')
    def without_tensor_devices(value):
        if isinstance(value, dict):
            tensor = {'shape', 'dtype', 'sha256', 'device'} <= value.keys()
            return {k: without_tensor_devices(v) for k, v in value.items()
                    if not (tensor and k == 'device')}
        if isinstance(value, list):
            return [without_tensor_devices(v) for v in value]
        return value
    actual_models = material(torch, trainer.state_dict()['models'])
    assert without_tensor_devices(actual_models) == without_tensor_devices(receipt['initial_material']['models']), \
        'initialized CUDA model parameters/buffers differ from CPU proof'
    dump(args.output / 'initial-model-cpu-cuda-proof.json', dict(status='EXACT_ALL_MODEL_BYTES',
         tensor_placement_only_ignored=True, models=actual_models,
         derived_geometry='independently checked on the executing device'))

    def checkpoint():
        return dict(trainer=trainer.state_dict(), data_rng=data.get_state(),
                    global_cpu_rng=torch.get_rng_state(), global_cuda_rng=torch.cuda.get_rng_state_all())

    def real(step, stream=data):
        with torch.device('cuda:0'):
            return host.sample_target(cfg, cfg['batch'], stream, step)

    initial = checkpoint()
    torch.save(initial, args.output / 'initial-state.pt')
    dump(args.output / 'initial.json', material(torch, initial))
    dump(args.output / 'optimizer-initial.json', optimizer_proof(torch, trainer, declaration))
    dry_data = torch.Generator(device='cuda:0').manual_seed(0)
    dry_latent = torch.Generator(device='cuda:0').manual_seed(1)
    batches = []
    for step in range(1, cfg['steps'] + 1):
        real_d = real(step, dry_data)
        idx_d = trainer.prior.sample_indices(cfg['batch'], generator=dry_latent)
        idx_g = trainer.prior.sample_indices(cfg['batch'], generator=dry_latent)
        real_g = real(step, dry_data)
        batches.append(dict(step=step, real_d=material(torch, real_d), real_g=material(torch, real_g),
                            latent_d=material(torch, idx_d), latent_g=material(torch, idx_g),
                            data_cursor=material(torch, dry_data.get_state()), latent_cursor=material(torch, dry_latent.get_state())))
    assert material(torch, checkpoint()) == material(torch, initial)
    dump(args.output / 'expected-batches.json', batches)
    expected = list(range(50, 1201, 50))
    assert cfg['steps'] == 1200 and len(batches) == 1200
    observations, started = [], time.monotonic()

    def measure(completed, ema=False):
        model, prior = ((trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior))
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            latent, indices = prior.sample(4096, generator=torch.Generator(device='cuda:0').manual_seed(990))
            stream = torch.Generator(device='cuda:0').manual_seed(402 + 1901)
            arguments = [model, latent, output_noise_std(trainer.recipe, completed), stream]
            if declaration['evaluation_generate'] == 'indexed':
                arguments.append(indices)
            elif declaration['evaluation_generate'] != 'plain':
                raise ValueError('unreviewed generation binding')
            fake = trainer._generate(*arguments)
            with torch.device('cuda:0'):
                return host.score_samples(fake, cfg, completed)

    try:
        with (args.output / 'metrics.jsonl').open('w', buffering=1) as metrics, (args.output / 'learning-rates.jsonl').open('w', buffering=1) as rates:
            for step, batch in enumerate(batches, 1):
                real_d = real(step)
                assert material(torch, real_d) == batch['real_d']
                future = torch.Generator(device='cuda:0')
                future.set_state(trainer.latent_generator.get_state())
                assert material(torch, trainer.prior.sample_indices(cfg['batch'], generator=future)) == batch['latent_d']
                assert material(torch, trainer.prior.sample_indices(cfg['batch'], generator=future)) == batch['latent_g']
                calls = []
                def generator_real():
                    value = real(step)
                    assert material(torch, value) == batch['real_g']
                    calls.append(1)
                    return value
                previous = torch.autograd.is_multithreading_enabled()
                with torch.autograd.set_multithreading_enabled(False):
                    trainer.step(real_d, generator_real=generator_real, collect_stats=step in expected)
                assert torch.autograd.is_multithreading_enabled() == previous
                assert str(torch.get_default_device()) == 'cpu'
                assert calls == [1] and trainer.completed_steps == step
                assert material(torch, data.get_state()) == batch['data_cursor']
                assert material(torch, trainer.latent_generator.get_state()) == batch['latent_cursor']
                row = dict(step=step, applied_group_rates=[[g['lr'] for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)],
                           policy=trainer.controller.diagnostics(), critic=trainer.penalty.diagnostics(),
                           input_noise=input_noise_std(trainer.recipe, step - 1), output_noise=output_noise_std(trainer.recipe, step - 1),
                           evaluation_output_noise=output_noise_std(trainer.recipe, step))
                rates.write(json.dumps(row, allow_nan=False) + '\n')
                if step in (1, 1200):
                    dump(args.output / f'optimizer-{step}.json', optimizer_proof(torch, trainer, declaration))
                if step in expected:
                    before = material(torch, checkpoint())
                    point = dict(step=step, **measure(step), ema=measure(step, True), seconds=time.monotonic() - started)
                    assert material(torch, checkpoint()) == before
                    observations.append(point)
                    metrics.write(json.dumps(point, allow_nan=False) + '\n')
                    print(json.dumps(point), flush=True)
        verdict = host.test_verdict(task['spec'], dict(observations=observations, live=observations[-1]))
        assert verdict['convergence']['complete']
        status = 'PASS' if verdict['passed'] and verdict['convergence']['passing_suffix'] >= 5 else 'FAIL'
        torch.save(checkpoint(), args.output / 'final-state.pt')
        result = dict(status=status, verdict=verdict, final=observations[-1])
    except Exception:
        torch.save(checkpoint(), args.output / 'error-state.pt')
        result = dict(status='ERROR', traceback=traceback.format_exc(), observations=len(observations))
    result.update(candidate=declaration['candidate'], task='vector_unequal_mass',
                  completed_steps=trainer.completed_steps, seconds=time.monotonic() - started,
                  recipe=trainer.recipe.to_dict(), imports=imports(args.package_root, declaration),
                  cpu_receipt_sha256=sha(args.cpu_receipt.read_bytes()))
    dump(args.output / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('candidate', 'task', 'status', 'completed_steps', 'seconds')}), flush=True)
    if result['status'] == 'ERROR':
        raise RuntimeError(result['traceback'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--cpu-receipt', type=Path)
    args = parser.parse_args()
    declaration, task = verify(args.package_root, args.declaration)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {name: args.package_root / name for name in declaration['package_sha256']}
    sources.update({f.name: f for f in ROOT.iterdir() if f.is_file()})
    sources['candidate-declaration.json'] = args.declaration
    source_hashes = {n: sha(f.read_bytes()) for n, f in sources.items()}
    with zipfile.ZipFile(args.output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, file in sorted(sources.items()):
            archive.write(file, name)
    dump(args.output / 'declaration.json', dict(candidate=declaration, task=task,
         changed_host_resources_only=dict(num_particles=256, z_dim=4, batch_size=128),
         old_initial_weights_loaded=False, evaluation_noise_seed=2303, source_sha256=source_hashes))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    assert not any(n == 'particlegan' or n.startswith('particlegan.') for n in sys.modules)
    sys.path.insert(0, str(args.package_root.resolve()))
    import torch
    import particlegan as package
    module_spec = importlib.util.spec_from_file_location('frozen_vector_host', ROOT / 'frozen_vector_host.py')
    host = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(host)
    imports(args.package_root, declaration)
    cfg = host.resolve(task['spec'])
    if args.cpu_only:
        result = cpu_check(torch, package, host, declaration, cfg, task['discriminator_card'])
        result.update(candidate_declaration_sha256=sha(args.declaration.read_bytes()), source_sha256=source_hashes,
                      imports=imports(args.package_root, declaration))
        dump(args.output / 'cpu-preflight.json', result)
        print(json.dumps({'status': result['status'], 'candidate': declaration['candidate']}), flush=True)
        return
    if args.cpu_receipt is None:
        raise ValueError('matching vector-host CPU zero-step receipt required')
    try:
        run_cuda(torch, package, host, declaration, task, cfg, args, source_hashes)
    except Exception:
        if not (args.output / 'result.json').exists():
            dump(args.output / 'result.json', dict(status='ERROR', candidate=declaration['candidate'],
                 task='vector_unequal_mass', traceback=traceback.format_exc(), phase='pre_update_runtime_checks'))
        raise
    finally:
        dump(args.output / 'artifact-sha256.json', {f.name: sha(f.read_bytes()) for f in args.output.iterdir()
             if f.is_file() and f.name != 'artifact-sha256.json'})


if __name__ == '__main__':
    main()
