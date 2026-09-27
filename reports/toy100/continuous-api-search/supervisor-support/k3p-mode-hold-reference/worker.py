"""One frozen mode_hold reference using the exact public K3P v0.8.0 package.

Requires independent source review and supervisor release before execution.
Importing this file performs no Torch import, model construction or training.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from preflight import verify
    verification = verify(BUNDLE)
    declaration = json.loads((BUNDLE / 'declaration.json').read_text())
    fixture = json.loads((BUNDLE / 'fixture-receipt.json').read_text())
    manifest = json.loads((BUNDLE / 'bundle-sha256.json').read_text())
    expected_batches = [json.loads(line) for line in gzip.decompress(
        (BUNDLE / 'fixtures/rp5/batch-receipts.jsonl.gz').read_bytes()).decode().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'declaration.json', declaration)
    dump(args.output / 'source-preflight.json', verification)
    with zipfile.ZipFile(args.output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in manifest['files']:
            archive.write(BUNDLE / name, name)
        archive.write(BUNDLE / 'bundle-sha256.json', 'bundle-sha256.json')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    if any(name == 'particlegan' or name.startswith('particlegan.') for name in sys.modules):
        raise RuntimeError('run in a fresh process: ParticleGAN was already imported')
    sys.path.insert(0, str(BUNDLE / 'source'))
    import torch
    from particlegan import GANTrainer, get_recipe
    from particlegan.training import input_noise_std, output_noise_std
    from particlegan.recipes import learning_rate_scales
    import frozen_host as host
    from execution_contract import EXECUTION, host_cuda, serial_step, validate_checkpoint

    identity = dict(bundle_manifest_sha256=hashlib.sha256((BUNDLE / 'bundle-sha256.json').read_bytes()).hexdigest(),
                    recipe=declaration['recipe'], fixture_receipt_sha256=declaration['fixture_receipt_sha256'],
                    evaluation=declaration['evaluation'])
    trainer = stream = None
    observations = []
    started = time.monotonic()
    original_generator = torch.Generator

    def tensor_receipt(tensor):
        cpu = tensor.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=hashlib.sha256(raw).hexdigest())

    def checkpoint():
        return dict(schema=1, execution=EXECUTION, identity=identity,
                    trainer=trainer.state_dict(), data_rng=stream.get_state())

    def restore_checkpoint(envelope):
        validate_checkpoint(envelope, identity)
        trainer.load_state_dict(envelope['trainer'])
        stream.set_state(envelope['data_rng'].cpu())

    def assert_cpu_defaults():
        assert str(torch.get_default_device()) == 'cpu', 'learner factories must remain on CPU'
        assert torch.Generator is original_generator, 'host generator routing leaked into learner'

    def optimizer_proof():
        proof = {}
        for role, opt in (('generator', trainer.opt_g), ('critic', trainer.opt_d)):
            entries = []
            for group in opt.param_groups:
                assert group['capturable'] is False and group['foreach'] is False and group['fused'] is False
                for parameter in group['params']:
                    state = opt.state.get(parameter)
                    if not state:
                        entries.append(dict(shape=list(parameter.shape), state_missing=True))
                        continue
                    entries.append(dict(shape=list(parameter.shape), parameter_device=str(parameter.device),
                        step_device=str(state['step'].device), step=float(state['step']),
                        exp_avg_device=str(state['exp_avg'].device), exp_avg_sq_device=str(state['exp_avg_sq'].device)))
            proof[role] = entries
        dump(args.output / f'optimizer-device-proof-{trainer.completed_steps:04d}.json', proof)
        assert all(not p.get('state_missing', False) and p['step_device'] == 'cpu'
                   and p['step'] == trainer.completed_steps
                   and p['parameter_device'] == p['exp_avg_device'] == p['exp_avg_sq_device'] == 'cuda:0'
                   for entries in proof.values() for p in entries), 'native Adam state mismatch; no repair permitted'

    try:
        assert_cpu_defaults()
        assert torch.cuda.is_available(), 'CUDA is required; no CPU score fallback'
        expected = declaration['runtime_expected']
        assert str(torch.__version__) == expected['torch']
        assert torch.version.cuda == expected['cuda']
        assert torch.cuda.get_device_name(0) == expected['gpu']
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
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if name == 'particlegan' or name.startswith('particlegan.'):
                path = Path(module.__file__).resolve()
                relative = str(path.relative_to(BUNDLE / 'source'))
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                assert actual == declaration['release']['source_sha256'][relative]
                imported[name] = dict(path=str(path), sha256=actual)
        dump(args.output / 'runtime.json', dict(torch=str(torch.__version__), torch_revision=torch.version.git_version,
             cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
             deterministic=True, tf32=False, threads=1, interop_threads=1, imported_package=imported,
             default_device_during_learner='cpu', serial_backward=True,
             environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
                          'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}))
        spec = declaration['spec']
        recipe = get_recipe(total_steps=1200, num_particles=12, z_dim=4, batch_size=128)
        assert json.loads(json.dumps(recipe.to_dict())) == declaration['recipe']
        assert input_noise_std(recipe, 120) == 0 and output_noise_std(recipe, 240) == .029
        torch.manual_seed(0)
        stream = torch.Generator(device='cuda:0').manual_seed(0)
        with host_cuda(torch):
            means = host.ring_means()
            prior = recipe.make_prior(init_std=.5, generator=stream)
            generator = host.SimpleMLPGenerator(4, 96, 3, 2)
            critic = host.SimpleMLPDiscriminator(2, 96, 3, 3)
        assert sum(p.numel() for m in (generator, prior) for p in m.parameters()) == 19346
        assert sum(p.numel() for p in critic.parameters()) == 20161
        assert_cpu_defaults()
        trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=0, latent_generator=stream,
                             optimizer_options={'foreach': False, 'fused': False})
        assert not trainer.opt_g.state and not trainer.opt_d.state, 'released Adam must start lazy and empty'
        initial = checkpoint()
        validate_checkpoint(initial, identity)
        torch.save(initial, args.output / 'initial-state.pt')
        state = initial['trainer']
        actual_models = {name: {key: tensor_receipt(value) for key, value in values.items()}
                         for name, values in state['models'].items()}
        raw_initial = fixture['expected_initial']
        initial_receipt = dict(backend='canonical CUDA shared-stream host; native CPU Adam counters',
            complete=host.digest(initial), models={k: host.digest(v) for k, v in state['models'].items()},
            raw_models=actual_models, streams={k: tensor_receipt(v) for k, v in state['streams'].items()},
            cpu_rng=tensor_receipt(state['cpu_rng']), cuda_rng=tensor_receipt(state['cuda_rng']),
            data_rng=tensor_receipt(initial['data_rng']), adam_state_empty=True)
        dump(args.output / 'initial.json', initial_receipt)
        assert initial_receipt['models'] == fixture['expected_initial_model_hashes']
        assert actual_models == raw_initial['models']
        assert initial_receipt['cpu_rng'] == raw_initial['cpu_rng']
        assert initial_receipt['cuda_rng'] == raw_initial['cuda_rng']
        assert initial_receipt['data_rng'] == raw_initial['data_rng']
        for name, value in initial_receipt['streams'].items():
            assert value == raw_initial['streams'][name], name

        def real_data(rng):
            with host_cuda(torch):
                return host.sample_ring(means, 128, host.SIGMA, rng)

        @torch.no_grad()
        def measure(ema=False):
            model, table = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
            modes = [(m, m.training) for root in (model, table) for m in root.modules()]
            try:
                model.eval()
                table.eval()
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(402 + trainer.completed_steps)
                    latent = table.sample(4096, generator=torch.Generator(device='cuda:0').manual_seed(9))[0]
                    fake = model(latent)
                    sigma = output_noise_std(recipe, trainer.completed_steps)
                    if sigma:
                        fake = fake + sigma * torch.randn_like(fake)
                    return host.diversity(fake, means)
            finally:
                for module, flag in modes:
                    module.training = flag

        with (args.output / 'metrics.jsonl').open('w', buffering=1) as obs, \
             (args.output / 'learning-rates.jsonl').open('w', buffering=1) as lr, \
             (args.output / 'batch-receipts.jsonl').open('w', buffering=1) as batches:
            for step in range(1, 1201):
                real = real_data(stream)
                future = torch.Generator(device='cuda:0')
                future.set_state(stream.get_state())
                latent_d = prior.sample_indices(128, generator=future)
                latent_g = prior.sample_indices(128, generator=future)
                before_g_real = future.get_state()
                real_g = real_data(future)
                after_g_real = future.get_state()
                # Fail before updating on any reconstructed batch/index drift.
                expected_batch = expected_batches[step - 1]
                batch = dict(step=step, real_d=host.digest(real), latent_d=host.digest(latent_d),
                             latent_g=host.digest(latent_g), real_g=host.digest(real_g),
                             accepted_cursor=host.digest(after_g_real))
                assert batch == expected_batch, 'frozen batch fixture mismatch before learner step'
                assert_cpu_defaults()
                with serial_step(torch):
                    stats = trainer.step(real, generator_real=real_g, collect_stats=step % 50 == 0)
                assert_cpu_defaults()
                assert torch.equal(stream.get_state(), before_g_real), 'public step must consume exactly two frozen latent draws'
                stream.set_state(after_g_real)
                assert host.digest(stream.get_state()) == expected_batch['accepted_cursor']
                batches.write(json.dumps(batch, allow_nan=False) + '\n')
                if step in (1, 1200):
                    optimizer_proof()
                network_scale, prior_scale = learning_rate_scales(step - 1, recipe)
                for optimizer, base, roles in zip((trainer.opt_g, trainer.opt_d), trainer.initial_lrs, trainer.roles):
                    for group, rate, role in zip(optimizer.param_groups, base, roles):
                        assert group['lr'] == rate * (prior_scale if role == 'prior' else network_scale)
                lr.write(json.dumps(dict(step=step, **host.rates(trainer),
                    input_noise=input_noise_std(recipe, step - 1), output_noise=output_noise_std(recipe, step - 1),
                    penalty=trainer.penalty.diagnostics(), losses={k: float(v) for k, v in stats.items()
                                                                   if k != 'penalty_stats'}), allow_nan=False) + '\n')
                if step % 50 == 0:
                    before = host.digest([trainer.state_dict(), stream.get_state()])
                    row = dict(step=step, **measure(), ema=measure(True), seconds=time.monotonic() - started)
                    assert host.digest([trainer.state_dict(), stream.get_state()]) == before
                    observations.append(row)
                    obs.write(json.dumps(row, allow_nan=False) + '\n')
                    print(json.dumps(row), flush=True)
        assert tensor_receipt(stream.get_state()) == fixture['expected_final_data_rng']
        verdict = host.test_verdict(spec, dict(live=observations[-1], observations=observations))
        assert verdict['convergence']['complete']
        status = 'PASS' if verdict['passed'] and verdict['convergence']['passing_suffix'] >= 5 else 'FAIL'
        metrics = dict(verdict=verdict, final=observations[-1], updates=trainer.completed_steps,
                       matched_batch_receipts=1200, policy=trainer.penalty.diagnostics())
        torch.save(checkpoint(), args.output / 'final-state.pt')
    except Exception as error:
        status = 'ERROR'
        metrics = dict(error=repr(error), traceback=traceback.format_exc(), observations=len(observations))
        if trainer is not None and stream is not None:
            try:
                torch.save(checkpoint(), args.output / 'error-state.pt')
            except Exception as checkpoint_error:
                metrics['checkpoint_error'] = repr(checkpoint_error)
    result = dict(candidate=declaration['candidate'], gate=declaration['gate'], status=status,
                  seconds=time.monotonic() - started, metrics=metrics, artifact=str(args.output.resolve()))
    dump(args.output / 'result.json', result)
    dump(args.output / 'artifact-sha256.json', {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact-sha256.json'})
    print(json.dumps(result), flush=True)
    if status == 'ERROR':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
