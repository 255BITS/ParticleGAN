"""Source-sealed fixed-init mode-hold screen; training is only in main()."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true', help='CUDA construction/sampling checks; no update')
    args = parser.parse_args()
    from preflight import verify, import_package, verify_imports, sha
    declaration, protocol, fixture, expected_batches, source_proof = verify(args.package_root, args.declaration)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'declaration.json', declaration)
    dump(args.output / 'protocol.json', protocol)
    dump(args.output / 'source-preflight.json', source_proof)
    inputs = {name: (args.package_root / name) for name in declaration['package_sha256']}
    inputs.update({name: ROOT / name for name in protocol['source_sha256']})
    inputs.update({name: ROOT / name for name in ('mode_hold_harness.py', 'mode_hold_contract.py',
                                                  'init_contract.py', 'preflight.py', 'protocol.json')})
    with zipfile.ZipFile(args.output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in inputs.items():
            archive.write(path, name)
        archive.write(args.declaration, 'candidate-declaration.json')
    identity = dict(candidate_declaration=sha(args.declaration.read_bytes()),
                    source_zip=sha((args.output / 'source.zip').read_bytes()),
                    initializer_commit=protocol['initializer_commit'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    import torch
    package = import_package(args.package_root, declaration)
    from particlegan.training import input_noise_std, output_noise_std
    from mode_hold_contract import construct, load_host, peek_batch, measure, serial_step
    from init_contract import tensor_receipt, initial_material, assert_geometry_after_initialization
    host = load_host()
    trainer = stream = None
    observations = []
    started = time.monotonic()
    generator_class = torch.Generator

    def cpu_default():
        assert str(torch.get_default_device()) == 'cpu'
        assert torch.Generator is generator_class

    def checkpoint():
        return dict(schema=1, execution={'serial_backward': True, 'scope': 'full public GANTrainer.step'},
                    identity=identity, trainer=trainer.state_dict(), data_rng=stream.get_state())

    def optimizer_proof():
        rows = {}
        for role, optimizer in (('G', trainer.opt_g), ('D', trainer.opt_d)):
            entries = []
            for group in optimizer.param_groups:
                assert group.get('capturable', False) is False
                assert group['foreach'] is False and group['fused'] is False
                for p in group['params']:
                    state = optimizer.state.get(p)
                    if not state:
                        entries.append(dict(shape=list(p.shape), state_missing=True))
                    else:
                        entries.append(dict(shape=list(p.shape), step=float(state['step']),
                            step_device=str(state['step'].device), parameter_device=str(p.device),
                            exp_avg_device=str(state['exp_avg'].device), exp_avg_sq_device=str(state['exp_avg_sq'].device)))
            rows[role] = entries
        dump(args.output / f'optimizer-device-proof-{trainer.completed_steps:04d}.json', rows)
        if trainer.completed_steps or declaration['initial_optimizer_state'] == 'declared_eager':
            assert all(not x.get('state_missing') and x['step_device'] == (
                           x['parameter_device'] if declaration['optimizer_step_devices'][role] == 'parameter' else 'cpu')
                       and x['step'] == trainer.completed_steps
                       and x['parameter_device'] == x['exp_avg_device'] == x['exp_avg_sq_device'] == 'cuda:0'
                       for role, values in rows.items() for x in values), 'declared Adam state mismatch; no repair allowed'

    try:
        cpu_default()
        assert torch.cuda.is_available(), 'frozen score requires CUDA; no CPU score fallback'
        expected = protocol['runtime_expected']
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
        torch.manual_seed(0)
        stream = torch.Generator(device='cuda:0').manual_seed(0)
        trainer, means = construct(torch, package, host, declaration, 'cuda:0', stream)
        cpu_default()
        assert trainer.completed_steps == 0
        if declaration['initial_optimizer_state'] == 'native_lazy':
            assert not trainer.opt_g.state and not trainer.opt_d.state
        initial = checkpoint()
        geometry = assert_geometry_after_initialization(torch, trainer)
        torch.save(initial, args.output / 'initial-state.pt')
        state = initial['trainer']
        raw = fixture['expected_initial']
        # Values changed by this epoch; constructor and sampling RNG did not.
        for key in ('cpu_rng', 'cuda_rng'):
            assert tensor_receipt(torch, state[key]) == raw[key], key
        assert tensor_receipt(torch, stream.get_state()) == raw['data_rng']
        for key, value in state['streams'].items():
            if key in raw['streams']:
                assert tensor_receipt(torch, value) == raw['streams'][key], key
        dump(args.output / 'initial.json', dict(material=initial_material(torch, trainer),
             models={k: host.digest(v) for k, v in state['models'].items()},
             data_rng=tensor_receipt(torch, stream.get_state()),
             cpu_rng=tensor_receipt(torch, state['cpu_rng']), cuda_rng=tensor_receipt(torch, state['cuda_rng']),
             old_sampling_rng_equal=True, old_model_values_loaded=False, **geometry))
        optimizer_proof()
        # Prove the complete frozen oracle sequence before any learner update.
        dry = torch.Generator(device='cuda:0')
        dry.set_state(stream.get_state())
        for step in range(1, 1201):
            _, _, _, after, receipt = peek_batch(torch, host, trainer, dry, means, 'cuda:0', step)
            assert receipt == expected_batches[step - 1], f'full fixture preflight mismatch at {step}'
            dry.set_state(after)
        assert tensor_receipt(torch, dry.get_state()) == fixture['expected_final_data_rng']
        dump(args.output / 'sampling-preflight.json', dict(status='PASS', matched_batches=1200,
             actual_cuda_draws=True, optimizer_updates=0, training_stream_unchanged=torch.equal(stream.get_state(), initial['data_rng'])))
        dump(args.output / 'runtime.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0), threads=1, interop_threads=1, deterministic=True,
             tf32=False, learner_default_device='cpu', serial_backward=True,
             imports=verify_imports(args.package_root, declaration), recipe=trainer.recipe.to_dict(),
             environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG')}))
        if args.preflight_only:
            status, metrics = 'PREFLIGHT_PASS', dict(updates=0, matched_batch_receipts=1200, quality='NOT_RUN')
        else:
            with (args.output / 'metrics.jsonl').open('w', buffering=1) as obs, \
                 (args.output / 'learning-rates.jsonl').open('w', buffering=1) as rates, \
                 (args.output / 'batch-receipts.jsonl').open('w', buffering=1) as batches:
                for step in range(1, 1201):
                    real, real_g, before_g, after_g, receipt = peek_batch(torch, host, trainer, stream, means, 'cuda:0', step)
                    assert receipt == expected_batches[step - 1]
                    cpu_default()
                    with serial_step(torch):
                        stats = trainer.step(real, generator_real=real_g, collect_stats=step % 50 == 0)
                    cpu_default()
                    assert trainer.completed_steps == step
                    assert torch.equal(stream.get_state(), before_g), 'candidate changed shared latent draw transaction'
                    stream.set_state(after_g)
                    batches.write(json.dumps(receipt) + '\n')
                    row = dict(step=step, applied_group_rates=[
                        [dict(lr=g['lr'], betas=list(g['betas']), parameters=sum(p.numel() for p in g['params']))
                         for g in opt.param_groups] for opt in (trainer.opt_g, trainer.opt_d)],
                        input_noise=input_noise_std(trainer.recipe, step - 1),
                        output_noise=output_noise_std(trainer.recipe, step - 1),
                        penalty=trainer.penalty.diagnostics(),
                        losses={k: float(v) for k, v in stats.items() if k != 'penalty_stats'})
                    for name in ('controller', 'precision'):
                        controller = getattr(trainer, name, None)
                        if controller is not None and hasattr(controller, 'diagnostics'):
                            row[name] = controller.diagnostics()
                    if hasattr(trainer, 'game_stats'):
                        row['game_stats'] = trainer.game_stats
                    rates.write(json.dumps(row, allow_nan=False) + '\n')
                    if step in (1, 1200):
                        optimizer_proof()
                    if step % 50 == 0:
                        before = host.digest([trainer.state_dict(), stream.get_state()])
                        measurement = dict(step=step, **measure(torch, host, trainer, means, declaration, output_noise_std),
                            ema=measure(torch, host, trainer, means, declaration, output_noise_std, True),
                            seconds=time.monotonic() - started)
                        assert host.digest([trainer.state_dict(), stream.get_state()]) == before
                        observations.append(measurement)
                        obs.write(json.dumps(measurement, allow_nan=False) + '\n')
                        print(json.dumps(measurement), flush=True)
            assert tensor_receipt(torch, stream.get_state()) == fixture['expected_final_data_rng']
            verdict = host.test_verdict(protocol['host'], dict(live=observations[-1], observations=observations))
            assert verdict['convergence']['complete']
            status = 'PASS' if verdict['passed'] and verdict['convergence']['passing_suffix'] >= 5 else 'FAIL'
            metrics = dict(verdict=verdict, final=observations[-1], updates=1200, matched_batch_receipts=1200)
            torch.save(checkpoint(), args.output / 'final-state.pt')
    except Exception as error:
        status = 'ERROR'
        metrics = dict(error=repr(error), traceback=traceback.format_exc(), observations=len(observations))
        if trainer is not None:
            torch.save(checkpoint(), args.output / 'error-state.pt')
    result = dict(candidate=declaration['candidate'], gate='deterministic_mode_hold', status=status,
                  seconds=time.monotonic() - started, metrics=metrics, old_quality_inherited=False)
    dump(args.output / 'result.json', result)
    dump(args.output / 'artifact-sha256.json', {str(p.relative_to(args.output)): sha(p.read_bytes())
        for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact-sha256.json'})
    print(json.dumps(result), flush=True)
    if status == 'ERROR':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
