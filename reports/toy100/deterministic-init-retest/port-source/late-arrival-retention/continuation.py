"""Unchanged DV1–DV4 supplementary retention, resuming its completed1200 checkpoint."""
from pathlib import Path
import argparse
import hashlib
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


def exact(torch, value):
    """Include placement as well as bytes: CPU Adam clocks are meaningful here."""
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    sha256=sha(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()))
    if isinstance(value, dict):
        return {str(k): exact(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [exact(torch, v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unknown checkpoint value {type(value)}')


def source_check(candidate):
    seal = json.loads((ROOT / 'bundle-sha256.json').read_text())
    actual = {str(f.relative_to(ROOT)): sha(f.read_bytes()) for f in ROOT.rglob('*')
              if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'
              and f.name != 'bundle-sha256.json'}
    assert actual == seal, 'reviewed supplementary runner bundle changed'
    plan = json.loads((ROOT / 'continuation-protocol.json').read_text())
    plan = {**plan, **plan['candidates'][candidate]}
    for row in plan['original_artifacts'].values():
        assert sha(Path(row['path']).read_bytes()) == row['sha256']
    from preflight import verify
    declaration_path = Path(plan['candidate_declaration'])
    assert sha(declaration_path.read_bytes()) == plan['candidate_declaration_sha256']
    declaration, protocol, fixture, batches, proof = verify(Path(plan['package_root']), declaration_path)
    assert declaration == json.loads(Path(plan['original_artifacts']['declaration.json']['path']).read_text())
    assert declaration['candidate'] == plan['candidate']
    assert declaration['resolved_recipe']['total_steps'] is None
    assert declaration['initial_optimizer_state'] == 'native_lazy'
    assert declaration['optimizer_step_devices'] == {'G': 'cpu', 'D': 'cpu'}
    assert (plan['start_completed'], plan['end_completed']) == (1200, 2400)
    assert plan['observations'] == list(range(1250, 2401, 50))
    assert protocol == json.loads(Path(plan['original_artifacts']['protocol.json']['path']).read_text())
    return plan, declaration, protocol, fixture, batches, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--candidate', choices=('api-dv1','api-dv2','api-dv3','api-dv4'), required=True)
    args = parser.parse_args()
    plan, declaration, protocol, fixture, expected_batches, source_proof = source_check(args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'continuation-protocol.json', plan)
    dump(args.output / 'source-preflight.json', source_proof)
    inputs = {str(f.relative_to(ROOT)): f for f in ROOT.rglob('*')
              if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc'}
    inputs.update({name: Path(plan['package_root']) / name for name in declaration['package_sha256']})
    inputs['candidate-declaration.json'] = Path(plan['candidate_declaration'])
    with zipfile.ZipFile(args.output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(inputs.items()):
            archive.write(path, name)
    if args.source_only:
        dump(args.output / 'source-only.json', dict(status='PASS_SOURCE_ONLY_NO_MODEL_EXECUTION',
             source_sha256={n: sha(f.read_bytes()) for n, f in inputs.items()}))
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    import torch
    from preflight import import_package, verify_imports
    from mode_hold_contract import construct, load_host, peek_batch, measure, serial_step
    package = import_package(Path(plan['package_root']), declaration)
    from particlegan.training import input_noise_std, output_noise_std
    host = load_host()
    trainer = stream = None
    observations = []
    started = time.monotonic()
    expected_runtime = protocol['runtime_expected']

    def checkpoint():
        return dict(schema=1, execution={'serial_backward': True, 'scope': 'full public GANTrainer.step'},
                    identity=loaded['identity'], trainer=trainer.state_dict(), data_rng=stream.get_state())

    def counters():
        rows = {}
        for role, optimizer in (('G', trainer.opt_g), ('D', trainer.opt_d)):
            rows[role] = []
            for group in optimizer.param_groups:
                assert group.get('capturable', False) is False
                assert group['foreach'] is False and group['fused'] is False
                for parameter in group['params']:
                    state = optimizer.state[parameter]
                    assert state['step'].device.type == 'cpu'
                    assert float(state['step']) == trainer.completed_steps
                    assert state['exp_avg'].device == state['exp_avg_sq'].device == parameter.device
                    rows[role].append(dict(shape=list(parameter.shape), step=float(state['step']),
                                           step_device=str(state['step'].device), parameter_device=str(parameter.device),
                                           exp_avg_device=str(state['exp_avg'].device), exp_avg_sq_device=str(state['exp_avg_sq'].device)))
        return rows

    try:
        assert str(torch.get_default_device()) == 'cpu'
        assert str(torch.__version__) == expected_runtime['torch']
        assert torch.version.cuda == expected_runtime['cuda']
        assert torch.cuda.get_device_name(0) == expected_runtime['gpu']
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
        # Preserve original heterogeneous placement, particularly CPU Adam step.
        loaded = torch.load(plan['original_artifacts']['final-state.pt']['path'], weights_only=False)
        old_initial = torch.load(plan['original_artifacts']['initial-state.pt']['path'], weights_only=False)
        assert loaded['identity']['candidate_declaration'] == plan['candidate_declaration_sha256']
        assert loaded['identity']['source_zip'] == plan['original_artifacts']['source.zip']['sha256']
        assert loaded['execution'] == {'serial_backward': True, 'scope': 'full public GANTrainer.step'}
        assert loaded['trainer']['completed_steps'] == 1200
        trainer.load_state_dict(loaded['trainer'])
        stream.set_state(loaded['data_rng'])
        assert exact(torch, checkpoint()) == exact(torch, loaded), 'full serialized state/device/RNG restore mismatch'
        dump(args.output / 'restore-proof.json', dict(status='EXACT_SERIALIZED_STATE_AND_DEVICES',
             state=exact(torch, checkpoint()), optimizer=counters(),
             original_checkpoint_sha256=plan['original_artifacts']['final-state.pt']['sha256']))
        # Derive the extension from the original fixed host draw sequence, first
        # reproducing all1200 frozen receipts and the actual saved caller cursor.
        dry = torch.Generator(device='cuda:0')
        dry.set_state(old_initial['data_rng'])
        future_batches = []
        for step in range(1, 2401):
            _, _, _, after, receipt = peek_batch(torch, host, trainer, dry, means, 'cuda:0', step)
            if step <= 1200:
                assert receipt == expected_batches[step - 1]
            else:
                future_batches.append(receipt)
            dry.set_state(after)
            if step == 1200:
                assert torch.equal(dry.get_state(), loaded['data_rng'])
        assert exact(torch, checkpoint()) == exact(torch, loaded)
        dump(args.output / 'sampling-preflight.json', dict(status='PASS', matched_original_batches=1200,
             declared_extension_batches=1200, learner_updates=0, final_cursor=exact(torch, dry.get_state())))
        dump(args.output / 'expected-extension-batches.json', future_batches)
        original_points = [json.loads(line) for line in Path(plan['original_artifacts']['metrics.jsonl']['path']).read_text().splitlines()]
        before = exact(torch, checkpoint())
        restored = measure(torch, host, trainer, means, declaration, output_noise_std)
        restored_ema = measure(torch, host, trainer, means, declaration, output_noise_std, True)
        assert all(restored[k] == original_points[-1][k] for k in restored)
        assert restored_ema == original_points[-1]['ema']
        assert exact(torch, checkpoint()) == before
        dump(args.output / 'restored-observation.json', dict(step=1200, **restored, ema=restored_ema))
        with (args.output / 'metrics.jsonl').open('w', buffering=1) as obs, \
             (args.output / 'learning-rates.jsonl').open('w', buffering=1) as rates, \
             (args.output / 'batch-receipts.jsonl').open('w', buffering=1) as batches:
            for step in range(1201, 2401):
                real, real_g, before_g, after_g, receipt = peek_batch(torch, host, trainer, stream, means, 'cuda:0', step)
                assert receipt == future_batches[step - 1201]
                assert str(torch.get_default_device()) == 'cpu'
                with serial_step(torch):
                    stats = trainer.step(real, generator_real=real_g, collect_stats=step % 50 == 0)
                assert str(torch.get_default_device()) == 'cpu'
                assert trainer.completed_steps == step
                assert torch.equal(stream.get_state(), before_g)
                stream.set_state(after_g)
                batches.write(json.dumps(receipt) + '\n')
                row = dict(step=step, applied_group_rates=[
                    [dict(lr=g['lr'], betas=list(g['betas']), parameters=sum(p.numel() for p in g['params']))
                     for g in opt.param_groups] for opt in (trainer.opt_g, trainer.opt_d)],
                    input_noise=input_noise_std(trainer.recipe, step - 1),
                    output_noise=output_noise_std(trainer.recipe, step - 1),
                    penalty=trainer.penalty.diagnostics(),
                    losses={k: float(v) for k, v in stats.items() if k != 'penalty_stats'})
                controller = getattr(trainer, 'controller', None)
                if controller is not None and hasattr(controller, 'diagnostics'):
                    row['controller'] = controller.diagnostics()
                rates.write(json.dumps(row, allow_nan=False) + '\n')
                if step in (1201, 2400):
                    dump(args.output / f'optimizer-device-proof-{step}.json', counters())
                if step % 50 == 0:
                    before = exact(torch, checkpoint())
                    point = dict(step=step, **measure(torch, host, trainer, means, declaration, output_noise_std),
                        ema=measure(torch, host, trainer, means, declaration, output_noise_std, True),
                        seconds=time.monotonic() - started)
                    assert exact(torch, checkpoint()) == before
                    observations.append(point)
                    obs.write(json.dumps(point, allow_nan=False) + '\n')
                    print(json.dumps(point), flush=True)
        assert [p['step'] for p in observations] == plan['observations']
        assert torch.equal(stream.get_state(), dry.get_state())
        combined = original_points + observations
        passing = lambda p: p['modes'] >= 8 and p['hq'] >= .9
        first = next(p['step'] for p in combined if passing(p))
        since = [p for p in combined if p['step'] >= first]
        suffix = 0
        for point in reversed(combined):
            if not passing(point):
                break
            suffix += 1
        torch.save(checkpoint(), args.output / 'final-state.pt')
        result = dict(status='COMPLETE_SUPPLEMENT', original_gate_status='FAIL',
            original_gate_unchanged=True, updates=trainer.completed_steps, new_observations=len(observations),
            first_arrival=first, passing_since_arrival=sum(passing(p) for p in since),
            observations_since_arrival=len(since), departures=[p['step'] for p in since if not passing(p)],
            min_hq_since_arrival=min(p['hq'] for p in since), min_modes_since_arrival=min(p['modes'] for p in since),
            final_passing_suffix=suffix, final=observations[-1], recipe=trainer.recipe.to_dict())
    except Exception:
        result = dict(status='ERROR', traceback=traceback.format_exc(), observations=len(observations))
        if trainer is not None and 'loaded' in locals():
            torch.save(checkpoint(), args.output / 'error-state.pt')
    result.update(candidate=declaration['candidate'], seconds=time.monotonic() - started,
                  gate='supplementary_late_arrival_retention', imports=verify_imports(Path(plan['package_root']), declaration))
    dump(args.output / 'result.json', result)
    dump(args.output / 'artifact-sha256.json', {p.name: sha(p.read_bytes()) for p in args.output.iterdir()
                                             if p.is_file() and p.name != 'artifact-sha256.json'})
    print(json.dumps(result), flush=True)
    if result['status'] == 'ERROR':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
