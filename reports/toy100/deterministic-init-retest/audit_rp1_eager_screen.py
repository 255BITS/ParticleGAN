"""Independently audit a completed API screen without Torch or learner imports."""
import argparse
import gzip
import hashlib
import io
import math
from audit_public3_runtime import Reader, Tensor
import json
from pathlib import Path
import zipfile

from audit_public3_runtime import checkpoint, read, sha
from archive_screen import summarize

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    bundle = ROOT / 'rp1-eager-screen'
    declaration = read(bundle / 'candidate-declaration.json')
    digest = sha((bundle / 'candidate-declaration.json').read_bytes())
    cpu = read(ROOT / 'rp1-eager-review/cpu-preflight.json')
    review = read(ROOT / 'rp1-eager-review/independent-audit.json')
    assert declaration['candidate'] == 'API-RP1-CUDA-EAGER-new-init-diagnostic'
    assert digest == review['declaration_sha256'] == '59a92e348eb1fc230a2ebbf3138cb53422453ebedfbdbfaae9ce1bf39b29bd7b'
    assert sha((bundle / 'manifest.json').read_bytes()) == review['manifest_sha256'] == 'bea404c3c04462117704c29eba1b550d2f3fcac36576e222339bc4784c5e3635'
    assert sha((ROOT / 'rp1-eager-review/cpu-preflight.json').read_bytes()) == review['cpu_receipt_sha256']
    assert declaration['qualification_scope'] == 'historical_worker_diagnostic_not_ordinary_public_API'
    assert declaration['initial_optimizer_state'] == 'declared_eager'
    assert declaration['optimizer_step_devices'] == {'G': 'parameter', 'D': 'parameter'}
    assert cpu['declaration_sha256'] == review['declaration_sha256'] == digest
    assert cpu['status'] == cpu['cpu_initialization']['status'] == 'PASS'
    assert review['status'] == 'PASS_INITIALIZATION_ONLY'
    assert read(source / 'declaration.json') == declaration
    manifest = read(source / 'artifact-sha256.json')
    assert {str(p.relative_to(source)): sha(p.read_bytes()) for p in source.rglob('*')
            if p.is_file() and p.name != 'artifact-sha256.json'} == manifest
    seal = read(bundle / 'manifest.json')['files']
    with zipfile.ZipFile(source / 'source.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        required = set(read(source / 'protocol.json')['source_sha256']) | {
            'mode_hold_harness.py', 'mode_hold_contract.py', 'init_contract.py',
            'preflight.py', 'protocol.json', 'candidate-declaration.json',
            'reviewed_setup.py', 'historical_eager_setup.py'}
        assert required <= set(archive.namelist()), 'missing required harness or host source'
        assert {n: sha(archive.read(n)) for n in archive.namelist()
                if n.startswith('particlegan/')} == declaration['package_sha256']
        for name, expected in seal.items():
            if name in archive.namelist():
                assert sha(archive.read(name)) == expected
        assert sha(archive.read('candidate-declaration.json')) == digest
    proof = read(source / 'source-preflight.json')
    assert proof['status'] == 'PASS' and proof['declaration_sha256'] == digest
    assert proof['harness_sha256'] == {name: seal[name] for name in proof['harness_sha256']}
    runtime = read(source / 'runtime.json')
    assert (runtime['torch'], runtime['cuda'], runtime['gpu']) == ('2.13.0+cu126', '12.6', 'NVIDIA RTX A6000')
    assert runtime['learner_default_device'] == 'cpu' and runtime['serial_backward']
    assert runtime['deterministic'] and not runtime['tf32']
    assert runtime['threads'] == runtime['interop_threads'] == 1
    for item in runtime['imports'].values():
        assert item['sha256'] == declaration['package_sha256']['particlegan/' + Path(item['path']).name]
        assert item['sha256'] == sha(Path(item['path']).read_bytes())
    assert read(source / 'sampling-preflight.json') == dict(status='PASS', matched_batches=1200,
        actual_cuda_draws=True, optimizer_updates=0, training_stream_unchanged=True)
    expected_batches = [json.loads(line) for line in gzip.decompress(
        (ROOT / 'mode-hold-source/batch-receipts.jsonl.gz').read_bytes()).splitlines()]
    batches = [json.loads(line) for line in (source / 'batch-receipts.jsonl').read_text().splitlines()]
    assert batches == expected_batches
    rows = [json.loads(line) for line in (source / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(50, 1201, 50))
    summary = summarize(rows)
    result = read(source / 'result.json')
    assert result['metrics']['final'] == rows[-1]
    assert result['metrics']['updates'] == result['metrics']['matched_batch_receipts'] == 1200
    assert result['status'] == ('PASS' if summary['final_suffix'] >= 5 else 'FAIL')
    convergence = result['metrics']['verdict']['convergence']
    assert convergence['passing_observations'] == summary['passing']
    assert convergence['passing_suffix'] == summary['final_suffix']
    rates = [json.loads(line) for line in (source / 'learning-rates.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rates] == list(range(1, 1201))
    initial, clocks0 = checkpoint(source / 'initial-state.pt')
    final, clocksN = checkpoint(source / 'final-state.pt')
    assert initial['trainer']['completed_steps'] == 0 and final['trainer']['completed_steps'] == 1200
    identity = dict(candidate_declaration=digest, source_zip=manifest['source.zip'],
                    initializer_commit='c720645ecae6b648e9fc6034e9d6b48ccff06ed3')
    assert initial['identity'] == final['identity'] == identity
    assert initial['execution'] == final['execution'] == dict(serial_backward=True, scope='full public GANTrainer.step')
    assert initial['trainer']['recipe'] == final['trainer']['recipe'] == runtime['recipe']
    init = read(source / 'initial.json')
    assert init['old_sampling_rng_equal'] and not init['old_model_values_loaded']
    state = initial['trainer']
    assert {key: value for key, value in state.items() if key not in ('streams', 'cpu_rng', 'cuda_rng')} == init['material']['state']
    cpu_material = cpu['cpu_initialization']['all_initial_material']
    assert state['models'] == cpu_material['state']['models']
    assert init['material']['all_parameters_and_buffers'] == cpu_material['all_parameters_and_buffers']
    fixture = read(ROOT / 'mode-hold-source/fixture-receipt.json')
    for key in ('cpu_rng', 'cuda_rng'):
        assert state[key] == fixture['expected_initial'][key]
    for key, value in fixture['expected_initial']['streams'].items():
        assert state['streams'][key] == value
    assert initial['data_rng'] == fixture['expected_initial']['data_rng']
    assert final['data_rng'] == fixture['expected_final_data_rng']
    for index, role in enumerate(('G', 'D')):
        device = 'cuda:0' if declaration['optimizer_step_devices'][role] == 'parameter' else 'cpu'
        assert clocksN[index] and all(row == dict(step=1200., device=device) for row in clocksN[index])
        if declaration['initial_optimizer_state'] == 'native_lazy':
            assert not clocks0[index]
        else:
            assert clocks0[index] and all(row == dict(step=0., device=device) for row in clocks0[index])
    for step in (0, 1, 1200):
        proof = read(source / f'optimizer-device-proof-{step:04d}.json')
        for role, values in proof.items():
            for value in values:
                if step == 0 and declaration['initial_optimizer_state'] == 'native_lazy':
                    assert value['state_missing']
                else:
                    device = 'cuda:0' if declaration['optimizer_step_devices'][role] == 'parameter' else 'cpu'
                    assert value['step'] == step and value['step_device'] == device
                    assert value['parameter_device'] == value['exp_avg_device'] == value['exp_avg_sq_device'] == 'cuda:0'
    setup = audit_eager_setup(source, declaration, initial, final, rates)
    output = dict(status='PASS', scope='Independent stdlib source/receipts/raw-state audit; no training',
        candidate=declaration['candidate'], result=result['status'], summary=summary, source=str(source),
        harness_version='historical-rp1-eager-diagnostic', historical_setup=setup,
        declaration_sha256=digest, source_zip_sha256=manifest['source.zip'],
        artifacts_verified=len(manifest), sampling_rows_verified=1200, observations_verified=24,
        applied_diagnostic_rows_preserved=1200, initial_tensors_equal_cpu_preflight=True,
        initial_state_sha256=manifest['initial-state.pt'], final_state_sha256=manifest['final-state.pt'],
        optimizer_clocks=clocksN,
        limits=['Historical external eager setup diagnostic, not ordinary package-owned public API qualification',
                'Per-row precision diagnostics absent in this historical package; validates applied open/quiet scale set and final state, not transition replay',
                'Short screen does not establish long stability, API breadth, or default eligibility'])
    target = ROOT / 'rp1-cuda-eager-diagnostic-runtime-audit.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(candidate=declaration['candidate'], audit='PASS', result=result['status'], summary=summary)))



def audit_eager_setup(source, declaration, initial, final, rates):
    setup = read(source / 'historical-eager-setup.json')
    assert setup['scope'] == 'HISTORICAL_WORKER_EAGER_DIAGNOSTIC_NOT_PUBLIC_API'
    assert setup['setup_sha256'] == declaration['post_constructor_setup']['sha256']
    assert setup['non_optimizer_and_rng_before'] == setup['non_optimizer_and_rng_after']
    assert setup['counts'] == {'G': 9, 'D': 8} and setup['completed_steps'] == 0
    assert setup['default_device'] == 'cpu' and setup['only_declared_zero_optimizer_state_created']
    # Independently recompute the historical portable digest from raw storage.
    with zipfile.ZipFile(source / 'initial-state.pt') as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = name[:-8]
        envelope = Reader(io.BytesIO(archive.read(name))).load()
        state = dict(envelope['trainer']); del state['optimizers']
        digest = hashlib.sha256()
        def visit(value):
            if isinstance(value, Tensor):
                kind, key, device, count = value.storage
                width, dtype = {'FloatStorage': (4, 'torch.float32'), 'ByteStorage': (1, 'torch.uint8')}[kind]
                binary = archive.read(prefix + 'data/' + key)
                assert len(binary) == count * width
                stride = 1
                for length, step in reversed(list(zip(value.shape, value.stride))):
                    assert length <= 1 or step == stride
                    stride *= length
                binary = binary[value.offset * width:(value.offset + math.prod(value.shape)) * width]
                digest.update(json.dumps(['tensor', dtype, list(value.shape)]).encode())
                digest.update(binary)
            elif isinstance(value, dict):
                digest.update(b'dict[')
                for key in sorted(value, key=lambda x: (type(x).__name__, str(x))):
                    visit(key); visit(value[key])
                digest.update(b']')
            elif isinstance(value, (list, tuple)):
                digest.update(type(value).__name__.encode() + b'[')
                for item in value: visit(item)
                digest.update(b']')
            else:
                digest.update(json.dumps([type(value).__name__, value], allow_nan=False).encode())
        visit([state, envelope['data_rng']])
        assert digest.hexdigest() == setup['non_optimizer_and_rng_after']
    for expected_count, optimizer in zip((9, 8), initial['trainer']['optimizers']):
        assert len(optimizer['state']) == expected_count
        for value in optimizer['state'].values():
            assert set(value) == {'step', 'exp_avg', 'exp_avg_sq'}
            for tensor in value.values():
                assert tensor['dtype'] == 'torch.float32'
                assert tensor['sha256'] == sha(bytes(4 * math.prod(tensor['shape'])))
    recipe = declaration['resolved_recipe']
    assert recipe['total_steps'] is None and recipe['continuous_precision'] == 'rp1'
    states = []
    for row in rates:
        actual = [[g['lr'] for g in groups] for groups in row['applied_group_rates']]
        matches = []
        for opened in (False, True):
            gain = .2 if opened else 0.
            network, prior = .01 + .99 * gain, .05 + .95 * gain
            expected = [[recipe['lr'] * network, recipe['lr'] * recipe['prior_lr_mult'] * prior],
                        [recipe['lr'] * recipe['d_lr_mult'] * network]]
            if actual == expected: matches.append(opened)
        assert len(matches) == 1
        states.append(matches[0])
        step = row['step'] - 1
        assert row['input_noise'] == recipe['input_noise_std'] * max(0., 1. - step / recipe['input_noise_init_steps'])
        assert row['output_noise'] == recipe['output_noise_std'] * min(1., step / recipe['output_noise_init_steps'])
    assert initial['trainer']['precision']['state']['updates'] == 0
    assert final['trainer']['precision']['state']['updates'] == 1200
    return dict(receipt=setup, raw_non_optimizer_and_rng_digest_verified=True,
                raw_initial_zero_eager_states=17, applied_rate_noise_rows_verified=1200,
                open_scale_rows=sum(states), quiet_scale_rows=len(states) - sum(states),
                final_precision_state=final['trainer']['precision']['state'])


if __name__ == '__main__':
    main()
