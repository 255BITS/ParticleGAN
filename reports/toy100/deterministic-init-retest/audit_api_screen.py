"""Independently audit a completed API screen without Torch or learner imports."""
import argparse
import gzip
import json
from pathlib import Path
import zipfile

from audit_public3_runtime import checkpoint, read, sha
from archive_screen import summarize

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    port = ROOT / 'port-source' / args.candidate
    declaration = read(port / 'candidate-declaration.json')
    digest = sha((port / 'candidate-declaration.json').read_bytes())
    cpu = read(ROOT / f'{args.candidate}-cpu-preflight.json')
    review = read(ROOT / f'{args.candidate}-independent-init-audit.json')
    assert cpu['declaration_sha256'] == review['declaration_sha256'] == digest
    assert cpu['status'] == cpu['cpu_initialization']['status'] == 'PASS'
    assert review['status'] == 'PASS_INITIALIZATION_ONLY'
    assert read(source / 'declaration.json') == declaration
    manifest = read(source / 'artifact-sha256.json')
    assert {str(p.relative_to(source)): sha(p.read_bytes()) for p in source.rglob('*')
            if p.is_file() and p.name != 'artifact-sha256.json'} == manifest
    seal = read(ROOT / 'harness-sha256.json')['files']
    with zipfile.ZipFile(source / 'source.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        required = set(read(source / 'protocol.json')['source_sha256']) | {
            'mode_hold_harness.py', 'mode_hold_contract.py', 'init_contract.py',
            'preflight.py', 'protocol.json', 'candidate-declaration.json'}
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
    output = dict(status='PASS', scope='Independent stdlib source/receipts/raw-state audit; no training',
        candidate=declaration['candidate'], result=result['status'], summary=summary, source=str(source),
        declaration_sha256=digest, source_zip_sha256=manifest['source.zip'],
        artifacts_verified=len(manifest), sampling_rows_verified=1200, observations_verified=24,
        applied_diagnostic_rows_preserved=1200, initial_tensors_equal_cpu_preflight=True,
        initial_state_sha256=manifest['initial-state.pt'], final_state_sha256=manifest['final-state.pt'],
        optimizer_clocks=clocksN,
        limits=['Candidate-specific LR/controller diagnostics preserved; not independently replayed by this audit',
                'Short screen does not establish long stability, API breadth, or default eligibility'])
    target = ROOT / f'{args.candidate}-runtime-audit.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(candidate=declaration['candidate'], audit='PASS', result=result['status'], summary=summary)))


if __name__ == '__main__':
    main()
